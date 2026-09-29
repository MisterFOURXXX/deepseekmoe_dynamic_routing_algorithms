import time
import subprocess
import math
import numpy as np
import pandas as pd
import psutil
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from torch.utils.flop_counter import FlopCounterMode


# ---------------------------------------------------------------------------
# Architecture helpers
# ---------------------------------------------------------------------------
def _find_transformer_layers(unwrapped):
    """
    Return the transformer layer list for any of the supported architectures.

      * HF-style:      unwrapped.model.layers                  (DeepSeek baseline + routing)
      * DynMoE custom: unwrapped.transformer.decoder.layers    (DYNMoE baseline)
      * Nested HF:     unwrapped.transformer.model.layers
      * Flat:          unwrapped.transformer.layers
      * Top-level:     unwrapped.layers
    """
    if hasattr(unwrapped, "model") and hasattr(unwrapped.model, "layers"):
        return unwrapped.model.layers

    if hasattr(unwrapped, "transformer"):
        t = unwrapped.transformer
        if hasattr(t, "decoder") and hasattr(t.decoder, "layers"):
            return t.decoder.layers
        if hasattr(t, "model") and hasattr(t.model, "layers"):
            return t.model.layers
        if hasattr(t, "layers"):
            return t.layers

    if hasattr(unwrapped, "layers"):
        return unwrapped.layers

    return None


def _collect_gates(unwrapped):
    """Collect every MoE gate module found on the model."""
    gates = []
    layers = _find_transformer_layers(unwrapped)
    if layers is not None:
        for layer in layers:
            mlp = getattr(layer, "mlp", None)
            if mlp is not None and hasattr(mlp, "gate"):
                gates.append(mlp.gate)

    if not gates:
        for name, module in unwrapped.named_modules():
            leaf = name.split(".")[-1]
            if leaf in ("gate", "router") and hasattr(module, "n_routed_experts"):
                gates.append(module)

    return gates


def _detect_architecture(unwrapped):
    """
    Return ``(is_pure_dynmoe, is_routing_prototype)``.

    Detection rules
    ---------------
    1. ``config.model_type == "dynmoe"`` OR any module named ``DynamicMoEGate``
       → pure DYNMoE baseline.
    2. Otherwise, any module that exposes ``update_loss_free_bias`` **or**
       ``thresholds`` → DeepSeekMoE + DYNMoE Top-Any routing prototype.
    3. Otherwise → plain DeepSeekMoE baseline.
    """
    is_pure_dynmoe = getattr(unwrapped.config, "model_type", "") == "dynmoe"

    if not is_pure_dynmoe:
        for module in unwrapped.modules():
            if type(module).__name__ == "DynamicMoEGate":
                is_pure_dynmoe = True
                break

    is_routing_prototype = False
    if not is_pure_dynmoe:
        for module in unwrapped.modules():
            # FusedMoEGate exposes ``thresholds`` and ``update_loss_free_bias``.
            # The DeepSeekMoE baseline gate exposes neither.
            if (hasattr(module, "update_loss_free_bias")
                    or hasattr(module, "thresholds")):
                is_routing_prototype = True
                break

    return is_pure_dynmoe, is_routing_prototype


# ---------------------------------------------------------------------------
# Expert-balance hook
# ---------------------------------------------------------------------------
class _ExpertHook:
    """
    Forward hook that extracts per-expert activation counts from a gate's
    output. Handles all three gate output conventions used by the three
    architectures under evaluation.
    """

    def __init__(self, n_exp, is_pure_dynmoe, is_routing, fallback_n_exp):
        self.global_counts = np.zeros(n_exp, dtype=np.float64)
        self.batch_vios = []
        self.batch_counts = None
        self.is_pure_dynmoe = is_pure_dynmoe
        self.is_routing = is_routing
        self.fallback_n_exp = fallback_n_exp

    # -- helpers --------------------------------------------------------
    @staticmethod
    def _counts_from_weights(w):
        """Count activations per expert from a weight tensor of shape [..., E]."""
        if w.dim() == 3:                       # [B, S, E]
            return (w > 1e-8).float().sum(dim=(0, 1)).detach().cpu().numpy()
        elif w.dim() == 2:                     # [N, E]
            return (w > 1e-8).float().sum(dim=0).detach().cpu().numpy()
        elif w.dim() == 1:                     # [E]
            return (w > 1e-8).float().detach().cpu().numpy()
        else:
            return (w > 1e-8).float().sum(dim=0).detach().cpu().numpy()

    @staticmethod
    def _align(a, b):
        """Return two numpy arrays padded to the same length."""
        if a.shape[0] == b.shape[0]:
            return a, b
        m = max(a.shape[0], b.shape[0])
        aa = np.zeros(m, dtype=a.dtype); aa[:a.shape[0]] = a
        bb = np.zeros(m, dtype=b.dtype); bb[:b.shape[0]] = b
        return aa, bb

    # -- hook -----------------------------------------------------------
    def __call__(self, module, inp, out):
        # 1. Extract the per-expert weight / index tensor.
        if self.is_pure_dynmoe:
            # DynamicMoEGate.forward → (topk_idx, topk_weight, aux_loss)
            w = out[1] if isinstance(out, tuple) else out
            counts = self._counts_from_weights(w)

        elif self.is_routing:
            # FusedMoEGate.forward → single tensor [B, S, E]
            # OR (topk_weight, aux_loss, token_counts)
            w = out[0] if isinstance(out, tuple) else out
            counts = self._counts_from_weights(w)

        else:
            # DeepSeekMoE baseline gate → (topk_idx, topk_weight, aux_loss)
            topk_idx = out[0] if isinstance(out, tuple) else out
            if topk_idx.is_floating_point():
                topk_idx = topk_idx.long()
            n_exp = getattr(module, "n_routed_experts", self.fallback_n_exp)
            counts = torch.bincount(
                topk_idx.flatten(), minlength=n_exp
            ).cpu().numpy()

        # 2. Accumulate into batch + global counters (aligned).
        if self.batch_counts is None:
            self.batch_counts = counts.astype(np.float64, copy=True)
        else:
            bc, cc = self._align(self.batch_counts, counts)
            self.batch_counts = bc + cc

        gc, cc = self._align(self.global_counts, counts)
        self.global_counts = gc + cc

    def reset_batch(self):
        self.batch_counts = None


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------
def evaluate_model(model, tokenizer, test_file, device, **kwargs):
    max_seq_len         = kwargs.get("max_seq_len", 256)
    eval_batch_size     = kwargs.get("eval_batch_size", 8)
    gen_max_new_tokens  = kwargs.get("gen_max_new_tokens", 256)
    repetition_penalty  = kwargs.get("repetition_penalty", 1.35)

    model.eval()
    unwrapped = model.module if hasattr(model, "module") else model

    is_pure_dynmoe, is_routing_prototype = _detect_architecture(unwrapped)

    n_experts = (
        getattr(unwrapped.config, "n_routed_experts", None)
        or getattr(unwrapped.config, "num_experts", 8)
    )

    # ------------------------------------------------------------------
    # Attach hooks to every MoE gate
    # ------------------------------------------------------------------
    hook_obj = _ExpertHook(
        n_exp=n_experts,
        is_pure_dynmoe=is_pure_dynmoe,
        is_routing=is_routing_prototype,
        fallback_n_exp=n_experts,
    )
    hooks = []
    for gate in _collect_gates(unwrapped):
        hooks.append(gate.register_forward_hook(hook_obj))

    num_moe_layers = len(hooks)
    print(f"Attached {num_moe_layers} expert-balance hooks "
          f"(pure_dynmoe={is_pure_dynmoe}, routing={is_routing_prototype})")
    if num_moe_layers == 0:
        print("[WARN] No MoE gates found — expert-balance metrics will be zero.")

    # ------------------------------------------------------------------
    # Load test data
    # ------------------------------------------------------------------
    user_utterances, references, current_user = [], [], None
    with open(test_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("User: "):
                current_user = line[6:].strip()
            elif line.startswith("System: ") and current_user:
                references.append(line[8:].strip())
                user_utterances.append(current_user)
                current_user = None

    print(f"Loaded {len(user_utterances)} generation samples")

    dataset = load_dataset("text", data_files={"test": test_file})["test"]
    tokenized = dataset.map(
        lambda ex: tokenizer(
            ex["text"], truncation=True, max_length=max_seq_len,
            padding=False, return_attention_mask=True,
        ),
        batched=True,
        remove_columns=["text"],
        num_proc=2,
    )
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8
    )
    loader = DataLoader(
        tokenized,
        batch_size=eval_batch_size,
        collate_fn=data_collator,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # ------------------------------------------------------------------
    # Perplexity, MaxVIO, FLOPs
    # ------------------------------------------------------------------
    total_loss, total_tokens = 0.0, 0
    flop_counter = FlopCounterMode(display=False)

    with flop_counter, torch.no_grad():
        for batch in loader:
            hook_obj.reset_batch()
            batch = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
            outputs = model(**batch)
            logits  = outputs.logits
            labels  = batch["labels"]

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="none",
                ignore_index=-100,
            )
            total_loss   += loss.sum().item()
            total_tokens += (shift_labels != -100).sum().item()

            if hook_obj.batch_counts is not None:
                tot = hook_obj.batch_counts.sum()
                if tot > 0:
                    expected = tot / hook_obj.batch_counts.shape[0]
                    hook_obj.batch_vios.append(
                        np.max(np.abs(hook_obj.batch_counts - expected)) / tot
                    )

    for h in hooks:
        h.remove()

    perplexity = math.exp(total_loss / total_tokens) if total_tokens > 0 else float("inf")

    # ------------------------------------------------------------------
    # Global MaxVIO
    # ------------------------------------------------------------------
    gs = hook_obj.global_counts.sum()
    if gs > 0:
        expected   = gs / hook_obj.global_counts.shape[0]
        global_vio = np.max(np.abs(hook_obj.global_counts - expected)) / gs
    else:
        global_vio = 0.0

    avg_batch_vio = np.mean(hook_obj.batch_vios) if hook_obj.batch_vios else 0.0
    min_batch_vio = np.min(hook_obj.batch_vios)  if hook_obj.batch_vios else 0.0
    max_batch_vio = np.max(hook_obj.batch_vios)  if hook_obj.batch_vios else 0.0

    measured_flops = flop_counter.get_total_flops()
    avg_flops      = measured_flops / 1e9 if measured_flops else 0.0

    # ------------------------------------------------------------------
    # Active parameters + average activated experts
    # ------------------------------------------------------------------
    if (is_pure_dynmoe or is_routing_prototype) and num_moe_layers > 0 and total_tokens > 0:
        total_activations = hook_obj.global_counts.sum()
        avg_activated = total_activations / (total_tokens * num_moe_layers)
    else:
        avg_activated = getattr(unwrapped.config, "num_experts_per_tok", 2) or 2

    n_shared = getattr(unwrapped.config, "n_shared_experts", 2) or 2
    expert_params = 3 * unwrapped.config.moe_intermediate_size * unwrapped.config.hidden_size
    active_params = (n_shared + avg_activated) * expert_params

    # ------------------------------------------------------------------
    # Generation + quality metrics
    # ------------------------------------------------------------------
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    smooth = SmoothingFunction().method4

    resources = []
    bleu_scores, rouge1_scores, rouge2_scores, rougeL_scores = [], [], [], []
    gen_tokens_total = 0
    gen_start = time.time()

    print("Generating responses...")
    for i in range(0, len(user_utterances), eval_batch_size):
        batch_u   = user_utterances[i:i + eval_batch_size]
        batch_ref = references[i:i + eval_batch_size]
        prompts   = [f"User: {u}\nSystem: " for u in batch_u]

        inputs = tokenizer(
            prompts, return_tensors="pt", padding=True,
            truncation=True, max_length=max_seq_len,
        ).to(device)

        with torch.no_grad():
            generated_ids = unwrapped.generate(
                **inputs,
                max_new_tokens=gen_max_new_tokens,
                do_sample=False,
                repetition_penalty=repetition_penalty,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        new_tokens = generated_ids[:, inputs["input_ids"].shape[1]:]
        gen_tokens_total += new_tokens.ne(tokenizer.pad_token_id).sum().item()
        responses = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

        for res, ref in zip(responses, batch_ref):
            res, ref = res.strip(), ref.strip()
            bleu_scores.append(
                sentence_bleu([ref.split()], res.split(), smoothing_function=smooth)
            )
            scores = scorer.score(ref, res)
            rouge1_scores.append(scores["rouge1"].fmeasure)
            rouge2_scores.append(scores["rouge2"].fmeasure)
            rougeL_scores.append(scores["rougeL"].fmeasure)

        cpu_pct    = psutil.cpu_percent(interval=None)
        sys_mem_gb = psutil.virtual_memory().used / (1024 ** 3)
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True,
        )
        lines = [l.strip() for l in smi.stdout.splitlines() if l.strip()]
        if lines:
            gpu_util   = np.mean([float(l.split(",")[0]) for l in lines])
            gpu_mem_gb = np.mean([float(l.split(",")[1]) for l in lines]) / 1024
            resources.append(
                {"cpu": cpu_pct, "gpu": gpu_util,
                 "gmem": gpu_mem_gb, "smem": sys_mem_gb}
            )

    gen_time = time.time() - gen_start
    avg_tps  = gen_tokens_total / gen_time if gen_time > 0 else 0

    avg_cpu   = np.mean([r["cpu"]  for r in resources]) if resources else 0
    avg_gpu   = np.mean([r["gpu"]  for r in resources]) if resources else 0
    avg_smem  = np.mean([r["smem"] for r in resources]) if resources else 0
    avg_gmem  = np.mean([r["gmem"] for r in resources]) if resources else 0
    avg_bleu   = np.mean(bleu_scores)  if bleu_scores  else 0
    avg_rouge1 = np.mean(rouge1_scores) if rouge1_scores else 0
    avg_rouge2 = np.mean(rouge2_scores) if rouge2_scores else 0
    avg_rougeL = np.mean(rougeL_scores) if rougeL_scores else 0

    results = {
        "Average TPS (generation)":           f"{avg_tps:.1f}",
        "Average CPU Usage (%)":              f"{avg_cpu:.1f}",
        "Average GPU Usage (%)":              f"{avg_gpu:.1f}",
        "Average System Memory (GB)":         f"{avg_smem:.2f}",
        "Average GPU Memory (GB)":            f"{avg_gmem:.2f}",
        "Average FLOPs (GFLOPS)":             f"{avg_flops:.1f}",
        "Perplexity":                         f"{perplexity:.2f}",
        "Average BLEU":                       f"{avg_bleu:.4f}",
        "Average ROUGE-1":                    f"{avg_rouge1:.4f}",
        "Average ROUGE-2":                    f"{avg_rouge2:.4f}",
        "Average ROUGE-L":                    f"{avg_rougeL:.4f}",
        "Global MaxVIO":                      f"{global_vio:.4f}",
        "Average Batch MaxVIO":               f"{avg_batch_vio:.4f}",
        "Min Batch MaxVIO":                   f"{min_batch_vio:.4f}",
        "Max Batch MaxVIO":                   f"{max_batch_vio:.4f}",
        "Avg Activated Experts":              f"{avg_activated:.2f}",
        "Total Active Parameters (inference)": f"{active_params:,.0f}",
    }

    df = pd.DataFrame(list(results.items()), columns=["Metric", "Value"])
    print("\n" + "═" * 70)
    print("EVALUATION RESULTS SUMMARY")
    print("═" * 70)
    print(df.to_string(index=False))
    print("═" * 70)

    return results