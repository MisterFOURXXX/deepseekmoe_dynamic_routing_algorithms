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


def evaluate_model(model, tokenizer, test_file, device, **kwargs):
    """
    Unified evaluation for all three architectures:
      - Baseline DeepSeekMoE
      - Pure DYNMoE
      - Prototype (DeepSeekMoE + DYNMoE routing)
    """
    # Default parameters
    max_seq_len = kwargs.get('max_seq_len', 256)
    eval_batch_size = kwargs.get('eval_batch_size', 8)
    gen_max_new_tokens = kwargs.get('gen_max_new_tokens', 256)
    repetition_penalty = kwargs.get('repetition_penalty', 1.35)

    model.eval()
    unwrapped = model.module if hasattr(model, 'module') else model

    # ------------------------------------------------------------
    # 1. Determine architecture type
    # ------------------------------------------------------------
    is_pure_dynmoe = hasattr(unwrapped.config, 'model_type') and unwrapped.config.model_type == "dynmoe"
    is_routing_prototype = False
    if not is_pure_dynmoe:
        # Check for the custom MoEGate used in the prototype (has update_biases)
        for module in unwrapped.modules():
            if hasattr(module, 'update_biases') and hasattr(module, 'thresholds'):
                is_routing_prototype = True
                break
    # For baseline, both flags remain False

    # Determine number of experts (works for all)
    n_experts = getattr(unwrapped.config, 'n_routed_experts', None) or getattr(unwrapped.config, 'num_experts', 8)

    # ------------------------------------------------------------
    # 2. Expert‑balance hook (handles all gate types)
    # ------------------------------------------------------------
    class ExpertHook:
        def __init__(self, n_exp):
            self.global_counts = np.zeros(n_exp, dtype=np.float64)
            self.batch_vios = []
            self.batch_counts = None
            self.is_pure_dynmoe = is_pure_dynmoe
            self.is_routing = is_routing_prototype

        def __call__(self, module, inp, out):
            # out is the tuple returned by the gate's forward
            if self.is_pure_dynmoe:
                # Pure DYNMoE gate returns (topk_idx, topk_weight, aux_loss)
                topk_weight = out[1]      # shape [B*S, K]
                counts = (topk_weight > 1e-8).float().sum(dim=0).detach().cpu().numpy()
            elif self.is_routing:
                # Routing prototype gate returns (topk_weight, aux_loss, token_counts)
                topk_weight = out[0]
                counts = (topk_weight > 1e-8).float().sum(dim=0).detach().cpu().numpy()
            else:
                # Baseline DeepSeekMoE gate returns (topk_idx, topk_weight, aux_loss)
                topk_idx = out[0]
                # In eval, topk_idx is a tensor of indices; flatten and bincount
                counts = torch.bincount(topk_idx.flatten(), minlength=module.n_routed_experts).cpu().numpy()

            if self.batch_counts is None:
                self.batch_counts = counts
            else:
                self.batch_counts += counts
            self.global_counts += counts

        def reset_batch(self):
            self.batch_counts = None

    hook_obj = ExpertHook(n_experts)
    hooks = []

    # Attach hooks to all gate modules
    for module in unwrapped.modules():
        if is_pure_dynmoe and module.__class__.__name__ == "DynamicMoEGate":
            hooks.append(module.register_forward_hook(hook_obj))
        elif is_routing_prototype and hasattr(module, 'update_biases') and hasattr(module, 'thresholds'):
            # This is the custom MoEGate in the prototype
            hooks.append(module.register_forward_hook(hook_obj))
        elif not is_pure_dynmoe and not is_routing_prototype and module.__class__.__name__ == "MoEGate":
            # Baseline DeepSeekMoE gate
            hooks.append(module.register_forward_hook(hook_obj))

    # ------------------------------------------------------------
    # 3. Count MoE layers from attached hooks (works for all)
    # ------------------------------------------------------------
    num_moe_layers = len(hooks)
    print(f"Attached {num_moe_layers} expert‑balance hooks")

    # ------------------------------------------------------------
    # 4. Load test data and prepare dataloader
    # ------------------------------------------------------------
    # Read user utterances and references
    user_utterances = []
    references = []
    current_user = None
    with open(test_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("User: "):
                current_user = line[6:].strip()
            elif line.startswith("System: ") and current_user:
                ref = line[8:].strip()
                user_utterances.append(current_user)
                references.append(ref)
                current_user = None

    print(f"Loaded {len(user_utterances)} generation samples")

    # Dataset for perplexity
    dataset = load_dataset("text", data_files={"test": test_file})["test"]
    tokenized = dataset.map(
        lambda ex: tokenizer(ex["text"], truncation=True, max_length=max_seq_len,
                             padding=False, return_attention_mask=True),
        batched=True,
        remove_columns=["text"],
        num_proc=2
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
        pin_memory=True
    )

    # ------------------------------------------------------------
    # 5. Perplexity, MaxVIO, FLOPs
    # ------------------------------------------------------------
    total_loss = 0.0
    total_tokens = 0
    flop_counter = FlopCounterMode(display=False)

    with flop_counter:
        with torch.no_grad():
            for batch in loader:
                hook_obj.reset_batch()
                batch = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
                outputs = model(**batch)
                logits = outputs.logits
                labels = batch["labels"]
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = torch.nn.functional.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    reduction="none",
                    ignore_index=-100
                )
                total_loss += loss.sum().item()
                total_tokens += (shift_labels != -100).sum().item()

                # Batch‑level MaxVIO
                if hook_obj.batch_counts is not None:
                    tot = hook_obj.batch_counts.sum()
                    if tot > 0:
                        n_exp = len(hook_obj.batch_counts)
                        expected = tot / n_exp
                        # Use bounded MaxVIO: max(|diff|) / total
                        batch_max = np.max(np.abs(hook_obj.batch_counts - expected)) / tot
                        hook_obj.batch_vios.append(batch_max)

    # Remove hooks
    for h in hooks:
        h.remove()

    perplexity = math.exp(total_loss / total_tokens) if total_tokens > 0 else float('inf')

    # Global MaxVIO
    gs = hook_obj.global_counts.sum()
    if gs > 0:
        n_exp = len(hook_obj.global_counts)
        expected = gs / n_exp
        global_vio = np.max(np.abs(hook_obj.global_counts - expected)) / gs
    else:
        global_vio = 0.0

    avg_batch_vio = np.mean(hook_obj.batch_vios) if hook_obj.batch_vios else 0.0
    min_batch_vio = np.min(hook_obj.batch_vios) if hook_obj.batch_vios else 0.0
    max_batch_vio = np.max(hook_obj.batch_vios) if hook_obj.batch_vios else 0.0

    measured_flops = flop_counter.get_total_flops()
    avg_flops = measured_flops / 1e9 if measured_flops else 0.0

    # ------------------------------------------------------------
    # 6. Active parameters & average activated experts
    # ------------------------------------------------------------
    if is_pure_dynmoe or is_routing_prototype:
        # For DYNMoE variants, compute average activated experts per token
        total_activations = hook_obj.global_counts.sum()
        # Use the number of MoE layers from hooks
        avg_activated = total_activations / (total_tokens * num_moe_layers) if total_tokens > 0 else 0.0
        expert_params = 3 * unwrapped.config.moe_intermediate_size * unwrapped.config.hidden_size
        active_params = num_moe_layers * avg_activated * expert_params
    else:
        # Baseline: fixed top‑k
        avg_activated = getattr(unwrapped.config, 'num_experts_per_tok', 2)
        expert_params = 3 * unwrapped.config.moe_intermediate_size * unwrapped.config.hidden_size
        n_shared = getattr(unwrapped.config, 'n_shared_experts', 0)
        active_params = num_moe_layers * (n_shared + avg_activated) * expert_params
        avg_activated = None  # Not applicable for fixed top‑k

    # ------------------------------------------------------------
    # 7. Generation and quality metrics (ROUGE, BLEU)
    # ------------------------------------------------------------
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    smooth = SmoothingFunction().method4

    resources = []
    bleu_scores = []
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    gen_tokens_total = 0
    gen_start = time.time()

    print("Generating responses...")
    for i in range(0, len(user_utterances), eval_batch_size):
        batch_u = user_utterances[i:i+eval_batch_size]
        batch_ref = references[i:i+eval_batch_size]
        prompts = [f"User: {u}\nSystem: " for u in batch_u]

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_len
        ).to(device)

        with torch.no_grad():
            generated_ids = unwrapped.generate(
                **inputs,
                max_new_tokens=gen_max_new_tokens,
                do_sample=False,
                repetition_penalty=repetition_penalty,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        new_tokens = generated_ids[:, inputs['input_ids'].shape[1]:]
        gen_tokens_total += new_tokens.ne(tokenizer.pad_token_id).sum().item()
        responses = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

        for res, ref in zip(responses, batch_ref):
            res = res.strip()
            ref = ref.strip()
            bleu_scores.append(sentence_bleu([ref.split()], res.split(), smoothing_function=smooth))
            scores = scorer.score(ref, res)
            rouge1_scores.append(scores["rouge1"].fmeasure)
            rouge2_scores.append(scores["rouge2"].fmeasure)
            rougeL_scores.append(scores["rougeL"].fmeasure)

        # Resource monitoring
        cpu_pct = psutil.cpu_percent(interval=None)
        sys_mem_gb = psutil.virtual_memory().used / (1024 ** 3)

        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True
        )
        lines = [l.strip() for l in smi.stdout.splitlines() if l.strip()]
        if lines:
            gpu_util = np.mean([float(l.split(",")[0]) for l in lines])
            gpu_mem_gb = np.mean([float(l.split(",")[1]) for l in lines]) / 1024
            resources.append({"cpu": cpu_pct, "gpu": gpu_util, "gmem": gpu_mem_gb, "smem": sys_mem_gb})

    gen_time = time.time() - gen_start
    avg_tps = gen_tokens_total / gen_time if gen_time > 0 else 0

    avg_cpu = np.mean([r["cpu"] for r in resources]) if resources else 0
    avg_gpu = np.mean([r["gpu"] for r in resources]) if resources else 0
    avg_smem = np.mean([r["smem"] for r in resources]) if resources else 0
    avg_gmem = np.mean([r["gmem"] for r in resources]) if resources else 0

    avg_bleu = np.mean(bleu_scores) if bleu_scores else 0
    avg_rouge1 = np.mean(rouge1_scores) if rouge1_scores else 0
    avg_rouge2 = np.mean(rouge2_scores) if rouge2_scores else 0
    avg_rougeL = np.mean(rougeL_scores) if rougeL_scores else 0

    # ------------------------------------------------------------
    # 8. Assemble results
    # ------------------------------------------------------------
    results = {
        "Average TPS (generation)": f"{avg_tps:.1f}",
        "Average CPU Usage (%)": f"{avg_cpu:.1f}",
        "Average GPU Usage (%)": f"{avg_gpu:.1f}",
        "Average System Memory (GB)": f"{avg_smem:.2f}",
        "Average GPU Memory (GB)": f"{avg_gmem:.2f}",
        "Average FLOPs (GFLOPS)": f"{avg_flops:.1f}",
        "Perplexity": f"{perplexity:.2f}",
        "Average BLEU": f"{avg_bleu:.4f}",
        "Average ROUGE-1": f"{avg_rouge1:.4f}",
        "Average ROUGE-2": f"{avg_rouge2:.4f}",
        "Average ROUGE-L": f"{avg_rougeL:.4f}",
        "Global MaxVIO": f"{global_vio:.4f}",
        "Average Batch MaxVIO": f"{avg_batch_vio:.4f}",
        "Min Batch MaxVIO": f"{min_batch_vio:.4f}",
        "Max Batch MaxVIO": f"{max_batch_vio:.4f}",
        "Avg Activated Experts": f"{avg_activated:.2f}" if avg_activated is not None else "N/A",
        "Total Active Parameters (inference)": f"{active_params:,.0f}"
    }

    # Print summary
    df = pd.DataFrame(list(results.items()), columns=["Metric", "Value"])
    print("\n" + "═" * 70)
    print("EVALUATION RESULTS SUMMARY")
    print("═" * 70)
    print(df.to_string(index=False))
    print("═" * 70)

    return results