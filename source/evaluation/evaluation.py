import sys
sys.path.append("..")

import math
import re
import string
import subprocess
import time
from collections import Counter

import numpy as np
import pandas as pd
import psutil
import torch
from torch.utils.data import DataLoader
from torch.utils.flop_counter import FlopCounterMode
from transformers import DataCollatorForLanguageModeling

# Answer Normalisation Helpers (EM / F1)
def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        return "".join(ch for ch in text if ch not in string.punctuation)
    return white_space_fix(remove_articles(remove_punc(s.lower())))


def compute_exact_match(pred, golds):
    pred = normalize_answer(pred)
    return max(int(pred == normalize_answer(g)) for g in golds)


def compute_f1(pred, golds):
    pred_toks = normalize_answer(pred).split()
    best = 0.0
    for g in golds:
        gold_toks = normalize_answer(g).split()
        if not pred_toks or not gold_toks:
            continue
        common   = Counter(pred_toks) & Counter(gold_toks)
        num_same = sum(common.values())
        if num_same == 0:
            continue
        prec = num_same / len(pred_toks)
        rec  = num_same / len(gold_toks)
        best = max(best, 2 * prec * rec / (prec + rec))
    return best


# Resource Snapshot Helper
def take_resource_snapshot(resources):
    cpu_pct = psutil.cpu_percent(interval=None)
    sys_mem = psutil.virtual_memory().used / (1024 ** 3)
    try:
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        )
        lines = [l.strip() for l in smi.stdout.splitlines() if l.strip()]
        if lines:
            gpu_util = np.mean([float(l.split(",")[0]) for l in lines])
            gpu_mem  = np.mean([float(l.split(",")[1]) for l in lines]) / 1024
            resources.append({"cpu": cpu_pct, "gpu": gpu_util,
                              "gmem": gpu_mem, "smem": sys_mem})
            return
    except Exception:
        pass
    resources.append({"cpu": cpu_pct, "gpu": 0.0, "gmem": 0.0, "smem": sys_mem})


# Core Evaluation Engine
def evaluate_model(model, tokenizer, dataset_name, dataset, device, **kwargs):
    max_seq_len        = kwargs.get("max_seq_len", 512)
    eval_batch_size    = kwargs.get("eval_batch_size", 8)
    gen_max_new_tokens = kwargs.get("gen_max_new_tokens", 32)
    gen_min_new_tokens = kwargs.get("gen_min_new_tokens", 1)
    repetition_penalty = kwargs.get("repetition_penalty", 1.1)
    num_beams          = kwargs.get("num_beams", 1)
    do_sample          = kwargs.get("do_sample", False)

    model.eval()
    unwrapped = model.module if hasattr(model, "module") else model

    is_pure_dynmoe = (
        hasattr(unwrapped.config, "model_type")
        and unwrapped.config.model_type == "dynmoe"
    )

    is_routing_prototype = False
    if not is_pure_dynmoe:
        for module in unwrapped.modules():
            if hasattr(module, "update_biases") or hasattr(module, "thresholds"):
                is_routing_prototype = True
                break

    n_experts = (
        getattr(unwrapped.config, "n_routed_experts", None)
        or getattr(unwrapped.config, "num_experts", 8)
    )

    class ExpertHook:
        def __init__(self, n_exp):
            self.global_counts   = np.zeros(n_exp, dtype=np.float64)
            self.batch_vios      = []
            self.batch_counts    = None
            self.is_pure_dynmoe  = is_pure_dynmoe
            self.is_routing      = is_routing_prototype

        def __call__(self, module, inp, out):
            if self.is_pure_dynmoe:
                topk_weight = out[1]
                counts = (topk_weight > 1e-8).float().sum(dim=0).detach().cpu().numpy()
            elif self.is_routing:
                topk_weight = out[0]
                counts = (topk_weight > 1e-8).float().sum(dim=0).detach().cpu().numpy()
            else:
                topk_idx = out[0]
                if topk_idx.is_floating_point():
                    topk_idx = topk_idx.long()
                counts = torch.bincount(
                    topk_idx.flatten(),
                    minlength=getattr(module, "n_routed_experts", n_experts),
                ).cpu().numpy()

            if self.batch_counts is None:
                self.batch_counts = counts.copy()
            else:
                self.batch_counts += counts
            self.global_counts += counts

        def reset_batch(self):
            self.batch_counts = None

    hook_obj = ExpertHook(n_experts)
    hooks = []

    if hasattr(unwrapped, "model") and hasattr(unwrapped.model, "layers"):
        layers = unwrapped.model.layers
    elif (hasattr(unwrapped, "transformer")
          and hasattr(unwrapped.transformer, "decoder")
          and hasattr(unwrapped.transformer.decoder, "layers")):
        layers = unwrapped.transformer.decoder.layers
    elif hasattr(unwrapped, "layers"):
        layers = unwrapped.layers
    else:
        layers = None

    if layers is not None:
        for layer in layers:
            if hasattr(layer, "mlp") and hasattr(layer.mlp, "gate"):
                hooks.append(layer.mlp.gate.register_forward_hook(hook_obj))
    else:
        for module in unwrapped.modules():
            if hasattr(module, "n_routed_experts") and (
                hasattr(module, "gate_proj") or hasattr(module, "thresholds")
            ):
                hooks.append(module.register_forward_hook(hook_obj))

    num_moe_layers = len(hooks)
    print(f"Attached {num_moe_layers} expert-balance hooks")

    task_metrics = {
        "Perplexity":  "N/A",
        "Accuracy":    "N/A",
        "Exact Match": "N/A",
        "F1":          "N/A",
    }

    total_tokens   = 0
    resources      = []
    eval_start     = time.time()
    measured_flops = 0

    # Language Modeling PPL (SlimPajama / MiniPile)
    if dataset_name in ("slimpajama", "minipile"):
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8
        )
        loader = DataLoader(
            dataset, batch_size=eval_batch_size, collate_fn=data_collator,
            shuffle=False, num_workers=0, pin_memory=True,
        )
        total_loss = 0.0
        flop_counter = FlopCounterMode(display=False)
        with flop_counter:
            with torch.no_grad():
                for batch in loader:
                    hook_obj.reset_batch()
                    batch = {k: v.to(device) for k, v in batch.items()
                             if torch.is_tensor(v)}
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
                    total_loss   += loss.float().sum().item()
                    total_tokens += (shift_labels != -100).sum().item()

                    if hook_obj.batch_counts is not None:
                        tot = hook_obj.batch_counts.sum()
                        if tot > 0:
                            exp = tot / len(hook_obj.batch_counts)
                            hook_obj.batch_vios.append(
                                np.max(np.abs(hook_obj.batch_counts - exp)) / tot
                            )
                    take_resource_snapshot(resources)
        perplexity = (math.exp(total_loss / total_tokens)
                      if total_tokens > 0 else float("inf"))
        task_metrics["Perplexity"] = f"{perplexity:.2f}"
        measured_flops = flop_counter.get_total_flops()

    # Multiple-Choice Accuracy (ARC / RACE)
    elif dataset_name in ("arc", "race"):
        correct = 0
        total   = 0
        flop_counter = FlopCounterMode(display=False)
        with flop_counter:
            with torch.no_grad():
                for ex in dataset:
                    hook_obj.reset_batch()
                    prompt        = ex["prompt"]
                    choices       = ex["choices"]
                    choice_labels = ex["choice_labels"]
                    label         = ex["label"]

                    losses = []
                    for c in choices:
                        full = prompt + " " + c
                        inputs = tokenizer(
                            full, return_tensors="pt",
                            truncation=True, max_length=max_seq_len,
                        ).to(device)
                        prompt_len = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
                        labels_t = inputs["input_ids"].clone()
                        labels_t[:, :prompt_len] = -100
                        out = model(**inputs, labels=labels_t)
                        losses.append(out.loss.float().item())
                        total_tokens += (labels_t != -100).sum().item()

                    best_idx = int(np.argmin(losses))
                    if choice_labels[best_idx] == label:
                        correct += 1
                    total += 1

                    if hook_obj.batch_counts is not None:
                        tot = hook_obj.batch_counts.sum()
                        if tot > 0:
                            exp = tot / len(hook_obj.batch_counts)
                            hook_obj.batch_vios.append(
                                np.max(np.abs(hook_obj.batch_counts - exp)) / tot
                            )
                    take_resource_snapshot(resources)
        acc = correct / total if total else 0.0
        task_metrics["Accuracy"] = f"{acc:.4f}"
        measured_flops = flop_counter.get_total_flops()

    # Generative QA EM / F1 (TriviaQA / AmbigQA)
    elif dataset_name in ("triviaqa", "ambigqa", "nq"):
        em_scores, f1_scores = [], []
        flop_counter = FlopCounterMode(display=False)
        with flop_counter:
            with torch.no_grad():
                for i in range(0, len(dataset), eval_batch_size):
                    batch_ex = dataset[i : i + eval_batch_size]
                    if isinstance(batch_ex, dict):
                        prompts = batch_ex["prompt"]
                        golds   = batch_ex["answers"]
                    else:
                        prompts = [ex["prompt"]  for ex in batch_ex]
                        golds   = [ex["answers"] for ex in batch_ex]

                    _prev_padding_side = tokenizer.padding_side
                    tokenizer.padding_side = "left"
                    inputs = tokenizer(
                        prompts, return_tensors="pt", padding=True,
                        truncation=True, max_length=max_seq_len,
                    ).to(device)
                    tokenizer.padding_side = _prev_padding_side

                    hook_obj.reset_batch()

                    generated = unwrapped.generate(
                        **inputs,
                        max_new_tokens=gen_max_new_tokens,
                        min_new_tokens=gen_min_new_tokens,
                        do_sample=do_sample,
                        num_beams=num_beams,
                        repetition_penalty=repetition_penalty,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        use_cache=True,
                    )
                    new_tokens = generated[:, inputs["input_ids"].shape[1]:]
                    preds = [
                        p.strip()
                        for p in tokenizer.batch_decode(
                            new_tokens, skip_special_tokens=True
                        )
                    ]

                    for pred, gold_list in zip(preds, golds):
                        em_scores.append(compute_exact_match(pred, gold_list))
                        f1_scores.append(compute_f1(pred, gold_list))

                    total_tokens += new_tokens.ne(tokenizer.pad_token_id).sum().item()

                    if hook_obj.batch_counts is not None:
                        tot = hook_obj.batch_counts.sum()
                        if tot > 0:
                            exp = tot / len(hook_obj.batch_counts)
                            hook_obj.batch_vios.append(
                                np.max(np.abs(hook_obj.batch_counts - exp)) / tot
                            )
                    take_resource_snapshot(resources)

        task_metrics["Exact Match"] = f"{np.mean(em_scores):.4f}" if em_scores else "0.0000"
        task_metrics["F1"]          = f"{np.mean(f1_scores):.4f}" if f1_scores else "0.0000"
        measured_flops = flop_counter.get_total_flops()

    else:
        print(f"[evaluate_model] Unknown dataset_name='{dataset_name}' - skipping.")
        for h in hooks:
            h.remove()
        return None

    for h in hooks:
        h.remove()

    # Global MaxVIO
    gs = hook_obj.global_counts.sum()
    if gs > 0:
        exp = gs / len(hook_obj.global_counts)
        global_vio = np.max(np.abs(hook_obj.global_counts - exp)) / gs
    else:
        global_vio = 0.0

    avg_batch_vio = float(np.mean(hook_obj.batch_vios)) if hook_obj.batch_vios else 0.0
    min_batch_vio = float(np.min(hook_obj.batch_vios))  if hook_obj.batch_vios else 0.0
    max_batch_vio = float(np.max(hook_obj.batch_vios))  if hook_obj.batch_vios else 0.0
    avg_flops     = measured_flops / 1e9 if measured_flops else 0.0

    # Avg Activated Experts / Active Params
    if is_pure_dynmoe or is_routing_prototype:
        total_activations = hook_obj.global_counts.sum()
        avg_activated = (
            total_activations / (total_tokens * num_moe_layers)
            if (total_tokens > 0 and num_moe_layers > 0) else 0.0
        )
    else:
        avg_activated = getattr(unwrapped.config, "num_experts_per_tok", 2)
        if avg_activated is None:
            avg_activated = 2

    n_shared = getattr(unwrapped.config, "n_shared_experts", 2)
    if n_shared is None:
        n_shared = 2

    moe_inter     = getattr(unwrapped.config, "moe_intermediate_size", 1408)
    hidden_sz     = getattr(unwrapped.config, "hidden_size", 2048)
    expert_params = 3 * moe_inter * hidden_sz
    active_params = (n_shared + avg_activated) * expert_params

    total_time = time.time() - eval_start
    avg_tps    = total_tokens / total_time if total_time > 0 else 0.0
    avg_cpu    = float(np.mean([r["cpu"]  for r in resources])) if resources else 0.0
    avg_gpu    = float(np.mean([r["gpu"]  for r in resources])) if resources else 0.0
    avg_smem   = float(np.mean([r["smem"] for r in resources])) if resources else 0.0
    avg_gmem   = float(np.mean([r["gmem"] for r in resources])) if resources else 0.0

    results = {
        "Dataset":                              dataset_name,
        "Average TPS (inference)":              f"{avg_tps:.1f}",
        "Average CPU Usage (%)":                f"{avg_cpu:.1f}",
        "Average GPU Usage (%)":                f"{avg_gpu:.1f}",
        "Average System Memory (GB)":           f"{avg_smem:.2f}",
        "Average GPU Memory (GB)":              f"{avg_gmem:.2f}",
        "Average FLOPs (GFLOPS)":               f"{avg_flops:.1f}",
        "Global MaxVIO":                        f"{global_vio:.4f}",
        "Average Batch MaxVIO":                 f"{avg_batch_vio:.4f}",
        "Min Batch MaxVIO":                     f"{min_batch_vio:.4f}",
        "Max Batch MaxVIO":                     f"{max_batch_vio:.4f}",
        "Avg Activated Experts":                f"{avg_activated:.2f}",
        "Total Active Parameters (inference)":  f"{active_params:,.0f}",
        **task_metrics,
    }

    df = pd.DataFrame(list(results.items()), columns=["Metric", "Value"])
    print("\n" + "=" * 70)
    print(f"EVALUATION RESULTS - {dataset_name.upper()}")
    print("=" * 70)
    print(df.to_string(index=False))
    print("=" * 70)
    return results