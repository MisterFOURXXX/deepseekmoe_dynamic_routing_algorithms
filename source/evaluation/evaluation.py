import sys
sys.path.append("..")

import warnings
# Suppress external library warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*seen_tokens.*")

import math
import subprocess
import time

import numpy as np
import pandas as pd
import psutil
import torch
from torch.utils.data import DataLoader
from torch.utils.flop_counter import FlopCounterMode
from transformers import DataCollatorForLanguageModeling


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
    
    # Note: gen_max_new_tokens, gen_min_new_tokens, repetition_penalty, 
    # num_beams, do_sample are safely ignored here since they were only
    # used for generative QA (TriviaQA/AmbigQA).

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