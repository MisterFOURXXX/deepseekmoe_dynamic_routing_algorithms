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

def evaluate_model(model, tokenizer, test_file, device, max_seq_len=256, eval_batch_size=8, gen_max_new_tokens=256, repetition_penalty=1.35):
    # Load prompts and references
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

    dataset = load_dataset("text", data_files={"test": test_file})["test"]
    tokenized = dataset.map(
        lambda ex: tokenizer(ex["text"], truncation=True, max_length=max_seq_len,
                             padding=False, return_attention_mask=True),
        batched=True,
        remove_columns=["text"],
        num_proc=2
    )
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8)
    loader = DataLoader(
        tokenized,
        batch_size=eval_batch_size,
        collate_fn=data_collator,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    class ExpertHook:
        def __init__(self, n_experts):
            self.global_counts = np.zeros(n_experts, dtype=np.float64)
            self.batch_vios = []
            self.batch_counts = None
            self.is_dynmoe = None
        def __call__(self, module, inp, out):
            if self.is_dynmoe is None:
                self.is_dynmoe = hasattr(module, 'update_biases')
            if self.is_dynmoe:
                weights = out[1]
                counts = (weights > 1e-8).float().sum(dim=0).detach().cpu().numpy()
            else:
                topk_idx = out[0]
                counts = torch.bincount(topk_idx.flatten(), minlength=module.n_routed_experts).cpu().numpy()
            if self.batch_counts is None:
                self.batch_counts = counts
            else:
                self.batch_counts += counts
            self.global_counts += counts
        def reset_batch(self):
            self.batch_counts = None

    hook_obj = ExpertHook(model.config.n_routed_experts)
    hooks = []
    for layer in model.model.layers:
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "gate"):
            hooks.append(layer.mlp.gate.register_forward_hook(hook_obj))

    total_loss = 0.0
    total_tokens = 0
    flop_counter = FlopCounterMode(mods=model, display=False)
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
                if hook_obj.batch_counts is not None:
                    tot = hook_obj.batch_counts.sum()
                    if tot > 0:
                        expected = tot / len(hook_obj.batch_counts)
                        batch_max = np.max(np.abs(hook_obj.batch_counts - expected) / expected)
                        hook_obj.batch_vios.append(batch_max)
    for h in hooks:
        h.remove()
    perplexity = math.exp(total_loss / total_tokens)
    gs = hook_obj.global_counts.sum()
    expected = gs / model.config.n_routed_experts
    global_vio = np.max(np.abs(hook_obj.global_counts - expected) / expected) if gs > 0 else 0.0
    avg_batch_vio = np.mean(hook_obj.batch_vios) if hook_obj.batch_vios else 0.0
    min_batch_vio = np.min(hook_obj.batch_vios) if hook_obj.batch_vios else 0.0
    max_batch_vio = np.max(hook_obj.batch_vios) if hook_obj.batch_vios else 0.0
    measured_flops = flop_counter.get_total_flops()
    avg_flops = measured_flops / 1e9

    n_shared = getattr(model.config, 'n_shared_experts', 2)
    n_routed_activated = getattr(model.config, 'num_experts_per_tok', 2)
    expert_size = model.config.moe_intermediate_size * model.config.hidden_size * 3
    total_active_params = (n_shared + n_routed_activated) * expert_size

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    smooth = SmoothingFunction().method4
    resources = []
    bleu_scores = []
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    gen_tokens_total = 0
    gen_start = time.time()
    for i in range(0, len(user_utterances), eval_batch_size):
        batch_u = user_utterances[i:i+eval_batch_size]
        batch_ref = references[i:i+eval_batch_size]
        prompts = [f"User: {u}\nSystem: " for u in batch_u]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            generated_ids = model.generate(
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
        cpu_pct = psutil.cpu_percent(interval=None)
        sys_mem_gb = psutil.virtual_memory().used / (1024**3)
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True
        )
        lines = [l.strip() for l in smi.stdout.splitlines() if l.strip()]
        gpu_util = np.mean([float(l.split(",")[0]) for l in lines]) if lines else 0.0
        gpu_mem_gb = np.mean([float(l.split(",")[1]) for l in lines]) / 1024 if lines else 0.0
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

    results = {
        "Average TPS (generation)": f"{avg_tps:.1f}",
        "Average CPU Usage (%)": f"{avg_cpu:.1f}",
        "Average GPU Usage (%)": f"{avg_gpu:.1f}",
        "Average System Memory (GB)": f"{avg_smem:.2f}",
        "Average GPU Memory (GB)": f"{avg_gmem:.2f}",
        "Average FLOPs (GFLOPS)": f"{avg_flops:.1f}",
        "Average perplexity": f"{perplexity:.2f}",
        "Average BLEU": f"{avg_bleu:.4f}",
        "Average ROUGE-1": f"{avg_rouge1:.4f}",
        "Average ROUGE-2": f"{avg_rouge2:.4f}",
        "Average ROUGE-L": f"{avg_rougeL:.4f}",
        "Average MaxVIO_global": f"{global_vio:.4f}",
        "Lowest MaxVIO_global": f"{global_vio:.4f}",
        "Highest MaxVIO_global": f"{global_vio:.4f}",
        "Average MaxVIO_batch": f"{avg_batch_vio:.4f}",
        "Lowest MaxVIO_batch": f"{min_batch_vio:.4f}",
        "Highest MaxVIO_batch": f"{max_batch_vio:.4f}",
        "Total Active Parameters (inference)": f"{total_active_params:,.0f}"
    }
    df = pd.DataFrame(list(results.items()), columns=["Metric", "Value"])
    print("\n" + "═" * 70)
    print("EVALUATION RESULTS SUMMARY")
    print("═" * 70)
    print(df.to_string(index=False))
    print("═" * 70)
    df.to_csv("evaluation_results.csv", index=False)
    return results