# ============================================================
# evaluation/evaluation_runner.py
# ============================================================
import sys
sys.path.append("..")

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import os
import gc
import torch
import pandas as pd
from datasets import load_from_disk

from .config import (
    DATASET_ROOT, EVAL_PARAMS,
    FINAL_FT_BASELINE, FINAL_FT_DYNMOE_ROUTING
)
from .model_loading import load_model_and_tokenizer
from .evaluation import evaluate_model

os.environ["HF_DATASETS_DISABLE_DILL"]     = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"]       = "false"
os.environ["CUBLAS_WORKSPACE_CONFIG"]      = ":4096:8"

def _p(name):
    return os.path.join(DATASET_ROOT, name)


def clear_gpu_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def run_evaluate(model_path, label="Fine-tuned Model"):
    print("=" * 60)
    print(f"EVALUATING: {label}")
    print(f"Model path: {model_path}")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load benchmarks from disk
    print("Loading benchmark datasets from disk...")
    datasets_to_run = [
        ("minipile",   load_from_disk(_p("minipile"))),
        ("arc",        load_from_disk(_p("arc"))),
        ("race",       load_from_disk(_p("race"))),
        ("triviaqa",   load_from_disk(_p("triviaqa"))),
        ("ambigqa",    load_from_disk(_p("ambigqa"))),
    ]

    for name, ds in datasets_to_run:
        print(f"  {name:12s} : {len(ds):,} samples")

    # Load Model and Tokenizer
    model, tokenizer = load_model_and_tokenizer(model_path)
    model = model.to(device)
    model.eval()

    # Run Evaluation
    all_results = []
    for name, ds in datasets_to_run:
        res = evaluate_model(
            model=model,
            tokenizer=tokenizer,
            dataset_name=name,
            dataset=ds,
            device=device,
            **EVAL_PARAMS,
        )
        if res is not None:
            all_results.append(res)

    # Cleanup
    del model
    del tokenizer
    clear_gpu_memory()

    # Print combined summary
    if all_results:
        print("\n" + "=" * 80)
        print(f"COMBINED BENCHMARK SUMMARY: {label}")
        print("=" * 80)
        print(pd.DataFrame(all_results).to_string(index=False))
        print("=" * 80)
    else:
        print("\nNo benchmarks were run.")


def evaluate_all_finetuned_models():
    """
    Automatically evaluates the two fine-tuned models saved in the checkpoints folder.
    """
    print("\n" + "#" * 80)
    print("# RUNNING ALL FINE-TUNED MODEL EVALUATIONS")
    print("#" * 80)
    
    # 1. Baseline
    run_evaluate(FINAL_FT_BASELINE, label="Baseline Fine-Tuned")
    
    # 2. DeepSeek Dynamics Routing
    run_evaluate(FINAL_FT_DYNMOE_ROUTING, label="DYNMoE Routing Fine-Tuned")