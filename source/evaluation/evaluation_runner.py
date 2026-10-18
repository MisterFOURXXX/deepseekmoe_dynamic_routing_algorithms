import sys
sys.path.append("..")

import gc
import torch
import pandas as pd
from datasets import load_from_disk

from .config import DATASET_ROOT, EVAL_PARAMS
from .model_loading import load_model_and_tokenizer
from .evaluation import evaluate_model

def _p(name):
    return f"{DATASET_ROOT}/{name}"

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
        ("slimpajama", load_from_disk(_p("slimpajama_test"))),
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
        print("COMBINED BENCHMARK SUMMARY")
        print("=" * 80)
        print(pd.DataFrame(all_results).to_string(index=False))
        print("=" * 80)
    else:
        print("\nNo benchmarks were run.")