import sys
import os
import gc
import torch
import pandas as pd
from transformers import AutoTokenizer

from .evaluation import evaluate_model
from .model_loading import load_model_and_tokenizer
from .config import EVAL_PARAMS

# Device
torch.cuda.empty_cache()
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Cleanup helper
def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

# Helper: evaluate one model using the dedicated loader 
def run_evaluate(label, model_path, EVAL_FILE_PATH):
    print("=" * 60)
    print(f"EVALUATING: {label}")
    print(f"Model path: {model_path}")
    print("=" * 60)

    # Load model and tokenizer using the robust utility
    model, tokenizer = load_model_and_tokenizer(model_path)

    # Move to device
    model = model.to(device)
    model.eval()

    # Run evaluation
    evaluate_model_func(
        model=model,
        tokenizer=tokenizer,
        test_file=EVAL_FILE_PATH,     
        device=device,
        **EVAL_PARAMS
    )

    # Cleanup 
    del model
    del tokenizer
    clear_gpu_memory()