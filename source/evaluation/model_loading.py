import os
import sys
repo_path =  ".."
os.chdir(repo_path)                 # Move into the repo
sys.path.insert(0, os.getcwd())     # Ensure the repo root is on sys.path

from transformers import AutoTokenizer
from source.deepseek_baseline.model import DeepseekForCausalLM

def load_model_and_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-moe-16b-base", use_fast=True, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = DeepseekForCausalLM.from_pretrained(model_path)
    return model, tokenizer