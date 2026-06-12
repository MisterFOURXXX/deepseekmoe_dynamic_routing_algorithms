from transformers import AutoTokenizer
from source.deepseek_baseline.model import DeepseekForCausalLM

def load_model_and_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-moe-16b-base", use_fast=True, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = DeepseekForCausalLM.from_pretrained(model_path)
    return model, tokenizer