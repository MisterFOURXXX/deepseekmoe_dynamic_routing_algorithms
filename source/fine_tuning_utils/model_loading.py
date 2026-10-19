# ============================================================
# fine_tuning_utils/model_loading.py
# ============================================================
import sys
sys.path.append("..")

import os
import json
import torch
from transformers import AutoTokenizer
from transformers.generation.utils import GenerationMixin

from ..deepseek_baseline.model import DeepseekForCausalLM as BaselineModel
from ..deepseek_dynamics_routing.model import DeepseekForCausalLM as RoutingModel


def load_model_and_tokenizer(model_path):
    # 1. Load Tokenizer first
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Read config to determine ModelClass
    config_path = os.path.join(model_path, "config.json")
    with open(config_path, "r") as f:
        config_dict = json.load(f)

    # 3. Distinguish baseline from routing based on config markers
    if "max_active_k" in config_dict or "threshold_init" in config_dict or "router_bias_update_rate" in config_dict:
        ModelClass = RoutingModel
    else:
        ModelClass = BaselineModel

    # 4. Prepare Config
    config = ModelClass.config_class.from_pretrained(model_path)
    config.pad_token_id = tokenizer.pad_token_id
    config.bos_token_id = tokenizer.bos_token_id
    config.eos_token_id = tokenizer.eos_token_id
    # FIX: Do NOT set config.dtype or pass torch_dtype here. 
    # Let TrainingArguments(bf16=True) handle autocasting to prevent dtype mismatch.

    # 5. Patch GenerationMixin if missing (future-proof for transformers >= 4.50)
    if not issubclass(ModelClass, GenerationMixin):
        ModelClass = type(
            ModelClass.__name__,
            (ModelClass, GenerationMixin),
            {"__module__": ModelClass.__module__},
        )
        print(f"Patched {ModelClass.__name__} with GenerationMixin.")

    # 6. Load Model in default precision (fp32)
    model = ModelClass.from_pretrained(model_path, config=config)
    
    # 7. Resize token embeddings
    model.resize_token_embeddings(len(tokenizer))
    
    return model, tokenizer