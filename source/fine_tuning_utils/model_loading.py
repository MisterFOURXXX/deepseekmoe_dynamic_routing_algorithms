import sys
sys.path.append("..")

import os
import json
import torch
from transformers import AutoTokenizer
from transformers.generation.utils import GenerationMixin

from ..deepseek_baseline.model import DeepseekForCausalLM as BaselineModel
from ..deepseek_dynamics_routing.model import DeepseekForCausalLM as RoutingModel
from ..DYNMoE_baseline.model import DynMoEForCausalLM as DynMoEModel


def load_model_and_tokenizer(model_path):
    # Load Tokenizer first
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Read config to determine ModelClass
    config_path = os.path.join(model_path, "config.json")
    with open(config_path, "r") as f:
        config_dict = json.load(f)

    if config_dict.get("model_type") == "dynmoe":
        ModelClass = DynMoEModel
    else:
        if "max_routed_experts" in config_dict:
            ModelClass = RoutingModel
        else:
            ModelClass = BaselineModel

    # Prepare Config
    config = ModelClass.config_class.from_pretrained(model_path)
    config.pad_token_id = tokenizer.pad_token_id
    config.bos_token_id = tokenizer.bos_token_id
    config.eos_token_id = tokenizer.eos_token_id
    config.dtype = torch.bfloat16

    # Patch GenerationMixin if missing 
    if not issubclass(ModelClass, GenerationMixin):
        ModelClass = type(
            ModelClass.__name__,
            (ModelClass, GenerationMixin),
            {"__module__": ModelClass.__module__},
        )
        print(f"Patched {ModelClass.__name__} with GenerationMixin.")

    # Load Model
    model = ModelClass.from_pretrained(model_path, config=config)
    
    # Resize token embeddings 
    model.resize_token_embeddings(len(tokenizer))
    
    return model, tokenizer