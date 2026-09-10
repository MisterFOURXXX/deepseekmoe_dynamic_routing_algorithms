import sys
sys.path.append("..")

import os
import json
import torch
from transformers import AutoTokenizer

from deepseek_baseline.config import DeepseekConfig as BaselineConfig
from DYNMoE_baseline.config import DynMoEConfig as DYNMoEBaseConfig
from deepseek_dynamics_routing.config import DeepseekConfig as DynmoeConfig

from deepseek_baseline.model import DeepseekForCausalLM as BaselineModel
from deepseek_dynamics_routing.model import DeepseekForCausalLM as RoutingModel
from DYNMoE_baseline.model import DynMoEForCausalLM as DynMoEModel


def load_model_and_tokenizer(model_path):
    # Determine which config and model classes to use
    if "dynmoe" in model_path:
        ConfigClass = DYNMoEBaseConfig
        ModelClass = DynMoEModel
    elif "routing" in model_path:
        ConfigClass = DynmoeConfig
        ModelClass = RoutingModel
    else:
        ConfigClass = BaselineConfig
        ModelClass = BaselineModel

    config = ConfigClass()
    model = ModelClass(config)

    config_path = os.path.join(model_path, "config.json")
    with open(config_path, "r") as f:
        config_dict = json.load(f)

    # Determine which model class to instantiate
    if config_dict.get("model_type") == "dynmoe":
        ModelClass = DynMoEModel
    else:
        # model_type is "deepseek" – differentiate baseline from routing
        if "max_routed_experts" in config_dict:
            ModelClass = RoutingModel
        else:
            ModelClass = BaselineModel

    model = ModelClass.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Sanity check: print total parameters to confirm successful loading
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded model with {total_params:,} parameters.")
    if total_params == 0:
        raise RuntimeError("Model has zero parameters – loading likely failed!")

    return model, tokenizer