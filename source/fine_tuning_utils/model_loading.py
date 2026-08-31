import os
import json
from transformers import AutoTokenizer

from deepseek_baseline.model import DeepseekForCausalLM as BaselineModel
from deepseek_dynamics_routing.model import DeepseekForCausalLM as RoutingModel
from DYNMoE_baseline.model import DynMoEForCausalLM as DynMoEModel


def load_model_and_tokenizer(model_path):
    config_path = os.path.join(model_path, "config.json")
    with open(config_path, "r") as f:
        config_dict = json.load(f)

    # Determine which model class to use
    if config_dict.get("model_type") == "dynmoe":
        ModelClass = DynMoEModel
    else:
        # model_type is "deepseek" – distinguish baseline from routing
        if "max_routed_experts" in config_dict:
            ModelClass = RoutingModel
        else:
            ModelClass = BaselineModel

    model = ModelClass.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer