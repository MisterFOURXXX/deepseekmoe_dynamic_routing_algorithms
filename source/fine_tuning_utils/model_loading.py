import os
import sys
repo_path =  ".."
os.chdir(repo_path)                 # Move into the repo
sys.path.insert(0, os.getcwd())     # Ensure the repo root is on sys.path

import os
import json
from transformers import AutoTokenizer
from deepseekmoe_dynamic_routing_algorithms.source.deepseek_baseline.model import DeepseekForCausalLM as BaselineModel
from deepseekmoe_dynamic_routing_algorithms.source.deepseek_dynamics_routing.model import DeepseekForCausalLM as RoutingModel
from deepseekmoe_dynamic_routing_algorithms.source.DYNMoE_baseline.model import DynMoEForCausalLM as DynMoEModel

def load_model_and_tokenizer(model_path):
    """
    Load the correct model class and its tokenizer from a saved checkpoint.
    """
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