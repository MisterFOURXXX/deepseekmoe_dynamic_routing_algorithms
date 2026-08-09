import os
import json
from transformers import AutoTokenizer

# Use the same import paths as the rest of the codebase
from deepseekmoe_dynamic_routing_algorithms.source.deepseek_baseline.model import DeepseekForCausalLM as BaselineModel
from deepseekmoe_dynamic_routing_algorithms.source.deepseek_dynamics_routing.model import DeepseekForCausalLM as RoutingModel
from deepseekmoe_dynamic_routing_algorithms.source.DYNMoE_baseline.model import DynMoEForCausalLM as DynMoEModel

def load_model_and_tokenizer(model_path):
    """
    Automatically detect and load the correct model class and tokenizer
    from a saved checkpoint directory.

    Args:
        model_path (str): Path to the saved model (contains config.json).

    Returns:
        model: The loaded PyTorch model.
        tokenizer: The corresponding tokenizer.
    """
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