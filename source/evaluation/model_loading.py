import sys
sys.path.append("..")

import os
import json
from transformers import AutoTokenizer

from ..deepseek_baseline.config       import DeepseekConfig as BaselineConfig
from ..DYNMoE_baseline.config          import DynMoEConfig   as DYNMoEBaseConfig
from ..deepseek_dynamics_routing.config import DeepseekConfig as RoutingConfig

from ..deepseek_baseline.model         import DeepseekForCausalLM as BaselineModel
from ..DYNMoE_baseline.model           import DynMoEForCausalLM as DYNMoEModel
from ..deepseek_dynamics_routing.model import DeepseekForCausalLM as RoutingModel


_ROUTING_MARKERS = (
    "max_active_k",
    "min_active_k",
    "threshold_init",
    "router_bias_update_rate",
    "router_sync_interval",
)


def _resolve_classes(model_path: str):
    """
    Pick the correct (ConfigClass, ModelClass) pair for a saved checkpoint
    by inspecting its config.json.

    Mapping
    -------
    model_type == "dynmoe"                              -> DynMoE baseline (Phi-2 based)
    model_type == "deepseek" + any routing marker       -> DeepSeekMoE + DYNMoE Top-Any routing
    model_type == "deepseek" (no routing markers)       -> DeepSeekMoE baseline
    """
    config_path = os.path.join(model_path, "config.json")
    with open(config_path, "r") as f:
        cfg = json.load(f)

    model_type = cfg.get("model_type", "")

    if model_type == "dynmoe":
        return DYNMoEBaseConfig, DYNMoEModel

    if model_type == "deepseek":
        if any(k in cfg for k in _ROUTING_MARKERS):
            return RoutingConfig, RoutingModel
        return BaselineConfig, BaselineModel

    raise ValueError(
        f"Unknown model_type '{model_type}' in {config_path}. "
        f"Expected 'deepseek' or 'dynmoe'."
    )


def load_model_and_tokenizer(model_path):
    """
    Load a fine-tuned checkpoint for evaluation.
    """
    ConfigClass, ModelClass = _resolve_classes(model_path)

    # Build the architecture from the saved config, then load the weights.
    config = ConfigClass.from_pretrained(model_path)
    model  = ModelClass.from_pretrained(model_path, config=config)

    # Tokenizer lives in the same checkpoint directory and already defined
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

    return model, tokenizer