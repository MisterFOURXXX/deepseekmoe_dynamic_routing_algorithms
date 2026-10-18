import sys
sys.path.append("..")

import os
import json
import torch
from transformers import AutoTokenizer
from transformers.cache_utils import DynamicCache
from transformers.generation.utils import GenerationMixin

from ..deepseek_baseline.model         import DeepseekForCausalLM as BaselineModel
from ..DYNMoE_baseline.model           import DynMoEForCausalLM as DYNMoEModel
from ..deepseek_dynamics_routing.model import DeepseekForCausalLM as RoutingModel


# TRANSFORMERS COMPATIBILITY SHIMS (DynamicCache)
if not hasattr(DynamicCache, "seen_tokens"):
    DynamicCache.seen_tokens = property(lambda self: self.get_seq_length())

if not hasattr(DynamicCache, "get_max_length"):
    DynamicCache.get_max_length = lambda self: None

if not hasattr(DynamicCache, "get_usable_length"):
    def _get_usable_length(self, new_seq_length, layer_idx=0):
        max_length = self.get_max_length()
        prev = self.get_seq_length(layer_idx)
        if max_length is not None and prev + new_seq_length > max_length:
            return max_length - new_seq_length
        return prev
    DynamicCache.get_usable_length = _get_usable_length


def _patch_dynmoe_prepare_inputs(model_cls):
    """Convert modern DynamicCache to legacy (k, v) tuples for DynMoE."""
    if model_cls is None or getattr(model_cls, "_dynmoe_prepare_patched", False):
        return

    def _patched(self, input_ids, past_key_values=None,
                 attention_mask=None, **kwargs):
        if past_key_values is not None and hasattr(past_key_values, "to_legacy_cache"):
            past_key_values = past_key_values.to_legacy_cache()

        if past_key_values is not None and not isinstance(past_key_values, (tuple, list)):
            layers = []
            for layer in past_key_values:
                k = getattr(layer, "key", None)
                v = getattr(layer, "value", None)
                if k is None and isinstance(layer, (tuple, list)) and len(layer) >= 2:
                    k, v = layer[0], layer[1]
                layers.append((k, v))
            past_key_values = tuple(layers)

        if isinstance(past_key_values, (tuple, list)) and len(past_key_values) > 0:
            if all(isinstance(p, (tuple, list)) and len(p) == 2
                   and p[0] is None and p[1] is None for p in past_key_values):
                past_key_values = None

        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'past_key_values': past_key_values,
            'use_cache': True,
        }

    model_cls.prepare_inputs_for_generation = _patched
    model_cls._dynmoe_prepare_patched = True


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
    """
    config_path = os.path.join(model_path, "config.json")
    with open(config_path, "r") as f:
        cfg = json.load(f)

    model_type = cfg.get("model_type", "")

    if model_type == "dynmoe":
        return DYNMoEModel
    if model_type == "deepseek":
        if any(k in cfg for k in _ROUTING_MARKERS):
            return RoutingModel
        return BaselineModel

    raise ValueError(f"Unknown model_type '{model_type}' in {config_path}.")


def load_model_and_tokenizer(model_path):
    """
    Load a fine-tuned checkpoint for evaluation.
    """
    ModelClass = _resolve_classes(model_path)

    # Patch GenerationMixin if missing
    if not issubclass(ModelClass, GenerationMixin):
        ModelClass = type(
            ModelClass.__name__,
            (ModelClass, GenerationMixin),
            {"__module__": ModelClass.__module__},
        )
        print(f"Patched {ModelClass.__name__} with GenerationMixin.")

    # Patch DynMoE prepare_inputs if needed
    if "DynMoE" in ModelClass.__name__ or "dynmoe" in ModelClass.__name__.lower():
        _patch_dynmoe_prepare_inputs(ModelClass)

    model = ModelClass.from_pretrained(model_path, dtype=torch.bfloat16)
    model.config.use_cache = True

    tok = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded {ModelClass.__name__} from {model_path} "
          f"({total_params:,} parameters, dtype={next(model.parameters()).dtype})")
    
    return model, tok