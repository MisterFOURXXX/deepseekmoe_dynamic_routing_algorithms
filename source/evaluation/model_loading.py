# ============================================================
# evaluation/model_loading.py
# ============================================================
import sys
sys.path.append("..")

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import json
import torch
from transformers import AutoTokenizer
from transformers.cache_utils import DynamicCache
from transformers.generation.utils import GenerationMixin

from ..deepseek_baseline.model         import DeepseekForCausalLM as BaselineModel
from ..deepseek_dynamics_routing.model import DeepseekForCausalLM as RoutingModel


# ============================================================
# TRANSFORMERS COMPATIBILITY SHIMS (DynamicCache)
# ============================================================
# The DeepSeek routing model's forward() still calls
#   past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
# which was removed from DynamicCache in newer transformers.
# Re-add it (and the helpers it depends on) so the model's
# native prepare_inputs_for_generation can pass a real
# DynamicCache through generate().
# ============================================================
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


_ROUTING_MARKERS = (
    "max_active_k",
    "min_active_k",
    "threshold_init",
    "router_bias_update_rate",
    "router_sync_interval",
)


def _resolve_classes(model_path: str):
    """
    Pick the correct ModelClass for a saved checkpoint by inspecting its config.json.
    """
    config_path = os.path.join(model_path, "config.json")
    with open(config_path, "r") as f:
        cfg = json.load(f)

    # Distinguish routing from baseline
    if any(k in cfg for k in _ROUTING_MARKERS):
        return RoutingModel
    return BaselineModel


def load_model_and_tokenizer(model_path):
    """
    Load a fine-tuned checkpoint for evaluation.

    NOTE: We do NOT patch prepare_inputs_for_generation for the
    routing model. It uses the modern DynamicCache API and its
    native implementation handles the Cache correctly. The
    DynamicCache shims above are enough to keep its forward()
    working across transformers versions.
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

    # Use torch_dtype for transformers 4.44.2 compatibility
    model = ModelClass.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    model.config.use_cache = True

    tok = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded {ModelClass.__name__} from {model_path} "
          f"({total_params:,} parameters, dtype={next(model.parameters()).dtype})")

    return model, tok