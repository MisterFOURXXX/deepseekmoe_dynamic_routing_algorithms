import os
import sys
import gc
import torch
import math
from transformers import TrainerCallback

from deepseekmoe_dynamic_routing_algorithms.source.deepseek_dynamics_routing.config import (
    AUDIT_STEPS,
    PRUNE_THRESHOLD,
    MIN_ACTIVE_EXPERTS,
    BIAS_UPDATE_INTERVAL,
    CLEAR_CACHE_EVERY
)

# AdaptiveExpertTuningCallback: Trainer callback for expert pool resizing
class AdaptiveExpertTuningCallback(TrainerCallback):
    """
    Hugging Face Trainer callback that triggers adaptive expert tuning
    (pruning/addition) every `audit_steps` training steps.

    It calls adaptive_tune() on all gates, then sync_experts() on all MoE
    layers. If the expert pool was resized, the optimizer is re‑created to
    avoid parameter‑size mismatches (Section 3.4).

    Attributes:
        audit_steps: number of steps between audits
        global_step: current training step counter
        trainer: reference to the Trainer (set after creation)
    """
    def __init__(
        self,
        audit_steps: int = AUDIT_STEPS,
        prune_threshold: float = PRUNE_THRESHOLD,
        min_active_experts: int = MIN_ACTIVE_EXPERTS,
        bias_update_interval: int = BIAS_UPDATE_INTERVAL,
        clear_cache_every: int = CLEAR_CACHE_EVERY,
    ):
        self.audit_steps = audit_steps
        self.prune_threshold = prune_threshold
        self.min_active_experts = min_active_experts
        self.bias_update_interval = bias_update_interval
        self.clear_cache_every = clear_cache_every
    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step == 0 or model is None:
            return
        if state.global_step % self.bias_update_interval == 0:
            unwrapped = model.module if hasattr(model, "module") else model
            for module in unwrapped.modules():
                if hasattr(module, "update_loss_free_bias"):
                    module.update_loss_free_bias()
        if state.global_step % self.audit_steps == 0:
            unwrapped = model.module if hasattr(model, "module") else model
            self._audit_and_soft_prune(unwrapped)
        # Periodically clear cache to reduce fragmentation
        if state.global_step % self.clear_cache_every == 0:
            torch.cuda.empty_cache()
    @torch.no_grad()
    def _audit_and_soft_prune(self, model):
        for module in model.modules():
            if hasattr(module, "is_active") and hasattr(module, "routing_counts") and hasattr(module, "num_experts"):
                counts = module.routing_counts.float()
                total = counts.sum()
                if total == 0:
                    continue
                usage = counts / total
                active = module.is_active.sum().item()
                for i in range(module.num_experts):
                    if active <= self.min_active_experts:
                        break
                    if module.is_active[i] and usage[i] < self.prune_threshold:
                        module.is_active[i] = False
                        module.gate_proj.weight[i].zero_()
                        active -= 1