import sys
import os
import gc
import torch
import math
from transformers import TrainerCallback

from .config import (
    AUDIT_STEPS,
    PRUNE_THRESHOLD,
    MIN_ACTIVE_EXPERTS,
    BIAS_UPDATE_INTERVAL,
    CLEAR_CACHE_EVERY,
    WARMUP_STEPS
)

# AdaptiveExpertTuningCallback: Trainer callback for expert pool resizing
class AdaptiveExpertTuningCallback(TrainerCallback):
    """
    Trainer callback that drives all DYNMoE state updates at step boundaries:
      * loss-free bias update (Eq. 11) every `bias_update_interval` steps
      * adaptive threshold update every `bias_update_interval` steps
      * soft pruning every `audit_steps` steps (after warmup)
      * CUDA cache clearing every `clear_cache_every` steps

    Hugging Face Trainer callback that triggers adaptive expert tuning
    (soft pruning / auto-tuning) every `audit_steps` training steps
    (Section 3.4).

    It calls `update_loss_free_bias()` on all gates at a higher frequency
    (Eq. 11), then performs periodic soft pruning by deactivating experts
    whose relative usage falls below `prune_threshold`, while keeping at
    least `min_active_experts` active.

    If the expert pool was resized, the optimizer is re-created to avoid
    parameter-size mismatches (Section 3.4).
    """
    def __init__(
        self,
        audit_steps: int = AUDIT_STEPS,
        prune_threshold: float = PRUNE_THRESHOLD,
        min_active_experts: int = MIN_ACTIVE_EXPERTS,
        bias_update_interval: int = BIAS_UPDATE_INTERVAL,
        clear_cache_every: int = CLEAR_CACHE_EVERY,
        warmup_steps: int = WARMUP_STEPS,
    ):
        self.audit_steps = audit_steps
        self.prune_threshold = prune_threshold
        self.min_active_experts = min_active_experts
        self.bias_update_interval = bias_update_interval
        self.clear_cache_every = clear_cache_every
        self.warmup_steps = warmup_steps

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step == 0 or model is None:
            return

        unwrapped = model.module if hasattr(model, "module") else model

        if state.global_step % self.bias_update_interval == 0:
            for module in unwrapped.modules():
                if hasattr(module, "update_loss_free_bias"):
                    module.update_loss_free_bias()
                if hasattr(module, "update_adaptive_threshold"):
                    module.update_adaptive_threshold()

        if state.global_step > self.warmup_steps and \
           state.global_step % self.audit_steps == 0:
            self._audit_and_soft_prune(unwrapped)

        if state.global_step % self.clear_cache_every == 0:
            torch.cuda.empty_cache()

    @torch.no_grad()
    def _audit_and_soft_prune(self, model):
        """
        Prune only experts whose relative usage is far below uniform AND
        only when the router has settled into a non-degenerate regime.
        """
        for module in model.modules():
            if not (hasattr(module, "is_active")
                    and hasattr(module, "routing_counts")
                    and hasattr(module, "num_experts")
                    and hasattr(module, "avg_k_running")):
                continue

            counts = module.routing_counts.float()
            total  = counts.sum()
            if total == 0:
                continue

            # Guard: do not prune while avg_k is still well below target.
            # The router has not yet converged, so pruning would be premature.
            avg_k = float(module.avg_k_running.item())
            target = getattr(module, "target_active_k", 2.0)
            if avg_k < 0.75 * target:
                continue

            N = module.num_experts
            uniform = 1.0 / N
            # Prune only experts below a fraction of uniform usage
            absolute_threshold = self.prune_threshold * uniform

            usage  = counts / total
            active = module.is_active.sum().item()
            for i in range(N):
                if active <= self.min_active_experts:
                    break
                if module.is_active[i] and usage[i] < absolute_threshold:
                    module.is_active[i] = False
                    module.gate_proj.weight[i].zero_()
                    active -= 1