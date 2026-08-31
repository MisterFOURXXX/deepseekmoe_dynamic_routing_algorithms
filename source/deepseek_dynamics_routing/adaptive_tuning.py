import os
import sys
import gc
import torch
import math
from transformers import TrainerCallback

from .config import ADAPTIVE_AUDIT_STEPS

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
    def __init__(self, audit_steps: int = ADAPTIVE_AUDIT_STEPS):
        self.audit_steps = audit_steps
        self.global_step = 0
        self.trainer = None          # set this after Trainer creation

    def on_step_end(self, args, state, control, model=None, **kwargs):
        self.global_step += 1
        if self.global_step % self.audit_steps != 0 or model is None:
            return

        unwrapped = model.module if hasattr(model, 'module') else model

        # Adaptive tune on all gates
        for module in unwrapped.modules():
            if hasattr(module, 'adaptive_tune'):
                module.adaptive_tune()

        # Sync ModuleLists (rebuild experts)
        resized = False
        for module in unwrapped.modules():
            if hasattr(module, 'sync_experts'):
                old_n = getattr(module, 'n_routed_experts', None)
                module.sync_experts()
                if getattr(module, 'n_routed_experts', None) != old_n:
                    resized = True

        # If anything was resized then rebuild optimizer from scratch
        if resized and self.trainer is not None:
            if self.trainer.optimizer is not None:
                del self.trainer.optimizer
                self.trainer.optimizer = None
            gc.collect()
            torch.cuda.empty_cache()

            # Re-create optimizer with the new parameter list
            self.trainer.create_optimizer()
        print("[DYNMoE] Expert pool resized – rebuilding optimizer …")