import os
import sys
import math
from transformers import TrainerCallback

from deepseekmoe_dynamic_routing_algorithms.source.DYNMoE_baseline.config import (
    ADAPTIVE_AUDIT_STEPS, 
)

class AdaptiveExpertTuningCallback(TrainerCallback):
    """
    This callback must be added to the Hugging Face Trainer to automatically
    trigger adaptive expert addition/removal every `audit_steps` training steps.
    """
    def __init__(self, audit_steps: int = ADAPTIVE_AUDIT_STEPS):
        self.audit_steps = audit_steps
        self.global_step = 0

    def on_step_end(self, args, state, control, model=None, **kwargs):
        self.global_step += 1
        if self.global_step % self.audit_steps == 0 and model is not None:
            self._apply_tuning(model)

    def _apply_tuning(self, model):
        unwrapped = model.module if hasattr(model, 'module') else model
        tuned_count = 0
        for module in unwrapped.modules():
            if hasattr(module, 'adaptive_tune'):
                module.adaptive_tune()
                tuned_count += 1
        if tuned_count > 0:
            print(f"[DYNMoE] Tuned {tuned_count} MoE layers")

