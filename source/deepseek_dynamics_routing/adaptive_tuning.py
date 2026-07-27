import os
import sys
repo_path =  ".."
os.chdir(repo_path)                 # Move into the repo
sys.path.insert(0, os.getcwd())     # Ensure the repo root is on sys.path

import math
from transformers import TrainerCallback

ADAPTIVE_AUDIT_STEPS = 10

class AdaptiveExpertTuningCallback(TrainerCallback):
    """
    Callback that triggers adaptive tuning on all modules that have an
    `adaptive_tune()` method (e.g., MoEGate). It then synchronises experts
    by calling `sync_experts()` on any module that provides it (e.g., DeepseekMoE).
    Works with any model structure by scanning all submodules.
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

        # 1. Call adaptive_tune on all gates that support it
        tuned_count = 0
        for module in unwrapped.modules():
            if hasattr(module, 'adaptive_tune'):
                module.adaptive_tune()
                tuned_count += 1

        # 2. Sync experts on all MoE modules that have sync_experts
        sync_count = 0
        for module in unwrapped.modules():
            if hasattr(module, 'sync_experts'):
                module.sync_experts()
                sync_count += 1

        if tuned_count > 0:
            print(f"[DYNMoE Adaptive] Tuned {tuned_count} MoE layers, synced {sync_count} MoE modules")