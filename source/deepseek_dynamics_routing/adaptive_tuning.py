import os
import sys
#repo_path =  ".."
#os.chdir(repo_path)                 # Move into the repo
#sys.path.insert(0, os.getcwd())     # Ensure the repo root is on sys.path

import math
from transformers import TrainerCallback

from deepseekmoe_dynamic_routing_algorithms.source.deepseek_dynamics_routing.config import (
    ADAPTIVE_AUDIT_STEPS 
)

class AdaptiveExpertTuningCallback(TrainerCallback):
    def __init__(self, audit_steps: int = ADAPTIVE_AUDIT_STEPS):
        self.audit_steps = audit_steps
        self.global_step = 0
        self.trainer = None          # set this after Trainer creation

    def on_step_end(self, args, state, control, model=None, **kwargs):
        self.global_step += 1
        if self.global_step % self.audit_steps != 0 or model is None:
            return

        unwrapped = model.module if hasattr(model, 'module') else model

        # Adaptive tune (may change parameter shapes)
        for module in unwrapped.modules():
            if hasattr(module, 'adaptive_tune'):
                module.adaptive_tune()

        # Sync ModuleLists
        resized = False
        for module in unwrapped.modules():
            if hasattr(module, 'sync_experts'):
                old_n = getattr(module, 'n_routed_experts', None)
                module.sync_experts()
                if getattr(module, 'n_routed_experts', None) != old_n:
                    resized = True

        # If anything was resized then rebuild optimizer from scratch
        if resized and self.trainer is not None:
            print("[DYNMoE] Expert pool resized – rebuilding optimizer …")
            if self.trainer.optimizer is not None:
                del self.trainer.optimizer
                self.trainer.optimizer = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Re-create optimizer with the new parameter list
            self.trainer.create_optimizer()