from transformers import TrainerCallback
from model import DeepseekForCausalLM
# from config import DeepseekConfig

ADAPTIVE_AUDIT_STEPS = 10

class AdaptiveExpertTuningCallback(TrainerCallback):
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
        for layer in unwrapped.model.layers:
            if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'gate'):
                if hasattr(layer.mlp.gate, 'adaptive_tune'):
                    layer.mlp.gate.adaptive_tune()
                    tuned_count += 1
        if tuned_count > 0:
            print(f"[DYNMoE Adaptive] Tuned {tuned_count} MoE layers")