import sys
sys.path.append("..")

import os
import math
import json
import torch
import warnings
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    GenerationConfig,
)
from datasets import load_from_disk

# Modular utilities
from .monitoring import ResourceMonitorCallback, MoEMetricsCallback
from .save_model import save_finetuned_model
from .summarization import print_finetuning_summary
from .model_loading import load_model_and_tokenizer

from .config import (
    PER_DEVICE_BATCH,
    GRAD_ACCUM,
    LEARNING_RATE,
    NUM_EPOCHS_FT,
    WARMUP_STEPS,
    WEIGHT_DECAY,
    EARLY_STOPPING_PATIENCE,
    EARLY_STOPPING_THRESHOLD,
    ADAM_BETA1,
    ADAM_BETA2,
    ADAM_EPS,
    MAX_GRAD_NORM,
    SCHEDULE,
    DECAY_FRACS,
    DECAY_GAMMA,
    world_size,
)
from ..data_preprocessing import (
    DEFAULT_FINE_PATH,
    DEFAULT_VAL_PATH,
)
from ..memory_utils import cleanup_trainer, clear_cached_data

# Model imports
from ..deepseek_baseline.config import DeepseekConfig as BaselineConfig
from ..deepseek_baseline.model import DeepseekForCausalLM as BaselineModel

from ..deepseek_dynamics_routing.config import DeepseekConfig as DynmoeConfig
from ..deepseek_dynamics_routing.model import DeepseekForCausalLM as DynmoeModel

from ..deepseek_dynamics_routing.adaptive_tuning import AdaptiveExpertTuningCallback as DynmoeRoutingCallback
from ..deepseek_dynamics_routing.config import (
    AUDIT_STEPS,
    PRUNE_THRESHOLD,
    MIN_ACTIVE_EXPERTS,
    BIAS_UPDATE_INTERVAL,
    CLEAR_CACHE_EVERY,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ---------------------------------------------------------------------------
# LR schedules
# ---------------------------------------------------------------------------
class WarmupStepDecayScheduler(torch.optim.lr_scheduler.LambdaLR):
    """Linear warm-up followed by multiplicative step decay.

    This is the DeepSeekMoE schedule: the learning rate is multiplied by
    ``decay_gamma`` at each fraction in ``decay_fracs`` (0.316 at 80% and
    again at 90% of the planned steps).

    ``total_steps`` is the scheduler horizon.  With early stopping it is
    usually better to pass the *expected* stop step (e.g. 5 epochs) rather
    than the 20-epoch ceiling, otherwise the decays are never reached.
    """

    def __init__(self, optimizer, warmup_steps, total_steps,
                 decay_fracs=(0.80, 0.90), decay_gamma=0.316, last_epoch=-1):
        self.total_steps = max(1, int(total_steps))
        self.warmup_steps = max(0, int(warmup_steps))
        # A warm-up longer than the run would keep the LR below its target for
        # the whole run, so cap it at 10% of the horizon in that case.
        if self.warmup_steps >= self.total_steps:
            capped = max(1, self.total_steps // 10)
            print(
                f"[WarmupStepDecayScheduler] WARMUP_STEPS={self.warmup_steps} "
                f">= planned steps ({self.total_steps}); capping warm-up at "
                f"{capped} steps."
            )
            self.warmup_steps = capped
        self.decay_fracs = tuple(float(frac) for frac in decay_fracs)
        self.decay_gamma = float(decay_gamma)
        super().__init__(optimizer, self._lr_lambda, last_epoch=last_epoch)

    def _lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / max(1, self.warmup_steps)

        factor = 1.0
        for frac in self.decay_fracs:
            if step >= int(round(frac * self.total_steps)):
                factor *= self.decay_gamma
        return factor


def _hf_scheduler_type(schedule, warmup_steps):
    """Map the notebook schedule name to a HuggingFace scheduler type."""
    if schedule == "warmup_step_decay":
        # Placeholder; UnifiedTrainer.create_scheduler installs the real one.
        return "constant"
    if schedule == "constant" and int(warmup_steps) <= 0:
        return "constant"
    # "cosine" / "linear" already include the warm-up; "constant" only does
    # when warm-up steps are requested.
    return {
        "cosine": "cosine",
        "linear": "linear",
        "constant": "constant_with_warmup",
    }.get(schedule, schedule)


def _build_ds_config(per_device_batch, grad_accum, max_grad_norm):
    """Fallback DeepSpeed configuration if none is passed from the notebook."""
    return {
        "train_batch_size": per_device_batch * max(world_size, 1) * grad_accum,
        "train_micro_batch_size_per_gpu": per_device_batch,
        "gradient_accumulation_steps": grad_accum,
        "bf16": {"enabled": True},
        "zero_optimization": {
            "stage": 2,
            "offload_optimizer": {"device": "cpu", "pin_memory": True},
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_bucket_size": 5e8,
            "allgather_bucket_size": 5e8,
        },
        "gradient_clipping": max_grad_norm,
        "steps_per_print": 10,
        "wall_clock_breakdown": False,
        "zero_allow_untested_optimizer": True,
    }


# Helper: Detect DYNMoE exactly like the Jupyter notebook
def _class_uses_dynmoe(cls):
    if "DynMoE" in cls.__name__ or "dynmoe" in cls.__name__.lower():
        return True
    names = ("OptimizedDeepSeekMoE", "FusedMoEGate")
    mod = sys.modules.get(getattr(cls, "__module__", None))
    if mod is not None and any(hasattr(mod, n) for n in names):
        return True
    return any(n in globals() for n in names)


def fine_tune_model(pretrained_path, output_dir, data_file_path=None, val_data_file_path=None, **kwargs):
    # Resolve Fine-Tuning Hyperparameters
    _max_seq_len             = kwargs.pop("MAX_SEQ_LEN", None)
    per_device_batch         = kwargs.pop("PER_DEVICE_BATCH", PER_DEVICE_BATCH)
    grad_accum               = kwargs.pop("GRAD_ACCUM", GRAD_ACCUM)
    learning_rate            = kwargs.pop("LEARNING_RATE", LEARNING_RATE)
    num_epochs               = kwargs.pop("NUM_EPOCHS_FT", NUM_EPOCHS_FT)
    warmup_steps             = kwargs.pop("WARMUP_STEPS", WARMUP_STEPS)
    weight_decay             = kwargs.pop("WEIGHT_DECAY", WEIGHT_DECAY)
    early_stopping_patience  = kwargs.pop("EARLY_STOPPING_PATIENCE", EARLY_STOPPING_PATIENCE)
    early_stopping_threshold = kwargs.pop("EARLY_STOPPING_THRESHOLD", EARLY_STOPPING_THRESHOLD)

    # Optimizer (AdamW)
    adam_beta1               = kwargs.pop("ADAM_BETA1", ADAM_BETA1)
    adam_beta2               = kwargs.pop("ADAM_BETA2", ADAM_BETA2)
    adam_eps                 = kwargs.pop("ADAM_EPS", ADAM_EPS)
    max_grad_norm            = kwargs.pop("MAX_GRAD_NORM", MAX_GRAD_NORM)

    # LR schedule
    schedule                 = str(kwargs.pop("SCHEDULE", SCHEDULE)).strip().lower()
    decay_fracs              = tuple(kwargs.pop("DECAY_FRACS", DECAY_FRACS))
    decay_gamma              = float(kwargs.pop("DECAY_GAMMA", DECAY_GAMMA))

    # Resolve DeepSpeed Config
    ds_config = kwargs.pop("ds_config", None)
    if ds_config is None:
        ds_config = _build_ds_config(per_device_batch, grad_accum, max_grad_norm)
    elif isinstance(ds_config, (str, os.PathLike)):
        with open(ds_config, "r") as f:
            ds_config = json.load(f)
    else:
        ds_config = dict(ds_config)
    # DeepSpeed clips gradients with this value, not TrainingArguments.
    ds_config["gradient_clipping"] = max_grad_norm

    # Load Model and Tokenizer
    print(f"Loading pretrained model from {pretrained_path} ...")
    model, tokenizer = load_model_and_tokenizer(pretrained_path)

    # Resolve Dataset Paths
    finetune_path = data_file_path if data_file_path is not None else DEFAULT_FINE_PATH
    val_path = val_data_file_path if val_data_file_path is not None else DEFAULT_VAL_PATH

    # Load Pre-tokenized Datasets from Disk 
    print(f"Loading train dataset from {finetune_path} ...")
    finetune_dataset = load_from_disk(finetune_path)
    
    print(f"Loading val dataset from {val_path} ...")
    val_dataset = load_from_disk(val_path)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8
    )

    # Setup Model 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.config.use_cache = False
    model.train()

    if getattr(model, "generation_config", None) is None:
        model.generation_config = GenerationConfig.from_model_config(model.config)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("\nModel Summary:")
    print(f"  Total parameters     : {total_params:,}")
    print(f"  Trainable parameters : {trainable_params:,}")
    print(f"  Model architecture:\n{model}\n")

    # DeepSpeed Configuration Save to JSON
    os.makedirs(output_dir, exist_ok=True)
    ds_config_path = os.path.join(output_dir, "ds_config_finetune.json")
    with open(ds_config_path, "w") as f:
        json.dump(ds_config, f, indent=2)
    print(f"Using DeepSpeed config at {ds_config_path}")

    # Callbacks 
    resource_monitor = ResourceMonitorCallback()
    moemetrics_callback = MoEMetricsCallback(
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        early_stop_patience=early_stopping_patience,
        early_stop_threshold=early_stopping_threshold,
    )
    callbacks = [resource_monitor, moemetrics_callback]

    # Detect DYNMoE based on the actual class structure
    is_dynmoe = _class_uses_dynmoe(model.__class__)

    if is_dynmoe:
        adaptive_callback = DynmoeRoutingCallback(
            audit_steps=AUDIT_STEPS,
            prune_threshold=PRUNE_THRESHOLD,
            min_active_experts=MIN_ACTIVE_EXPERTS,
            bias_update_interval=BIAS_UPDATE_INTERVAL,
            clear_cache_every=CLEAR_CACHE_EVERY,
        )
        callbacks.append(adaptive_callback)
        print("AdaptiveExpertTuningCallback enabled.")
    else:
        print("Adaptive expert tuning disabled for this run.")

    # Training Arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=per_device_batch,
        per_device_eval_batch_size=per_device_batch,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        max_grad_norm=max_grad_norm,
        lr_scheduler_type=_hf_scheduler_type(schedule, warmup_steps),
        adam_beta1=adam_beta1,
        adam_beta2=adam_beta2,
        adam_epsilon=adam_eps,
        fp16=False,
        bf16=True,
        bf16_full_eval=False,
        fp16_full_eval=False,
        half_precision_backend="auto",
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="no",
        load_best_model_at_end=False,
        report_to="none",
        deepspeed=ds_config_path, 
        ddp_find_unused_parameters=False if world_size > 1 else None,
        gradient_checkpointing=False,
        dataloader_num_workers=2,
        remove_unused_columns=True,
        optim="adamw_torch",
        logging_dir=os.path.join(output_dir, "logs"),
        seed=42,
        save_only_model=True,
    )

    # Trainer
    class UnifiedTrainer(Trainer):
        def create_scheduler(self, num_training_steps, optimizer=None):
            if schedule == "warmup_step_decay":
                if optimizer is None:
                    optimizer = self.optimizer
                self.lr_scheduler = WarmupStepDecayScheduler(
                    optimizer,
                    warmup_steps=warmup_steps,
                    total_steps=num_training_steps,
                    decay_fracs=decay_fracs,
                    decay_gamma=decay_gamma,
                )
                return self.lr_scheduler
            return super().create_scheduler(num_training_steps, optimizer)

        def create_optimizer_and_scheduler(self, num_training_steps):
            super().create_optimizer_and_scheduler(num_training_steps)
            for opt in (self.optimizer, getattr(self.optimizer, "optimizer", None)):
                if opt is not None and hasattr(opt, "_opt_called"):
                    opt._opt_called = True

    trainer = UnifiedTrainer(
        model=model,
        args=training_args,
        train_dataset=finetune_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=callbacks,
    )

    # Train & Evaluate 
    print("\nStarting fine-tuning on finetune_dataset...")
    train_result = trainer.train()
    print("\nFine-tuning finished.")

    # Save & Summarize 
    save_finetuned_model(trainer, output_dir)
    print_finetuning_summary(trainer, resource_monitor, moemetrics_callback, train_result)

    return trainer


def run_fine_tuning(pretrained_path, output_dir, data_file_path=None, val_data_file_path=None, **kwargs):
    print("=" * 60)
    print(f"FINE-TUNING from {pretrained_path}")
    print("=" * 60)

    trainer = fine_tune_model(
        pretrained_path, 
        output_dir, 
        data_file_path=data_file_path,
        val_data_file_path=val_data_file_path,
        **kwargs
    )
    cleanup_trainer(trainer)
    clear_cached_data()