# source/fine_tuning_utils/fine_tuning_runner.py
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


def _build_ds_config(per_device_batch, grad_accum):
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
        "gradient_clipping": 1.0,
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


def fine_tune_model(pretrained_path, output_dir, **kwargs):
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

    # Resolve DeepSpeed Config
    ds_config = kwargs.pop("ds_config", None)
    if ds_config is None:
        ds_config = _build_ds_config(per_device_batch, grad_accum)

    # Load Model and Tokenizer
    print(f"Loading pretrained model from {pretrained_path} ...")
    model, tokenizer = load_model_and_tokenizer(pretrained_path)

    # Load Pre-tokenized Datasets from Disk 
    print(f"Loading train dataset from {DEFAULT_FINE_PATH} ...")
    finetune_dataset = load_from_disk(DEFAULT_FINE_PATH)
    print(f"Loading val dataset from {DEFAULT_VAL_PATH} ...")
    val_dataset = load_from_disk(DEFAULT_VAL_PATH)

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
    if isinstance(ds_config, str):
        ds_config_path = ds_config
    else:
        ds_config_path = os.path.join(output_dir, "ds_config_finetune.json")
        with open(ds_config_path, "w") as f:
            json.dump(ds_config, f, indent=2)
    print(f"Using DeepSpeed config at {ds_config_path}")

    # Callbacks 
    resource_monitor = ResourceMonitorCallback()
    moemetrics_callback = MoEMetricsCallback(
        val_dataset, tokenizer, data_collator,
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
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
        fp16=False,
        bf16=True,
        bf16_full_eval=False,
        fp16_full_eval=False,
        half_precision_backend="auto",
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        load_best_model_at_end=False,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
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

    eval_results = trainer.evaluate()
    final_loss = eval_results.get("eval_loss", float('inf'))
    perplexity = math.exp(final_loss) if 0 < final_loss < 30 else float('inf')
    print(f"Final validation loss : {final_loss:.4f}")
    print(f"Validation Perplexity : {perplexity:.2f}")

    # Save & Summarize 
    save_finetuned_model(trainer, output_dir)
    print_finetuning_summary(trainer, resource_monitor, moemetrics_callback, train_result, eval_results, perplexity)

    return trainer


def run_fine_tuning(pretrained_path, output_dir, **kwargs):
    print("=" * 60)
    print(f"FINE-TUNING from {pretrained_path}")
    print("=" * 60)

    trainer = fine_tune_model(pretrained_path, output_dir, **kwargs)
    cleanup_trainer(trainer)
    clear_cached_data()