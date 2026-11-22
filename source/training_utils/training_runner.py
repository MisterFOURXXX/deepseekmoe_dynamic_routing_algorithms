import os
import sys
import gc
import math
import inspect
import random
import warnings

import torch
from datasets import Dataset, DatasetDict, load_from_disk
from transformers import (
    AutoTokenizer, Trainer, TrainingArguments,
    DataCollatorForLanguageModeling, GenerationConfig,
)
from transformers.generation.utils import GenerationMixin

# Model imports
from ..deepseek_baseline.config import DeepseekConfig as BaselineConfig
from ..deepseek_baseline.model  import DeepseekForCausalLM as BaselineModel

from ..deepseek_dynamics_routing.config import DeepseekConfig as DynmoeConfig
from ..deepseek_dynamics_routing.model  import DeepseekForCausalLM as DynmoeModel
from ..deepseek_dynamics_routing.adaptive_tuning import (
    AdaptiveExpertTuningCallback as DynmoeRoutingCallback,
)
from ..deepseek_dynamics_routing.config import (
    AUDIT_STEPS, PRUNE_THRESHOLD, MIN_ACTIVE_EXPERTS,
    BIAS_UPDATE_INTERVAL, CLEAR_CACHE_EVERY,
)

from .monitoring import ResourceMonitorCallback, MoEMetricsCallback
from .save_model import save_model_and_tokenizer
from .summarization import print_training_summary
from .config import (
    MAX_SEQ_LEN, PER_DEVICE_BATCH, GRAD_ACCUM, LEARNING_RATE,
    NUM_EPOCHS, WARMUP_STEPS, WEIGHT_DECAY,
    EARLY_STOPPING_PATIENCE, EARLY_STOPPING_THRESHOLD, world_size,
    # optimizer
    ADAM_BETA1, ADAM_BETA2, ADAM_EPS, MAX_GRAD_NORM,
    # schedule
    SCHEDULE, DECAY_FRACS, DECAY_GAMMA,
)

# Memory utilities
from ..memory_utils import cleanup_trainer, clear_cached_data

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# transformers >= 4.46 renamed Trainer(tokenizer=...) -> Trainer(processing_class=...)
_TRAINER_TAKES_PROCESSING_CLASS = (
    "processing_class" in inspect.signature(Trainer.__init__).parameters
)

# Schedule names accepted by `SCHEDULE` in the notebook.
VALID_SCHEDULES = (
    "warmup_step_decay",
    "linear", "cosine", "cosine_with_restarts", "cosine_with_min_lr",
    "polynomial", "constant", "constant_with_warmup",
    "reduce_lr_on_plateau", "warmup_stable_decay",
)

# Weight-decay exclusions: norms, biases, embeddings, tied lm_head.
NO_DECAY_KEYWORDS = ("bias", "norm", "embed_tokens", "lm_head")


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


# Optimizer / LR-schedule
def _build_param_groups(model, weight_decay):
    """Decay matmul weights only: no decay on norms, biases, embeddings, lm_head."""
    decay, no_decay, seen = [], [], set()
    for name, param in model.named_parameters():
        if not param.requires_grad or id(param) in seen:   # tied embeddings appear twice
            continue
        seen.add(id(param))
        if any(k in name.lower() for k in NO_DECAY_KEYWORDS):
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": decay,    "weight_decay": float(weight_decay)},
        {"params": no_decay, "weight_decay": 0.0},
    ]


class DeepSeekMoETrainer(Trainer):
    """AdamW(betas, eps) from TrainingArguments + DeepSeekMoE LR schedule.

    schedule="warmup_step_decay"
        linear warmup from 0 to peak, then lr *= decay_gamma at every fraction
        listed in decay_fracs (paper §4.1.3: x0.316 at 80 % and 90 %).
    schedule=<any HF SchedulerType>
        delegated to Trainer.create_scheduler / transformers.get_scheduler.
    """

    def __init__(self, *args, schedule=SCHEDULE, decay_fracs=DECAY_FRACS,
                 decay_gamma=DECAY_GAMMA, **kwargs):
        # Must be set BEFORE super().__init__(): Trainer.__init__ builds the
        # optimizer + scheduler and therefore calls the two overrides below.
        fracs = tuple(sorted(float(f) for f in decay_fracs))
        if not fracs or any(not 0.0 < f < 1.0 for f in fracs):
            raise ValueError(
                f"DECAY_FRACS must be fractions in (0, 1); got {decay_fracs}"
            )
        if float(decay_gamma) <= 0.0:
            raise ValueError(f"DECAY_GAMMA must be > 0; got {decay_gamma}")

        self.schedule    = str(schedule).lower()
        self.decay_fracs = fracs
        self.decay_gamma = float(decay_gamma)
        super().__init__(*args, **kwargs)

    # weight decay: matmul weights only 
    def create_optimizer(self):
        if getattr(self, "optimizer", None) is not None:
            return self.optimizer

        optim_name = getattr(self.args.optim, "value", self.args.optim)
        if optim_name != "adamw_torch":
            # Respect fused / 8-bit / custom optimizers: HF builds them.
            return super().create_optimizer()

        self.optimizer = torch.optim.AdamW(
            _build_param_groups(self.model, self.args.weight_decay),
            lr=self.args.learning_rate,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            eps=self.args.adam_epsilon,
        )
        return self.optimizer

    # schedule 
    def create_scheduler(self, num_training_steps: int, optimizer=None):
        if getattr(self, "lr_scheduler", None) is not None:
            return self.lr_scheduler
        if num_training_steps <= 0:
            raise ValueError(f"num_training_steps must be > 0; got {num_training_steps}")

        optimizer = optimizer or self.optimizer

        if self.schedule != "warmup_step_decay":
            self.lr_scheduler = super().create_scheduler(num_training_steps, optimizer)
            return self.lr_scheduler

        warmup = int(self.args.get_warmup_steps(num_training_steps))
        warmup = max(1, min(warmup, max(1, num_training_steps - 1)))
        milestones = [math.floor(f * num_training_steps) for f in self.decay_fracs]

        def lr_lambda(step: int) -> float:
            if step < warmup:                       # linear 0 -> peak
                return (step + 1) / warmup
            factor = 1.0
            for m in milestones:                    # x gamma, then x gamma again
                if step >= m:
                    factor *= self.decay_gamma
            return factor

        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return self.lr_scheduler


def _class_uses_dynmoe(cls):
    if "DynMoE" in cls.__name__ or "dynmoe" in cls.__name__.lower():
        return True
    names = ("OptimizedDeepSeekMoE", "FusedMoEGate")
    mod = sys.modules.get(getattr(cls, "__module__", None))
    if mod is not None and any(hasattr(mod, n) for n in names):
        return True
    return any(n in globals() for n in names)


def _prepare_data(
    data_file_path: str,
    val_data_file_path: str = None,
    model_class=None,
    tokenizer_name: str = None,
    split_ratio: float = 0.8,
    random_seed: int = 42,
    max_seq_len: int = MAX_SEQ_LEN,
):
    """Return (tokenizer, tokenized_datasets, data_collator)."""
    if tokenizer_name is None:
        tokenizer_name = "deepseek-ai/deepseek-moe-16b-base"

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name, trust_remote_code=True, use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    if os.path.isdir(data_file_path):
        train_dataset = load_from_disk(data_file_path)

        if val_data_file_path is None:
            derived = data_file_path.replace("_train", "_val")
            if os.path.isdir(derived):
                val_data_file_path = derived

        if val_data_file_path is None or not os.path.isdir(val_data_file_path):
            raise ValueError(
                f"Could not locate the validation dataset for {data_file_path}. "
                f"Pass `val_data_file_path` explicitly."
            )
        val_dataset = load_from_disk(val_data_file_path)

        tokenized_datasets = DatasetDict({
            "train": train_dataset,
            "validation": val_dataset,
        })

        if "input_ids" not in train_dataset.column_names:
            raise ValueError(
                f"{data_file_path} does not look like a tokenized dataset "
                f"(missing 'input_ids'). Expected Arrow files from "
                f"prepare_datasets.py."
            )
    else:
        with open(data_file_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        if not lines:
            raise ValueError(f"No data found in {data_file_path}")

        rng = random.Random(random_seed)
        rng.shuffle(lines)
        split_idx = int(len(lines) * split_ratio)
        train_lines = lines[:split_idx]
        val_lines = lines[split_idx:]

        train_dataset = Dataset.from_dict({"text": train_lines})
        val_dataset = Dataset.from_dict({"text": val_lines})
        dataset_dict = DatasetDict({"train": train_dataset, "validation": val_dataset})

        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                max_length=max_seq_len,
                padding=False,
                return_attention_mask=True,
            )

        tokenized_datasets = dataset_dict.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"],
            num_proc=2,
            desc="Tokenizing datasets",
        )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8,
    )

    return tokenizer, tokenized_datasets, data_collator


def train_model(
    ModelClass, ConfigClass, output_dir, is_dynmoe=False,
    tokenizer=None, tokenized_datasets=None, data_collator=None,
    data_file_path=None, val_data_file_path=None,
    tokenizer_name=None, split_ratio=0.9, random_seed=42,
    **kwargs,
):
    # Resolve Training Hyperparameters
    max_seq_len              = kwargs.pop("MAX_SEQ_LEN", MAX_SEQ_LEN)
    per_device_batch         = kwargs.pop("PER_DEVICE_BATCH", PER_DEVICE_BATCH)
    grad_accum               = kwargs.pop("GRAD_ACCUM", GRAD_ACCUM)
    learning_rate            = kwargs.pop("LEARNING_RATE", LEARNING_RATE)
    num_epochs               = kwargs.pop("NUM_EPOCHS", NUM_EPOCHS)
    warmup_steps             = kwargs.pop("WARMUP_STEPS", WARMUP_STEPS)
    weight_decay             = kwargs.pop("WEIGHT_DECAY", WEIGHT_DECAY)
    early_stopping_patience  = kwargs.pop("EARLY_STOPPING_PATIENCE", EARLY_STOPPING_PATIENCE)
    early_stopping_threshold = kwargs.pop("EARLY_STOPPING_THRESHOLD", EARLY_STOPPING_THRESHOLD)

    # Resolve Optimizer Hyperparameters
    adam_beta1    = kwargs.pop("ADAM_BETA1", ADAM_BETA1)
    adam_beta2    = kwargs.pop("ADAM_BETA2", ADAM_BETA2)
    adam_eps      = kwargs.pop("ADAM_EPS", ADAM_EPS)
    max_grad_norm = kwargs.pop("MAX_GRAD_NORM", MAX_GRAD_NORM)

    # Resolve LR-Schedule Hyperparameters
    schedule    = str(kwargs.pop("SCHEDULE", SCHEDULE)).lower()
    decay_fracs = kwargs.pop("DECAY_FRACS", DECAY_FRACS)
    decay_gamma = kwargs.pop("DECAY_GAMMA", DECAY_GAMMA)
    if schedule not in VALID_SCHEDULES:
        raise ValueError(
            f"SCHEDULE must be one of {VALID_SCHEDULES}; got {schedule!r}"
        )

    # Resolve Adaptive-Tuning Hyperparameters for DYNMoE
    audit_steps          = kwargs.pop("AUDIT_STEPS", AUDIT_STEPS)
    prune_threshold      = kwargs.pop("PRUNE_THRESHOLD", PRUNE_THRESHOLD)
    min_active_experts   = kwargs.pop("MIN_ACTIVE_EXPERTS", MIN_ACTIVE_EXPERTS)
    bias_update_interval = kwargs.pop("BIAS_UPDATE_INTERVAL", BIAS_UPDATE_INTERVAL)
    clear_cache_every    = kwargs.pop("CLEAR_CACHE_EVERY", CLEAR_CACHE_EVERY)

    # Remaining kwargs are model-config overrides
    if "MIN_ROUTED_EXPERTS" in kwargs:
        kwargs["min_routed_experts"] = kwargs.pop("MIN_ROUTED_EXPERTS")

    # Resolve DeepSpeed Config
    ds_config = kwargs.pop("ds_config", None)
    if ds_config is None:
        ds_config = _build_ds_config(per_device_batch, grad_accum)

    model_config_kwargs = kwargs

    # Prepare data
    if tokenizer is None or tokenized_datasets is None or data_collator is None:
        tokenizer, tokenized_datasets, data_collator = _prepare_data(
            data_file_path=data_file_path,
            val_data_file_path=val_data_file_path,
            model_class=ModelClass,
            tokenizer_name=tokenizer_name,
            split_ratio=split_ratio,
            random_seed=random_seed,
            max_seq_len=max_seq_len,
        )

    # Instantiate model with config overrides
    config = ConfigClass(**model_config_kwargs)
    config.pad_token_id = tokenizer.pad_token_id
    config.bos_token_id = tokenizer.bos_token_id
    config.eos_token_id = tokenizer.eos_token_id

    if not issubclass(ModelClass, GenerationMixin):
        ModelClass = type(
            ModelClass.__name__,
            (ModelClass, GenerationMixin),
            {"__module__": ModelClass.__module__},
        )
        print(f"Patched {ModelClass.__name__} with GenerationMixin.")

    model = ModelClass(config)
    model.resize_token_embeddings(len(tokenizer))
    model.config.use_cache = False
    model.train()

    if getattr(model, "generation_config", None) is None:
        model.generation_config = GenerationConfig.from_model_config(model.config)

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("\nModel Summary:")
    print(f" Total parameters    : {total_params:,}")
    print(f" Trainable parameters: {trainable_params:,}")
    print(f" Model architecture  :\n{model}\n")

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

        # AdamW betas / eps (read by DeepSeekMoETrainer.create_optimizer)
        adam_beta1=adam_beta1,
        adam_beta2=adam_beta2,
        adam_epsilon=adam_eps,

        # Marker only. "constant" == "no built-in schedule"; the custom
        # LambdaLR is built in DeepSeekMoETrainer.create_scheduler().
        # "warmup_step_decay" is NOT a valid SchedulerType and would raise here.
        lr_scheduler_type="constant" if schedule == "warmup_step_decay" else schedule,

        fp16=False,
        bf16=True,
        bf16_full_eval=False,
        fp16_full_eval=False,
        half_precision_backend="auto",
        logging_steps=1,
        logging_first_step=True,
        save_strategy="epoch",
        eval_strategy="no",
        load_best_model_at_end=False,
        report_to="none",
        deepspeed=ds_config,
        ddp_find_unused_parameters=False if world_size > 1 else None,
        gradient_checkpointing=False,
        dataloader_num_workers=2,
        remove_unused_columns=True,
        optim="adamw_torch",
        logging_dir=os.path.join(output_dir, "logs"),
        seed=42,
        save_only_model=True,
    )

    # Callbacks
    resource_monitor = ResourceMonitorCallback()
    moemetrics = MoEMetricsCallback(
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        early_stop_patience=early_stopping_patience,
        early_stop_threshold=early_stopping_threshold,
    )
    callbacks = [resource_monitor, moemetrics]

    if is_dynmoe:
        adaptive_callback = DynmoeRoutingCallback(
            audit_steps=audit_steps,
            prune_threshold=prune_threshold,
            min_active_experts=min_active_experts,
            bias_update_interval=bias_update_interval,
            clear_cache_every=clear_cache_every,
        )
        callbacks.append(adaptive_callback)
        print("AdaptiveExpertTuningCallback enabled.")
    else:
        print("Adaptive expert tuning disabled for this run.")

    # Trainer
    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        callbacks=callbacks,
        schedule=schedule,
        decay_fracs=decay_fracs,
        decay_gamma=decay_gamma,
    )
    if _TRAINER_TAKES_PROCESSING_CLASS:
        trainer = DeepSeekMoETrainer(processing_class=tokenizer, **trainer_kwargs)
    else:
        trainer = DeepSeekMoETrainer(tokenizer=tokenizer, **trainer_kwargs)

    print("Starting training...")
    train_result = trainer.train()

    save_model_and_tokenizer(trainer, output_dir)
    print_training_summary(
        resource_monitor, moemetrics, train_result,
    )

    return trainer


def run_training(
    ModelClass, ConfigClass, output_dir, is_dynmoe=False,
    tokenizer=None, tokenized_datasets=None, data_collator=None,
    data_file_path=None, val_data_file_path=None,
    tokenizer_name=None, split_ratio=0.9, random_seed=42,
    **kwargs,
):
    print("=" * 60)
    print(f"TRAINING {ModelClass.__name__}")
    print("=" * 60)

    trainer = train_model(
        ModelClass, ConfigClass, output_dir, is_dynmoe,
        tokenizer=tokenizer,
        tokenized_datasets=tokenized_datasets,
        data_collator=data_collator,
        data_file_path=data_file_path,
        val_data_file_path=val_data_file_path,
        tokenizer_name=tokenizer_name,
        split_ratio=split_ratio,
        random_seed=random_seed,
        **kwargs,
    )

    cleanup_trainer(trainer)
    clear_cached_data()