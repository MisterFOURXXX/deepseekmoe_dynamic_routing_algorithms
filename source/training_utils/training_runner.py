import os
import gc
import math
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

from ..DYNMoE_baseline.config import DynMoEConfig as DYNMoEBaseConfig
from ..DYNMoE_baseline.model  import DynMoEForCausalLM as DYNMoEBaseModel
from ..DYNMoE_baseline.adaptive_tuning import (
    AdaptiveExpertTuningCallback as DYNMoEBaseCallback,
    ADAPTIVE_AUDIT_STEPS as DYNMoE_BASE_ADAPTIVE_AUDIT_STEPS,
)

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
)

# Memory utilities
from ..memory_utils import cleanup_trainer, clear_cached_data

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# DeepSpeed configuration (ZeRO-2 + bf16) - matches Jupyter notebook
ds_config = {
    "train_batch_size": PER_DEVICE_BATCH * max(world_size, 1) * GRAD_ACCUM,
    "train_micro_batch_size_per_gpu": PER_DEVICE_BATCH,
    "gradient_accumulation_steps": GRAD_ACCUM,
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
):
    if tokenizer is None or tokenized_datasets is None or data_collator is None:
        tokenizer, tokenized_datasets, data_collator = _prepare_data(
            data_file_path=data_file_path,
            val_data_file_path=val_data_file_path,
            model_class=ModelClass,
            tokenizer_name=tokenizer_name,
            split_ratio=split_ratio,
            random_seed=random_seed,
        )

    # Instantiate model 
    config = ConfigClass()
    config.pad_token_id = tokenizer.pad_token_id
    config.bos_token_id = tokenizer.bos_token_id
    config.eos_token_id = tokenizer.eos_token_id
    # NOTE: Do NOT set config.dtype = torch.bfloat16 to avoid JSON serialization errors during checkpointing

    # Patch GenerationMixin (Jupyter notebook logic)
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

    # TrainingArguments (matches Jupyter notebook)
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_BATCH,
        per_device_eval_batch_size=PER_DEVICE_BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_steps=WARMUP_STEPS,
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
        fp16=False,
        bf16=True,
        bf16_full_eval=False,
        fp16_full_eval=False,
        half_precision_backend="auto",
        logging_steps=1,
        logging_first_step=True,
        save_strategy="epoch",
        eval_strategy="epoch",
        load_best_model_at_end=False,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
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
        tokenized_datasets["validation"], tokenizer, data_collator,
        early_stop_patience=EARLY_STOPPING_PATIENCE,
        early_stop_threshold=EARLY_STOPPING_THRESHOLD,
    )
    callbacks = [resource_monitor, moemetrics]

    if is_dynmoe:
        if "DYNMoE_baseline" in ModelClass.__module__:
            adaptive_callback = DYNMoEBaseCallback(
                audit_steps=DYNMoE_BASE_ADAPTIVE_AUDIT_STEPS,
            )
        else:
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

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,          # transformers==4.44.2 still uses `tokenizer=`
        callbacks=callbacks,
    )

    print("Starting training...")
    train_result = trainer.train()
    eval_results = trainer.evaluate()

    final_loss   = eval_results.get("eval_loss", float("inf"))
    perplexity   = math.exp(final_loss) if 0 < final_loss < 30 else float("inf")
    print(f"Final validation loss: {final_loss:.4f}")
    print(f"Validation Perplexity: {perplexity:.2f}")

    save_model_and_tokenizer(trainer, output_dir)
    print_training_summary(
        resource_monitor, moemetrics, train_result, eval_results, perplexity,
    )

    return trainer


def run_training(
    ModelClass, ConfigClass, output_dir, is_dynmoe=False,
    tokenizer=None, tokenized_datasets=None, data_collator=None,
    data_file_path=None, val_data_file_path=None,
    tokenizer_name=None, split_ratio=0.9, random_seed=42,
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
    )

    cleanup_trainer(trainer)
    clear_cached_data()