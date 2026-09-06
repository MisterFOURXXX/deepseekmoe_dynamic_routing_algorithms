import os
import gc
import math
import torch
import warnings
import random
from transformers import AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset, DatasetDict

# Model imports (assumes repo root is on sys.path)
from deepseekmoe_dynamic_routing_algorithms.source.deepseek_baseline.config import DeepseekConfig as BaselineConfig
from deepseekmoe_dynamic_routing_algorithms.source.deepseek_baseline.model import DeepseekForCausalLM as BaselineModel

from deepseekmoe_dynamic_routing_algorithms.source.DYNMoE_baseline.config import DynMoEConfig as DYNMoEBaseConfig
from deepseekmoe_dynamic_routing_algorithms.source.DYNMoE_baseline.model import DynMoEForCausalLM as DYNMoEBaseModel
from deepseekmoe_dynamic_routing_algorithms.source.DYNMoE_baseline.adaptive_tuning import AdaptiveExpertTuningCallback as DYNMoEBaseCallback
from deepseekmoe_dynamic_routing_algorithms.source.DYNMoE_baseline.adaptive_tuning import ADAPTIVE_AUDIT_STEPS as DYNMoE_BASE_ADAPTIVE_AUDIT_STEPS

from deepseekmoe_dynamic_routing_algorithms.source.deepseek_dynamics_routing.config import DeepseekConfig as DynmoeConfig
from deepseekmoe_dynamic_routing_algorithms.source.deepseek_dynamics_routing.model import DeepseekForCausalLM as DynmoeModel
from deepseekmoe_dynamic_routing_algorithms.source.deepseek_dynamics_routing.adaptive_tuning import AdaptiveExpertTuningCallback as DynmoeRoutingCallback
from deepseekmoe_dynamic_routing_algorithms.source.deepseek_dynamics_routing.config import (
    AUDIT_STEPS
    PRUNE_THRESHOLD
    MIN_ACTIVE_EXPERTS
    BIAS_UPDATE_INTERVAL
    CLEAR_CACHE_EVERY
)

from deepseekmoe_dynamic_routing_algorithms.source.fine_tuning_utils.monitoring import ResourceMonitorCallback, MoEMetricsCallback
from deepseekmoe_dynamic_routing_algorithms.source.fine_tuning_utils.save_model import save_finetuned_model
from deepseekmoe_dynamic_routing_algorithms.source.fine_tuning_utils.summarization import print_finetuning_summary
from deepseekmoe_dynamic_routing_algorithms.source.fine_tuning_utils.model_loading import load_model_and_tokenizer
from deepseekmoe_dynamic_routing_algorithms.source.fine_tuning_utils.config import (
    MAX_SEQ_LEN,
    PER_DEVICE_BATCH,
    GRAD_ACCUM,
    LEARNING_RATE,
    NUM_EPOCHS_FT,
    WARMUP_STEPS,
    WEIGHT_DECAY,
    world_size
)

from deepseekmoe_dynamic_routing_algorithms.source.memory_utils import cleanup_trainer, clear_cached_data

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# DeepSpeed config
ds_config = {
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "gradient_accumulation_steps": "auto",
    "fp16": {
        "enabled": True,
        "loss_scale": 0,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    },
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {"device": "cpu", "pin_memory": True},
        "offload_param": {"device": "cpu", "pin_memory": True},
        "overlap_comm": True,
        "contiguous_gradients": True,
        "reduce_bucket_size": 5e8,
        "stage3_prefetch_bucket_size": 5e8,
        "stage3_param_persistence_threshold": 1e6,
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_16bit_weights_on_model_save": True
    },
    "gradient_clipping": 0.5,
    "steps_per_print": 10,
    "wall_clock_breakdown": False,
    "zero_allow_untested_optimizer": True
}

# Data preparation (aligned with training)
def _prepare_data(
    data_file_path: str,
    tokenizer,
    split_ratio: float = 0.8,
    random_seed: int = 42,
    max_seq_len: int = MAX_SEQ_LEN
):
    # Load lines
    with open(data_file_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
        raise ValueError(f"No data found in {data_file_path}")

    # Shuffle and split
    rng = random.Random(random_seed)
    rng.shuffle(lines)
    split_idx = int(len(lines) * split_ratio)
    train_lines = lines[:split_idx]
    val_lines = lines[split_idx:]

    # Build DatasetDict
    train_dataset = Dataset.from_dict({"text": train_lines})
    val_dataset = Dataset.from_dict({"text": val_lines})
    dataset_dict = DatasetDict({"train": train_dataset, "validation": val_dataset})

    # Tokenize
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
        desc="Tokenizing datasets"
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8
    )

    return tokenized_datasets, data_collator

# Core fine-tuning function
def fine_tune_model(pretrained_path, output_dir,
                    data_file_path=None, split_ratio=0.8, random_seed=42,
                    tokenizer=None, tokenized_datasets=None, data_collator=None):

    # Load pretrained model and its tokenizer
    model, tokenizer = load_model_and_tokenizer(pretrained_path)

    tokenized_datasets, data_collator = _prepare_data(
        data_file_path=data_file_path,
        tokenizer=tokenizer,
        split_ratio=split_ratio,
        random_seed=random_seed
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.config.use_cache = False
    model.train()

    is_dynmoe = "DynMoE" in model.__class__.__name__

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("\nModel Summary:")
    print(f" Total parameters: {total_params:,}")
    print(f" Trainable parameters: {trainable_params:,}")

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=NUM_EPOCHS_FT,
        per_device_train_batch_size=PER_DEVICE_BATCH,
        per_device_eval_batch_size=PER_DEVICE_BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_steps=WARMUP_STEPS,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        load_best_model_at_end=False,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        deepspeed=ds_config,
        ddp_find_unused_parameters=False if world_size > 1 else None,
        max_grad_norm=0.5,
        gradient_checkpointing=False,
        dataloader_num_workers=2,
        remove_unused_columns=True,
        optim="adamw_torch",
        logging_dir=f"{output_dir}/logs",
        seed=42,
        save_only_model=True
    )

    resource_monitor = ResourceMonitorCallback()
    moemetrics = MoEMetricsCallback(tokenized_datasets["validation"], tokenizer, data_collator)
    callbacks = [resource_monitor, moemetrics]

    if is_dynmoe:
        if "DYNMoE_baseline" in model.__class__.__name__:
            adaptive_callback = DYNMoEBaseCallback(audit_steps=DYNMoE_BASE_ADAPTIVE_AUDIT_STEPS)
        else:
            adaptive_callback = DynmoeRoutingCallback(
                audit_steps = AUDIT_STEPS,
                prune_threshold = PRUNE_THRESHOLD,
                min_active_experts = MIN_ACTIVE_EXPERTS,
                bias_update_interval = BIAS_UPDATE_INTERVAL,
                clear_cache_every = CLEAR_CACHE_EVERY,
            )
        callbacks.append(adaptive_callback)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=callbacks
    )

    print("Starting fine‑tuning...")
    train_result = trainer.train()
    print("Fine‑tuning finished.")

    eval_results = trainer.evaluate()
    final_loss = eval_results.get("eval_loss", float('inf'))
    perplexity = math.exp(final_loss) if 0 < final_loss < 30 else float('inf')
    print(f"Final validation loss: {final_loss:.4f}")
    print(f"Validation Perplexity: {perplexity:.2f}")

    final_output_dir = os.path.join(output_dir, "final")
    unwrapped = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model
    unwrapped.save_pretrained(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)
    print(f"Fine‑tuned model saved to {final_output_dir}")

    save_finetuned_model(trainer, output_dir)
    print_finetuning_summary(resource_monitor, moemetrics, train_result, eval_results, perplexity)

    return trainer

def run_fine_tuning(pretrained_path, output_dir,
                               data_file_path=None, split_ratio=0.8, random_seed=42,
                               tokenizer=None, tokenized_datasets=None, data_collator=None):

    print("=" * 60)
    print(f"FINE-TUNING from {pretrained_path}")
    print("=" * 60)

    trainer = fine_tune_model(pretrained_path, output_dir,
                              data_file_path, split_ratio, random_seed,
                              tokenizer, tokenized_datasets, data_collator)
    cleanup_trainer(trainer)
    clear_cached_data()   # frees the dataset cache