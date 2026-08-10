import os
import math
import torch
import warnings
import random
from transformers import AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset

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
from deepseekmoe_dynamic_routing_algorithms.source.deepseek_dynamics_routing.adaptive_tuning import ADAPTIVE_AUDIT_STEPS as DYNMOE_ROUTING_ADAPTIVE_AUDIT_STEPS

from deepseekmoe_dynamic_routing_algorithms.source.training_utils.monitoring import ResourceMonitorCallback, MoEMetricsCallback
from deepseekmoe_dynamic_routing_algorithms.source.training_utils.save_model import save_model_and_tokenizer
from deepseekmoe_dynamic_routing_algorithms.source.training_utils.summarization import print_training_summary

from deepseekmoe_dynamic_routing_algorithms.source.training_utils.config import (
    MAX_SEQ_LEN, PER_DEVICE_BATCH, GRAD_ACCUM, LEARNING_RATE,
    NUM_EPOCHS, WARMUP_STEPS, WEIGHT_DECAY,
    EARLY_STOPPING_PATIENCE, EARLY_STOPPING_THRESHOLD, world_size
)

## Import memory utilities
from deepseekmoe_dynamic_routing_algorithms.source.memory_utils import cleanup_trainer, clear_cached_data

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ---- DeepSpeed configuration ----
ds_config = {
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "gradient_accumulation_steps": "auto",
    "fp16": {"enabled": True, "loss_scale": 0, "initial_scale_power": 16, "hysteresis": 2, "min_loss_scale": 1},
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
    "gradient_clipping": 1.0,
    "steps_per_print": 10,
    "wall_clock_breakdown": False,
    "zero_allow_untested_optimizer": True
}

# Lazy dataset loading (cached)
_tokenizer = None
_tokenized_datasets = None
_data_collator = None

def _prepare_data(
    data_file_path: str,
    model_class=None,
    tokenizer_name: str = None,
    split_ratio: float = 0.8,
    random_seed: int = 42,
    max_seq_len: int = MAX_SEQ_LEN   # defined elsewhere
):
    """
    Load a preprocessed train_sequences.txt, split into train/val,
    and tokenize using the appropriate tokenizer for the model.
    
    Args:
        data_file_path: Path to the preprocessed text file (one sequence per line).
        model_class: The model class (e.g., BaselineModel, DynmoeModel, DYNMoEBaseModel).
        tokenizer_name: Explicit tokenizer name (overrides automatic selection).
        split_ratio: Fraction of data to use for training (rest for validation).
        random_seed: Seed for shuffling.
        max_seq_len: Maximum sequence length for tokenization.
    
    Returns:
        tokenizer, tokenized_datasets, data_collator
    """
    # 1. Choose tokenizer based on model class or explicit name
    if tokenizer_name is None:
        if model_class is not None and model_class.__name__ == "DYNMoEBaseModel":
            # Use the tokenizer specified in the DYNMoE research paper.
            tokenizer_name = "microsoft/phi-2" 
        else:
            # BaselineModel and DynmoeModel use the DeepSeek tokenizer.
            tokenizer_name = "deepseek-ai/deepseek-moe-16b-base"

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Load the preprocessed data
    with open(data_file_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
        raise ValueError(f"No data found in {data_file_path}")

    # 3. Shuffle and split
    rng = random.Random(random_seed)
    rng.shuffle(lines)
    split_idx = int(len(lines) * split_ratio)
    train_lines = lines[:split_idx]
    val_lines = lines[split_idx:]

    # 4. Build DatasetDict
    from datasets import Dataset, DatasetDict
    train_dataset = Dataset.from_dict({"text": train_lines})
    val_dataset = Dataset.from_dict({"text": val_lines})
    dataset_dict = DatasetDict({"train": train_dataset, "validation": val_dataset})

    # 5. Tokenize
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

    # 6. Data collator for language modelling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8
    )

    return tokenizer, tokenized_datasets, data_collator

# ---- Core training function ----
def train_model(ModelClass, ConfigClass, output_dir, is_dynmoe=False,
                tokenizer=None, tokenized_datasets=None, data_collator=None,
                data_file_path=None, split_ratio=0.9, random_seed=42):
    """
    Train a model. If tokenizer/datasets are provided, use them; otherwise,
    load and tokenize from the preprocessed text file.
    """
    tokenizer, tokenized_datasets, data_collator = _prepare_data(
            data_file_path=data_file_path,
            model_class=ModelClass,
            split_ratio=split_ratio,
            random_seed=random_seed
        )
    # Instantiate model
    config = ConfigClass()
    model = ModelClass(config)
    model.resize_token_embeddings(len(tokenizer))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.config.use_cache = False
    model.train()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("\nModel Summary:")
    print(f" Total parameters: {total_params:,}")
    print(f" Trainable parameters: {trainable_params:,}")
    print(f" Model architecture:\n{model}\n")

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=NUM_EPOCHS,
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
        if "DYNMoE_baseline" in str(ModelClass):
            adaptive_callback = DYNMoEBaseCallback(audit_steps=DYNMoE_BASE_ADAPTIVE_AUDIT_STEPS)
        else:
            adaptive_callback = DynmoeRoutingCallback(audit_steps=DYNMOE_ROUTING_ADAPTIVE_AUDIT_STEPS)
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

    print("Starting training...")
    trainer.train()
    eval_results = trainer.evaluate()
    final_loss = eval_results.get("eval_loss", float('inf'))
    perplexity = math.exp(final_loss) if 0 < final_loss < 30 else float('inf')
    print(f"Final validation loss: {final_loss:.4f}")
    print(f"Validation Perplexity: {perplexity:.2f}")

    save_model_and_tokenizer(trainer, output_dir)
    print_training_summary(resource_monitor, moemetrics, trainer.state, eval_results, perplexity)

    return trainer

def run_training(ModelClass, ConfigClass, output_dir, is_dynmoe=False,
                 tokenizer=None, tokenized_datasets=None, data_collator=None,
                 data_file_path=None, split_ratio=0.9, random_seed=42):
    """
    Run training and then fully clean up GPU/CPU memory and the dataset cache.
    Returns nothing; the trainer is deleted internally.
    """
    print("=" * 60)
    print(f"TRAINING {ModelClass.__name__}")
    print("=" * 60)

    trainer = train_model(ModelClass, ConfigClass, output_dir, is_dynmoe,
                          tokenizer, tokenized_datasets, data_collator,
                          data_file_path=data_file_path,
                          split_ratio=split_ratio,
                          random_seed=random_seed)

    # ---- Cleanup ----
    cleanup_trainer(trainer)
    clear_cached_data()