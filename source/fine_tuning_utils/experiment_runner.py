import os
import sys

repo_path =  ".."
os.chdir(repo_path)                 # Move into the repo
sys.path.insert(0, os.getcwd())     # Ensure the repo root is on sys.path

import torch
import math
from transformers import AutoTokenizer
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from transformers import AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset

# Imports for models
from deepseekmoe_dynamic_routing_algorithms.source.deepseek_baseline.config import DeepseekConfig as BaselineConfig
from deepseekmoe_dynamic_routing_algorithms.source.deepseek_baseline.model import DeepseekForCausalLM as BaselineModel

from deepseekmoe_dynamic_routing_algorithms.source.DYNMoE_baseline.config import DynMoEConfig as DYNMoEBaseConfig
from deepseekmoe_dynamic_routing_algorithms.source.DYNMoE_baseline.model import DynMoEForCausalLM as DYNMoEBaseModel
from deepseekmoe_dynamic_routing_algorithms.source.DYNMoE_baseline.adaptive_tuning import AdaptiveExpertTuningCallback as DYNMoEBaseCallback

from deepseekmoe_dynamic_routing_algorithms.source.deepseek_dynamics_routing.config import DeepseekConfig as DynmoeConfig
from deepseekmoe_dynamic_routing_algorithms.source.deepseek_dynamics_routing.model import DeepseekForCausalLM as DynmoeModel
from deepseekmoe_dynamic_routing_algorithms.source.deepseek_dynamics_routing.adaptive_tuning import AdaptiveExpertTuningCallback as DynmoeRoutingCallback

# Utilities
from deepseekmoe_dynamic_routing_algorithms.source.fine_tuning_utils.monitoring import ResourceMonitorCallback, MoEMetricsCallback
from deepseekmoe_dynamic_routing_algorithms.source.fine_tuning_utils.save_model import save_finetuned_model
from deepseekmoe_dynamic_routing_algorithms.source.fine_tuning_utils.summarization import print_finetuning_summary
from deepseekmoe_dynamic_routing_algorithms.source.data_preprocessing import load_and_preprocess_multiwoz

import gc

from deepseekmoe_dynamic_routing_algorithms.source.fine_tuning_utils.config import (
    MAX_SEQ_LEN,
    PER_DEVICE_BATCH,
    GRAD_ACCUM,
    LEARNING_RATE,
    NUM_EPOCHS_FT,
    WARMUP_STEPS,
    WEIGHT_DECAY,
    EARLY_STOPPING_PATIENCE,
    EARLY_STOPPING_THRESHOLD,
    world_size
)

# Add after other imports
from deepseekmoe_dynamic_routing_algorithms.source.fine_tuning_utils.model_loading import load_model_and_tokenizer

# Model Training Configurations
import warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

torch.cuda.empty_cache()
#world_size = torch.cuda.device_count()
print(f"Number of GPUs: {world_size}")

# DeepSpeed config (same as used in fine‑tuning example)
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
    "gradient_clipping": 0.5,   # float is fine here
    "steps_per_print": 10,
    "wall_clock_breakdown": False,
    "zero_allow_untested_optimizer": True   # if needed
}

# Load dataset
torch.cuda.empty_cache()
world_size = torch.cuda.device_count()
print(f"Number of GPUs: {world_size}")

train_sequences, val_sequences, test_sequences = load_and_preprocess_multiwoz(
    zip_path="/kaggle/working/deepseekmoe_dynamic_routing_algorithms/dataset/MultiWOZ-coref/MultiWOZ2_3.zip",
    sample_size=300,
    random_seed=42
)
print(f"Train sequences: {len(train_sequences)}")
print(f"Validation sequences: {len(val_sequences)}")
print(f"Test sequences: {len(test_sequences)}")

# Tokenizer 
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-moe-16b-base", use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Tokenize dataset
dataset = load_dataset("text", data_files={"train": "train_sequences.txt", "validation": "val_sequences.txt"})

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_SEQ_LEN,
        padding=False,
        return_attention_mask=True,
    )

tokenized_datasets = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"],
    num_proc=2,
    desc="Tokenizing datasets"
)
print(f"Train samples: {len(tokenized_datasets['train'])}")
print(f"Validation samples: {len(tokenized_datasets['validation'])}")

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
    pad_to_multiple_of=8
)

def fine_tune_model(pretrained_path, output_dir):
    """
    Load the correct model automatically from the checkpoint,
    then fine‑tune it on the MultiWOZ dataset.
    """
    #print(f"\n{'='*60}")
    #print(f"FINE‑TUNING from {pretrained_path}")
    #print(f"{'='*60}")

    # Load model and tokenizer using the auto‑detection function
    model, tokenizer = load_model_and_tokenizer(pretrained_path)

    # Move to GPU and set training mode
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.config.use_cache = False
    model.train()

    # Detect whether this is a DYNMoE variant (for callback selection)
    # We check the config we just loaded – we can also inspect the model class name
    is_dynmoe = "DynMoE" in model.__class__.__name__  # or check config_dict

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("\nModel Summary:")
    print(f" Total parameters: {total_params:,}")
    print(f" Trainable parameters: {trainable_params:,}")
    print(f" Model architecture:\n{model}\n")

    # Training arguments (same as before)
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

    # Callbacks
    resource_monitor = ResourceMonitorCallback()
    moemetrics = MoEMetricsCallback(
        tokenized_datasets["validation"],
        tokenizer,
        data_collator
    )
    callbacks = [resource_monitor, moemetrics]

    if is_dynmoe:
        # Use the appropriate callback based on model class name
        if "DYNMoE_baseline" in model.__class__.__name__:
            adaptive_callback = DYNMoEBaseCallback(audit_steps=10)
        else:
            adaptive_callback = DynmoeRoutingCallback(audit_steps=50)
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

    # Save the fine‑tuned model
    final_output_dir = os.path.join(output_dir, "final")
    unwrapped = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model
    unwrapped.save_pretrained(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)
    print(f"Fine‑tuned model saved to {final_output_dir}")

    save_finetuned_model(trainer, output_dir)
    print_finetuning_summary(resource_monitor, moemetrics, train_result, eval_results, perplexity)

    return trainer

def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

def cleanup_trainer(trainer):
    if trainer is not None:
        if hasattr(trainer, 'model'):
            del trainer.model
        del trainer
    clear_gpu_memory()

def run_and_cleanup_experiment(pretrained_path, output_dir):
    """
    Run a fine‑tuning experiment for a model loaded from the given checkpoint,
    then clean up GPU memory completely.
    """
    print("=" * 60)
    print(f"FINE-TUNING from {pretrained_path}")
    print("=" * 60)

    trainer = fine_tune_model(pretrained_path, output_dir)

    # Aggressive cleanup
    cleanup_trainer(trainer)
