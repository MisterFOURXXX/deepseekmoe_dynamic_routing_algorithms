import os
from datasets import load_dataset_builder
from .statistical_sample import DOMAIN_SHARES

HF_TOKEN             = "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"
TOKENIZER_NAME       = "deepseek-ai/deepseek-moe-16b-base"

# Default parameters (can be overridden from notebook)
MAX_SEQ_LEN              = 384
MINORITY_CONFIDENCE      = 0.99
MINORITY_MARGIN_ERROR    = 0.0130

BENCH_SAMPLE_RATIO   = 1.0   # 100% of MiniPile test set

PRETRAIN_SEED      = 42
FINETUNE_SEED      = 43

# MiniPile evaluation
# Dynamically fetch the exact number of rows in the MiniPile dataset
try:
    print("Fetching MiniPile dataset metadata...")
    _builder_mp = load_dataset_builder("JeanKaddour/minipile", split="test")
    MINIPILE_TOTAL_ROWS = _builder_mp.info.splits["test"].num_examples
    print(f"Dynamically fetched MiniPile test size: {MINIPILE_TOTAL_ROWS:,} rows")
except Exception as e:
    print(f"Warning: Could not fetch MiniPile size dynamically ({e}). Defaulting to 10,000.")
    MINIPILE_TOTAL_ROWS = 10_000

BOOTSTRAP_ITERATIONS    = 1_000

# Statistical verification thresholds
CHI_SQUARE_P_THRESHOLD  = 0.95
KL_DIVERGENCE_THRESHOLD = 0.001
VOCAB_SIZE              = 102400

DATASET_ROOT = "/content/deepseekmoe_dynamic_routing_algorithms/dataset"
REPO_ROOT    = "/content/deepseekmoe_dynamic_routing_algorithms"

DEFAULT_TOKENIZER_PATH  = os.path.join(DATASET_ROOT, "tokenizer")
DEFAULT_TRAIN_PATH      = os.path.join(DATASET_ROOT, "slimpajama_train")
DEFAULT_FINE_PATH       = os.path.join(DATASET_ROOT, "slimpajama_finetune")
DEFAULT_VAL_PATH        = os.path.join(DATASET_ROOT, "slimpajama_val")
DEFAULT_FINE_VAL_PATH   = os.path.join(DATASET_ROOT, "slimpajama_finetune_val")
DEFAULT_MINIPILE_PATH   = os.path.join(DATASET_ROOT, "minipile")

PATHS = {
    "tokenizer":               DEFAULT_TOKENIZER_PATH,
    "slimpajama_train":        DEFAULT_TRAIN_PATH,
    "slimpajama_finetune":     DEFAULT_FINE_PATH,
    "slimpajama_val":          DEFAULT_VAL_PATH,
    "slimpajama_finetune_val": DEFAULT_FINE_VAL_PATH,
    "minipile":                DEFAULT_MINIPILE_PATH,
}

META_FILE           = os.path.join(DATASET_ROOT, "_preprocessing_meta.json")