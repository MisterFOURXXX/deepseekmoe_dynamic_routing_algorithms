import os

# Config constants (defined in .config)
from .config import (
    HF_TOKEN,
    TOKENIZER_NAME,
    MAX_SEQ_LEN,
    SAMPLE_RATIO,
    BENCH_SAMPLE_RATIO,
    AMBIGQA_MAX_EXAMPLES,
    DATASET_ROOT,
)

# Pre-computed dataset paths 
DEFAULT_TOKENIZER_PATH = os.path.join(DATASET_ROOT, "tokenizer")

# Language-modeling splits (SlimPajama-6B)
DEFAULT_TRAIN_PATH     = os.path.join(DATASET_ROOT, "slimpajama_train")
DEFAULT_FINE_PATH      = os.path.join(DATASET_ROOT, "slimpajama_finetune")
DEFAULT_VAL_PATH       = os.path.join(DATASET_ROOT, "slimpajama_val")
DEFAULT_TEST_PATH      = os.path.join(DATASET_ROOT, "slimpajama_test")

# Evaluation benchmarks
DEFAULT_MINIPILE_PATH  = os.path.join(DATASET_ROOT, "minipile")
DEFAULT_ARC_PATH       = os.path.join(DATASET_ROOT, "arc")
DEFAULT_RACE_PATH      = os.path.join(DATASET_ROOT, "race")
DEFAULT_TRIVIAQA_PATH  = os.path.join(DATASET_ROOT, "triviaqa")
DEFAULT_AMBIGQA_PATH   = os.path.join(DATASET_ROOT, "ambigqa")

# Dict (useful when looping over benchmarks)
DATASET_PATHS = {
    "tokenizer": DEFAULT_TOKENIZER_PATH,
    "train":     DEFAULT_TRAIN_PATH,
    "finetune":  DEFAULT_FINE_PATH,
    "val":       DEFAULT_VAL_PATH,
    "test":      DEFAULT_TEST_PATH,
    "minipile":  DEFAULT_MINIPILE_PATH,
    "arc":       DEFAULT_ARC_PATH,
    "race":      DEFAULT_RACE_PATH,
    "triviaqa":  DEFAULT_TRIVIAQA_PATH,
    "ambigqa":   DEFAULT_AMBIGQA_PATH,
}

# Preprocessing module 
from . import preprocessing

# Public API
__all__ = [
    # config constants
    "HF_TOKEN",
    "TOKENIZER_NAME",
    "MAX_SEQ_LEN",
    "SAMPLE_RATIO",
    "BENCH_SAMPLE_RATIO",
    "AMBIGQA_MAX_EXAMPLES",
    "DATASET_ROOT",

    # individual paths
    "DEFAULT_TOKENIZER_PATH",
    "DEFAULT_TRAIN_PATH",
    "DEFAULT_FINE_PATH",
    "DEFAULT_VAL_PATH",
    "DEFAULT_TEST_PATH",
    "DEFAULT_MINIPILE_PATH",
    "DEFAULT_ARC_PATH",
    "DEFAULT_RACE_PATH",
    "DEFAULT_TRIVIAQA_PATH",
    "DEFAULT_AMBIGQA_PATH",

    # dict of paths
    "DATASET_PATHS",

    # modules
    "preprocessing",
]