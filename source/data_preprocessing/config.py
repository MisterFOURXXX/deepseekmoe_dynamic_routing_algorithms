# ============================================================
# source/data_preprocessing/config.py
# ============================================================
import os

HF_TOKEN             = "hf_..." 
TOKENIZER_NAME       = "deepseek-ai/deepseek-moe-16b-base"

MAX_SEQ_LEN          = 384
SAMPLE_RATIO         = 0.0001
BENCH_SAMPLE_RATIO   = 0.005
AMBIGQA_MAX_EXAMPLES = 50

DATASET_ROOT = "/content/deepseekmoe_dynamic_routing_algorithms/dataset"

DEFAULT_TOKENIZER_PATH = os.path.join(DATASET_ROOT, "tokenizer")
DEFAULT_TRAIN_PATH     = os.path.join(DATASET_ROOT, "slimpajama_train")
DEFAULT_FINE_PATH      = os.path.join(DATASET_ROOT, "slimpajama_finetune")
DEFAULT_VAL_PATH       = os.path.join(DATASET_ROOT, "slimpajama_val")

DEFAULT_MINIPILE_PATH  = os.path.join(DATASET_ROOT, "minipile")
DEFAULT_ARC_PATH       = os.path.join(DATASET_ROOT, "arc")
DEFAULT_RACE_PATH      = os.path.join(DATASET_ROOT, "race")
DEFAULT_TRIVIAQA_PATH  = os.path.join(DATASET_ROOT, "triviaqa")
DEFAULT_AMBIGQA_PATH   = os.path.join(DATASET_ROOT, "ambigqa")

PATHS = {
    "tokenizer":           DEFAULT_TOKENIZER_PATH,
    "slimpajama_train":    DEFAULT_TRAIN_PATH,
    "slimpajama_finetune": DEFAULT_FINE_PATH,
    "slimpajama_val":      DEFAULT_VAL_PATH,
    "minipile":            DEFAULT_MINIPILE_PATH,
    "arc":                 DEFAULT_ARC_PATH,
    "race":                DEFAULT_RACE_PATH,
    "triviaqa":            DEFAULT_TRIVIAQA_PATH,
    "ambigqa":             DEFAULT_AMBIGQA_PATH,
}