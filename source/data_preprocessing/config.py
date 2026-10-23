import os

HF_TOKEN             = "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" 
TOKENIZER_NAME       = "deepseek-ai/deepseek-moe-16b-base"

MAX_SEQ_LEN          = 384
SAMPLE_RATIO         = 0.0001
BENCH_SAMPLE_RATIO   = 0.005

DATASET_ROOT = "/content/deepseekmoe_dynamic_routing_algorithms/dataset"
REPO_ROOT = "/content/deepseekmoe_dynamic_routing_algorithms"

DEFAULT_TOKENIZER_PATH = os.path.join(DATASET_ROOT, "tokenizer")
DEFAULT_TRAIN_PATH     = os.path.join(DATASET_ROOT, "slimpajama_train")
DEFAULT_FINE_PATH      = os.path.join(DATASET_ROOT, "slimpajama_finetune")
DEFAULT_VAL_PATH       = os.path.join(DATASET_ROOT, "slimpajama_val")

DEFAULT_MINIPILE_PATH  = os.path.join(DATASET_ROOT, "minipile")
DEFAULT_ARC_PATH       = os.path.join(DATASET_ROOT, "arc")
DEFAULT_TRIVIAQA_PATH  = os.path.join(DATASET_ROOT, "triviaqa")

PATHS = {
    "tokenizer":           DEFAULT_TOKENIZER_PATH,
    "slimpajama_train":    DEFAULT_TRAIN_PATH,
    "slimpajama_finetune": DEFAULT_FINE_PATH,
    "slimpajama_val":      DEFAULT_VAL_PATH,
    "minipile":            DEFAULT_MINIPILE_PATH,
    "arc":                 DEFAULT_ARC_PATH,
    "triviaqa":            DEFAULT_TRIVIAQA_PATH,
}