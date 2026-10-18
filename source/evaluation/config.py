import os

DATASET_ROOT = "/content/deepseekmoe_dynamic_routing_algorithms/dataset"
MAX_SEQ_LEN  = 512

# EVAL_PARAMS - identical to the DeepSeekMoE evaluation code
EVAL_PARAMS = {
    "max_seq_len":        512,
    "eval_batch_size":    8,
    "gen_max_new_tokens": 256,
    "gen_min_new_tokens": 1,
    "repetition_penalty": 1.1,
    "num_beams":          1,
    "do_sample":          False,
}

# Default paths (Can be overridden by the runner)
_VARIANT_PATHS = {
    "deepseek_moe":        "./deepseek-moe-1b-checkpoints/final",
    "deepseek_moe_dynmoe": "./deepseek-moe-dyn-1b-checkpoints/final",
    "dynmoe":              "./dyn-1b-checkpoints-finetuned/final",
}