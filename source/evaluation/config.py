import os
from ..fine_tuning_utils import (
    OUTPUT_FT_BASELINE,
    OUTPUT_FT_DYNMOE_ROUTING
)
from ..data_preprocessing import DATASET_ROOT

MAX_SEQ_LEN = 256

# EVAL_PARAMS - identical to the DeepSeekMoE evaluation code
EVAL_PARAMS = {
    "max_seq_len":        256,
    "eval_batch_size":    8,
    "gen_max_new_tokens": 256,
    "gen_min_new_tokens": 1,
    "repetition_penalty": 1.1,
    "num_beams":          1,
    "do_sample":          False,
}

# Paths to the final saved fine-tuned models
FINAL_FT_BASELINE       = os.path.join(OUTPUT_FT_BASELINE,       "final")
FINAL_FT_DYNMOE_ROUTING = os.path.join(OUTPUT_FT_DYNMOE_ROUTING, "final")