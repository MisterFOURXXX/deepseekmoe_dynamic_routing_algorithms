from fine_tuning_utils.config import (
    OUTPUT_FT_BASELINE,
    OUTPUT_FT_DYNMOE_BASE,
    OUTPUT_FT_DYNMOE_ROUTING
)

EVAL_PARAMS = {
    "max_seq_len": 256,
    "eval_batch_size": 8,
    "gen_max_new_tokens": 256,
    "repetition_penalty": 1.35
}

# Default paths revise follow your storage path
OUTPUT_FT_BASELINE = OUTPUT_FT_BASELINE + "/final"
OUTPUT_FT_DYNMOE_BASE = OUTPUT_FT_DYNMOE_BASE + "/final"
OUTPUT_FT_DYNMOE_ROUTING = OUTPUT_FT_DYNMOE_ROUTING + "/final"