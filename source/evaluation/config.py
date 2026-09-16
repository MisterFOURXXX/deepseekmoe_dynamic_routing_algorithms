import sys
sys.path.append("..")

#from ..fine_tuning_utils.config import (
#    OUTPUT_FT_BASELINE,
#    OUTPUT_FT_DYNMOE_BASE,
#    OUTPUT_FT_DYNMOE_ROUTING
#)

from ..training_utils.config import (
    OUTPUT_BASELINE,
    OUTPUT_DYNMOE_BASE,
    OUTPUT_DEEPSEEK_DYNMOE
)

EVAL_PARAMS = {
    "max_seq_len": 256,
    "eval_batch_size": 8,
    "gen_max_new_tokens": 256,
    "repetition_penalty": 1.35
}

# Default paths revise follow your storage path
# OUTPUT_FT_BASELINE = OUTPUT_FT_BASELINE + "/final"
# OUTPUT_FT_DYNMOE_BASE = OUTPUT_FT_DYNMOE_BASE + "/final"
# OUTPUT_FT_DYNMOE_ROUTING = OUTPUT_FT_DYNMOE_ROUTING + "/final"

# Paths to pre‑trained models (output from training comparison)
PRETRAINED_BASELINE = OUTPUT_BASELINE + "/final"

# Paths to pre‑trained models (output from training comparison)
PRETRAINED_DYNMOE_BASE = OUTPUT_DYNMOE_BASE + "/final"

# Paths to pre‑trained models (output from training comparison)
PRETRAINED_DYNMOE_ROUTING = OUTPUT_DEEPSEEK_DYNMOE + "/final"