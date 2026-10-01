import sys
sys.path.append("..")

import torch

from ..training_utils.config import (
    OUTPUT_BASELINE,
    OUTPUT_DYNMOE_BASE,
    OUTPUT_DEEPSEEK_DYNMOE
)

# Fine-tuning hyperparameters
MAX_SEQ_LEN = 256
PER_DEVICE_BATCH = 4          # Reduced for stability
GRAD_ACCUM = 16               # Effective batch = 4 * 2 GPUs * 16 = 128
LEARNING_RATE = 3e-5
NUM_EPOCHS_FT = 5             # for testing, set to 100
WARMUP_STEPS = 50
WEIGHT_DECAY = 0.01
EARLY_STOPPING_PATIENCE = 3
EARLY_STOPPING_THRESHOLD = 0.005
world_size = torch.cuda.device_count()            

# Fine-tuning config
PRETRAINED_BASELINE       = OUTPUT_BASELINE        + "/final"   
PRETRAINED_DYNMOE_BASE    = OUTPUT_DYNMOE_BASE     + "/final"
PRETRAINED_DYNMOE_ROUTING = OUTPUT_DEEPSEEK_DYNMOE + "/final"

# Output directories for fine-tuned models
OUTPUT_FT_BASELINE       = OUTPUT_BASELINE        + "/baseline-ft"
OUTPUT_FT_DYNMOE_BASE    = OUTPUT_DYNMOE_BASE     + "/dynmoe-baseline-ft"
OUTPUT_FT_DYNMOE_ROUTING = OUTPUT_DEEPSEEK_DYNMOE + "/dynmoe-routing-ft"
