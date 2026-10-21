import sys
sys.path.append("..")

import torch

from ..training_utils.config import (
    OUTPUT_BASELINE,
    OUTPUT_DEEPSEEK_DYNMOE
)

# Fine-tuning hyperparameters 
MAX_SEQ_LEN = 384
PER_DEVICE_BATCH = 8
GRAD_ACCUM = 8
LEARNING_RATE = 8e-5
NUM_EPOCHS_FT = 1
WARMUP_STEPS = 100
WEIGHT_DECAY = 0.05
EARLY_STOPPING_PATIENCE = 3
EARLY_STOPPING_THRESHOLD = 0.001
world_size = torch.cuda.device_count()            

# Fine-tuning pretrained checkpoints
PRETRAINED_BASELINE       = OUTPUT_BASELINE        + "/final"   
PRETRAINED_DYNMOE_ROUTING = OUTPUT_DEEPSEEK_DYNMOE + "/final"

# Output directories for fine-tuned models
OUTPUT_FT_BASELINE       = OUTPUT_BASELINE        + "/baseline-ft"
OUTPUT_FT_DYNMOE_ROUTING = OUTPUT_DEEPSEEK_DYNMOE + "/dynmoe-routing-ft"