# Fine-Tuning Notebook Configuration
import sys
import os
import json
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
LEARNING_RATE = 4e-4
NUM_EPOCHS_FT = 20          # ceiling; early stopping normally stops at ~2-5
WARMUP_STEPS = 163          # 2% of the 20-epoch step count for ~10M tokens
WEIGHT_DECAY = 0.1
EARLY_STOPPING_PATIENCE = 3
EARLY_STOPPING_THRESHOLD = 0.001

# Optimizer (AdamW)
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.95
ADAM_EPS = 1e-8
MAX_GRAD_NORM = 1.0

# LR schedule
# From the DeepSeekMoE training approach: the learning rate is multiplied by
# 0.316 at 80% of the training steps, and again by 0.316 at 90% of them.
SCHEDULE = "warmup_step_decay"   # "warmup_step_decay" | "cosine" | "linear" | "constant"
DECAY_FRACS = (0.80, 0.90)
DECAY_GAMMA = 0.316

world_size = torch.cuda.device_count()

# Fine-tuning pretrained checkpoints
PRETRAINED_BASELINE       = OUTPUT_BASELINE        + "/final"   
PRETRAINED_DYNMOE_ROUTING = OUTPUT_DEEPSEEK_DYNMOE + "/final"

# Output directories for fine-tuned models
OUTPUT_FT_BASELINE       = OUTPUT_BASELINE        + "/baseline-ft"
OUTPUT_FT_DYNMOE_ROUTING = OUTPUT_DEEPSEEK_DYNMOE + "/dynmoe-routing-ft"