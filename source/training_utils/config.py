import os
import torch

# Dataset / preprocessing
MAX_SEQ_LEN              = 384
MINORITY_CONFIDENCE      = 0.99
MINORITY_MARGIN_ERROR    = 0.0130

# Training hyperparameter
PER_DEVICE_BATCH         = 8
GRAD_ACCUM               = 8
LEARNING_RATE            = 1e-3          # peak LR (pretrain from scratch)
NUM_EPOCHS               = 2
WARMUP_STEPS             = 100
WEIGHT_DECAY             = 0.1           # DeepSeekMoE paper §4.1.3 (matmul weights only)
EARLY_STOPPING_PATIENCE  = 3
EARLY_STOPPING_THRESHOLD = 0.001

# AdamW 
ADAM_BETA1               = 0.9
ADAM_BETA2               = 0.95          # paper §4.1.3
ADAM_EPS                 = 1e-8
MAX_GRAD_NORM            = 1.0

# LR schedule
SCHEDULE                 = "warmup_step_decay"
DECAY_FRACS              = (0.80, 0.90)  # fractions of total optimizer steps
DECAY_GAMMA              = 0.316         # 10**-0.5, applied once per DECAY_FRACS point

# Runtime
world_size               = torch.cuda.device_count()

# Output directories
OUTPUT_BASELINE          = "/content/deepseekmoe_dynamic_routing_algorithms/checkpoints/baseline"
OUTPUT_DEEPSEEK_DYNMOE   = "/content/deepseekmoe_dynamic_routing_algorithms/checkpoints/dynmoe_routing"