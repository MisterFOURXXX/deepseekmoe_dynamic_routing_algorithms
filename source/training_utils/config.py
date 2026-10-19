import os
import torch

# Training hyper-parameters  
MAX_SEQ_LEN              = 256
PER_DEVICE_BATCH         = 8
GRAD_ACCUM               = 8
LEARNING_RATE            = 8e-5
NUM_EPOCHS               = 1
WARMUP_STEPS             = 100
WEIGHT_DECAY             = 0.05
EARLY_STOPPING_PATIENCE  = 3
EARLY_STOPPING_THRESHOLD = 0.001
world_size               = torch.cuda.device_count()

# Output directories  
OUTPUT_BASELINE        = "/content/deepseekmoe_dynamic_routing_algorithms/checkpoints/baseline"
OUTPUT_DYNMOE_BASE     = "/content/deepseekmoe_dynamic_routing_algorithms/checkpoints/dynmoe_baseline"
OUTPUT_DEEPSEEK_DYNMOE = "/content/deepseekmoe_dynamic_routing_algorithms/checkpoints/dynmoe_routing"