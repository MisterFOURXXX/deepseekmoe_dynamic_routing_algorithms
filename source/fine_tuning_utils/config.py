# Fine‑tuning hyperparameters
import torch


MAX_SEQ_LEN = 256
PER_DEVICE_BATCH = 4          # Reduced for stability
GRAD_ACCUM = 16               # Effective batch = 4 * 2 GPUs * 16 = 128
LEARNING_RATE = 5e-5
NUM_EPOCHS_FT = 3
WARMUP_STEPS = 50
WEIGHT_DECAY = 0.01
EARLY_STOPPING_PATIENCE = 3
EARLY_STOPPING_THRESHOLD = 0.005
world_size = torch.cuda.device_count()

# Dynamic MoE parameters
ADAPTIVE_AUDIT_STEPS = 10
MAX_ROUTED_EXPERTS = 6
DYNMOE_THRESHOLD_INIT = -0.08
BIAS_UPDATE_RATE = 0.001