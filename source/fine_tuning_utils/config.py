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

# DeepSeek DYNMoE Configs
# GLOBAL DYNMOE CONFIGS (optimised for dynamic routing & resource efficiency)
ADAPTIVE_AUDIT_STEPS = 100          # allow biases to stabilise before pruning
MAX_ROUTED_EXPERTS = 6             # allow up to 6 experts (but dynamic routing will activate fewer)
MIN_ROUTED_EXPERTS = 2             # keep at least 2 experts to avoid collapse
DYNMOE_THRESHOLD_INIT = -0.02 #-0.01 #-0.01 #-0.02 #-0.03 #-0.01 #0.02 #-0.03   #-0.05    # positive threshold → sigmoid(0.5)≈0.62, harder to activate
SPARSITY_ALPHA = 0.08 #0.4 #0.05     #0.1     #0.4  #0.5   #0.5  #0.2  # stronger penalty for activating many experts (less number, less activate -> more number more activate)
BIAS_UPDATE_RATE = 0.001           # moderate bias update to balance load

#  DYNMoE CONFIGS 
ADAPTIVE_AUDIT_STEPS = 100           # paper suggests 100–300
MAX_ROUTED_EXPERTS = 6
DYNMOE_THRESHOLD_INIT = -0.02