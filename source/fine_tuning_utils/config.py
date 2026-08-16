import torch

MAX_SEQ_LEN = 256
PER_DEVICE_BATCH = 4          # Reduced for stability
GRAD_ACCUM = 16               # Effective batch = 4 * 2 GPUs * 16 = 128
LEARNING_RATE = 5e-5
NUM_EPOCHS_FT = 1
WARMUP_STEPS = 50
WEIGHT_DECAY = 0.01
EARLY_STOPPING_PATIENCE = 3
EARLY_STOPPING_THRESHOLD = 0.005
world_size = torch.cuda.device_count()

# DeepSeek DYNMoE Configs
# GLOBAL DYNMOE CONFIGS (optimised for dynamic routing & resource efficiency)
ADAPTIVE_AUDIT_STEPS = 100          
MAX_ROUTED_EXPERTS   = 8
MIN_ROUTED_EXPERTS   = 2
DYNMOE_THRESHOLD_INIT = -0.05 
BIAS_UPDATE_RATE   = 0.0005   

#  DYNMoE CONFIGS 
ADAPTIVE_AUDIT_STEPS = 100           
MAX_ROUTED_EXPERTS = 32              
DYNMOE_THRESHOLD_INIT = -0.05
INITIAL_EXPERTS = 19                  

# Paths to pre‑trained models (output from training comparison)
PRETRAINED_BASELINE = "/kaggle/working/deepseekmoe_dynamic_routing_algorithms/checkpoints/baseline/final"
# Output directories for fine‑tuned versions
OUTPUT_FT_BASELINE = "/kaggle/working/deepseekmoe_dynamic_routing_algorithms/checkpoints/baseline-ft"

# Paths to pre‑trained models (output from training comparison)
PRETRAINED_DYNMOE_BASE = "/kaggle/working/deepseekmoe_dynamic_routing_algorithms/checkpoints/dynmoe_baseline/final"
# Output directories for fine‑tuned versions
OUTPUT_FT_DYNMOE_BASE = "/kaggle/working/deepseekmoe_dynamic_routing_algorithms/checkpoints/dynmoe_baseline-ft"

# Paths to pre‑trained models (output from training comparison)
PRETRAINED_DYNMOE_ROUTING = "/kaggle/working/deepseekmoe_dynamic_routing_algorithms/checkpoints/dynmoe_routing/final"
# Output directories for fine‑tuned versions
OUTPUT_FT_DYNMOE_ROUTING = "/kaggle/working/deepseekmoe_dynamic_routing_algorithms/checkpoints/dynmoe_routing-ft"

