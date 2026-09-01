import torch
from deepseekmoe_dynamic_routing_algorithms.source.deepseek_dynamics_routing.config import (
    DR_ADAPTIVE_AUDIT_STEPS,         
    DR_MAX_ROUTED_EXPERTS,      
    DR_MIN_ROUTED_EXPERTS,          
    DR_DYNMOE_THRESHOLD_INIT, 
    DR_BIAS_UPDATE_RATE,   
)

from deepseekmoe_dynamic_routing_algorithms.source.DYNMoE_baseline.config import (
    DYN_ADAPTIVE_AUDIT_STEPS,
    DYN_MAX_ROUTED_EXPERTS,
    DYN_DYNMOE_THRESHOLD_INIT,
    DYN_INITIAL_EXPERTS
)

from deepseekmoe_dynamic_routing_algorithms.source.training_utils.config import (
    OUTPUT_BASELINE,
    OUTPUT_DYNMOE_BASE,
    OUTPUT_DEEPSEEK_DYNMOE
)

# Fine-tuning hyperparameters
MAX_SEQ_LEN = 256
PER_DEVICE_BATCH = 4          # Reduced for stability
GRAD_ACCUM = 16               # Effective batch = 4 * 2 GPUs * 16 = 128
LEARNING_RATE = 5e-5
NUM_EPOCHS_FT = 1             # for testing, set to 100
WARMUP_STEPS = 50
WEIGHT_DECAY = 0.01
EARLY_STOPPING_PATIENCE = 3
EARLY_STOPPING_THRESHOLD = 0.005
world_size = torch.cuda.device_count()

# DeepSeek DYNMoE Configs
# GLOBAL DYNMOE CONFIGS (optimised for dynamic routing & resource efficiency)
ADAPTIVE_AUDIT_STEPS = DR_ADAPTIVE_AUDIT_STEPS         # for testing, set to 100
MAX_ROUTED_EXPERTS   = DR_MAX_ROUTED_EXPERTS
MIN_ROUTED_EXPERTS   = DR_MIN_ROUTED_EXPERTS
DYNMOE_THRESHOLD_INIT = DR_DYNMOE_THRESHOLD_INIT
BIAS_UPDATE_RATE   = DR_BIAS_UPDATE_RATE   

#  DYNMoE CONFIGS 
ADAPTIVE_AUDIT_STEPS = DYN_ADAPTIVE_AUDIT_STEPS         # for testing, set to 100
MAX_ROUTED_EXPERTS = DYN_MAX_ROUTED_EXPERTS              
DYNMOE_THRESHOLD_INIT = DYN_DYNMOE_THRESHOLD_INIT
INITIAL_EXPERTS = DYN_INITIAL_EXPERTS             

# Paths to pre‑trained models (output from training comparison)
PRETRAINED_BASELINE = OUTPUT_BASELINE + "/final"
# Output directories for fine‑tuned versions
OUTPUT_FT_BASELINE = "/kaggle/working/deepseekmoe_dynamic_routing_algorithms/checkpoints/baseline-ft"

# Paths to pre‑trained models (output from training comparison)
PRETRAINED_DYNMOE_BASE = OUTPUT_DYNMOE_BASE + "/final"
# Output directories for fine‑tuned versions
OUTPUT_FT_DYNMOE_BASE = "/kaggle/working/deepseekmoe_dynamic_routing_algorithms/checkpoints/dynmoe_baseline-ft"

# Paths to pre‑trained models (output from training comparison)
PRETRAINED_DYNMOE_ROUTING = OUTPUT_DEEPSEEK_DYNMOE + "/final"
# Output directories for fine‑tuned versions
OUTPUT_FT_DYNMOE_ROUTING = "/kaggle/working/deepseekmoe_dynamic_routing_algorithms/checkpoints/dynmoe_routing-ft"

