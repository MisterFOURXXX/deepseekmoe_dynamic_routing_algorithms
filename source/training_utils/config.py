import torch
from deepseekmoe_dynamic_routing_algorithms.source.deepseek_dynamics_routing.config import (
    ADAPTIVE_AUDIT_STEPS as DR_ADAPTIVE_AUDIT_STEPS,         
    MAX_ROUTED_EXPERTS as DR_MAX_ROUTED_EXPERTS,      
    MIN_ROUTED_EXPERTS as DR_MIN_ROUTED_EXPERTS,          
    DYNMOE_THRESHOLD_INIT as DR_DYNMOE_THRESHOLD_INIT, 
    BIAS_UPDATE_RATE as DR_BIAS_UPDATE_RATE,   
)

from deepseekmoe_dynamic_routing_algorithms.source.DYNMoE_baseline.config import (
    ADAPTIVE_AUDIT_STEPS as DYN_ADAPTIVE_AUDIT_STEPS,
    MAX_ROUTED_EXPERTS as DYN_MAX_ROUTED_EXPERTS,
    DYNMOE_THRESHOLD_INIT as DYN_DYNMOE_THRESHOLD_INIT,
    INITIAL_EXPERTS as DYN_INITIAL_EXPERTS
)

# Training hyperparameters
MAX_SEQ_LEN = 256
PER_DEVICE_BATCH = 8
GRAD_ACCUM = 8
LEARNING_RATE = 1e-4
NUM_EPOCHS = 1         # for testing, set to 100
WARMUP_STEPS = 100
WEIGHT_DECAY = 0.01
EARLY_STOPPING_PATIENCE = 3
EARLY_STOPPING_THRESHOLD = 0.001
world_size = torch.cuda.device_count()         

# Paths to pre‑trained models (output from training comparison)
OUTPUT_BASELINE = "/kaggle/working/deepseekmoe_dynamic_routing_algorithms/checkpoints/baseline"
OUTPUT_DYNMOE_BASE = "/kaggle/working/deepseekmoe_dynamic_routing_algorithms/checkpoints/dynmoe_baseline"
OUTPUT_DEEPSEEK_DYNMOE = "/kaggle/working/deepseekmoe_dynamic_routing_algorithms/checkpoints/dynmoe_routing"