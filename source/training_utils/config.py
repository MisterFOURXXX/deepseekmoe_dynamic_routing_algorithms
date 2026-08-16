import torch

MAX_SEQ_LEN = 256
PER_DEVICE_BATCH = 8
GRAD_ACCUM = 8
LEARNING_RATE = 1e-4
NUM_EPOCHS = 3
WARMUP_STEPS = 100
WEIGHT_DECAY = 0.01
EARLY_STOPPING_PATIENCE = 3
EARLY_STOPPING_THRESHOLD = 0.001
world_size = torch.cuda.device_count()

# DeepSeek DYNMoE Configs
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
OUTPUT_BASELINE = "/kaggle/working/deepseekmoe_dynamic_routing_algorithms/checkpoints/baseline"
OUTPUT_DYNMOE_BASE = "/kaggle/working/deepseekmoe_dynamic_routing_algorithms/checkpoints/dynmoe_baseline"
OUTPUT_DEEPSEEK_DYNMOE = "/kaggle/working/deepseekmoe_dynamic_routing_algorithms/checkpoints/dynmoe_routing"