import torch

# Training hyperparameters
MAX_SEQ_LEN = 256
PER_DEVICE_BATCH = 8
GRAD_ACCUM = 8
LEARNING_RATE = 1e-6 #1e-4
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