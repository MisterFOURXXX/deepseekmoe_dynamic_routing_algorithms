EVAL_PARAMS = {
    "max_seq_len": 256,
    "eval_batch_size": 8,
    "gen_max_new_tokens": 256,
    "repetition_penalty": 1.35
}

# Default paths revise follow your storage path
OUTPUT_FT_BASELINE = "/kaggle/working/deepseekmoe_dynamic_routing_algorithms/checkpoints/baseline-ft/final"
OUTPUT_FT_DYNMOE_BASE = "/kaggle/working/deepseekmoe_dynamic_routing_algorithms/checkpoints/dynmoe_baseline-ft/final"
OUTPUT_FT_DYNMOE_ROUTING = "/kaggle/working/deepseekmoe_dynamic_routing_algorithms/checkpoints/dynmoe_routing-ft/final"