# config.py
import os
import sys
#repo_path =  ".."
#os.chdir(repo_path)
#sys.path.insert(0, os.getcwd())

import math
from typing import Optional, Dict, Any
from transformers.utils import logging
from transformers.configuration_utils import PretrainedConfig

# ====================== GLOBAL DYNMoE CONFIGS ======================
ADAPTIVE_AUDIT_STEPS = 100           # paper suggests 100–300
MAX_ROUTED_EXPERTS = 32              # max experts for Phi‑2
DYNMOE_THRESHOLD_INIT = 0.02
INITIAL_EXPERTS = 14                  # FIXED: paper starts with 2 experts for Phi‑2

# ====================== PHI‑2 CONFIG (with DYNMoE extensions) ======================
class DynMoEConfig(PretrainedConfig):
    model_type = "dynmoe"
    
    def __init__(
        self,
        vocab_size=102400,
        hidden_size=1024,
        num_hidden_layers=6,
        num_attention_heads=32,
        intermediate_size=4096,
        hidden_act="gelu_new",
        max_position_embeddings=2048,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        rms_norm_eps=1e-5,
        pad_token_id=None,
        bos_token_id=100000,
        eos_token_id=100001,
        # DYNMoE specific
        num_experts=INITIAL_EXPERTS,      # FIXED: start with 2 experts
        max_expert_num=MAX_ROUTED_EXPERTS,
        replace_layers=None,              # default to all layers
        moe_intermediate_size=1024,
        moe_aux_loss_weight=0.01,
        adaptive_experts=True,
        # compatibility
        is_decoder=True,
        use_cache=True,
        attention_bias=True,
        attention_dropout=0.0,
        pretraining_tp=1,
        tie_word_embeddings=True,
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.rms_norm_eps = rms_norm_eps
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        if replace_layers is None:
            replace_layers = list(range(num_hidden_layers))
        self.replace_layers = replace_layers
        self.num_experts = num_experts
        self.max_expert_num = max_expert_num
        self.moe_intermediate_size = moe_intermediate_size
        self.moe_aux_loss_weight = moe_aux_loss_weight
        self.adaptive_experts = adaptive_experts

        self.is_decoder = is_decoder
        self.use_cache = use_cache
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.pretraining_tp = pretraining_tp
        self.tie_word_embeddings = tie_word_embeddings
        self.n_routed_experts = num_experts
        self._attn_implementation = "eager"
