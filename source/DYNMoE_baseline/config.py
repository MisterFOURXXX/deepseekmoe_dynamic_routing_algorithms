# config.py
import os
import sys
#repo_path =  ".."
#os.chdir(repo_path)
#sys.path.insert(0, os.getcwd())

import math
from transformers.utils import logging
from transformers.configuration_utils import PretrainedConfig

# ====================== GLOBAL DYNMoE CONFIGS ======================
ADAPTIVE_AUDIT_STEPS = 100           # paper suggests 100–300
MAX_ROUTED_EXPERTS = 6
DYNMOE_THRESHOLD_INIT = -0.02

# ====================== DYNMoE CONFIG ======================
class DynMoEConfig(PretrainedConfig):
    model_type = "dynmoe"
    
    def __init__(
        self,
        vocab_size=50257,
        hidden_size=1024,
        num_hidden_layers=12,
        num_attention_heads=16,
        intermediate_size=4096,
        hidden_act="gelu",
        max_position_embeddings=1024,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        pad_token_id=50256,
        bos_token_id=50256,
        eos_token_id=50256,
        # DYNMoE specific
        num_experts=8,
        max_expert_num=16,
        replace_layers=None,
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
        rms_norm_eps=1e-5,
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
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        self.num_experts = num_experts
        self.max_expert_num = max_expert_num
        self.replace_layers = replace_layers or list(range(6, 12))
        self.moe_intermediate_size = moe_intermediate_size
        self.moe_aux_loss_weight = moe_aux_loss_weight
        self.adaptive_experts = adaptive_experts

        self.is_decoder = is_decoder
        self.use_cache = use_cache
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.pretraining_tp = pretraining_tp
        self.tie_word_embeddings = tie_word_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.n_routed_experts = num_experts
        self._attn_implementation = "eager"