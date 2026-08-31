import os
import sys
import torch
import transformers
import accelerate
import deepspeed
import math

import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_attn_mask_utils import (
    AttentionMaskConverter,
    _prepare_4d_attention_mask,
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    SequenceClassifierOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS, is_torch_greater_or_equal_than_1_13
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from transformers import GenerationConfig
from transformers.utils.import_utils import is_torch_fx_available

# Flash Attention (optional, conditionally imported)
if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input

# FX wrapping
if is_torch_fx_available():
    if not is_torch_greater_or_equal_than_1_13:
        import torch.fx
        _prepare_4d_causal_attention_mask = torch.fx.wrap(_prepare_4d_causal_attention_mask)

import math
import torch
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.utils import logging
from transformers import GenerationConfig
from typing import List, Optional, Tuple, Union, Dict, Any

# Test if flash attention is available through PyTorch
from torch.backends.cuda import flash_sdp_enabled, mem_efficient_sdp_enabled

# Relative imports
from .config import DynMoEConfig
from .adaptive_tuning import AdaptiveExpertTuningCallback
from .config import (
    ADAPTIVE_AUDIT_STEPS,
    MAX_ROUTED_EXPERTS,
    DYNMOE_THRESHOLD_INIT,
    INITIAL_EXPERTS,
)

# RMSNorm (Phi‑2)
class DynMoERMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


# Rotary Embeddings
class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor,
                         cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# Dynamic MoE Gate (Top‑Any) 
class DynamicMoEGate(nn.Module):
    def __init__(self, config: DynMoEConfig):
        super().__init__()
        self.config = config
        self.n_routed_experts = config.num_experts
        self.hidden_size = config.hidden_size
        self.max_expert_num = config.max_expert_num

        # Expert representations (weights) and per‑expert thresholds
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.hidden_size)))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.thresholds = nn.Parameter(
            torch.full((self.n_routed_experts,), DYNMOE_THRESHOLD_INIT)
        )

        # Buffers for adaptive tuning
        self.register_buffer('routing_records', torch.zeros(self.n_routed_experts))
        self.register_buffer('dropped_embeddings', torch.zeros(self.hidden_size))
        self.audit_counter = 0
        self.audit_interval = ADAPTIVE_AUDIT_STEPS

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        bsz, seq_len, h = hidden_states.shape
        x = hidden_states.view(-1, h)                     # (N, d)

        norm_x = F.normalize(x, p=2, dim=-1)
        norm_w = F.normalize(self.weight, p=2, dim=-1)
        scores = torch.matmul(norm_x, norm_w.T)           # (N, K)

        g = torch.sigmoid(scores) - torch.sigmoid(self.thresholds)  # (N, K)
        g_bin = (g > 0).float()                           # (N, K) 0/1
        k_r = g_bin.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        if self.training:
            g_bin = g_bin + (g - g.detach())  # STE

        topk_weight = g_bin / k_r             # (N, K)
        topk_idx = torch.arange(self.n_routed_experts, device=g_bin.device).unsqueeze(0).expand_as(g_bin)

        # Test‑time fallback
        if not self.training:
            zero_mask = (k_r.squeeze(-1) == 0)
            if zero_mask.any():
                max_idx = torch.argmax(scores, dim=-1)   # (N,)
                g_bin[zero_mask] = 0
                g_bin[zero_mask, max_idx[zero_mask]] = 1.0
                k_r = g_bin.sum(dim=-1, keepdim=True).clamp(min=1e-8)
                topk_weight = g_bin / k_r

        # Routing recording for adaptive tuning
        if self.training:
            self.audit_counter += 1
            with torch.no_grad():
                batch_activ = g_bin.mean(dim=0) * bsz * seq_len
                self.routing_records += batch_activ
                dropped_mask = (g_bin.sum(dim=-1) == 0)
                if dropped_mask.any():
                    self.dropped_embeddings += x[dropped_mask].sum(dim=0)

        aux_loss = self._sparse_simple_loss() if self.training else None
        return topk_idx, topk_weight, aux_loss

    def _sparse_simple_loss(self) -> torch.Tensor:
        Wg = self.weight
        K = self.n_routed_experts
        gram = torch.matmul(Wg, Wg.T)
        diversity = torch.norm(gram - torch.eye(K, device=Wg.device), p='fro') ** 2
        simplicity = torch.mean(torch.norm(Wg, p=2, dim=1))
        return diversity + simplicity

    # Internal tuning method (called by DynMoEMLP.adaptive_tune)
    def _tune_experts(self) -> Dict[str, Any]:
        """Called periodically to add/remove experts based on routing statistics."""
        if self.audit_counter < self.audit_interval:
            return {'added': False, 'removed_idx': None}

        added = False
        removed_idx = None
        with torch.no_grad():
            # Experts Removal
            if self.n_routed_experts > 2:
                active = self.routing_records > 0
                if not active.all():
                    inactive_idx = (~active).nonzero(as_tuple=True)[0]
                    idx = inactive_idx[0].item()
                    keep = torch.ones(self.n_routed_experts, dtype=torch.bool, device=self.weight.device)
                    keep[idx] = False
                    self.weight.data = self.weight.data[keep]
                    self.thresholds.data = self.thresholds.data[keep]
                    self.routing_records = self.routing_records[keep]
                    self.n_routed_experts = int(keep.sum().item())
                    removed_idx = idx
                    print(f"[DYNMoE] Removed expert {idx} → {self.n_routed_experts} experts")

            # Experts Addition
            if torch.norm(self.dropped_embeddings) > 1e-6 and self.n_routed_experts < self.max_expert_num:
                new_w = F.normalize(self.dropped_embeddings.unsqueeze(0), p=2, dim=-1)
                self.weight.data = torch.cat([self.weight.data, new_w], dim=0)
                self.thresholds.data = torch.cat([
                    self.thresholds.data,
                    torch.zeros(1, device=self.thresholds.device)
                ], dim=0)
                self.routing_records = torch.cat([
                    self.routing_records,
                    torch.zeros(1, device=self.routing_records.device)
                ], dim=0)
                self.n_routed_experts += 1
                added = True
                print(f"[DYNMoE] Added new expert → {self.n_routed_experts} experts")

            # Reset buffers
            self.routing_records.zero_()
            self.dropped_embeddings.zero_()
            self.audit_counter = 0

        return {'added': added, 'removed_idx': removed_idx}


# Auxiliary Loss Wrapper
class AddAuxiliaryLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, loss: torch.Tensor) -> torch.Tensor:
        ctx.dtype = loss.dtype
        ctx.required_aux_loss = loss.requires_grad
        return x

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        grad_loss = torch.ones(1, dtype=ctx.dtype, device=grad_output.device) if ctx.required_aux_loss else None
        return grad_output, grad_loss


# DYNMoE MLP (Mixture of Experts)
class DynMoEMLP(nn.Module):
    def __init__(self, config: DynMoEConfig):
        super().__init__()
        self.config = config
        self.experts = nn.ModuleList([
            self._create_expert(config) for _ in range(config.num_experts)
        ])
        self.gate = DynamicMoEGate(config)

    def _create_expert(self, config: DynMoEConfig, intermediate_size: Optional[int] = None) -> nn.Module:
        if intermediate_size is None:
            intermediate_size = config.moe_intermediate_size
        return nn.Sequential(
            nn.Linear(config.hidden_size, intermediate_size),
            nn.GELU(approximate='tanh') if config.hidden_act == "gelu_new" else nn.GELU(),
            nn.Linear(intermediate_size, config.hidden_size),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        identity = hidden_states
        orig_shape = hidden_states.shape
        x = hidden_states.view(-1, hidden_states.shape[-1])  # (N, d)

        topk_idx, topk_weight, aux_loss = self.gate(hidden_states)

        y = torch.zeros_like(x)
        K = self.gate.n_routed_experts
        for i in range(K):
            weight = topk_weight[:, i:i+1]  # (N, 1)
            if weight.sum() > 1e-8:
                expert_out = self.experts[i](x)
                y = y + expert_out * weight

        y = y.view(*orig_shape)  # (bsz, seq_len, hidden_size)

        if self.training and aux_loss is not None:
            y = AddAuxiliaryLoss.apply(y, self.config.moe_aux_loss_weight * aux_loss)

        return y + identity  # residual

    # Public adaptive_tune – called by the callback (only from this module)
    def adaptive_tune(self) -> None:
        result = self.gate._tune_experts()          # call internal gate tuning
        removed_idx = result.get('removed_idx')
        if removed_idx is not None:
            del self.experts[removed_idx]
        if result.get('added'):
            new_expert = self._create_expert(self.config)
            new_expert = new_expert.to(self.gate.weight.device)
            self.experts.append(new_expert)


# Causal Attention (with Rotary)
class DynMoEGPTAttention(nn.Module):
    def __init__(self, config: DynMoEConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        self.attention_dropout = config.attention_dropout

        self.rotary_emb = RotaryEmbedding(self.head_dim, max_position_embeddings=config.max_position_embeddings)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(query_states, seq_len=q_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states = torch.cat([past_key_values[0], key_states], dim=-2)
            value_states = torch.cat([past_key_values[1], value_states], dim=-2)
        kv_seq_len = key_states.shape[-2]

        # Scaled dot-product attention with causal mask
        attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal_mask = torch.triu(torch.ones(q_len, kv_seq_len, device=hidden_states.device), diagonal=1).bool()
        min_value = torch.finfo(attn_weights.dtype).min
        attn_weights = attn_weights.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), min_value)

        if attention_mask is not None:
            if attention_mask.dtype != attn_weights.dtype:
                attention_mask = attention_mask.to(attn_weights.dtype)
            if attention_mask.dim() == 4:
                attn_weights = attn_weights + attention_mask
            else:
                attn_mask_4d = attention_mask[:, None, None, :]  # (bsz, 1, 1, kv_len)
                attn_mask_4d = (1.0 - attn_mask_4d) * min_value
                attn_weights = attn_weights + attn_mask_4d

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        present_key_values = (key_states, value_states) if use_cache else None
        return attn_output, present_key_values


# Transformer Block
class DynMoEGPTBlock(nn.Module):
    def __init__(self, config: DynMoEConfig, layer_idx: int = 0):
        super().__init__()
        self.layer_idx = layer_idx
        self.ln1 = DynMoERMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention = DynMoEGPTAttention(config)
        self.ln2 = DynMoERMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        if layer_idx in config.replace_layers:
            self.mlp = DynMoEMLP(config)
        else:
            self.mlp = nn.Sequential(
                nn.Linear(config.hidden_size, config.intermediate_size),
                nn.GELU(approximate='tanh') if config.hidden_act == "gelu_new" else nn.GELU(),
                nn.Linear(config.intermediate_size, config.hidden_size)
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        residual = hidden_states
        hidden_states = self.ln1(hidden_states)
        attn_output, present_kv = self.attention(hidden_states, attention_mask, past_key_values, use_cache)
        hidden_states = residual + attn_output

        residual = hidden_states
        hidden_states = self.ln2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states, present_kv


# Decoder (Transformer)
class DynMoEGPTDecoder(nn.Module):
    def __init__(self, config: DynMoEConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            DynMoEGPTBlock(config, layer_idx=i) for i in range(config.num_hidden_layers)
        ])
        self.ln_f = DynMoERMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[Tuple[torch.Tensor, torch.Tensor]]]]:
        hidden_states = self.embed_tokens(input_ids)
        # Ensure correct dtype (float32 or fp16)
        if hasattr(self.config, '_dtype'):
            hidden_states = hidden_states.to(dtype=self.config._dtype)

        presents = [] if use_cache else None
        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values is not None else None
            hidden_states, present_kv = layer(hidden_states, attention_mask, past_kv, use_cache)
            if use_cache:
                presents.append(present_kv)

        hidden_states = self.ln_f(hidden_states)
        return hidden_states, presents


# PreTrainedModel Base 
class DynMoEPreTrainedModel(PreTrainedModel):
    config_class = DynMoEConfig
    base_model_prefix = "transformer"
    supports_gradient_checkpointing = True
    _no_split_modules = ["DynMoEGPTBlock"]

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


# Model (without LM head) 
class DynMoEGPTModel(DynMoEPreTrainedModel):
    def __init__(self, config: DynMoEConfig):
        super().__init__(config)
        self.config = config
        self.decoder = DynMoEGPTDecoder(config)
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
    ) -> BaseModelOutputWithPast:
        hidden_states, presents = self.decoder(input_ids, attention_mask, past_key_values, use_cache)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=presents,
        )


# Causal LM (for text generation)
class DynMoEForCausalLM(DynMoEPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: DynMoEConfig):
        super().__init__(config)
        self.config = config
        self.transformer = DynMoEGPTModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

        # For generation config
        self.generation_config = GenerationConfig.from_model_config(config)

    def get_input_embeddings(self):
        return self.transformer.decoder.embed_tokens

    def set_input_embeddings(self, value):
        self.transformer.decoder.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        **kwargs
    ) -> CausalLMOutputWithPast:
        if input_ids is not None:
            input_ids = input_ids.long()

        # Prepare attention mask (4D for causal)
        if attention_mask is not None and attention_mask.dim() == 2:
            # Expand to 4D: (bsz, 1, 1, seq_len)
            attention_mask = attention_mask[:, None, None, :]
            min_value = torch.finfo(self.lm_head.weight.dtype).min
            attention_mask = (1.0 - attention_mask.float()) * min_value
            attention_mask = attention_mask.to(self.lm_head.weight.dtype)

        outputs = self.transformer(input_ids, attention_mask, past_key_values, use_cache)
        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **kwargs):
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'past_key_values': past_key_values,
            'use_cache': True,
        }

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device))
                      for past_state in layer_past),
            )
        return reordered_past

    def save_pretrained(self, save_directory, **kwargs):
        # Ensure generation config is saved
        self.generation_config = GenerationConfig.from_model_config(self.config)
        super().save_pretrained(save_directory, **kwargs)