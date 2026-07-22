import sys
import os
sys.path.insert(0, os.path.abspath('..'))
from DYNMoE_baseline.config import DynMoEConfig
from DYNMoE_baseline.adaptive_tuning import AdaptiveExpertTuningCallback

import torch
import transformers
import accelerate
import deepspeed

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

# FX wrapping (optional)
if is_torch_fx_available():
    if not is_torch_greater_or_equal_than_1_13:
        import torch.fx
        _prepare_4d_causal_attention_mask = torch.fx.wrap(_prepare_4d_causal_attention_mask)

# DYNMoE RMS NORM 
class DynMoERMSNorm(nn.Module):
    """RMSNorm for DYNMoE models."""
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        # Use the weight's dtype for consistency
        weight_dtype = self.weight.dtype
        hidden_states = hidden_states.to(dtype=weight_dtype)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

# DYNAMIC MoE GATE 
    """
    Original DYNMoE dynamic top-any gating from the ICLR 2025 paper.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_routed_experts = config.num_experts
        self.hidden_size = config.hidden_size
        self.max_expert_num = config.max_expert_num
        
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.hidden_size)))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
        self.thresholds = nn.Parameter(torch.full((self.n_routed_experts,), DYNMOE_THRESHOLD_INIT))
        self.biases = nn.Parameter(torch.zeros(self.n_routed_experts), requires_grad=False)
        
        self.register_buffer('routing_records', torch.zeros(self.n_routed_experts))
        self.register_buffer('dropped_embeddings', torch.zeros(self.hidden_size))
        self.audit_counter = 0
        self.audit_interval = ADAPTIVE_AUDIT_STEPS
        
    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        x = hidden_states.view(-1, h)
        
        norm_x = F.normalize(x, p=2, dim=-1)
        norm_w = F.normalize(self.weight, p=2, dim=-1)
        s = torch.matmul(norm_x, norm_w.T)
        
        biased_s = s + self.biases
        g = torch.sign(torch.sigmoid(biased_s) - torch.sigmoid(self.thresholds))
        g = (g > 0).float()
        k_r = g.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        
        if self.training:
            soft_g = torch.sigmoid(biased_s) - torch.sigmoid(self.thresholds)
            g = g + (soft_g - soft_g.detach())
        
        topk_weight = g / k_r
        topk_idx = torch.arange(self.n_routed_experts, device=g.device).unsqueeze(0).expand_as(g)
        
        if not self.training:
            zero_mask = (k_r.squeeze(-1) == 0)
            if zero_mask.any():
                max_affinity_idx = torch.argmax(s, dim=-1)
                for i in zero_mask.nonzero(as_tuple=True)[0]:
                    g[i, :] = 0
                    g[i, max_affinity_idx[i]] = 1.0
                k_r = g.sum(dim=-1, keepdim=True).clamp(min=1e-8)
                topk_weight = g / k_r
        
        if self.training:
            self.audit_counter += 1
            with torch.no_grad():
                batch_activ = g.mean(dim=0) * bsz * seq_len
                self.routing_records += batch_activ
                dropped_mask = (g.sum(dim=-1) == 0)
                if dropped_mask.any():
                    dropped_emb = x[dropped_mask].mean(dim=0)
                    self.dropped_embeddings += dropped_emb
        
        aux_loss = self._sparse_simple_loss() if self.training else None
        return topk_idx, topk_weight, aux_loss
    
    def _sparse_simple_loss(self):
        Wg = self.weight
        gram = torch.matmul(Wg, Wg.T)
        diversity = torch.norm(gram - torch.eye(self.n_routed_experts, device=Wg.device), p='fro') ** 2
        simplicity = torch.mean(torch.norm(Wg, p=2, dim=1))
        return diversity + simplicity
    
    def update_biases(self, token_counts_per_expert):
        if not self.training:
            return
        total = token_counts_per_expert.sum().float()
        avg = total / self.n_routed_experts
        violation = token_counts_per_expert - avg
        with torch.no_grad():
            self.biases += BIAS_UPDATE_RATE * torch.sign(violation)
    
    def adaptive_tune(self):
        if self.audit_counter < self.audit_interval:
            return
        
        with torch.no_grad():
            if self.n_routed_experts > 2:
                active = self.routing_records > 0
                if not active.all():
                    inactive_idx = (~active).nonzero(as_tuple=True)[0]
                    if len(inactive_idx) > 0:
                        keep = torch.ones(self.n_routed_experts, dtype=torch.bool, device=self.weight.device)
                        keep[inactive_idx[0]] = False
                        self.weight.data = self.weight.data[keep]
                        self.thresholds.data = self.thresholds.data[keep]
                        self.biases.data = self.biases.data[keep]
                        self.routing_records = self.routing_records[keep]
                        self.n_routed_experts = int(keep.sum().item())
                        print(f"[DYNMoE] Removed expert → {self.n_routed_experts}")
            
            if torch.norm(self.dropped_embeddings) > 1e-6 and self.n_routed_experts < self.max_expert_num:
                new_w = F.normalize(self.dropped_embeddings.unsqueeze(0), p=2, dim=-1)
                self.weight.data = torch.cat([self.weight.data, new_w], dim=0)
                self.thresholds.data = torch.cat([self.thresholds.data, torch.zeros(1, device=self.thresholds.device)])
                self.biases.data = torch.cat([self.biases.data, torch.zeros(1, device=self.biases.device)])
                self.routing_records = torch.cat([self.routing_records, torch.zeros(1, device=self.routing_records.device)])
                self.n_routed_experts += 1
                print(f"[DYNMoE] Added new expert → {self.n_routed_experts}")
            
            self.routing_records.zero_()
            self.dropped_embeddings.zero_()
            self.audit_counter = 0

# AUXILIARY LOSS 
class AddAuxiliaryLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, loss):
        assert loss.numel() == 1
        ctx.dtype = loss.dtype
        ctx.required_aux_loss = loss.requires_grad
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_loss = None
        if ctx.required_aux_loss:
            grad_loss = torch.ones(1, dtype=ctx.dtype, device=grad_output.device)
        return grad_output, grad_loss

# DYNMoE MLP 
class DynMoEMLP(nn.Module):
    """FFN replacement with DYNMoE."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_routed_experts = config.num_experts
        self.n_shared_experts = config.n_shared_experts
        
        self.experts = nn.ModuleList([
            self._create_expert(config)
            for _ in range(config.num_experts)
        ])
        self.gate = DynamicMoEGate(config)
        
        if self.n_shared_experts is not None and self.n_shared_experts > 0:
            inter_size = config.moe_intermediate_size * self.n_shared_experts
            self.shared_experts = self._create_expert(config, intermediate_size=inter_size)
    
    def _create_expert(self, config, intermediate_size=None):
        if intermediate_size is None:
            intermediate_size = config.moe_intermediate_size
        
        return nn.Sequential(
            nn.Linear(config.hidden_size, intermediate_size),
            nn.GELU() if config.hidden_act == "gelu" else ACT2FN[config.hidden_act],
            nn.Linear(intermediate_size, config.hidden_size),
        )
    
    def forward(self, hidden_states):
        identity = hidden_states
        orig_shape = hidden_states.shape
        
        topk_idx, topk_weight, aux_loss = self.gate(hidden_states)
        x = hidden_states.view(-1, hidden_states.shape[-1])
        
        y = torch.zeros_like(x)
        for i in range(self.n_routed_experts):
            expert_weight = topk_weight[:, i:i+1]
            if expert_weight.sum() > 1e-8:
                expert_out = self.experts[i](x)
                y = y + expert_out * expert_weight
        
        y = y.view(*orig_shape)
        
        if self.training and aux_loss is not None:
            y = AddAuxiliaryLoss.apply(y, aux_loss)
        
        if hasattr(self, 'shared_experts'):
            y = y + self.shared_experts(identity)
        y = y + identity
        
        return y

# GPT DECODER WITH DYNMoE
    """GPT-style causal attention."""
    def __init__(self, config, layer_idx=None):
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
        
    def forward(self, hidden_states, attention_mask=None, past_key_values=None, use_cache=False):
        bsz, q_len, _ = hidden_states.size()
        
        # Ensure hidden_states has correct dtype
        model_dtype = hidden_states.dtype
        
        # Project and reshape
        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        if past_key_values is not None:
            key_states = torch.cat([past_key_values[0], key_states], dim=-2)
            value_states = torch.cat([past_key_values[1], value_states], dim=-2)
        
        kv_seq_len = key_states.shape[-2]
        
        # Compute attention scores with proper dtype
        attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Causal mask
        causal_mask = torch.triu(torch.ones(q_len, kv_seq_len, device=hidden_states.device), diagonal=1).bool()
        # Use a large negative number compatible with the dtype
        min_value = torch.finfo(attn_weights.dtype).min
        attn_weights = attn_weights.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), min_value)
        
        if attention_mask is not None:
            # Ensure attention_mask has the same dtype as attn_weights
            if attention_mask.dtype != attn_weights.dtype:
                attention_mask = attention_mask.to(attn_weights.dtype)
            # Handle 4D attention mask
            if attention_mask.dim() == 4:
                attn_weights = attn_weights + attention_mask
            else:
                # For 2D mask, expand to 4D
                attn_mask_4d = attention_mask[:, None, None, :]
                attn_mask_4d = (1.0 - attn_mask_4d) * torch.finfo(attn_weights.dtype).min
                attn_weights = attn_weights + attn_mask_4d
        
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        
        present_key_values = (key_states, value_states) if use_cache else None
        return attn_output, present_key_values

class DynMoEGPTBlock(nn.Module):
    """GPT block with DYNMoE replacing MLP."""
    def __init__(self, config, layer_idx=0):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        
        self.ln1 = DynMoERMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention = DynMoEGPTAttention(config)
        self.ln2 = DynMoERMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        if layer_idx in config.replace_layers:
            self.mlp = DynMoEMLP(config)
        else:
            self.mlp = nn.Sequential(
                nn.Linear(config.hidden_size, config.intermediate_size),
                nn.GELU() if config.hidden_act == "gelu" else ACT2FN[config.hidden_act],
                nn.Linear(config.intermediate_size, config.hidden_size)
            )
    
    def forward(self, hidden_states, attention_mask=None, past_key_values=None, use_cache=False):
        # Self-attention
        residual = hidden_states
        hidden_states = self.ln1(hidden_states)
        attn_output, present_key_values = self.attention(
            hidden_states, attention_mask, past_key_values, use_cache
        )
        hidden_states = residual + attn_output
        
        # FFN or MoE
        residual = hidden_states
        hidden_states = self.ln2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states, present_key_values

class DynMoEGPTDecoder(nn.Module):
    """GPT Decoder with DYNMoE layers."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            DynMoEGPTBlock(config, layer_idx=i)
            for i in range(config.num_hidden_layers)
        ])
        self.ln_f = DynMoERMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # Store the dtype as a buffer to avoid parameter iteration issues
        self.register_buffer('_dtype_buffer', torch.tensor(1.0, dtype=torch.float32))
    
    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False):
        hidden_states = self.embed_tokens(input_ids)
        
        # Get dtype from the first parameter safely
        try:
            model_dtype = next(self.parameters()).dtype
        except StopIteration:
            # If no parameters, use float32 as fallback
            model_dtype = torch.float32
        
        # Only cast if model_dtype is available
        if model_dtype is not None:
            hidden_states = hidden_states.to(dtype=model_dtype)
        
        presents = [] if use_cache else None
        
        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values is not None else None
            hidden_states, present_kv = layer(
                hidden_states, attention_mask, past_kv, use_cache
            )
            if use_cache:
                presents.append(present_kv)
        
        hidden_states = self.ln_f(hidden_states)
        return hidden_states, presents

# MAIN DYNMoE MODELS 
class DynMoEPreTrainedModel(PreTrainedModel):
    """Base class for DYNMoE models."""
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

class DynMoEGPTModel(DynMoEPreTrainedModel):
    """DYNMoE GPT model (decoder-only)."""
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.decoder = DynMoEGPTDecoder(config)
        self.post_init()
    
    def forward(self, input_ids=None, attention_mask=None, past_key_values=None,
                use_cache=False, **kwargs):
        hidden_states, presents = self.decoder(
            input_ids, attention_mask, past_key_values, use_cache
        )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=presents
        )

class DynMoEForCausalLM(DynMoEPreTrainedModel):
    """DYNMoE model for causal language modeling."""
    _tied_weights_keys = ["lm_head.weight"]
    
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.transformer = DynMoEGPTModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()
        # Store dtype as a buffer to avoid parameter iteration issues
        self.register_buffer('_dtype_buffer', torch.tensor(1.0, dtype=torch.float32))
        
        # Create and attach a GenerationConfig to avoid warnings
        self.generation_config = GenerationConfig.from_model_config(config)
        # If you want default generation parameters, set them here, e.g.:
        # self.generation_config.top_k = None
        # self.generation_config.temperature = 1.0
    
    def get_input_embeddings(self):
        return self.transformer.decoder.embed_tokens
    
    def set_input_embeddings(self, value):
        self.transformer.decoder.embed_tokens = value
    
    def get_output_embeddings(self):
        return self.lm_head
    
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings
    
    def _get_model_dtype(self):
        """Safely get the model's dtype without iterating parameters."""
        try:
            for param in self.parameters():
                return param.dtype
        except StopIteration:
            pass
        if hasattr(self, '_dtype_buffer'):
            return self._dtype_buffer.dtype
        return torch.float32
    
    def forward(self, input_ids=None, attention_mask=None, labels=None,
                past_key_values=None, use_cache=False, **kwargs):
        # Ensure input_ids are long type
        if input_ids is not None:
            input_ids = input_ids.long()
        
        model_dtype = self._get_model_dtype()
        
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                attention_mask = attention_mask[:, None, None, :]
                min_value = torch.finfo(model_dtype).min
                attention_mask = (1.0 - attention_mask.float()) * min_value
            attention_mask = attention_mask.to(dtype=model_dtype)
        
        outputs = self.transformer(input_ids, attention_mask, past_key_values, use_cache)
        hidden_states = outputs.last_hidden_state
        hidden_states = hidden_states.to(dtype=model_dtype)
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1)
            )
        
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      attention_mask=None, **kwargs):
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'past_key_values': past_key_values,
            'use_cache': True
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
    
    # Override save_pretrained to refresh generation config before saving
    def save_pretrained(self, save_directory, **kwargs):
        # Update the generation config to match the current model config
        self.generation_config = GenerationConfig.from_model_config(self.config)

        super().save_pretrained(save_directory, **kwargs)