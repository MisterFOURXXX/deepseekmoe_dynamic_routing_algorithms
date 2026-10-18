import sys
sys.path.append("..")

import time
import subprocess
import numpy as np
import psutil
import torch
import torch.nn as nn
import math
from torch.utils.data import DataLoader
from transformers import TrainerCallback
from torch.utils.flop_counter import FlopCounterMode

from .config import (
    MAX_SEQ_LEN,
    PER_DEVICE_BATCH,
    GRAD_ACCUM,
    LEARNING_RATE,
    NUM_EPOCHS_FT,
    WARMUP_STEPS,
    WEIGHT_DECAY,
    EARLY_STOPPING_PATIENCE,
    EARLY_STOPPING_THRESHOLD,
    world_size
)

#from ..DYNMoE_baseline.config import MAX_ROUTED_EXPERTS as DYN_MAX_ROUTED_EXPERTS
#from ..deepseek_dynamics_routing.config import MAX_ROUTED_EXPERTS as DR_MAX_ROUTED_EXPERTS

class ResourceMonitorCallback(TrainerCallback):
    def __init__(self):
        self.epoch_start_time = None
        self.epoch_tokens = 0
        self.resource_metrics = []
        self.cpu_samples = []
        self.gpu_util_samples = []

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_start_time = time.time()
        self.epoch_tokens = 0
        self.cpu_samples = []
        self.gpu_util_samples = []

    def on_step_end(self, args, state, control, **kwargs):
        self.epoch_tokens += PER_DEVICE_BATCH * max(world_size, 1) * GRAD_ACCUM * MAX_SEQ_LEN
        self.cpu_samples.append(psutil.cpu_percent(interval=None))
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, check=True,
            )
            lines = [l.strip() for l in result.stdout.strip().split("\n") if l.strip()]
            self.gpu_util_samples.append(
                np.mean([float(l) for l in lines]) if lines else 0.0
            )
        except Exception:
            self.gpu_util_samples.append(0.0)

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch_time = time.time() - self.epoch_start_time
        tokens_per_sec = self.epoch_tokens / epoch_time if epoch_time > 0 else 0
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True,
        )
        lines = [l.strip() for l in result.stdout.strip().split("\n") if l.strip()]
        if lines:
            gpu_mem_gb = np.mean([float(l.split(", ")[1]) for l in lines]) / 1024
            gpu_util = (np.mean(self.gpu_util_samples) if self.gpu_util_samples
                        else np.mean([float(l.split(", ")[0]) for l in lines]))
        else:
            gpu_mem_gb = 0.0
            gpu_util = np.mean(self.gpu_util_samples) if self.gpu_util_samples else 0.0
        cpu_pct = np.mean(self.cpu_samples) if self.cpu_samples else 0.0
        sys_mem_gb = psutil.virtual_memory().used / (1024 ** 3)
        self.resource_metrics.append({
            'epoch': state.epoch,
            'gpu_mem_gb': gpu_mem_gb,
            'sys_mem_gb': sys_mem_gb,
            'gpu_util': gpu_util,
            'cpu_pct': cpu_pct,
            'tokens_per_sec': tokens_per_sec,
        })
        print(f"\nEpoch {state.epoch:.2f} Resources:")
        print(f" GPU Mem : {gpu_mem_gb:.1f} GB")
        print(f" Sys Mem : {sys_mem_gb:.1f} GB")
        print(f" GPU Usage : {gpu_util:.1f} %")
        print(f" CPU Usage : {cpu_pct:.1f} %")
        print(f" Tokens/s : {tokens_per_sec:.0f}")


class MoEMetricsCallback(TrainerCallback):
    def __init__(self, eval_dataset, tokenizer, data_collator,
                 early_stop_patience=3, early_stop_threshold=0.001):
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.data_collator = data_collator
        self.last_train_loss = 0.0
        self.metrics_history = []
        self.max_vio_values = []
        self.batch_max_vio_values = []
        self.gflops_values = []
        self.hooks = []
        self.gates = []
        self.batch_counts = None
        self.step_layer_counts = {}
        self.epoch_batch_max_vios = []
        self.best_eval_loss = float('inf')
        self.patience_counter = 0
        self.early_stop_patience = early_stop_patience
        self.early_stop_threshold = early_stop_threshold
        self.is_dynmoe = None
        self._epoch_metrics_printed = False

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and 'loss' in logs:
            self.last_train_loss = logs['loss']

    def on_train_begin(self, args, state, control, **kwargs):
        self._attach_hooks_with_model(kwargs.get('model'))

    def on_train_end(self, args, state, control, **kwargs):
        self._remove_hooks()

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_batch_max_vios = []
        self.batch_counts = None
        self.step_layer_counts = {}
        self._epoch_metrics_printed = False

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if model is None:
            return
        if self.is_dynmoe:
            step_vios = []
            for layer_idx, counts in self.step_layer_counts.items():
                total = counts.sum()
                n_experts = len(counts)
                expected = total / n_experts if total > 0 else 0.0
                step_vios.append(
                    np.max(np.abs(counts - expected)) / (expected + 1e-12)
                    if expected > 0 else 0.0
                )
                if (layer_idx < len(self.gates)
                        and hasattr(self.gates[layer_idx], 'update_biases')):
                    self.gates[layer_idx].update_biases(
                        torch.from_numpy(counts).to(model.device)
                    )
            if step_vios:
                self.epoch_batch_max_vios.append(np.mean(step_vios))
            self.step_layer_counts = {}
        else:
            if self.batch_counts is not None:
                n_experts = model.config.n_routed_experts
                total = self.batch_counts.sum()
                expected = total / n_experts if total > 0 else 0.0
                self.epoch_batch_max_vios.append(
                    np.max(np.abs(self.batch_counts - expected)) / (expected + 1e-12)
                    if expected > 0 else 0.0
                )
                self.batch_counts = None

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        if model is None or self._epoch_metrics_printed:
            return
        model.eval()
        unwrapped = model.module if hasattr(model, 'module') else model
        config = unwrapped.config
        moe_layers = self._get_moe_layers(unwrapped)
        num_moe_layers = len(moe_layers)
        if num_moe_layers == 0:
            print("[MoEMetricsCallback] No MoE layers found; skipping metrics.")
            model.train()
            return

        if self.is_dynmoe:
            layer_expert_counts = []
            val_hooks = []

            def create_val_hook(gate):
                arr = np.zeros(gate.n_routed_experts, dtype=np.float64)
                layer_expert_counts.append(arr)

                def val_hook_fn(module, input, output):
                    if len(output) >= 2 and isinstance(output[1], torch.Tensor):
                        weights = output[1]
                    elif len(output) >= 1 and isinstance(output[0], torch.Tensor):
                        weights = output[0]
                    else:
                        return
                    reduce_dims = tuple(range(weights.dim() - 1))
                    counts = ((weights > 1e-8).float()
                              .sum(dim=reduce_dims).detach().cpu().numpy())
                    arr[:] += counts

                return val_hook_fn

            for layer in moe_layers:
                val_hooks.append(
                    layer.mlp.gate.register_forward_hook(create_val_hook(layer.mlp.gate))
                )
        else:
            n_experts = config.n_routed_experts
            expert_counts = np.zeros(n_experts, dtype=np.float64)
            val_hooks = []

            def val_hook_fn(module, input, output):
                topk_idx = output[0]
                if topk_idx.is_floating_point():
                    topk_idx = topk_idx.long()
                counts = torch.bincount(
                    topk_idx.flatten(), minlength=n_experts
                ).cpu().numpy()
                expert_counts[:] += counts

            for layer in moe_layers:
                val_hooks.append(layer.mlp.gate.register_forward_hook(val_hook_fn))

        from torch.utils.flop_counter import FlopCounterMode
        flop_counter = FlopCounterMode(display=False)

        total_loss = 0.0
        total_valid_tokens = 0
        eval_dataloader = DataLoader(
            self.eval_dataset,
            batch_size=args.per_device_eval_batch_size,
            collate_fn=self.data_collator,
            shuffle=False,
        )

        with flop_counter, torch.no_grad():
            for batch in eval_dataloader:
                batch = {k: v.to(model.device) for k, v in batch.items()
                         if isinstance(v, torch.Tensor)}
                outputs = model(**batch)
                if 'labels' in batch:
                    shift_logits = outputs.logits[..., :-1, :].contiguous()
                    shift_labels = batch['labels'][..., 1:].contiguous()
                    loss_fct = torch.nn.CrossEntropyLoss(
                        reduction='sum', ignore_index=-100,
                    )
                    total_loss += float(loss_fct(
                        shift_logits.float().view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                    ).item())
                    total_valid_tokens += int((shift_labels != -100).sum().item())

        for h in val_hooks:
            h.remove()

        eval_loss = (total_loss / total_valid_tokens
                     if total_valid_tokens > 0 else float('inf'))
        perplexity = math.exp(eval_loss) if eval_loss < 20 else float('inf')
        gflops = flop_counter.get_total_flops() / 1e9

        if self.is_dynmoe:
            avg_activated = (sum(cnt.sum() for cnt in layer_expert_counts)
                             / (total_valid_tokens * num_moe_layers + 1e-12))
            layer_vios = []
            for cnt in layer_expert_counts:
                total = cnt.sum()
                expected = total / len(cnt) if total > 0 else 0.0
                layer_vios.append(
                    np.max(np.abs(cnt - expected)) / (expected + 1e-12)
                    if expected > 0 else 0.0
                )
            max_vio_global = np.mean(layer_vios) if layer_vios else 0.0
            max_vio_batch = (np.mean(self.epoch_batch_max_vios)
                             if self.epoch_batch_max_vios else 0.0)
            expert_params = (2 * config.moe_intermediate_size * config.hidden_size
                             + config.moe_intermediate_size + config.hidden_size)
            active_params = num_moe_layers * avg_activated * expert_params
            avg_routed_str = f"(avg routed: {avg_activated:.2f})"
        else:
            total_activated = getattr(config, 'num_experts_per_tok', 2)
            total = expert_counts.sum()
            expected = total / n_experts if total > 0 else 0.0
            max_vio_global = (np.max(np.abs(expert_counts - expected)) / (expected + 1e-12)
                              if expected > 0 else 0.0)
            max_vio_batch = (np.mean(self.epoch_batch_max_vios)
                             if self.epoch_batch_max_vios else 0.0)
            expert_params = 3 * config.moe_intermediate_size * config.hidden_size
            active_params = num_moe_layers * total_activated * expert_params
            avg_routed_str = ""

        train_loss = self.last_train_loss
        if train_loss == 0.0:
            for entry in reversed(state.log_history):
                if 'loss' in entry:
                    train_loss = entry['loss']
                    break

        metrics = {
            'epoch': state.epoch,
            'train_loss': train_loss,
            'eval_loss': eval_loss,
            'perplexity': perplexity,
            'gflops': gflops,
            'max_vio_global': max_vio_global,
            'max_vio_batch': max_vio_batch,
            'active_params': active_params,
        }
        if self.is_dynmoe:
            metrics['avg_activated'] = avg_activated

        self.metrics_history.append(metrics)
        self.max_vio_values.append(max_vio_global)
        self.batch_max_vio_values.append(max_vio_batch)
        self.gflops_values.append(gflops)
        self._epoch_metrics_printed = True

        print(f"\nEpoch {state.epoch:.2f} Metrics:")
        print(f" Loss          : {train_loss:8.4f}")
        print(f" Val_Loss      : {eval_loss:8.4f}")
        print(f" Perplexity    : {perplexity:8.2f}")
        print(f" GFLOPs        : {gflops:8.2f}")
        print(f" MaxVIO_global : {max_vio_global:8.6f}")
        print(f" MaxVIO_batch  : {max_vio_batch:8.6f}")
        print(f" Active Params : {active_params:,.0f} {avg_routed_str}")

        if eval_loss < self.best_eval_loss - self.early_stop_threshold:
            self.best_eval_loss = eval_loss
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        if self.patience_counter >= self.early_stop_patience:
            print("Early stopping triggered.")
            control.should_training_stop = True

        model.train()

    def _get_layer_list(self, model):
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            return model.model.layers
        if (hasattr(model, 'transformer') and hasattr(model.transformer, 'decoder')
                and hasattr(model.transformer.decoder, 'layers')):
            return model.transformer.decoder.layers
        if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            return model.transformer.h
        for name, module in model.named_modules():
            if name.endswith('.layers') or name == 'layers':
                if isinstance(module, torch.nn.ModuleList):
                    return module
        raise AttributeError("Could not locate the layer container in the model.")

    def _get_moe_layers(self, model):
        return [layer for layer in self._get_layer_list(model)
                if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'gate')]

    def _remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.gates = []

    def _attach_hooks_with_model(self, model):
        self._remove_hooks()
        if model is None:
            return
        unwrapped = model.module if hasattr(model, 'module') else model
        moe_layers = self._get_moe_layers(unwrapped)
        if not moe_layers:
            print("[MoEMetricsCallback] No MoE layers found. No hooks attached.")
            return

        self.is_dynmoe = any(
            hasattr(layer.mlp.gate, attr)
            for layer in moe_layers
            for attr in ('thresholds', 'update_biases', 'adaptive_tune')
        )

        if self.is_dynmoe:
            self.gates = []
            for idx, layer in enumerate(moe_layers):
                gate = layer.mlp.gate
                self.gates.append(gate)
                self.hooks.append(gate.register_forward_hook(
                    lambda m, i, o, idx=idx: self._batch_hook_dynmoe(m, i, o, idx)
                ))
            print(f"[MoEMetricsCallback] Attached {len(self.hooks)} hooks for DYNMoE")
        else:
            for layer in moe_layers:
                self.hooks.append(
                    layer.mlp.gate.register_forward_hook(self._batch_hook_standard)
                )
            print(f"[MoEMetricsCallback] Attached {len(self.hooks)} hooks for standard DeepSeekMoE")

    def _batch_hook_standard(self, module, input, output):
        topk_idx = output[0]
        if topk_idx.is_floating_point():
            topk_idx = topk_idx.long()
        counts = torch.bincount(
            topk_idx.flatten(), minlength=module.n_routed_experts
        ).cpu().numpy()
        if self.batch_counts is None:
            self.batch_counts = counts
        else:
            self.batch_counts += counts

    def _batch_hook_dynmoe(self, module, input, output, layer_idx):
        if len(output) >= 2 and isinstance(output[1], torch.Tensor):
            weights = output[1]
        elif len(output) >= 1 and isinstance(output[0], torch.Tensor):
            weights = output[0]
        else:
            return
        reduce_dims = tuple(range(weights.dim() - 1))
        counts = ((weights > 1e-8).float()
                  .sum(dim=reduce_dims).detach().cpu().numpy())
        if layer_idx not in self.step_layer_counts:
            self.step_layer_counts[layer_idx] = counts
        else:
            self.step_layer_counts[layer_idx] += counts