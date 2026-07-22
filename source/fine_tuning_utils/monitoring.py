import time
import subprocess
import numpy as np
import psutil
import torch
import math
from torch.utils.data import DataLoader
from transformers import TrainerCallback
from torch.utils.flop_counter import FlopCounterMode

from deepseekmoe_dynamic_routing_algorithms.source.fine_tuning_utils.config import (
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

class ResourceMonitorCallback(TrainerCallback):
    def __init__(self):
        self.epoch_start_time = None
        self.epoch_tokens = 0
        self.resource_metrics = []
        self.cpu_samples = []
        self.gpu_util_samples = []          # store per‑step GPU utilisation
        self.step_count = 0

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_start_time = time.time()
        self.epoch_tokens = 0
        self.cpu_samples = []
        self.gpu_util_samples = []
        self.step_count = 0

    def on_step_end(self, args, state, control, **kwargs):
        # Accumulate tokens for throughput
        step_tokens = PER_DEVICE_BATCH * world_size * GRAD_ACCUM * MAX_SEQ_LEN
        self.epoch_tokens += step_tokens

        # CPU usage sample
        self.cpu_samples.append(psutil.cpu_percent(interval=None))

        # GPU utilisation sample (average across all GPUs)
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, check=True
            )
            lines = [l.strip() for l in result.stdout.strip().split("\n") if l.strip()]
            if lines:
                gpu_utils = [float(l) for l in lines]
                self.gpu_util_samples.append(np.mean(gpu_utils))
            else:
                self.gpu_util_samples.append(0.0)
        except Exception:
            # If nvidia-smi fails, log 0 for this step (or skip)
            self.gpu_util_samples.append(0.0)

        self.step_count += 1

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch_time = time.time() - self.epoch_start_time
        tokens_per_sec = self.epoch_tokens / epoch_time if epoch_time > 0 else 0

        # Query final memory usage (still useful, but use averaged GPU util)
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True
        )
        lines = [l.strip() for l in result.stdout.strip().split("\n") if l.strip()]
        if lines:
            gpu_mem_gb = np.mean([float(l.split(", ")[1]) for l in lines]) / 1024
            # fallback if no samples were collected (unlikely during training)
            if not self.gpu_util_samples:
                gpu_util = np.mean([float(l.split(", ")[0]) for l in lines])
            else:
                gpu_util = np.mean(self.gpu_util_samples)
        else:
            gpu_mem_gb = 0.0
            gpu_util = np.mean(self.gpu_util_samples) if self.gpu_util_samples else 0.0

        cpu_pct = np.mean(self.cpu_samples) if self.cpu_samples else 0.0
        sys_mem_gb = psutil.virtual_memory().used / (1024 ** 3)

        stats = {
            'epoch': state.epoch,
            'gpu_mem_gb': gpu_mem_gb,
            'sys_mem_gb': sys_mem_gb,
            'gpu_util': gpu_util,
            'cpu_pct': cpu_pct,
            'tokens_per_sec': tokens_per_sec
        }
        self.resource_metrics.append(stats)

        print(f"\nEpoch {state.epoch:.2f} Resources:")
        print(f" GPU Mem : {gpu_mem_gb:.1f} GB")
        print(f" Sys Mem : {sys_mem_gb:.1f} GB")
        print(f" GPU Usage : {gpu_util:.1f} %")
        print(f" CPU Usage : {cpu_pct:.1f} %")
        print(f" Tokens/s : {tokens_per_sec:.0f}")


class MoEMetricsCallback(TrainerCallback):
    """
    Unified callback for monitoring:
      - Standard DeepSeekMoE (fixed top‑k, no bias update)
      - DYNMoE (adaptive routing with loss‑free balancing)
      - DeepSeekMoE with DYNMoE routing (same as DYNMoE)
    """
    def __init__(self, eval_dataset, tokenizer, data_collator):
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.data_collator = data_collator

        self.last_train_loss = 0.0
        self.metrics_history = []
        self.max_vio_values = []
        self.batch_max_vio_values = []
        self.gflops_values = []            # store GFLOPs per epoch

        self.hooks = []
        self.gates = []                    # list of gate modules (for DYNMoE)
        self.batch_counts = None           # used for standard DeepSeekMoE
        self.step_layer_counts = {}        # used for DYNMoE
        self.epoch_batch_max_vios = []

        self.best_eval_loss = float('inf')
        self.patience_counter = 0
        self.early_stop_patience = EARLY_STOPPING_PATIENCE
        self.early_stop_threshold = EARLY_STOPPING_THRESHOLD

        self.is_dynmoe = None              # set on first hook attachment
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
            # DYNMoE: process per‑layer counts from step_layer_counts
            n_experts = getattr(model.config, 'n_routed_experts', MAX_ROUTED_EXPERTS)
            step_vios = []
            for layer_idx, counts in self.step_layer_counts.items():
                total = counts.sum()
                expected = total / n_experts
                vio = np.max(np.abs(counts - expected)) / (expected + 1e-12)
                step_vios.append(vio)

                # Update Loss‑Free Balancing biases using the stored gate
                if layer_idx < len(self.gates):
                    self.gates[layer_idx].update_biases(torch.from_numpy(counts).to(model.device))

            if step_vios:
                self.epoch_batch_max_vios.append(np.mean(step_vios))
            self.step_layer_counts = {}

        else:
            # Standard DeepSeekMoE: process batch_counts
            if self.batch_counts is not None:
                n_experts = model.config.n_routed_experts
                total = self.batch_counts.sum()
                expected = total / n_experts
                batch_max = float(np.max(np.abs(self.batch_counts - expected) / (expected + 1e-12)))
                self.epoch_batch_max_vios.append(batch_max)
                self.batch_counts = None

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        if model is None or self._epoch_metrics_printed:
            return

        model.eval()
        unwrapped = model.module if hasattr(model, 'module') else model
        config = unwrapped.config
        n_experts = config.n_routed_experts

        # Determine architecture (safe in case it changed)
        is_dynmoe = False
        for gate in self.gates:
            if hasattr(gate, 'update_biases'):
                is_dynmoe = True
                break
        self.is_dynmoe = is_dynmoe

        # Get the list of MoE layers in order
        moe_layers = self._get_moe_layers(unwrapped)
        num_moe_layers = len(moe_layers)

        # Collect expert activation counts over the evaluation set 
        if is_dynmoe:
            layer_expert_counts = [np.zeros(n_experts, dtype=np.float64) for _ in range(num_moe_layers)]
            val_hooks = []
            def create_val_hook(idx):
                def val_hook_fn(module, input, output):
                    weights = output[1]
                    activated = (weights > 1e-8).float().sum(dim=0).detach().cpu().numpy()
                    layer_expert_counts[idx] += activated
                return val_hook_fn
            for idx, layer in enumerate(moe_layers):
                hook = layer.mlp.gate.register_forward_hook(create_val_hook(idx))
                val_hooks.append(hook)
        else:
            expert_counts = np.zeros(n_experts, dtype=np.float64)
            val_hooks = []
            def val_hook_fn(module, input, output):
                nonlocal expert_counts
                topk_idx = output[0]
                counts = torch.bincount(topk_idx.flatten(), minlength=n_experts).cpu().numpy()
                expert_counts += counts
            for layer in moe_layers:
                hook = layer.mlp.gate.register_forward_hook(val_hook_fn)
                val_hooks.append(hook)

        # Evaluation loop with FLOP counter
        from torch.utils.flop_counter import FlopCounterMode
        flop_counter = FlopCounterMode(unwrapped, display=False)

        total_loss = 0.0
        total_valid_tokens = 0
        eval_dataloader = DataLoader(
            self.eval_dataset,
            batch_size=args.per_device_eval_batch_size,
            collate_fn=self.data_collator,
            shuffle=False
        )

        with flop_counter:
            with torch.no_grad():
                for batch in eval_dataloader:
                    batch = {k: v.to(model.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                    outputs = model(**batch)
                    if 'labels' in batch:
                        shift_logits = outputs.logits[..., :-1, :].contiguous()
                        shift_labels = batch['labels'][..., 1:].contiguous()
                        loss_fct = torch.nn.CrossEntropyLoss(reduction='sum', ignore_index=-100)
                        total_loss += loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).item()
                        total_valid_tokens += (shift_labels != -100).sum().item()

        for h in val_hooks:
            h.remove()

        eval_loss = total_loss / total_valid_tokens
        perplexity = math.exp(eval_loss) if eval_loss < 20 else float('inf')
        measured_flops = flop_counter.get_total_flops()
        gflops = measured_flops / 1e9

        # Compute metrics based on architecture type
        if is_dynmoe:
            total_activations = sum(cnt.sum() for cnt in layer_expert_counts)
            avg_activated = total_activations / (total_valid_tokens * num_moe_layers + 1e-12)
        
            layer_vios = [np.max(np.abs(cnt - cnt.sum() / n_experts)) / (cnt.sum() / n_experts + 1e-12)
                          for cnt in layer_expert_counts]
            max_vio_global = np.mean(layer_vios)
            max_vio_batch = np.mean(self.epoch_batch_max_vios) if self.epoch_batch_max_vios else 0.0
        
            # Add bias terms
            expert_params = 2 * config.moe_intermediate_size * config.hidden_size \
                            + config.moe_intermediate_size + config.hidden_size
            n_shared = getattr(config, 'n_shared_experts', 0)
            active_params = num_moe_layers * (n_shared + avg_activated) * expert_params
            avg_routed_str = f"(avg routed: {avg_activated:.2f})"
        else:
            # Standard DeepSeekMoE: fixed number of activated experts per token
            total_activated = config.num_experts_per_tok
            if expert_counts.sum() > 0:
                expected_per_expert = expert_counts.sum() / n_experts
                max_vio_global = float(np.max(np.abs(expert_counts - expected_per_expert) / (expected_per_expert + 1e-12)))
            else:
                max_vio_global = 0.0
            max_vio_batch = np.mean(self.epoch_batch_max_vios) if self.epoch_batch_max_vios else 0.0
        
            # Standard DeepSeekMoE: 3 linear layers per expert (gate, up, down)
            expert_params = 3 * config.moe_intermediate_size * config.hidden_size
            n_shared = getattr(config, 'n_shared_experts', 0)
            active_params = num_moe_layers * (n_shared + total_activated) * expert_params   # <-- added num_moe_layers
            avg_routed_str = ""

        # Build metrics dict
        metrics = {
            'epoch': state.epoch,
            'train_loss': self.last_train_loss,
            'eval_loss': eval_loss,
            'perplexity': perplexity,
            'gflops': gflops,
            'max_vio_global': max_vio_global,
            'max_vio_batch': max_vio_batch,
            'active_params': active_params
        }
        if is_dynmoe:
            metrics['avg_activated'] = avg_activated

        self.metrics_history.append(metrics)
        self.max_vio_values.append(max_vio_global)
        self.batch_max_vio_values.append(max_vio_batch)
        self.gflops_values.append(gflops)
        self._epoch_metrics_printed = True

        # Print summary
        print(f"\nEpoch {state.epoch:.2f} Metrics:")
        print(f" Loss          : {self.last_train_loss:8.4f}")
        print(f" Val_Loss      : {eval_loss:8.4f}")
        print(f" Perplexity    : {perplexity:8.2f}")
        print(f" GFLOPs        : {gflops:8.2f}")
        print(f" MaxVIO_global : {max_vio_global:8.4f}")
        print(f" MaxVIO_batch  : {max_vio_batch:8.4f}")
        print(f" Active Params : {active_params:,.0f} {avg_routed_str}")

        # Early stopping
        if eval_loss < self.best_eval_loss - self.early_stop_threshold:
            self.best_eval_loss = eval_loss
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        if self.patience_counter >= self.early_stop_patience:
            print("Early stopping triggered.")
            control.should_training_stop = True

        model.train()

    # Helper methods
    def _get_layer_list(self, model):
        """Return the list of layer blocks in the model (in order)."""
        # Try common attribute names
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            return model.model.layers
        if hasattr(model, 'transformer') and hasattr(model.transformer, 'decoder') and hasattr(model.transformer.decoder, 'layers'):
            return model.transformer.decoder.layers
        if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            return model.transformer.h
        # Fallback: use named_modules to collect layers by name pattern
        layers = []
        for name, module in model.named_modules():
            if name.endswith('.layers') or name == 'layers':
                if isinstance(module, torch.nn.ModuleList):
                    return module
        # If nothing found, raise an error
        raise AttributeError("Could not locate the layer container in the model. "
                             "Expected attributes: model.layers, transformer.decoder.layers, or transformer.h")

    def _get_moe_layers(self, model):
        """Return a list of MoE layer modules (those with mlp that is an MoE)."""
        all_layers = self._get_layer_list(model)
        moe_layers = []
        for layer in all_layers:
            if hasattr(layer, 'mlp') and (
                layer.mlp.__class__.__name__ in ('DynMoEMLP', 'DeepseekMoE') or
                hasattr(layer.mlp, 'gate')
            ):
                moe_layers.append(layer)
        return moe_layers

    # Hook management
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

        # Get MoE layers in order
        moe_layers = self._get_moe_layers(unwrapped)
        if not moe_layers:
            print("[MoEMetricsCallback] No MoE layers found. No hooks attached.")
            return

        # Detect DYNMoE by checking if any gate has update_biases
        is_dynmoe = False
        for layer in moe_layers:
            if hasattr(layer.mlp, 'gate') and hasattr(layer.mlp.gate, 'update_biases'):
                is_dynmoe = True
                break
        self.is_dynmoe = is_dynmoe

        if is_dynmoe:
            # DYNMoE: attach per‑layer hooks with index, store gates
            self.gates = []
            for idx, layer in enumerate(moe_layers):
                gate = layer.mlp.gate
                self.gates.append(gate)
                hook = gate.register_forward_hook(
                    lambda m, i, o, idx=idx: self._batch_hook_dynmoe(m, i, o, idx)
                )
                self.hooks.append(hook)
            print(f"[MoEMetricsCallback] Attached {len(self.hooks)} hooks for DYNMoE (gates stored: {len(self.gates)})")
        else:
            # Standard DeepSeekMoE: attach simple batch hook
            for layer in moe_layers:
                gate = layer.mlp.gate
                hook = gate.register_forward_hook(self._batch_hook_standard)
                self.hooks.append(hook)
            print(f"[MoEMetricsCallback] Attached {len(self.hooks)} hooks for standard DeepSeekMoE")

    # Batch hooks for each architecture
    def _batch_hook_standard(self, module, input, output):
        """For standard DeepSeekMoE: output[0] contains expert indices."""
        topk_idx = output[0]
        counts = torch.bincount(topk_idx.flatten(), minlength=module.n_routed_experts).cpu().numpy()
        if self.batch_counts is None:
            self.batch_counts = counts
        else:
            self.batch_counts += counts

    def _batch_hook_dynmoe(self, module, input, output, layer_idx):
        """For DYNMoE: output[1] contains expert weights."""
        weights = output[1]
        counts = (weights > 1e-8).float().sum(dim=0).detach().cpu().numpy()
        self.step_layer_counts[layer_idx] = self.step_layer_counts.get(
            layer_idx, np.zeros_like(counts)
        ) + counts