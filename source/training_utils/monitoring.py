import sys
sys.path.append("..")

import os
import json
import time
import subprocess
import numpy as np
import psutil
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import TrainerCallback
from torch.utils.flop_counter import FlopCounterMode

from .config import (
    MAX_SEQ_LEN,
    PER_DEVICE_BATCH,
    GRAD_ACCUM,
    EARLY_STOPPING_PATIENCE,
    EARLY_STOPPING_THRESHOLD,
    world_size,
)

# Files written by the callbacks at the end of training and read by
# plot_training_comparison().
_RESOURCE_METRICS_FILE = "resource_metrics.json"
_MOE_METRICS_FILE = "moe_metrics.json"


# ── Monitoring persistence helpers ──────────────────────────────────────────
def _tokens_per_step():
    """Tokens consumed by one optimizer step (same formula as the callbacks)."""
    return (
        int(PER_DEVICE_BATCH)
        * max(int(world_size), 1)
        * int(GRAD_ACCUM)
        * int(MAX_SEQ_LEN)
    )


def _steps_per_epoch(args, state):
    if state is None:
        return None
    max_steps = getattr(state, "max_steps", None)
    epochs = float(getattr(args, "num_train_epochs", 0) or 0)
    if max_steps is not None and float(max_steps) > 0 and epochs > 0:
        return float(max_steps) / epochs
    return None


def _is_main_process(args, state):
    if state is not None and getattr(state, "is_world_process_zero", True) is False:
        return False
    return int(getattr(args, "process_index", 0) or 0) == 0


def _write_monitoring_file(filename, metrics, args, state=None):
    """Persist one metric list next to the checkpoint so it survives cleanup."""
    if args is None or not _is_main_process(args, state):
        return None

    output_dir = getattr(args, "output_dir", None)
    if not output_dir:
        return None

    os.makedirs(output_dir, exist_ok=True)
    tokens_per_step = _tokens_per_step()
    steps_per_epoch = _steps_per_epoch(args, state)

    payload = {
        "metrics": list(metrics or []),
        "tokens_per_step": tokens_per_step,
        "steps_per_epoch": steps_per_epoch,
        "tokens_per_epoch": (
            tokens_per_step * steps_per_epoch if steps_per_epoch else None
        ),
        "global_step": getattr(state, "global_step", None) if state is not None else None,
        "num_train_epochs": float(getattr(args, "num_train_epochs", 0) or 0),
    }

    path = os.path.join(output_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=float)
    return path


class ResourceMonitorCallback(TrainerCallback):
    def __init__(self):
        self.epoch_start_time = None
        self.epoch_tokens = 0
        self.resource_metrics = []
        self.cpu_samples = []
        self.gpu_util_samples = []
        self.step_count = 0

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_start_time = time.time()
        self.epoch_tokens = 0
        self.cpu_samples = []
        self.gpu_util_samples = []
        self.step_count = 0

    def on_step_end(self, args, state, control, **kwargs):
        step_tokens = PER_DEVICE_BATCH * world_size * GRAD_ACCUM * MAX_SEQ_LEN
        self.epoch_tokens += step_tokens

        self.cpu_samples.append(psutil.cpu_percent(interval=None))
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, check=True
            )
            lines = [l.strip() for l in result.stdout.strip().split("\n") if l.strip()]
            if lines:
                self.gpu_util_samples.append(np.mean([float(l) for l in lines]))
            else:
                self.gpu_util_samples.append(0.0)
        except Exception:
            self.gpu_util_samples.append(0.0)
        self.step_count += 1

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch_time = time.time() - self.epoch_start_time
        tokens_per_sec = self.epoch_tokens / epoch_time if epoch_time > 0 else 0

        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True
        )
        lines = [l.strip() for l in result.stdout.strip().split("\n") if l.strip()]
        if lines:
            gpu_mem_gb = np.mean([float(l.split(", ")[1]) for l in lines]) / 1024
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

    def on_train_end(self, args, state, control, **kwargs):
        # Persist metrics so they can be compared after the trainer is freed.
        self.save(args, state)

    def save(self, args, state=None):
        """Write ``resource_metrics.json`` into ``args.output_dir``."""
        return _write_monitoring_file(
            _RESOURCE_METRICS_FILE, self.resource_metrics, args, state
        )


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
        # Persist metrics before the hooks are torn down.
        self.save(args, state)
        self._remove_hooks()

    def save(self, args, state=None):
        """Write ``moe_metrics.json`` into ``args.output_dir``."""
        return _write_monitoring_file(
            _MOE_METRICS_FILE, self.metrics_history, args, state
        )

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
                if expected > 0:
                    vio = np.max(np.abs(counts - expected)) / (expected + 1e-12)
                else:
                    vio = 0.0
                step_vios.append(vio)

                if layer_idx < len(self.gates) and hasattr(self.gates[layer_idx], 'update_biases'):
                    self.gates[layer_idx].update_biases(torch.from_numpy(counts).to(model.device))

            if step_vios:
                self.epoch_batch_max_vios.append(np.mean(step_vios))
            self.step_layer_counts = {}
        else:
            if self.batch_counts is not None:
                n_experts = model.config.n_routed_experts
                total = self.batch_counts.sum()
                expected = total / n_experts if total > 0 else 0.0
                if expected > 0:
                    batch_max = np.max(np.abs(self.batch_counts - expected)) / (expected + 1e-12)
                else:
                    batch_max = 0.0
                self.epoch_batch_max_vios.append(batch_max)
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
            def create_val_hook(idx, gate):
                size = gate.n_routed_experts
                arr = np.zeros(size, dtype=np.float64)
                layer_expert_counts.append(arr)
                def val_hook_fn(module, input, output):
                    nonlocal arr
                    if len(output) >= 2 and isinstance(output[1], torch.Tensor):
                        weights = output[1]
                    elif len(output) >= 1 and isinstance(output[0], torch.Tensor):
                        weights = output[0]
                    else:
                        return
                    reduce_dims = tuple(range(weights.dim() - 1))
                    activated = (weights > 1e-8).float()
                    counts = activated.sum(dim=reduce_dims).detach().cpu().numpy()
                    arr += counts
                return val_hook_fn
            for idx, layer in enumerate(moe_layers):
                gate = layer.mlp.gate
                hook = gate.register_forward_hook(create_val_hook(idx, gate))
                val_hooks.append(hook)
        else:
            n_experts = config.n_routed_experts
            expert_counts = np.zeros(n_experts, dtype=np.float64)
            val_hooks = []
            def val_hook_fn(module, input, output):
                nonlocal expert_counts
                topk_idx = output[0]
                if topk_idx.is_floating_point():
                    topk_idx = topk_idx.long()
                counts = torch.bincount(topk_idx.flatten(), minlength=n_experts).cpu().numpy()
                expert_counts += counts
            for layer in moe_layers:
                hook = layer.mlp.gate.register_forward_hook(val_hook_fn)
                val_hooks.append(hook)

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
                        valid = (shift_labels != -100).sum().item()
                        total_valid_tokens += valid

        for h in val_hooks:
            h.remove()

        eval_loss = total_loss / total_valid_tokens if total_valid_tokens > 0 else float('inf')
        measured_flops = flop_counter.get_total_flops()
        gflops = measured_flops / 1e9

        if self.is_dynmoe:
            total_activations = sum(cnt.sum() for cnt in layer_expert_counts)
            avg_activated = total_activations / (total_valid_tokens * num_moe_layers + 1e-12)

            layer_vios = []
            for cnt in layer_expert_counts:
                total = cnt.sum()
                n_exp = len(cnt)
                expected = total / n_exp if total > 0 else 0.0
                if expected > 0:
                    vio = np.max(np.abs(cnt - expected)) / (expected + 1e-12)
                else:
                    vio = 0.0
                layer_vios.append(vio)
            max_vio_global = np.mean(layer_vios) if layer_vios else 0.0
            max_vio_batch = np.mean(self.epoch_batch_max_vios) if self.epoch_batch_max_vios else 0.0

            expert_params = 2 * config.moe_intermediate_size * config.hidden_size \
                            + config.moe_intermediate_size + config.hidden_size
            active_params = num_moe_layers * avg_activated * expert_params
            avg_routed_str = f"(avg routed: {avg_activated:.2f})"
        else:
            total_activated = getattr(config, 'num_experts_per_tok', 2)
            total = expert_counts.sum()
            expected = total / n_experts if total > 0 else 0.0
            if expected > 0:
                max_vio_global = np.max(np.abs(expert_counts - expected)) / (expected + 1e-12)
            else:
                max_vio_global = 0.0
            max_vio_batch = np.mean(self.epoch_batch_max_vios) if self.epoch_batch_max_vios else 0.0
            expert_params = 3 * config.moe_intermediate_size * config.hidden_size
            active_params = num_moe_layers * total_activated * expert_params
            avg_routed_str = ""

        metrics = {
            'epoch': state.epoch,
            'train_loss': self.last_train_loss,
            'eval_loss': eval_loss,
            'gflops': gflops,
            'max_vio_global': max_vio_global,
            'max_vio_batch': max_vio_batch,
            'active_params': active_params
        }
        if self.is_dynmoe:
            metrics['avg_activated'] = avg_activated

        self.metrics_history.append(metrics)
        self.max_vio_values.append(max_vio_global)
        self.batch_max_vio_values.append(max_vio_batch)
        self.gflops_values.append(gflops)
        self._epoch_metrics_printed = True

        print(f"\nEpoch {state.epoch:.2f} Metrics:")
        print(f" Loss          : {self.last_train_loss:8.4f}")
        print(f" Val_Loss      : {eval_loss:8.4f}")
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
        if hasattr(model, 'transformer') and hasattr(model.transformer, 'decoder') and hasattr(model.transformer.decoder, 'layers'):
            return model.transformer.decoder.layers
        if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            return model.transformer.h
        for name, module in model.named_modules():
            if name.endswith('.layers') or name == 'layers':
                if isinstance(module, torch.nn.ModuleList):
                    return module
        raise AttributeError("Could not locate the layer container in the model.")

    def _get_moe_layers(self, model):
        all_layers = self._get_layer_list(model)
        moe_layers = []
        for layer in all_layers:
            if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'gate'):
                moe_layers.append(layer)
        return moe_layers

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

        is_dynmoe = False
        for layer in moe_layers:
            gate = layer.mlp.gate
            if hasattr(gate, 'thresholds') or hasattr(gate, 'update_biases') or hasattr(gate, 'adaptive_tune'):
                is_dynmoe = True
                break
        self.is_dynmoe = is_dynmoe

        if is_dynmoe:
            self.gates = []
            for idx, layer in enumerate(moe_layers):
                gate = layer.mlp.gate
                self.gates.append(gate)
                hook = gate.register_forward_hook(
                    lambda m, i, o, idx=idx: self._batch_hook_dynmoe(m, i, o, idx)
                )
                self.hooks.append(hook)
            print(f"[MoEMetricsCallback] Attached {len(self.hooks)} hooks for DYNMoE")
        else:
            for layer in moe_layers:
                gate = layer.mlp.gate
                hook = gate.register_forward_hook(self._batch_hook_standard)
                self.hooks.append(hook)
            print(f"[MoEMetricsCallback] Attached {len(self.hooks)} hooks for standard DeepSeekMoE")

    def _batch_hook_standard(self, module, input, output):
        topk_idx = output[0]
        if topk_idx.is_floating_point():
            topk_idx = topk_idx.long()
        counts = torch.bincount(topk_idx.flatten(), minlength=module.n_routed_experts).cpu().numpy()
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
        counts = (weights > 1e-8).float().sum(dim=reduce_dims).detach().cpu().numpy()

        if layer_idx not in self.step_layer_counts:
            self.step_layer_counts[layer_idx] = counts
        else:
            self.step_layer_counts[layer_idx] += counts



# Post-training comparison plots
_PANEL_SPECS = (
    ("val_loss", "Val_Loss", "cross-entropy (nats/token)",
     ("eval_loss", "val_loss", "validation_loss")),
    ("gflops", "GFLOPs", "GFLOPs",
     ("gflops", "g_flops")),
    ("max_vio_global", "MaxVIO_global", "MaxVIO",
     ("max_vio_global", "maxvio_global")),
    ("max_vio_batch", "MaxVIO_batch", "MaxVIO",
     ("max_vio_batch", "maxvio_batch")),
    ("gpu_mem", "GPU Mem", "GB",
     ("gpu_mem_gb", "gpu_mem", "gpu_memory_gb")),
    ("sys_mem", "Sys Mem", "GB",
     ("sys_mem_gb", "sys_mem", "system_mem_gb")),
    ("gpu_util", "GPU Usage", "%",
     ("gpu_util", "gpu_usage", "gpu_util_pct")),
    ("cpu_util", "CPU Usage", "%",
     ("cpu_pct", "cpu_util", "cpu_usage")),
)


def _read_json_file(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _first_not_none(*values):
    for value in values:
        if value is not None:
            return value
    return None


def _epoch_key(record):
    try:
        return round(float(record.get("epoch", 0.0)), 6)
    except (TypeError, ValueError):
        return 0.0


def _merge_records(resource_metrics, moe_metrics):
    """Outer-join the two per-epoch lists on ``epoch``."""
    merged = {}
    for record in list(resource_metrics or []):
        if isinstance(record, dict):
            merged.setdefault(_epoch_key(record), {}).update(record)
    for record in list(moe_metrics or []):
        if isinstance(record, dict):
            merged.setdefault(_epoch_key(record), {}).update(record)
    return [merged[key] for key in sorted(merged)]


def _build_run(resource_payload, moe_payload, name):
    resource_payload = resource_payload or {}
    moe_payload = moe_payload or {}

    records = _merge_records(
        resource_payload.get("metrics"),
        moe_payload.get("metrics"),
    )
    return {
        "name": name,
        "records": records,
        "tokens_per_step": _first_not_none(
            resource_payload.get("tokens_per_step"),
            moe_payload.get("tokens_per_step"),
        ),
        "steps_per_epoch": _first_not_none(
            resource_payload.get("steps_per_epoch"),
            moe_payload.get("steps_per_epoch"),
        ),
        "tokens_per_epoch": _first_not_none(
            resource_payload.get("tokens_per_epoch"),
            moe_payload.get("tokens_per_epoch"),
        ),
    }


def _load_from_callbacks(source):
    items = source if isinstance(source, (list, tuple)) else [source]
    resource_metrics, moe_metrics = [], []
    found = False
    for item in items:
        if hasattr(item, "resource_metrics"):
            resource_metrics = list(getattr(item, "resource_metrics") or [])
            found = True
        if hasattr(item, "metrics_history"):
            moe_metrics = list(getattr(item, "metrics_history") or [])
            found = True
    if not found:
        return None
    return _build_run(
        {"metrics": resource_metrics, "tokens_per_step": _tokens_per_step()},
        {"metrics": moe_metrics, "tokens_per_step": _tokens_per_step()},
        "callbacks",
    )


def load_monitoring_metrics(source):
    """
    Load one run's monitoring metrics.
    """
    from pathlib import Path

    if isinstance(source, (str, os.PathLike)):
        path = Path(source)
        if path.is_dir():
            resource_payload = _read_json_file(path / _RESOURCE_METRICS_FILE)
            moe_payload = _read_json_file(path / _MOE_METRICS_FILE)
            if resource_payload is None and moe_payload is None:
                raise FileNotFoundError(
                    f"No {_RESOURCE_METRICS_FILE} or {_MOE_METRICS_FILE} in {path}. "
                    "Re-run training with this version of monitoring.py so the "
                    "callbacks can write their metrics."
                )
            return _build_run(resource_payload, moe_payload, str(path))

        if path.is_file():
            payload = _read_json_file(path)
            if payload is None:
                raise FileNotFoundError(f"Could not read {path}")
            if "resource" in path.name:
                return _build_run(payload, None, str(path))
            return _build_run(None, payload, str(path))

        raise FileNotFoundError(f"Monitoring source not found: {path}")

    loaded = _load_from_callbacks(source)
    if loaded is not None:
        return loaded

    raise TypeError(
        "Unsupported monitoring source. Pass a run directory, a metrics JSON "
        f"file, or the callbacks. Got {type(source)!r}"
    )


def _record_value(record, fields):
    for field in fields:
        if field in record:
            try:
                value = float(record[field])
            except (TypeError, ValueError):
                continue
            if np.isfinite(value):
                return value
    return None


def _series(records, fields, tokens_per_epoch):
    xs, ys = [], []
    for index, record in enumerate(records):
        value = _record_value(record, fields)
        if value is None:
            continue

        if record.get("tokens_seen") is not None:
            x = float(record["tokens_seen"])
        elif tokens_per_epoch:
            x = float(record.get("epoch", index)) * float(tokens_per_epoch)
        else:
            x = float(record.get("epoch", index))

        xs.append(x)
        ys.append(value)

    if not xs:
        return None, None
    order = np.argsort(xs)
    return np.asarray(xs, dtype=float)[order], np.asarray(ys, dtype=float)[order]


def _choose_token_unit(max_value):
    if max_value >= 1e9:
        return "B", 1e9
    if max_value >= 1e6:
        return "M", 1e6
    if max_value >= 1e3:
        return "K", 1e3
    return "", 1.0


def plot_training_comparison(
    baseline,
    dynmoe,
    output_path="training_comparison.png",
    tokens_per_step=None,
    steps_per_epoch=None,
    figsize=(15.0, 16.0),
    dpi=150,
    title="DeepSeekMoE vs DeepSeekMoE + DYNMoE Routing",
    baseline_label="DeepSeekMoE (baseline)",
    dynmoe_label="DeepSeekMoE + DYNMoE",
    colors=("#E69F00", "#0072B2"),
    annotate_final=True,
    show=False,
    verbose=True,
):
    """
    Plot the eight training metrics for baseline vs DYNMoE.
    """
    import matplotlib.pyplot as plt

    base = load_monitoring_metrics(baseline)
    dyn = load_monitoring_metrics(dynmoe)

    tokens_per_step = _first_not_none(
        tokens_per_step, base["tokens_per_step"], dyn["tokens_per_step"]
    )
    steps_per_epoch = _first_not_none(
        steps_per_epoch, base["steps_per_epoch"], dyn["steps_per_epoch"]
    )

    if tokens_per_step and steps_per_epoch:
        tokens_per_epoch = float(tokens_per_step) * float(steps_per_epoch)
    else:
        tokens_per_epoch = _first_not_none(
            base["tokens_per_epoch"], dyn["tokens_per_epoch"]
        )

    # Resolve the x unit once, from the largest x across both runs.
    max_x = 0.0
    for run in (base, dyn):
        for _, _, _, fields in _PANEL_SPECS:
            xs, _ = _series(run["records"], fields, tokens_per_epoch)
            if xs is not None and xs.size:
                max_x = max(max_x, float(xs.max()))

    if tokens_per_epoch:
        x_unit, x_scale = _choose_token_unit(max_x)
        xlabel = f"Training tokens ({x_unit})" if x_unit else "Training tokens"
    else:
        x_unit, x_scale = "", 1.0
        xlabel = "Epoch"

    n_panels = len(_PANEL_SPECS)
    ncols = 2
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes_grid = plt.subplots(
        nrows, ncols, figsize=figsize, squeeze=False, sharex=False
    )
    axes = list(axes_grid.ravel())

    summary = []
    series = {}

    for ax, (key, panel_title, ylabel, fields) in zip(axes, _PANEL_SPECS):
        panel_series = {}

        for model_index, (run, color, label) in enumerate((
            (base, colors[0], baseline_label),
            (dyn, colors[1], dynmoe_label),
        )):
            xs, ys = _series(run["records"], fields, tokens_per_epoch)
            if xs is None:
                continue

            xs = xs / x_scale
            panel_series[label] = (xs, ys)
            marker = "o" if xs.size <= 20 else None

            ax.plot(
                xs, ys,
                color=color, linewidth=2.0,
                marker=marker, markersize=5,
                label=label, zorder=3,
            )

            if annotate_final:
                ax.annotate(
                    f"{ys[-1]:.4g}",
                    xy=(xs[-1], ys[-1]),
                    # nudge the two labels apart when the curves coincide
                    xytext=(4, 6 if model_index == 0 else -6),
                    textcoords="offset points",
                    color=color, fontsize=8, va="center",
                    annotation_clip=False,
                )

            summary.append({
                "metric": key,
                "panel": panel_title,
                "model": label,
                "first": float(ys[0]),
                "final": float(ys[-1]),
                "min": float(np.min(ys)),
                "max": float(np.max(ys)),
                "n_points": int(ys.size),
            })

        series[key] = panel_series

        ax.set_title(panel_title, fontsize=12)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3, linewidth=0.6)
        ax.margins(x=0.08)

        if not panel_series:
            ax.text(
                0.5, 0.5, "not recorded",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=10, color="0.45",
            )
            ax.set_yticks([])
        elif key in ("gpu_util", "cpu_util"):
            ax.set_ylim(0.0, 100.0)
        elif key in ("gpu_mem", "sys_mem"):
            values = np.concatenate([v for _, v in panel_series.values()])
            finite = values[np.isfinite(values)]
            if finite.size and np.nanmin(finite) < 0.25 * np.nanmax(finite):
                ax.set_ylim(bottom=0.0)

    for ax in axes[n_panels:]:
        ax.set_visible(False)

    for index, ax in enumerate(axes[:n_panels]):
        if index >= (nrows - 1) * ncols:
            ax.set_xlabel(xlabel)

    from matplotlib.lines import Line2D

    label_colors = {baseline_label: colors[0], dynmoe_label: colors[1]}
    plotted = list(dict.fromkeys(row["model"] for row in summary))
    handles = [
        Line2D([0], [0], color=label_colors.get(label, "0.4"),
               linewidth=2.0, marker="o")
        for label in plotted
    ]
    if handles:
        fig.legend(
            handles, plotted,
            loc="upper center", ncol=len(handles), frameon=False,
            fontsize=11, bbox_to_anchor=(0.5, 0.995),
        )

    fig.suptitle(title, fontsize=15, fontweight="bold", y=1.0)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.965), h_pad=2.0, w_pad=1.5)

    if output_path:
        output_path = str(output_path)
        parent = os.path.dirname(output_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        if verbose:
            print(f"Saved comparison figure: {output_path}")

    if show:
        plt.show()

    return {
        "fig": fig,
        "axes": axes,
        "series": series,
        "summary": summary,
        "tokens_per_epoch": tokens_per_epoch,
        "x_unit": x_unit,
        "output_path": output_path,
    }


def print_monitoring_summary(result):
    """Print first/final/min/max per metric and the DYNMoE - baseline delta."""
    summary = result.get("summary") or []
    if not summary:
        print("No monitoring metrics to summarise.")
        return

    models = list(dict.fromkeys(row["model"] for row in summary))
    metrics = list(dict.fromkeys(row["metric"] for row in summary))

    print("\nTraining monitoring comparison")
    print("=" * 100)
    print(
        f"{'metric':<20s}{'model':<30s}"
        f"{'first':>12s}{'final':>12s}{'min':>12s}{'max':>12s}"
    )
    print("-" * 100)

    by_metric = {}
    for row in summary:
        by_metric.setdefault(row["metric"], {})[row["model"]] = row
        print(
            f"{row['panel']:<20s}{row['model']:<30s}"
            f"{row['first']:>12.4g}{row['final']:>12.4g}"
            f"{row['min']:>12.4g}{row['max']:>12.4g}"
        )

    if len(models) >= 2:
        print("-" * 100)
        first, second = models[0], models[1]
        print(
            f"{'Delta (final, DYNMoE - baseline)':<56s}"
            f"{'absolute':>14s}{'relative':>14s}"
        )
        for metric in metrics:
            a = by_metric.get(metric, {}).get(first)
            b = by_metric.get(metric, {}).get(second)
            if not a or not b:
                continue
            delta = b["final"] - a["final"]
            pct = (delta / a["final"] * 100.0) if a["final"] else float("nan")
            print(f"  {a['panel']:<54s}{delta:>+14.4g}{pct:>+13.1f}%")

    print("=" * 100)
    if result.get("tokens_per_epoch"):
        print(f"tokens/epoch: {result['tokens_per_epoch']:,.0f}")
    print(
        "Note: one point per epoch (the callbacks record at epoch end); "
        "increase NUM_EPOCHS for a denser curve."
    )


# Convenience alias
plot_comparison = plot_training_comparison