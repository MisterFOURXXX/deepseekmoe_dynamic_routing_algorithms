import os
import sys
import numpy as np
import pandas as pd

def print_training_summary(resource_monitor, moemetrics_callback, train_result, eval_results, perplexity):
    avg_gpu_memory = np.mean([m['gpu_mem_gb'] for m in resource_monitor.resource_metrics])
    avg_system_memory = np.mean([m['sys_mem_gb'] for m in resource_monitor.resource_metrics])
    avg_gpu_usage = np.mean([m['gpu_util'] for m in resource_monitor.resource_metrics])
    avg_cpu_usage = np.mean([m['cpu_pct'] for m in resource_monitor.resource_metrics])
    avg_tps = np.mean([m['tokens_per_sec'] for m in resource_monitor.resource_metrics])

    max_vio_values = moemetrics_callback.max_vio_values
    avg_max_vio = np.mean(max_vio_values) if max_vio_values else 0.0
    min_max_vio = np.min(max_vio_values) if max_vio_values else 0.0
    max_max_vio = np.max(max_vio_values) if max_vio_values else 0.0

    batch_max_vio_values = moemetrics_callback.batch_max_vio_values
    avg_batch_max_vio = np.mean(batch_max_vio_values) if batch_max_vio_values else 0.0
    min_batch_max_vio = np.min(batch_max_vio_values) if batch_max_vio_values else 0.0
    max_batch_max_vio = np.max(batch_max_vio_values) if batch_max_vio_values else 0.0

    active_params_values = [m.get('active_params', 0) for m in moemetrics_callback.metrics_history]
    avg_active_params = np.mean(active_params_values) if active_params_values else 0

    final_train_loss = train_result.training_loss if hasattr(train_result, 'training_loss') else None
    if final_train_loss is None and moemetrics_callback.metrics_history:
        final_train_loss = moemetrics_callback.metrics_history[-1]['train_loss']

    final_val_loss = eval_results.get('eval_loss', float('inf'))
    final_perplexity = perplexity

    results_table = {
        'Metric': ['Average TPS', 'Final Training Loss', 'Final Validation Loss',
                   'Average CPU Usage (%)', 'Average GPU Usage (%)',
                   'Average System Memory (GB)', 'Average GPU Memory (GB)',
                   'Average FLOPs (GFLOPS)', 'Final Validation Perplexity',
                   'Average MaxVIO_global', 'Lowest MaxVIO_global', 'Highest MaxVIO_global',
                   'Average MaxVIO_batch', 'Lowest MaxVIO_batch', 'Highest MaxVIO_batch',
                   'Average Active Parameters'],
        'Value': [
            f"{avg_tps:.0f}",
            f"{final_train_loss:.4f}" if final_train_loss else 'N/A',
            f"{final_val_loss:.4f}",
            f"{avg_cpu_usage:.1f}",
            f"{avg_gpu_usage:.1f}",
            f"{avg_system_memory:.2f}",
            f"{avg_gpu_memory:.2f}",
            f"{np.mean([m['gflops'] for m in moemetrics_callback.metrics_history]):.2f}" if moemetrics_callback.metrics_history else 'N/A',
            f"{final_perplexity:.2f}",
            f"{avg_max_vio:.4f}",
            f"{min_max_vio:.4f}",
            f"{max_max_vio:.4f}",
            f"{avg_batch_max_vio:.4f}",
            f"{min_batch_max_vio:.4f}",
            f"{max_batch_max_vio:.4f}",
            f"{avg_active_params:,.0f}"
        ]
    }
    df = pd.DataFrame(results_table)
    print("\n" + "="*60)
    print("Training Results Summary")
    print("="*60)
    print(df.to_string(index=False))
    print("="*60)