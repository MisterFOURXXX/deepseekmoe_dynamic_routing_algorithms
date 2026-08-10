import gc
import torch

def clear_cached_data():
    """Clear the global cached dataset to free memory."""
    global _tokenizer, _tokenized_datasets, _data_collator
    _tokenizer = None
    _tokenized_datasets = None
    _data_collator = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def clear_gpu_memory():
    """Clear GPU memory and reset peak stats."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

def cleanup_trainer(trainer):
    """
    Aggressively delete a trainer, its model and tokenizer,
    and clear GPU/CPU caches.
    """
    if trainer is not None:
        if hasattr(trainer, 'model'):
            del trainer.model
        if hasattr(trainer, 'tokenizer'):
            del trainer.tokenizer
        del trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()