"""
GPU memory utilities for clearing caches and debugging VRAM usage.

Best practice: delete tensors/models first, then call clear_gpu(). For full VRAM
release, move the model to CPU and delete it before calling clear_gpu(). In
production, consider a worker process that loads the model, runs inference, then
exits so the OS frees VRAM and avoids fragmentation.
"""

import gc
from typing import Optional, Tuple, Union

import torch


def clear_gpu(
    device: Optional[torch.device] = None,
    ipc_collect: bool = True,
    reset_peak_stats: bool = False,
) -> None:
    """
    Run garbage collection and clear device caches.

    Frees Python references (gc.collect()) and releases cached but unused
    memory on the device. Does not free memory still held by active tensors;
    delete those first or move the model to CPU and delete it before calling.

    Args:
        device: torch device (e.g. from self.device). If None or CPU, only
            gc.collect() is run.
        ipc_collect: If True and device is CUDA, call torch.cuda.ipc_collect()
            (useful when multiprocessing is used). Ignored for MPS/CPU.
        reset_peak_stats: If True and device is CUDA, call
            torch.cuda.reset_peak_memory_stats(). Ignored for MPS/CPU.
    """
    gc.collect()
    if device is None:
        return
    if device.type == "cuda":
        torch.cuda.empty_cache()
        if ipc_collect:
            torch.cuda.ipc_collect()
        if reset_peak_stats:
            torch.cuda.reset_peak_memory_stats()
    elif device.type == "mps":
        torch.mps.empty_cache()


def get_cuda_memory_info(
    device: Optional[Union[torch.device, str]] = None,
) -> Optional[Tuple[int, int]]:
    """
    Return (allocated, reserved) bytes for CUDA, for debugging.

    Useful to check if reserved >> allocated (cache holding memory).
    Returns None if device is not CUDA or CUDA is not available.

    Args:
        device: torch device or string (e.g. "cuda"). If None, uses current
            CUDA device.

    Returns:
        (memory_allocated, memory_reserved) in bytes, or None if not CUDA.
    """
    if not torch.cuda.is_available():
        return None
    if device is not None:
        d = device if isinstance(device, torch.device) else torch.device(device)
        if d.type != "cuda":
            return None
    return (
        torch.cuda.memory_allocated(),
        torch.cuda.memory_reserved(),
    )
