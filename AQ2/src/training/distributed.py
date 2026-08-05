"""Distributed training utilities.

This module provides common utilities for PyTorch distributed training.
"""

import datetime
import os
from typing import Optional

import torch
import torch.distributed as dist


def is_distributed() -> bool:
    """Check if distributed training is active."""
    return dist.is_available() and dist.is_initialized()


def is_main_process() -> bool:
    """Check if this is the main (rank 0) process."""
    return not is_distributed() or dist.get_rank() == 0


def get_rank() -> int:
    """Get the current process rank."""
    return dist.get_rank() if is_distributed() else 0


def get_world_size() -> int:
    """Get the total number of processes."""
    return dist.get_world_size() if is_distributed() else 1


def setup_distributed(device_type: str) -> tuple[int, int, int]:
    """Initialize distributed training if RANK is set in environment.

    Args:
        device_type: One of "cuda", "mps", or "cpu".

    Returns:
        Tuple of (rank, world_size, local_rank). Returns (0, 1, 0) when
        not running in a distributed context.
    """
    if "RANK" not in os.environ:
        return 0, 1, 0

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    device_id: Optional[torch.device] = None
    if device_type == "cuda" and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device_id = torch.device(f"cuda:{local_rank}")
    backend = "nccl" if device_type == "cuda" and torch.cuda.is_available() else "gloo"
    dist.init_process_group(
        backend=backend,
        init_method="env://",
        world_size=world_size,
        rank=rank,
        timeout=datetime.timedelta(minutes=30),
        device_id=device_id,
    )
    return rank, world_size, local_rank


def get_device(device_type: str) -> torch.device:
    """Get the appropriate torch device for the given device type.
    
    Args:
        device_type: One of "cuda", "mps", or "cpu".
        
    Returns:
        The torch.device for the current process.
    """
    if device_type == "cuda" and torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        return torch.device(f"cuda:{local_rank}")
    if device_type == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_device_type(device: str) -> str:
    """Resolve 'auto' device selection to a concrete device type.

    Args:
        device: One of "auto", "cuda", "mps", or "cpu".

    Returns:
        The resolved device type string.
    """
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device


def resolve_device(device_arg: str) -> torch.device:
    """Resolve device argument to a torch.device.

    Args:
        device_arg: One of "auto", "cuda", "mps", or "cpu".

    Returns:
        The resolved torch.device.
    """
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_arg)
