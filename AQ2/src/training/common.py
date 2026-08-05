"""Shared training utilities.

Common constants and helper functions used across training, finetuning,
and distillation scripts.
"""

import math
from typing import Any, Optional

import numpy as np
import torch
import torch.distributed as dist

from src.training.distributed import is_distributed, is_main_process


# ---------------------------------------------------------------------------
# Noise curriculum
# ---------------------------------------------------------------------------
NOISE_CURRICULUM_ENABLED = True
NOISE_CURRICULUM_PARAMS = {
    "w_c": 12.0,
    "sigma_c": 0.05,
    "f_c_min": 0.0,
    "t_c": 12.8e6,
    "s_c": 1.0,
}
NOISE_CURRICULUM_SCALE_FACTORS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


# ---------------------------------------------------------------------------
# Learning-rate schedule
# ---------------------------------------------------------------------------
BASE_DECAY_STEPS = [80_000, 200_000, 400_000, 1_000_000, 2_000_000]
BASE_BATCH_SIZE = 1024


def get_lr(
    step: int,
    warmup_steps: int,
    base_lr: float,
    decay_steps: list[int],
    decay_factor: float = 0.7,
) -> float:
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    lr = base_lr
    for ds in decay_steps:
        if step >= ds:
            lr *= decay_factor
    return lr


def compute_decay_steps(final_batch_size: int) -> list[int]:
    """Compute decay steps scaled by batch size relative to base batch of 1024."""
    scale = BASE_BATCH_SIZE / final_batch_size
    return [int(step * scale) for step in BASE_DECAY_STEPS]


# ---------------------------------------------------------------------------
# Default hyperparameters per dataset
# ---------------------------------------------------------------------------
def default_sycamore_lr(distance: int) -> float:
    if distance == 3:
        return 3.46e-4
    if distance == 5:
        return 2.45e-4
    return 3.46e-4


# ---------------------------------------------------------------------------
# Curriculum helpers
# ---------------------------------------------------------------------------
def curriculum_peak(
    examples_seen: float,
    f_c_min: float,
    t_c: float,
    s_c: float,
) -> float:
    if t_c <= 0:
        return 1.0
    x = examples_seen / t_c - 1.0
    return f_c_min + (1.0 - f_c_min) / (1.0 + math.exp(-s_c * x))


def curriculum_weights(
    scale_factors: list[float],
    peak: float,
    sigma_c: float,
    w_c: float,
) -> list[float]:
    if sigma_c <= 0:
        raise ValueError("sigma_c must be > 0")
    weights = []
    for f in scale_factors:
        g = math.exp(-0.5 * ((f - peak) / sigma_c) ** 2)
        weights.append(1.0 + w_c * g)
    total = sum(weights)
    return [w / total for w in weights]


# ---------------------------------------------------------------------------
# RNG state helpers (dataset-agnostic)
# ---------------------------------------------------------------------------
def gather_rng_states(
    datasets: dict[int, Any],
    val_datasets: dict[int, Any],
    curriculum_rng: np.random.Generator,
    device_type: str,
) -> Optional[dict]:
    """Gather RNG states from all ranks for checkpoint reproducibility.

    Args:
        datasets:     {distance: SI1000DEMDataset} for training.
        val_datasets: {distance: SI1000DEMDataset} for validation.
        curriculum_rng: NumPy Generator used for curriculum sampling.
        device_type:  "cuda", "mps", or "cpu".
    """
    local_state: dict[str, Any] = {
        "curriculum_np": curriculum_rng.bit_generator.state,
        "torch_cpu": torch.get_rng_state().cpu(),
    }
    for d, ds in datasets.items():
        local_state[f"dataset_np_d{d}"] = ds.rng.bit_generator.state
    if is_main_process():
        for d, vds in val_datasets.items():
            local_state[f"val_dataset_np_d{d}"] = vds.rng.bit_generator.state
    if device_type == "cuda" and torch.cuda.is_available():
        local_state["torch_cuda"] = torch.cuda.get_rng_state().cpu()

    if not is_distributed():
        return {"world_size": 1, "by_rank": {0: local_state}}

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    gathered: Optional[list] = [None for _ in range(world_size)] if rank == 0 else None
    try:
        dist.gather_object(local_state, object_gather_list=gathered, dst=0)
    except Exception:
        if rank == 0:
            return {"world_size": world_size, "by_rank": {0: local_state}}
        return None

    if rank != 0 or gathered is None:
        return None

    return {"world_size": world_size, "by_rank": {i: gathered[i] for i in range(world_size)}}


def restore_rng_states(
    gathered: Optional[dict],
    datasets: dict[int, Any],
    val_datasets: dict[int, Any],
    curriculum_rng: np.random.Generator,
    device_type: str,
) -> None:
    """Restore RNG states from checkpoint.

    Args:
        gathered:       RNG states blob from ``gather_rng_states``.
        datasets:       {distance: SI1000DEMDataset} for training.
        val_datasets:   {distance: SI1000DEMDataset} for validation.
        curriculum_rng: NumPy Generator for curriculum sampling.
        device_type:    "cuda", "mps", or "cpu".
    """
    if not gathered:
        return
    expected_ws = gathered.get("world_size")
    if is_distributed() and isinstance(expected_ws, int) and expected_ws != dist.get_world_size():
        return
    by_rank = gathered.get("by_rank")
    if not isinstance(by_rank, dict):
        return

    rank = dist.get_rank() if is_distributed() else 0
    state = by_rank.get(rank)
    if not isinstance(state, dict):
        return

    try:
        for d, ds in datasets.items():
            key = f"dataset_np_d{d}"
            if key in state:
                ds.rng.bit_generator.state = state[key]
        if is_main_process():
            for d, vds in val_datasets.items():
                key = f"val_dataset_np_d{d}"
                if key in state:
                    vds.rng.bit_generator.state = state[key]
        if "curriculum_np" in state:
            curriculum_rng.bit_generator.state = state["curriculum_np"]
        if "torch_cpu" in state and isinstance(state["torch_cpu"], torch.Tensor):
            torch.set_rng_state(state["torch_cpu"])
        if device_type == "cuda" and "torch_cuda" in state and isinstance(state["torch_cuda"], torch.Tensor):
            torch.cuda.set_rng_state(state["torch_cuda"])
    except Exception:
        return
