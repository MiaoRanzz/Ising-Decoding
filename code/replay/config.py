# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration resolution for the internal continual-replay feature."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import torch
from omegaconf import OmegaConf


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def _select(cfg, path: str, default):
    return OmegaConf.select(cfg, path, default=default)


@dataclass(frozen=True)
class ReplaySettings:
    enabled: bool
    task_id: str
    buffer_dir: Path
    global_capacity: int
    seed: int
    ratio: float
    replay_lambda: float
    storage_dtype: torch.dtype
    save_every_epoch: bool
    strict_world_size: bool

    def local_capacity(self, rank: int, world_size: int) -> int:
        base, remainder = divmod(self.global_capacity, int(world_size))
        capacity = base + (1 if int(rank) < remainder else 0)
        if capacity <= 0:
            raise ValueError(
                "Replay global capacity must be at least the distributed world size"
            )
        return capacity


def resolve_replay_settings(cfg) -> ReplaySettings:
    enabled = _env_bool(
        "PREDECODER_REPLAY_ENABLED", bool(_select(cfg, "replay.enabled", False))
    )
    task_id = str(
        os.environ.get(
            "PREDECODER_REPLAY_TASK_ID", _select(cfg, "replay.task_id", "") or ""
        )
    ).strip()
    output = str(_select(cfg, "output", "outputs/pre-decoder"))
    default_dir = str(_select(cfg, "replay.buffer_dir", f"{output}/replay"))
    buffer_dir = Path(os.environ.get("PREDECODER_REPLAY_DIR", default_dir)).expanduser()
    global_capacity = int(
        os.environ.get(
            "PREDECODER_REPLAY_CAPACITY", _select(cfg, "replay.capacity", 65536)
        )
    )
    seed = int(
        os.environ.get("PREDECODER_REPLAY_SEED", _select(cfg, "replay.seed", 12345))
    )
    ratio = float(
        os.environ.get("PREDECODER_REPLAY_RATIO", _select(cfg, "replay.ratio", 0.5))
    )
    replay_lambda = float(
        os.environ.get(
            "PREDECODER_REPLAY_LAMBDA", _select(cfg, "replay.lambda_replay", 1.0)
        )
    )
    dtype_name = str(
        os.environ.get(
            "PREDECODER_REPLAY_STORAGE_DTYPE",
            _select(cfg, "replay.storage_dtype", "float16"),
        )
    ).strip().lower()
    dtype_by_name = {"float16": torch.float16, "float32": torch.float32}
    if dtype_name not in dtype_by_name:
        raise ValueError("replay.storage_dtype must be float16 or float32")
    save_every_epoch = _env_bool(
        "PREDECODER_REPLAY_SAVE_EVERY_EPOCH",
        bool(_select(cfg, "replay.save_every_epoch", True)),
    )
    strict_world_size = _env_bool(
        "PREDECODER_REPLAY_STRICT_WORLD_SIZE",
        bool(_select(cfg, "replay.strict_world_size", True)),
    )
    if enabled and not task_id:
        raise ValueError(
            "Replay is enabled but no task id was provided; set "
            "PREDECODER_REPLAY_TASK_ID (for example T0_base)"
        )
    if global_capacity <= 0:
        raise ValueError("replay.capacity must be positive")
    if not 0.0 <= ratio < 1.0:
        raise ValueError("replay.ratio must be in [0, 1)")
    if replay_lambda < 0.0:
        raise ValueError("replay.lambda_replay must be non-negative")
    return ReplaySettings(
        enabled=enabled,
        task_id=task_id,
        buffer_dir=buffer_dir,
        global_capacity=global_capacity,
        seed=seed,
        ratio=ratio,
        replay_lambda=replay_lambda,
        storage_dtype=dtype_by_name[dtype_name],
        save_every_epoch=save_every_epoch,
        strict_world_size=strict_world_size,
    )


__all__ = ["ReplaySettings", "resolve_replay_settings"]
