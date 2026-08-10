# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A bounded, persistent reservoir buffer for continual-learning replay."""

from __future__ import annotations

import json
import os
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch


SCHEMA_VERSION = 1


@dataclass
class ReplayBatch:
    """A batch retrieved from historical replay memory."""

    train_x: torch.Tensor
    train_y: torch.Tensor
    task_ids: list[str]
    bases: list[str]
    sample_ids: list[str]

    def __len__(self) -> int:
        return int(self.train_x.shape[0])


class ReplayBuffer:
    """Per-rank global-reservoir memory.

    Samples from ``current_task_id`` are admitted but excluded from retrieval.
    They automatically become historical when the next task starts with a new
    task id. This makes task finalization robust across separate launcher runs.
    """

    def __init__(
        self,
        *,
        capacity: int,
        seed: int,
        rank: int = 0,
        world_size: int = 1,
        storage_dtype: torch.dtype = torch.float16,
    ) -> None:
        if int(capacity) <= 0:
            raise ValueError("Replay capacity must be positive")
        if int(world_size) <= 0 or not 0 <= int(rank) < int(world_size):
            raise ValueError("Invalid replay rank/world_size")
        self.capacity = int(capacity)
        self.seed = int(seed)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.storage_dtype = storage_dtype
        self.seen_count = 0
        self._admission_rng = random.Random(self.seed + self.rank * 1_000_003)
        self._retrieval_rng = random.Random(self.seed + self.rank * 1_000_003 + 1)
        self._train_x: Optional[torch.Tensor] = None
        self._train_y: Optional[torch.Tensor] = None
        self._size = 0
        self._task_ids: list[str] = [""] * self.capacity
        self._bases: list[str] = [""] * self.capacity
        self._sample_ids: list[str] = [""] * self.capacity
        self.snapshot_epoch = 0
        self.snapshot_global_step = 0

    def __len__(self) -> int:
        return self._size

    @property
    def train_x_shape(self) -> tuple[int, ...] | None:
        return None if self._train_x is None else tuple(self._train_x.shape[1:])

    @property
    def train_y_shape(self) -> tuple[int, ...] | None:
        return None if self._train_y is None else tuple(self._train_y.shape[1:])

    def _ensure_storage(self, train_x: torch.Tensor, train_y: torch.Tensor) -> None:
        x_shape = tuple(train_x.shape[1:])
        y_shape = tuple(train_y.shape[1:])
        if self._train_x is None:
            self._train_x = torch.empty(
                (self.capacity, *x_shape), dtype=self.storage_dtype, device="cpu"
            )
            self._train_y = torch.empty(
                (self.capacity, *y_shape), dtype=torch.uint8, device="cpu"
            )
            return
        if x_shape != self.train_x_shape or y_shape != self.train_y_shape:
            raise ValueError(
                "Replay tensor shape mismatch: "
                f"buffer X/Y={self.train_x_shape}/{self.train_y_shape}, "
                f"batch X/Y={x_shape}/{y_shape}"
            )

    def eligible_indices(self, current_task_id: str) -> list[int]:
        current_task_id = str(current_task_id)
        return [i for i in range(self._size) if self._task_ids[i] != current_task_id]

    def eligible_count(self, current_task_id: str) -> int:
        return sum(
            1 for i in range(self._size) if self._task_ids[i] != str(current_task_id)
        )

    def observe(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        *,
        task_id: str,
        basis: str,
        step: int,
    ) -> int:
        """Admit a batch with standard reservoir sampling.

        Only rows that survive reservoir decisions are copied from the training
        device to CPU. When multiple incoming rows target the same slot, the
        later row wins, matching sequential reservoir semantics.
        """
        if train_x.ndim < 1 or train_y.ndim < 1 or train_x.shape[0] != train_y.shape[0]:
            raise ValueError("Replay observe expects matching non-scalar X/Y batches")
        batch_size = int(train_x.shape[0])
        if batch_size == 0:
            return 0
        self._ensure_storage(train_x, train_y)

        final_assignments: dict[int, int] = {}
        for row in range(batch_size):
            stream_index = self.seen_count
            self.seen_count += 1
            if self._size < self.capacity:
                destination = self._size
                self._size += 1
            else:
                candidate = self._admission_rng.randint(0, stream_index)
                if candidate >= self.capacity:
                    continue
                destination = candidate
            final_assignments[destination] = row

        if not final_assignments:
            return 0
        destinations = list(final_assignments)
        rows = [final_assignments[d] for d in destinations]
        row_index = torch.tensor(rows, dtype=torch.long, device=train_x.device)
        selected_x = train_x.detach().index_select(0, row_index).to(
            device="cpu", dtype=self.storage_dtype
        ).contiguous()
        selected_y = train_y.detach().index_select(0, row_index).to(
            device="cpu", dtype=torch.uint8
        ).contiguous()
        destination_index = torch.tensor(destinations, dtype=torch.long)
        assert self._train_x is not None and self._train_y is not None
        self._train_x.index_copy_(0, destination_index, selected_x)
        self._train_y.index_copy_(0, destination_index, selected_y)
        for destination, row in final_assignments.items():
            self._task_ids[destination] = str(task_id)
            self._bases[destination] = str(basis).upper()
            self._sample_ids[destination] = (
                f"{task_id}:r{self.rank}:s{int(step)}:i{int(row)}:n{self.seen_count - batch_size + row}"
            )
        return len(final_assignments)

    def sample(
        self,
        size: int,
        *,
        current_task_id: str,
        device: torch.device | str,
    ) -> ReplayBatch | None:
        """Uniformly retrieve historical samples without replacement."""
        requested = max(0, int(size))
        eligible = self.eligible_indices(current_task_id)
        count = min(requested, len(eligible))
        if count <= 0:
            return None
        selected = self._retrieval_rng.sample(eligible, count)
        index = torch.tensor(selected, dtype=torch.long)
        assert self._train_x is not None and self._train_y is not None
        train_x = self._train_x.index_select(0, index).to(
            device=device, dtype=torch.float32, non_blocking=True
        )
        train_y = self._train_y.index_select(0, index).to(
            device=device, dtype=torch.float32, non_blocking=True
        )
        return ReplayBatch(
            train_x=train_x,
            train_y=train_y,
            task_ids=[self._task_ids[i] for i in selected],
            bases=[self._bases[i] for i in selected],
            sample_ids=[self._sample_ids[i] for i in selected],
        )

    def state_dict(self) -> dict:
        state = {
            "schema_version": SCHEMA_VERSION,
            "capacity": self.capacity,
            "seed": self.seed,
            "rank": self.rank,
            "world_size": self.world_size,
            "storage_dtype": str(self.storage_dtype).removeprefix("torch."),
            "seen_count": self.seen_count,
            "size": self._size,
            "task_ids": self._task_ids[:self._size],
            "bases": self._bases[:self._size],
            "sample_ids": self._sample_ids[:self._size],
            "admission_rng_state": self._admission_rng.getstate(),
            "retrieval_rng_state": self._retrieval_rng.getstate(),
            "snapshot_epoch": self.snapshot_epoch,
            "snapshot_global_step": self.snapshot_global_step,
        }
        if self._train_x is not None:
            state["train_x"] = self._train_x[:self._size].clone()
            state["train_y"] = self._train_y[:self._size].clone()
        return state

    def load_state_dict(self, state: dict, *, strict_world_size: bool = True) -> None:
        if int(state.get("schema_version", -1)) != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported replay schema_version={state.get('schema_version')!r}"
            )
        if strict_world_size and int(state["world_size"]) != self.world_size:
            raise ValueError(
                "Replay world_size mismatch: "
                f"snapshot={state['world_size']}, current={self.world_size}"
            )
        if int(state["rank"]) != self.rank:
            raise ValueError(
                f"Replay rank mismatch: snapshot={state['rank']}, current={self.rank}"
            )
        if int(state["capacity"]) != self.capacity:
            raise ValueError(
                f"Replay capacity mismatch: snapshot={state['capacity']}, current={self.capacity}"
            )
        self.seen_count = int(state["seen_count"])
        self._size = int(state["size"])
        self._task_ids[:self._size] = list(state["task_ids"])
        self._bases[:self._size] = list(state["bases"])
        self._sample_ids[:self._size] = list(state["sample_ids"])
        self._admission_rng.setstate(state["admission_rng_state"])
        self._retrieval_rng.setstate(state["retrieval_rng_state"])
        self.snapshot_epoch = int(state.get("snapshot_epoch", 0))
        self.snapshot_global_step = int(state.get("snapshot_global_step", 0))
        if self._size:
            saved_x = state["train_x"]
            saved_y = state["train_y"]
            self._ensure_storage(saved_x, saved_y)
            assert self._train_x is not None and self._train_y is not None
            self._train_x[:self._size].copy_(saved_x.to(dtype=self.storage_dtype))
            self._train_y[:self._size].copy_(saved_y.to(dtype=torch.uint8))

    @staticmethod
    def shard_path(directory: str | Path, rank: int) -> Path:
        return Path(directory) / f"rank_{int(rank):03d}.pt"

    def save(self, directory: str | Path, *, epoch: int, global_step: int) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.snapshot_epoch = int(epoch)
        self.snapshot_global_step = int(global_step)
        destination = self.shard_path(directory, self.rank)
        fd, tmp_name = tempfile.mkstemp(prefix=destination.name + ".", suffix=".tmp", dir=directory)
        os.close(fd)
        try:
            torch.save(self.state_dict(), tmp_name)
            os.replace(tmp_name, destination)
        finally:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
        if self.rank == 0:
            manifest = {
                "schema_version": SCHEMA_VERSION,
                "capacity_per_rank": self.capacity,
                "global_capacity": self.capacity * self.world_size,
                "world_size": self.world_size,
                "storage_dtype": str(self.storage_dtype).removeprefix("torch."),
                "snapshot_epoch": self.snapshot_epoch,
                "snapshot_global_step": self.snapshot_global_step,
            }
            manifest_path = directory / "manifest.json"
            fd, manifest_tmp = tempfile.mkstemp(
                prefix="manifest.", suffix=".tmp", dir=directory
            )
            os.close(fd)
            try:
                Path(manifest_tmp).write_text(
                    json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
                )
                os.replace(manifest_tmp, manifest_path)
            finally:
                if os.path.exists(manifest_tmp):
                    os.unlink(manifest_tmp)
        return destination

    def load(self, directory: str | Path, *, strict_world_size: bool = True) -> bool:
        path = self.shard_path(directory, self.rank)
        if not path.exists():
            return False
        state = torch.load(path, map_location="cpu", weights_only=False)
        self.load_state_dict(state, strict_world_size=strict_world_size)
        return True


__all__ = ["ReplayBatch", "ReplayBuffer", "SCHEMA_VERSION"]
