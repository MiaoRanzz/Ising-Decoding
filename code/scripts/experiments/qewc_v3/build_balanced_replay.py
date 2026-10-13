#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build a deterministic, exactly task-balanced QEWC v3 Replay buffer."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[4]
CODE_ROOT = REPO_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from replay.buffer import ReplayBuffer  # noqa: E402
from scripts.compute_ewc_fisher import _build_training_generator, _load_public_config  # noqa: E402
from scripts.experiments.qewc_v3.run_continual import Task, parse_task  # noqa: E402


def task_quotas(capacity: int, task_names: list[str]) -> dict[str, int]:
    if capacity < len(task_names):
        raise ValueError("Replay capacity must be at least the number of tasks")
    base, remainder = divmod(capacity, len(task_names))
    return {
        name: base + (1 if index < remainder else 0)
        for index, name in enumerate(task_names)
    }


def basis_quotas(task_capacity: int) -> dict[str, int]:
    x_count = (int(task_capacity) + 1) // 2
    return {"X": x_count, "Z": int(task_capacity) - x_count}


def task_seed(base_seed: int, task_name: str) -> int:
    return int(base_seed) + sum(
        (index + 1) * byte for index, byte in enumerate(task_name.encode())
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", action="append", type=parse_task, required=True)
    parser.add_argument("--capacity", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--target-bytes", type=int, default=5202656)
    parser.add_argument("--byte-tolerance", type=float, default=0.10)
    return parser.parse_args()


def build_task_generator(task: Task, device: torch.device, seed: int):
    cfg = _load_public_config(task.config_name)
    return _build_training_generator(cfg, device=device, seed=seed, verbose=False)


def main() -> None:
    args = parse_args()
    if len({task.name for task in args.task}) != len(args.task):
        raise ValueError("Replay task names must be unique")
    output = args.output_dir if args.output_dir.is_absolute() else REPO_ROOT / args.output_dir
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    tasks = sorted(args.task, key=lambda task: task.name)
    names = [task.name for task in tasks]
    quotas = task_quotas(args.capacity, names)
    buffer = ReplayBuffer(capacity=args.capacity, seed=args.seed, storage_dtype=torch.float16)
    planned_basis_counts = {}
    for task in tasks:
        generator = build_task_generator(
            task, device, task_seed(args.seed, task.name)
        )
        planned_basis_counts[task.name] = {}
        admitted_for_task = 0
        for basis_step, sample_count in enumerate(basis_quotas(quotas[task.name]).values()):
            if sample_count == 0:
                continue
            train_x, train_y = generator.generate_batch(
                step=basis_step, batch_size=sample_count
            )
            basis = generator.get_current_basis(basis_step)
            planned_basis_counts[task.name][basis] = sample_count
            admitted_for_task += buffer.observe(
                train_x, train_y, task_id=task.name, basis=basis, step=basis_step
            )
        if admitted_for_task != quotas[task.name]:
            raise RuntimeError(
                f"Expected {quotas[task.name]} samples for {task.name}, "
                f"admitted {admitted_for_task}"
            )
    shard = buffer.save(output, epoch=0, global_step=0)
    actual_bytes = shard.stat().st_size
    lower = args.target_bytes * (1.0 - args.byte_tolerance)
    upper = args.target_bytes * (1.0 + args.byte_tolerance)
    state = buffer.state_dict()
    counts = Counter(state["task_ids"])
    basis_counts = Counter(zip(state["task_ids"], state["bases"]))
    if dict(counts) != quotas:
        raise RuntimeError(f"Replay task balance mismatch: {dict(counts)} != {quotas}")
    expected_basis_counts = {
        (task_name, basis): count
        for task_name, per_basis in planned_basis_counts.items()
        for basis, count in per_basis.items()
    }
    if dict(basis_counts) != expected_basis_counts:
        raise RuntimeError(
            f"Replay task/basis balance mismatch: {dict(basis_counts)} != "
            f"{expected_basis_counts}"
        )
    if not lower <= actual_bytes <= upper:
        raise RuntimeError(
            f"Serialized Replay size {actual_bytes} is outside [{lower:.0f}, {upper:.0f}]"
        )
    manifest = {
        "tasks": [task.name for task in tasks],
        "configs": {task.name: task.config_name for task in tasks},
        "seed": args.seed,
        "capacity": args.capacity,
        "task_counts": quotas,
        "task_basis_counts": planned_basis_counts,
        "storage_dtype": "float16",
        "serialized_file": str(shard),
        "serialized_bytes": actual_bytes,
        "target_bytes": args.target_bytes,
        "byte_tolerance": args.byte_tolerance,
    }
    (output / "balanced_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[QEWC v3] balanced Replay buffer: {output} ({actual_bytes} bytes)")


if __name__ == "__main__":
    main()
