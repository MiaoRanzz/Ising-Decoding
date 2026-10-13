#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run a QEWC v3 Mix-total oracle from the common Stage-0 checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
LOCAL_RUN = REPO_ROOT / "code" / "scripts" / "local_run.sh"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--stage0", type=Path, required=True)
    parser.add_argument("--steps-per-task", type=int, required=True)
    parser.add_argument("--num-continual-tasks", type=int, default=4)
    parser.add_argument("--total-steps", type=int)
    parser.add_argument(
        "--method",
        choices=("mix_total", "mix_exposure"),
        default="mix_total",
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--validation-samples", type=int, default=65536)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument(
        "--python",
        type=Path,
        default=Path("/root/miniconda3/envs/ising-decoding/bin/python"),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    args.stage0 = args.stage0 if args.stage0.is_absolute() else REPO_ROOT / args.stage0
    args.config_path = REPO_ROOT / "conf" / f"{args.config_name}.yaml"
    args.run_root = REPO_ROOT / "outputs" / args.run_name
    if (
        args.steps_per_task <= 0
        or args.num_continual_tasks <= 0
        or (args.total_steps is not None and args.total_steps <= 0)
    ):
        parser.error("step counts must be positive")
    return args


def main() -> None:
    args = parse_args()
    if not args.stage0.is_file():
        raise FileNotFoundError(args.stage0)
    if not args.config_path.is_file():
        raise FileNotFoundError(args.config_path)
    total_steps = (
        args.total_steps
        if args.total_steps is not None
        else args.steps_per_task * args.num_continual_tasks
    )
    if total_steps % args.num_continual_tasks:
        raise ValueError("total steps must be divisible by checkpoint intervals")
    steps_per_interval = total_steps // args.num_continual_tasks
    checkpoints = {"M0": args.stage0}
    checkpoints.update({
        f"M{index}": args.run_root / "models" / f"HTnet.0.{index}.pt"
        for index in range(1, args.num_continual_tasks + 1)
    })
    endpoint = checkpoints[f"M{args.num_continual_tasks}"]
    env = dict(os.environ)
    env.update(
        CUDA_VISIBLE_DEVICES=str(args.gpu),
        GPUS="1",
        PREDECODER_PYTHON=str(args.python),
        PREDECODER_SEED=str(args.seed),
        PREDECODER_TRAIN_EPOCHS=str(args.num_continual_tasks),
        PREDECODER_TRAIN_SAMPLES=str(steps_per_interval * args.batch_size),
        PREDECODER_BATCH_SIZE=str(args.batch_size),
        PREDECODER_ACCUMULATE_STEPS="1",
        PREDECODER_VAL_SAMPLES=str(args.validation_samples),
        PREDECODER_VALIDATION_LER="0",
        PREDECODER_EWC_ENABLED="0",
        PREDECODER_REPLAY_ENABLED="0",
        CONFIG_NAME=args.config_name,
        WORKFLOW="train",
        EXPERIMENT_NAME=args.run_name,
        FRESH_START="1",
        EXTRA_PARAMS=f"++init_model_checkpoint_file={args.stage0}",
    )
    command = ["bash", str(LOCAL_RUN), "9", "9"]
    print(
        f"[QEWC v3] {args.method} steps={total_steps} command={' '.join(command)}",
        flush=True,
    )
    if args.dry_run:
        return
    args.run_root.mkdir(parents=True, exist_ok=True)
    if not (args.resume and all(path.is_file() for path in checkpoints.values())):
        subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)
    missing = [str(path) for path in checkpoints.values() if not path.is_file()]
    if missing:
        raise RuntimeError(f"Missing Mix checkpoints: {missing}")
    manifest = {
        "method": args.method,
        "seed": args.seed,
        "steps_per_task_reference": args.steps_per_task,
        "steps_per_checkpoint_interval": steps_per_interval,
        "num_continual_tasks": args.num_continual_tasks,
        "checkpoints": {
            name: {
                "fraction": index / args.num_continual_tasks,
                "path": str(path),
                "sha256": sha256(path),
            }
            for index, (name, path) in enumerate(checkpoints.items())
        },
        "total_optimizer_steps": total_steps,
        "batch_size": args.batch_size,
        "samples_processed": total_steps * args.batch_size,
        "stage0": {"path": str(args.stage0), "sha256": sha256(args.stage0)},
        "config": {"path": str(args.config_path), "sha256": sha256(args.config_path)},
        "endpoint": {"path": str(endpoint), "sha256": sha256(endpoint)},
        "resource_metrics": str(args.run_root / "resource_metrics.json"),
    }
    (args.run_root / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[QEWC v3] {args.method} complete: {args.run_root}")


if __name__ == "__main__":
    main()
