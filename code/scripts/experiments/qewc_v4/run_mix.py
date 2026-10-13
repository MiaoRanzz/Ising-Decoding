#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run a QEWC v4 Mix oracle from the common Stage-0 checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

from scripts.experiments.qewc_v4 import resume_stage


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
    parser.add_argument("--legacy-epochs-per-task", type=int, required=True)
    parser.add_argument("--num-continual-tasks", type=int, default=4)
    parser.add_argument("--total-legacy-epochs", type=int)
    parser.add_argument("--checkpoint-count", type=int, default=4)
    parser.add_argument(
        "--method",
        choices=("mix_total", "mix_exposure"),
        default="mix_total",
    )
    parser.add_argument("--requested-samples-per-epoch", type=int, default=4194304)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--accumulate-steps", type=int, default=2)
    parser.add_argument("--checkpoint-prefix", default="HTnet")
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
        args.legacy_epochs_per_task <= 0
        or args.num_continual_tasks <= 0
        or args.checkpoint_count <= 0
        or args.requested_samples_per_epoch <= 0
        or args.batch_size <= 0
        or args.accumulate_steps <= 0
        or (args.total_legacy_epochs is not None and args.total_legacy_epochs <= 0)
    ):
        parser.error("epoch, checkpoint, sample, and batch counts must be positive")
    return args


def main() -> None:
    args = parse_args()
    if not args.stage0.is_file():
        raise FileNotFoundError(args.stage0)
    if not args.config_path.is_file():
        raise FileNotFoundError(args.config_path)
    total_epochs = (
        args.total_legacy_epochs
        if args.total_legacy_epochs is not None
        else args.legacy_epochs_per_task * args.num_continual_tasks
    )
    if total_epochs % args.checkpoint_count:
        raise ValueError("total legacy epochs must be divisible by checkpoint count")
    epochs_per_interval = total_epochs // args.checkpoint_count
    update_denominator = (
        args.batch_size * args.accumulate_steps * args.accumulate_steps
    )
    if args.requested_samples_per_epoch % update_denominator:
        raise ValueError(
            "requested samples per epoch must be divisible by "
            "batch_size * accumulate_steps squared"
        )
    optimizer_updates_per_epoch = (
        args.requested_samples_per_epoch // update_denominator
    )
    total_optimizer_steps = total_epochs * optimizer_updates_per_epoch
    actual_samples_per_epoch = (
        args.requested_samples_per_epoch // args.accumulate_steps
    )
    checkpoints = {"M0": args.stage0}
    checkpoints.update({
        f"M{index}": args.run_root / "models" / f"{args.checkpoint_prefix}.0.{index * epochs_per_interval}.pt"
        for index in range(1, args.checkpoint_count + 1)
    })
    endpoint = checkpoints[f"M{args.checkpoint_count}"]
    env = dict(os.environ)
    env.update(
        CUDA_VISIBLE_DEVICES=str(args.gpu),
        GPUS="1",
        PREDECODER_PYTHON=str(args.python),
        PREDECODER_SEED=str(args.seed),
        PREDECODER_TRAIN_EPOCHS=str(total_epochs),
        PREDECODER_TRAIN_SAMPLES=str(args.requested_samples_per_epoch),
        PREDECODER_BATCH_SIZE=str(args.batch_size),
        PREDECODER_ACCUMULATE_STEPS=str(args.accumulate_steps),
        PREDECODER_VAL_SAMPLES=str(args.validation_samples),
        PREDECODER_VALIDATION_LER="0",
        PREDECODER_EWC_ENABLED="0",
        PREDECODER_REPLAY_ENABLED="0",
        CONFIG_NAME=args.config_name,
        WORKFLOW="train",
        EXPERIMENT_NAME=args.run_name,
        FRESH_START="1",
        EXTRA_PARAMS=f"++init_model_checkpoint_file={args.stage0} ++early_stopping.enabled=False",
    )
    command = ["bash", str(LOCAL_RUN), "9", "9"]
    print(
        f"[QEWC v4] {args.method} legacy_epochs={total_epochs} command={' '.join(command)}",
        flush=True,
    )
    if args.dry_run:
        return
    args.run_root.mkdir(parents=True, exist_ok=True)
    recovery = None
    if args.resume and not endpoint.is_file():
        args.epochs_per_task = total_epochs
        args.steps_per_task = total_optimizer_steps
        task = SimpleNamespace(name=args.method)
        # Reject missing historical evaluation endpoints before any further training.
        existing = list((args.run_root / 'models').glob(f'{args.checkpoint_prefix}.0.*.pt'))
        if existing:
            latest = max(int(p.name.split('.')[-2]) for p in existing)
            for index in range(1, args.checkpoint_count + 1):
                epoch = index * epochs_per_interval
                if epoch <= latest:
                    resume_stage.validate_pair(checkpoints[f'M{index}'], epoch, optimizer_updates_per_epoch)
        recovery = resume_stage.prepare(args, task, None, env, endpoint, REPO_ROOT)
    if not (args.resume and all(path.is_file() for path in checkpoints.values())):
        subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)
    if recovery is not None:
        resume_stage.finish(recovery, args, task, None, endpoint)
    missing = [str(path) for path in checkpoints.values() if not path.is_file()]
    if missing:
        raise RuntimeError(f"Missing Mix checkpoints: {missing}")
    manifest = {
        "method": args.method,
        "seed": args.seed,
        "legacy_epochs_per_task_reference": args.legacy_epochs_per_task,
        "total_legacy_epochs": total_epochs,
        "legacy_epochs_per_checkpoint_interval": epochs_per_interval,
        "requested_samples_per_epoch": args.requested_samples_per_epoch,
        "actual_generated_samples_per_epoch": actual_samples_per_epoch,
        "num_continual_tasks": args.num_continual_tasks,
        "checkpoints": {
            name: {
                "fraction": index / args.checkpoint_count,
                "path": str(path),
                "sha256": sha256(path),
            }
            for index, (name, path) in enumerate(checkpoints.items())
        },
        "optimizer_updates_per_epoch": optimizer_updates_per_epoch,
        "total_optimizer_steps": total_optimizer_steps,
        "micro_batch_size": args.batch_size,
        "accumulate_steps": args.accumulate_steps,
        "effective_batch_size": args.batch_size * args.accumulate_steps,
        "samples_processed": total_epochs * actual_samples_per_epoch,
        "stage0": {"path": str(args.stage0), "sha256": sha256(args.stage0)},
        "config": {"path": str(args.config_path), "sha256": sha256(args.config_path)},
        "endpoint": {"path": str(endpoint), "sha256": sha256(endpoint)},
        "resource_metrics": str(args.run_root / "resource_metrics.json"),
    }
    (args.run_root / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[QEWC v4] {args.method} complete: {args.run_root}")


if __name__ == "__main__":
    main()
