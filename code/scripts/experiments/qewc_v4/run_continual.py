#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run a QEWC v4 SEQ, Online-EWC, or Replay continual sequence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
LOCAL_RUN = REPO_ROOT / "code" / "scripts" / "local_run.sh"
FISHER_SCRIPT = REPO_ROOT / "code" / "scripts" / "compute_ewc_fisher.py"
CONSOLIDATE_SCRIPT = REPO_ROOT / "code" / "scripts" / "consolidate_online_ewc.py"
ANALYZE_EWC_SCRIPT = (
    REPO_ROOT / "code" / "scripts" / "experiments" / "qewc_v3" / "analyze_ewc_stage.py"
)
BUILD_REPLAY_SCRIPT = (
    REPO_ROOT / "code" / "scripts" / "experiments" / "qewc_v3" / "build_balanced_replay.py"
)


@dataclass(frozen=True)
class Task:
    name: str
    config_name: str

    @property
    def config_path(self) -> Path:
        return REPO_ROOT / "conf" / f"{self.config_name}.yaml"


def parse_task(value: str) -> Task:
    name, separator, config = value.partition(":")
    if not separator or not name.strip() or not config.strip():
        raise argparse.ArgumentTypeError("--task must be TASK:CONFIG_NAME")
    return Task(name.strip(), config.strip())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stage_environment(
    args: argparse.Namespace,
    task: Task,
    stage: int,
    initial_checkpoint: Path,
) -> tuple[dict[str, str], Path]:
    experiment_name = f"{args.run_name}/stage_{stage}_{task.name}"
    output = REPO_ROOT / "outputs" / experiment_name
    env = dict(os.environ)
    env.update(
        CUDA_VISIBLE_DEVICES=str(args.gpu),
        GPUS="1",
        PREDECODER_PYTHON=str(args.python),
        PREDECODER_SEED=str(args.seed),
        PREDECODER_TRAIN_EPOCHS=str(args.epochs_per_task),
        PREDECODER_TRAIN_SAMPLES=str(args.requested_samples_per_epoch),
        PREDECODER_BATCH_SIZE=str(args.batch_size),
        PREDECODER_ACCUMULATE_STEPS=str(args.accumulate_steps),
        PREDECODER_VAL_SAMPLES=str(args.validation_samples),
        PREDECODER_VALIDATION_LER="0",
        CONFIG_NAME=task.config_name,
        WORKFLOW="train",
        EXPERIMENT_NAME=experiment_name,
        FRESH_START="1",
        EXTRA_PARAMS=f"++init_model_checkpoint_file={initial_checkpoint} ++early_stopping.enabled=False",
        PREDECODER_EWC_ENABLED="1" if args.method == "ewc" else "0",
        PREDECODER_REPLAY_ENABLED="1" if args.method == "replay" else "0",
    )
    if args.method == "ewc":
        env.update(
            PREDECODER_EWC_DIR=str(args.run_root / "fisher" / f"C{stage - 1}"),
            PREDECODER_EWC_LAMBDA=str(args.ewc_lambda),
            PREDECODER_POSITIVE_WEIGHT=str(args.positive_weight),
        )
    elif args.method == "replay":
        env.update(
            PREDECODER_REPLAY_TASK_ID=task.name,
            PREDECODER_REPLAY_DIR=str(args.replay_dir),
            PREDECODER_REPLAY_CAPACITY=str(args.replay_capacity),
            PREDECODER_REPLAY_RATIO=str(args.replay_ratio),
            PREDECODER_REPLAY_SEED=str(args.replay_seed),
            PREDECODER_REPLAY_LOAD_EXISTING="1",
            PREDECODER_REPLAY_ADMIT_CURRENT="0",
        )
    return env, output / "models" / f"{args.checkpoint_prefix}.0.{args.epochs_per_task}.pt"


def run_command(
    command: list[str], *, env: dict[str, str] | None, dry_run: bool
) -> float:
    print("[QEWC v4] " + " ".join(command), flush=True)
    if dry_run:
        return 0.0
    minimum_free = float(os.environ.get("QEWC_V4_MIN_FREE_GIB", "0"))
    if minimum_free and shutil.disk_usage(REPO_ROOT / "outputs").free < minimum_free * 1024**3:
        raise RuntimeError(f"QEWC v4 output storage below {minimum_free} GiB; refusing next command")
    started = time.monotonic()
    subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)
    return time.monotonic() - started


def estimate_fisher(
    args: argparse.Namespace,
    task: Task,
    checkpoint: Path,
    output: Path,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(args.python), "-u", str(FISHER_SCRIPT),
        "--config-name", task.config_name,
        "--distance", "9", "--n-rounds", "9",
        "--checkpoint", str(checkpoint), "--output", str(output),
        "--task-name", task.name, "--num-samples", str(args.fisher_samples),
        "--batch-size", str(args.fisher_batch_size),
        "--per-sample-chunk-size", str(args.fisher_per_sample_chunk_size),
        "--positive-weight", str(args.positive_weight),
        "--seed", str(args.fisher_seed), "--device", "cuda:0",
    ]
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["PYTHONPATH"] = str(REPO_ROOT / "code")
    elapsed = run_command(command, env=env, dry_run=args.dry_run)
    args.fisher_timings.append({
        "task": task.name,
        "checkpoint": str(checkpoint),
        "output": str(output),
        "wall_seconds": elapsed,
    })


def ensure_initial_fisher(args: argparse.Namespace) -> Path:
    destination = args.run_root / "fisher" / "C0" / "online.pt"
    if destination.is_file() and args.resume:
        return destination
    if args.initial_fisher is not None:
        if not args.initial_fisher.is_file():
            raise FileNotFoundError(args.initial_fisher)
        if not args.dry_run:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(args.initial_fisher, destination)
        args.initial_fisher_reused = {
            "path": str(args.initial_fisher),
            "sha256": sha256(args.initial_fisher),
        }
        return destination
    estimate_fisher(args, args.base_task, args.stage0, destination)
    return destination


def update_fisher(
    args: argparse.Namespace,
    task: Task,
    stage: int,
    endpoint: Path,
    previous: Path,
) -> Path:
    raw = args.run_root / "fisher" / "raw" / f"C{stage}_{task.name}.pt"
    destination = args.run_root / "fisher" / f"C{stage}" / "online.pt"
    if destination.is_file() and args.resume:
        return destination
    estimate_fisher(args, task, endpoint, raw)
    diagnostic = args.run_root / "diagnostics" / f"C{stage}_{task.name}.json"
    run_command(
        [
            str(args.python), "-u", str(ANALYZE_EWC_SCRIPT),
            "--checkpoint", str(endpoint),
            "--reference", str(previous),
            "--current-fisher", str(raw),
            "--output", str(diagnostic),
        ],
        env=None,
        dry_run=args.dry_run,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            str(args.python), "-u", str(CONSOLIDATE_SCRIPT),
            "--previous", str(previous), "--current", str(raw),
            "--output", str(destination), "--gamma", str(args.ewc_gamma),
        ],
        env=None,
        dry_run=args.dry_run,
    )
    return destination


def rebuild_replay(args: argparse.Namespace, tasks: list[Task]) -> None:
    command = [
        str(args.python), "-u", str(BUILD_REPLAY_SCRIPT),
        "--capacity", str(args.replay_capacity),
        "--output-dir", str(args.replay_dir),
        "--seed", str(args.replay_seed),
        "--device", "cuda:0",
        "--target-bytes", str(args.replay_target_bytes),
        "--byte-tolerance", str(args.replay_byte_tolerance),
    ]
    for task in tasks:
        command.extend(("--task", f"{task.name}:{task.config_name}"))
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["PYTHONPATH"] = str(REPO_ROOT / "code")
    elapsed = run_command(command, env=env, dry_run=args.dry_run)
    args.replay_rebuild_timings.append({
        "tasks": [task.name for task in tasks],
        "output": str(args.replay_dir),
        "wall_seconds": elapsed,
    })


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        choices=("t0_only", "seq", "ewc", "replay"),
        required=True,
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--base-task", type=parse_task, required=True)
    parser.add_argument("--task", action="append", type=parse_task, required=True)
    parser.add_argument("--order", required=True)
    parser.add_argument("--expected-stages", type=int, default=4)
    parser.add_argument("--stage0", type=Path, required=True)
    parser.add_argument("--epochs-per-task", type=int, default=1)
    parser.add_argument("--requested-samples-per-epoch", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--accumulate-steps", type=int, default=1)
    parser.add_argument("--checkpoint-prefix", default="HTnet")
    parser.add_argument("--validation-samples", type=int, default=65536)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--python", type=Path, default=Path("/root/miniconda3/envs/ising-decoding/bin/python"))
    parser.add_argument("--positive-weight", type=float, default=1.0)
    parser.add_argument("--fisher-samples", type=int, default=65536)
    parser.add_argument("--fisher-batch-size", type=int, default=2048)
    parser.add_argument("--fisher-per-sample-chunk-size", type=int, default=16)
    parser.add_argument("--fisher-seed", type=int, default=20260912)
    parser.add_argument("--initial-fisher", type=Path, default=None)
    parser.add_argument("--ewc-lambda", type=float, default=None)
    parser.add_argument("--ewc-gamma", type=float, default=1.0)
    parser.add_argument("--replay-dir", type=Path, default=None)
    parser.add_argument("--replay-capacity", type=int, default=None)
    parser.add_argument("--replay-ratio", type=float, default=None)
    parser.add_argument("--replay-target-bytes", type=int, default=None)
    parser.add_argument("--replay-byte-tolerance", type=float, default=0.05)
    parser.add_argument("--replay-seed", type=int, default=20260913)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    args.stage0 = args.stage0 if args.stage0.is_absolute() else REPO_ROOT / args.stage0
    if args.initial_fisher is not None and not args.initial_fisher.is_absolute():
        args.initial_fisher = REPO_ROOT / args.initial_fisher
    if args.epochs_per_task <= 0 or args.batch_size <= 0 or args.accumulate_steps <= 0:
        parser.error("epochs, batch size, and accumulate steps must be positive")
    if args.requested_samples_per_epoch is None:
        parser.error("--requested-samples-per-epoch is required")
    if args.requested_samples_per_epoch % (args.batch_size * args.accumulate_steps):
        parser.error("requested samples per epoch must be divisible by batch_size * accumulate_steps")
    args.steps_per_task = int(
        args.epochs_per_task * args.requested_samples_per_epoch
        / (args.batch_size * args.accumulate_steps * args.accumulate_steps)
    )
    args.run_root = REPO_ROOT / "outputs" / args.run_name
    tasks = {task.name: task for task in args.task}
    order = [item.strip() for item in args.order.split(",") if item.strip()]
    if args.expected_stages <= 0 or len(order) != args.expected_stages or set(order) != set(tasks):
        parser.error("--order must list every --task name exactly once and match --expected-stages")
    args.tasks = tasks
    args.order_names = order
    if args.method == "ewc" and args.ewc_lambda is None:
        parser.error("--ewc-lambda is required for EWC")
    if args.method == "replay":
        if (
            args.replay_dir is None
            or args.replay_capacity is None
            or args.replay_ratio is None
            or args.replay_target_bytes is None
        ):
            parser.error(
                "Replay requires --replay-dir, --replay-capacity, "
                "--replay-ratio, and --replay-target-bytes"
            )
        if args.replay_target_bytes <= 0 or not 0 <= args.replay_byte_tolerance < 1:
            parser.error("Replay target bytes must be positive and tolerance must be in [0, 1)")
        args.replay_dir = args.replay_dir if args.replay_dir.is_absolute() else REPO_ROOT / args.replay_dir
    return args


def main() -> None:
    args = parse_args()
    if not args.stage0.is_file():
        raise FileNotFoundError(args.stage0)
    for task in [args.base_task, *args.tasks.values()]:
        if not task.config_path.is_file():
            raise FileNotFoundError(task.config_path)
    args.run_root.mkdir(parents=True, exist_ok=True)
    args.fisher_timings = []
    args.initial_fisher_reused = None

    args.replay_rebuild_timings = []
    previous_checkpoint = args.stage0
    previous_fisher = ensure_initial_fisher(args) if args.method == "ewc" else None
    learned_tasks = [args.base_task]
    from scripts.experiments.qewc_v4 import resume_stage
    replay_partial_stage = resume_stage.replay_resume_stage(args)
    if args.method == "replay" and not replay_partial_stage:
        rebuild_replay(args, learned_tasks)
    endpoints = {"C0": str(args.stage0)}
    for stage, task_name in enumerate(args.order_names, 1):
        task = args.tasks[task_name]
        env, endpoint = stage_environment(args, task, stage, previous_checkpoint)
        recovery = None
        if args.resume and not args.dry_run:
            from scripts.experiments.qewc_v4 import resume_stage
            if endpoint.is_file():
                resume_stage.validate_pair(endpoint, args.epochs_per_task,
                                           args.steps_per_task // args.epochs_per_task)
            else:
                recovery = resume_stage.prepare(args, task, stage, env, endpoint, REPO_ROOT)
        if not (args.resume and endpoint.is_file()):
            run_command(["bash", str(LOCAL_RUN), "9", "9"], env=env, dry_run=args.dry_run)
        if recovery is not None:
            resume_stage.finish(recovery, args, task, stage, endpoint)
        if not args.dry_run and not endpoint.is_file():
            raise RuntimeError(f"Missing stage endpoint: {endpoint}")
        endpoints[f"C{stage}"] = str(endpoint)
        previous_checkpoint = endpoint
        if args.method == "ewc":
            previous_fisher = update_fisher(
                args, task, stage, endpoint, previous_fisher
            )
        elif args.method == "replay":
            learned_tasks.append(task)
            if stage >= replay_partial_stage:
                rebuild_replay(args, learned_tasks)

    if args.dry_run:
        return
    method_parameters = {}
    if args.method == "ewc":
        final_fisher = args.run_root / "fisher" / f"C{len(args.order_names)}" / "online.pt"
        method_parameters = {
            "lambda": args.ewc_lambda,
            "gamma": args.ewc_gamma,
            "positive_weight": args.positive_weight,
            "fisher_estimator": "mean_squared_per_sample_bce_gradient",
            "fisher_samples": args.fisher_samples,
            "fisher_batch_size": args.fisher_batch_size,
            "fisher_per_sample_chunk_size": args.fisher_per_sample_chunk_size,
            "fisher_seed": args.fisher_seed,
            "fisher_timings": args.fisher_timings,
            "initial_fisher_reused": args.initial_fisher_reused,
            "online_state_bytes": final_fisher.stat().st_size,
        }
    elif args.method == "replay":
        replay_manifest_path = args.replay_dir / "balanced_manifest.json"
        method_parameters = {
            "ratio": args.replay_ratio,
            "capacity": args.replay_capacity,
            "target_bytes": args.replay_target_bytes,
            "byte_tolerance": args.replay_byte_tolerance,
            "seed": args.replay_seed,
            "admit_current_during_training": False,
            "selection": "fixed_deterministic_task_and_basis_balanced",
            "rebuild_timings": args.replay_rebuild_timings,
            "final_buffer_manifest": json.loads(
                replay_manifest_path.read_text(encoding="utf-8")
            ),
        }
    resource_metrics = {
        stage: str(Path(path).parent.parent / "resource_metrics.json")
        for stage, path in endpoints.items()
        if stage != "C0"
    }
    manifest = {
        "method": args.method,
        "seed": args.seed,
        "order": args.order_names,
        "legacy_epochs_per_task": args.epochs_per_task,
        "requested_samples_per_epoch": args.requested_samples_per_epoch,
        "actual_generated_samples_per_epoch": args.requested_samples_per_epoch // args.accumulate_steps,
        "steps_per_task": args.steps_per_task,
        "micro_batch_size": args.batch_size,
        "accumulate_steps": args.accumulate_steps,
        "effective_batch_size": args.batch_size * args.accumulate_steps,
        "total_optimizer_steps": args.steps_per_task * len(args.order_names),
        "samples_processed": args.steps_per_task * len(args.order_names) * args.batch_size * args.accumulate_steps,
        "validation_samples_per_stage": args.validation_samples,
        "python": str(args.python),
        "method_parameters": method_parameters,
        "resource_metrics": resource_metrics,
        "endpoints": {
            stage: {"path": path, "sha256": sha256(Path(path))}
            for stage, path in endpoints.items()
        },
        "configs": {
            task.name: {"path": str(task.config_path), "sha256": sha256(task.config_path)}
            for task in [args.base_task, *args.tasks.values()]
        },
    }
    (args.run_root / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[QEWC v4] continual run complete: {args.run_root}")


if __name__ == "__main__":
    main()
