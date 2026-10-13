#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Launch one frozen QEWC-v3 formal run by its preregistered matrix ID."""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[4]
MANIFEST_PATH = REPO_ROOT / "experiment" / "qewc-v3" / "experiment_manifest.yaml"
RUN_CONTINUAL = REPO_ROOT / "code" / "scripts" / "experiments" / "qewc_v3" / "run_continual.py"
RUN_MIX = REPO_ROOT / "code" / "scripts" / "experiments" / "qewc_v3" / "run_mix_total.py"
EXPECTED_PREFIX = Path("/root/miniconda3/envs/ising-decoding")


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    track: str
    method: str
    order_kind: str
    oracle_task: str | None = None


RUNS = {
    "R1": RunSpec("R1", "realistic", "seq", "natural"),
    "R2": RunSpec("R2", "realistic", "ewc", "natural"),
    "R3": RunSpec("R3", "realistic", "mix", "joint"),
    "S1": RunSpec("S1", "stress", "seq", "high_conflict"),
    "S2": RunSpec("S2", "stress", "ewc", "high_conflict"),
    "S3": RunSpec("S3", "stress", "replay", "high_conflict"),
    "S4": RunSpec("S4", "stress", "mix", "joint"),
    "O1": RunSpec("O1", "stress", "seq", "single", "T1"),
    "O2": RunSpec("O2", "stress", "seq", "single", "T2"),
    "O3": RunSpec("O3", "stress", "seq", "single", "T3"),
    "O4": RunSpec("O4", "stress", "seq", "single", "T4"),
    "S5": RunSpec("S5", "stress", "seq", "reverse"),
    "S6": RunSpec("S6", "stress", "ewc", "reverse"),
    "S7": RunSpec("S7", "stress", "mix_exposure", "joint_exposure"),
    "R4": RunSpec("R4", "realistic", "replay", "natural"),
    "O5": RunSpec("O5", "realistic", "seq", "single", "T1"),
    "O6": RunSpec("O6", "realistic", "seq", "single", "T2"),
    "O7": RunSpec("O7", "realistic", "seq", "single", "T3"),
    "O8": RunSpec("O8", "realistic", "seq", "single", "T4"),
    "R5": RunSpec("R5", "realistic", "mix_exposure", "joint_exposure"),
}


def hydra_config_name(config_path: str) -> str:
    prefix, suffix = "conf/", ".yaml"
    if not config_path.startswith(prefix) or not config_path.endswith(suffix):
        raise ValueError(f"task config must be conf/**/*.yaml: {config_path}")
    return config_path[len(prefix):-len(suffix)]


def assert_frozen(manifest) -> None:
    required = (
        "realistic_tasks",
        "stress_tasks",
        "stress_orders",
        "continual_seed",
        "optimizer_steps_per_task",
        "ewc_lambda",
        "replay_ratio",
        "evaluation_streams",
        "success_criteria",
    )
    missing = [name for name in required if not bool(manifest.freeze.get(name, False))]
    if missing:
        raise RuntimeError(f"formal runs are blocked until manifest freeze: {missing}")
    values = {
        "optimizer steps": manifest.pilot.selected_optimizer_steps_per_task,
        "EWC lambda": manifest.ewc.get("lambda"),
        "Replay ratio": manifest.replay.get("ratio"),
    }
    unset = [name for name, value in values.items() if value is None]
    if unset:
        raise RuntimeError(f"formal hyperparameters are unset: {unset}")


def load_track(spec: RunSpec):
    path = REPO_ROOT / "experiment" / "qewc-v3" / f"tasks_{spec.track}.yaml"
    task_doc = OmegaConf.load(path)
    configs = {
        name: hydra_config_name(str(item.config))
        for name, item in task_doc.tasks.items()
    }
    if spec.oracle_task:
        order = [spec.oracle_task]
    elif spec.order_kind == "natural":
        order = [str(item) for item in task_doc.order]
    elif spec.order_kind == "high_conflict":
        order = [str(item) for item in task_doc.high_conflict_order]
    elif spec.order_kind == "reverse":
        order = [str(item) for item in task_doc.reverse_order]
    else:
        order = []
    return configs, order


def build_command(spec: RunSpec, manifest, *, gpu: str, resume: bool) -> list[str]:
    configs, order = load_track(spec)
    python = str(EXPECTED_PREFIX / "bin" / "python")
    stage0 = str(REPO_ROOT / str(manifest.stage0.checkpoint))
    steps = int(manifest.pilot.selected_optimizer_steps_per_task)
    batch_size = int(manifest.pilot.batch_size_per_gpu)
    validation_samples = int(
        manifest.pilot.get("formal_validation_samples", manifest.pilot.validation_samples)
    )
    seed = int(manifest.seeds.continual_train)
    run_name = f"qewc_v3/formal/{spec.run_id}"
    if spec.method in {"mix", "mix_exposure"}:
        mix_config = (
            "experiments/qewc_v3/realistic/mix_t0_t4"
            if spec.track == "realistic"
            else "experiments/qewc_v3/mix/stress_selected"
        )
        command = [
            python, "-u", str(RUN_MIX),
            "--run-name", run_name,
            "--config-name", mix_config,
            "--stage0", stage0,
            "--steps-per-task", str(steps),
            "--num-continual-tasks", "4",
            "--batch-size", str(batch_size),
            "--validation-samples", str(validation_samples),
            "--seed", str(seed),
            "--gpu", gpu,
        ]
        if spec.method == "mix_exposure":
            command.extend(
                (
                    "--method", "mix_exposure",
                    "--total-steps", str(5 * steps),
                )
            )
    else:
        command = [
            python, "-u", str(RUN_CONTINUAL),
            "--method", spec.method,
            "--run-name", run_name,
            "--base-task", f"T0:{configs['T0']}",
        ]
        for task in order:
            command.extend(("--task", f"{task}:{configs[task]}"))
        command.extend(
            (
                "--order", ",".join(order),
                "--expected-stages", str(len(order)),
                "--stage0", stage0,
                "--steps-per-task", str(steps),
                "--batch-size", str(batch_size),
                "--validation-samples", str(validation_samples),
                "--seed", str(seed),
                "--gpu", gpu,
            )
        )
        if spec.method == "ewc":
            command.extend(
                (
                    "--positive-weight", str(manifest.ewc.positive_weight),
                    "--fisher-samples", str(manifest.ewc.fisher_samples),
                    "--fisher-batch-size", str(manifest.ewc.fisher_batch_size),
                    "--fisher-per-sample-chunk-size", str(manifest.ewc.per_sample_chunk_size),
                    "--fisher-seed", str(manifest.ewc.fisher_seed),
                    "--ewc-lambda", str(manifest.ewc.get("lambda")),
                    "--ewc-gamma", str(manifest.ewc.gamma),
                )
            )
        elif spec.method == "replay":
            command.extend(
                (
                    "--replay-dir", str(REPO_ROOT / "outputs" / run_name / "replay"),
                    "--replay-capacity", str(manifest.replay.capacity_samples),
                    "--replay-ratio", str(manifest.replay.get("ratio")),
                    "--replay-seed", str(manifest.replay.get("seed", 20260913)),
                )
            )
    if resume:
        command.append("--resume")
    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", choices=tuple(RUNS), required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if Path(sys.prefix).resolve() != EXPECTED_PREFIX.resolve():
        raise RuntimeError(f"wrong Python environment: executable={sys.executable}, prefix={sys.prefix}")
    manifest = OmegaConf.load(args.manifest)
    assert_frozen(manifest)
    command = build_command(RUNS[args.run_id], manifest, gpu=args.gpu, resume=args.resume)
    print("[QEWC v3 formal] " + " ".join(command), flush=True)
    if not args.dry_run:
        subprocess.run(command, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
