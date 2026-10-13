#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluate every endpoint of one frozen QEWC-v3 formal run."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[4]
PYTHON = Path("/root/miniconda3/envs/ising-decoding/bin/python")
EVALUATOR = REPO_ROOT / "code" / "scripts" / "experiments" / "qewc_v3" / "evaluate_realistic_matrix.py"
EXPERIMENT_MANIFEST = REPO_ROOT / "experiment" / "qewc-v3" / "experiment_manifest.yaml"


def task_items(track: str) -> list[tuple[str, Path]]:
    document = OmegaConf.load(REPO_ROOT / "experiment" / "qewc-v3" / f"tasks_{track}.yaml")
    return [(name, REPO_ROOT / str(item.config)) for name, item in document.tasks.items()]


def checkpoint_items(run_id: str, training_manifest: dict) -> list[tuple[str, Path]]:
    if training_manifest["method"] in {"mix_total", "mix_exposure"}:
        raw = training_manifest["checkpoints"]
    else:
        raw = training_manifest["endpoints"]
    items = [(name, Path(item["path"])) for name, item in raw.items()]
    if run_id.startswith("O") and len(items) == 2:
        items[-1] = (run_id, items[-1][1])
    return items


def build_command(
    run_id: str,
    *,
    gpus: str,
    parallelism: int,
    resume: bool,
) -> list[str]:
    experiment = OmegaConf.load(EXPERIMENT_MANIFEST)
    matrix = OmegaConf.load(REPO_ROOT / "experiment" / "qewc-v3" / "run_matrix.yaml")
    match = [row for row in matrix.runs if str(row.id) == run_id]
    if len(match) != 1:
        raise ValueError(f"unknown run id: {run_id}")
    track = str(match[0].track)
    training_path = REPO_ROOT / "outputs" / "qewc_v3" / "formal" / run_id / "run_manifest.json"
    if not training_path.is_file():
        raise FileNotFoundError(training_path)
    training = json.loads(training_path.read_text(encoding="utf-8"))
    sample_root = REPO_ROOT / "outputs" / "qewc_v3" / "evaluation_streams" / "formal" / track
    output_dir = REPO_ROOT / "outputs" / "qewc_v3" / "evaluations" / "formal" / run_id
    command = [
        str(PYTHON), "-u", str(EVALUATOR),
        "--track", track,
        "--sample-root", str(sample_root),
        "--num-samples", str(experiment.evaluation.formal_samples_per_basis),
        "--latency-num-samples", "1000",
        "--batch-size", "2048",
        "--seed", str(experiment.seeds.formal_test),
        "--gpus", gpus,
        "--parallelism", str(parallelism),
        "--output-dir", str(output_dir),
    ]
    for name, path in checkpoint_items(run_id, training):
        command.extend(("--checkpoint", f"{name}:{path}"))
    for name, path in task_items(track):
        command.extend(("--task", f"{name}:{path}"))
    if resume:
        command.append("--resume")
    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--parallelism", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    command = build_command(
        args.run_id,
        gpus=args.gpus,
        parallelism=args.parallelism,
        resume=args.resume,
    )
    print("[QEWC v3 formal eval] " + " ".join(command), flush=True)
    if not args.dry_run:
        subprocess.run(command, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
