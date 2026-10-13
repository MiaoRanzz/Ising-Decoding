#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluate final endpoints from different formal methods on one fixed stream set."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
import sys

from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[4]
CODE_ROOT = REPO_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from scripts.experiments.qewc_v3.evaluate_formal_run import task_items


PYTHON = Path("/root/miniconda3/envs/ising-decoding/bin/python")
EVALUATOR = REPO_ROOT / "code" / "scripts" / "experiments" / "qewc_v3" / "evaluate_realistic_matrix.py"
MANIFEST = REPO_ROOT / "experiment" / "qewc-v3" / "experiment_manifest.yaml"
RUN_IDS = {
    "realistic": ("R1", "R2", "R3"),
    "stress": ("S1", "S2", "S3", "S4"),
    "stress_reverse": ("S5", "S6"),
    "stress_stretch": ("S4", "S7"),
    "realistic_stretch": ("R1", "R2", "R3", "R4", "R5"),
}
TASK_TRACKS = {
    "stress": "stress",
    "stress_reverse": "stress",
    "stress_stretch": "stress",
    "realistic": "realistic",
    "realistic_stretch": "realistic",
}


def final_checkpoint(training: dict) -> Path:
    if training["method"] in {"mix_total", "mix_exposure"}:
        return Path(training["checkpoints"]["M4"]["path"])
    endpoints = training["endpoints"]
    final_name = max(endpoints, key=lambda name: int(name[1:]))
    return Path(endpoints[final_name]["path"])


def build_command(track: str, *, gpus: str, parallelism: int, resume: bool) -> list[str]:
    task_track = TASK_TRACKS[track]
    experiment = OmegaConf.load(MANIFEST)
    command = [
        str(PYTHON), "-u", str(EVALUATOR),
        "--track", f"formal_{track}_cross_method",
        "--sample-root", str(REPO_ROOT / "outputs" / "qewc_v3" / "evaluation_streams" / "formal" / task_track),
        "--num-samples", str(experiment.evaluation.formal_samples_per_basis),
        "--latency-num-samples", "1000",
        "--batch-size", "2048",
        "--seed", str(experiment.seeds.formal_test),
        "--gpus", gpus,
        "--parallelism", str(parallelism),
        "--output-dir", str(REPO_ROOT / "outputs" / "qewc_v3" / "evaluations" / "formal_cross_method" / track),
        "--checkpoint", f"C0:{REPO_ROOT / str(experiment.stage0.checkpoint)}",
    ]
    for run_id in RUN_IDS[track]:
        path = REPO_ROOT / "outputs" / "qewc_v3" / "formal" / run_id / "run_manifest.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        training = json.loads(path.read_text(encoding="utf-8"))
        command.extend(("--checkpoint", f"{run_id}:{final_checkpoint(training)}"))
    for task, path in task_items(task_track):
        command.extend(("--task", f"{task}:{path}"))
    if resume:
        command.append("--resume")
    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", choices=tuple(RUN_IDS), required=True)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--parallelism", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    command = build_command(args.track, gpus=args.gpus, parallelism=args.parallelism, resume=args.resume)
    print("[QEWC v3 formal comparison] " + " ".join(command), flush=True)
    if not args.dry_run:
        subprocess.run(command, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
