#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluate all saved endpoints of one QEWC-v4 formal run on frozen ID streams."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
CODE_ROOT = REPO_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from scripts.experiments.qewc_v4.evaluation_common import (  # noqa: E402
    ARCHITECTURES,
    FINAL_STREAM_ROOT,
    TASKS,
    checkpoint_items,
    config_path,
    load_training,
    run_ids,
)


EVALUATOR = REPO_ROOT / "code" / "scripts" / "experiments" / "qewc_v3" / "evaluate_realistic_matrix.py"
PYTHON = Path("/root/miniconda3/envs/ising-decoding/bin/python")
ALL_RUN_IDS = (*run_ids("htnet_a101"), *run_ids("ising_fast"))


def architecture_for_run(run_id: str) -> str:
    if run_id.startswith("H_"):
        return "htnet_a101"
    if run_id.startswith("I_"):
        return "ising_fast"
    raise ValueError(f"unknown V4 run id: {run_id}")


def build_command(
    run_id: str,
    *,
    gpu: str,
    python: Path = PYTHON,
    resume: bool,
) -> list[str]:
    architecture = architecture_for_run(run_id)
    spec = ARCHITECTURES[architecture]
    training = load_training(run_id)
    command = [
        str(python),
        "-u",
        str(EVALUATOR),
        "--track",
        f"formal_id_{run_id}",
        "--sample-root",
        str(FINAL_STREAM_ROOT),
        "--num-samples",
        "65536",
        "--latency-num-samples",
        "1000",
        "--batch-size",
        "2048",
        "--model-id",
        str(spec.model_id),
        "--seed",
        "20260916",
        "--gpus",
        gpu,
        "--parallelism",
        "1",
        "--python",
        str(python),
        "--output-dir",
        str(REPO_ROOT / "outputs" / "qewc_v4" / "evaluations" / "id" / run_id),
    ]
    for name, path in checkpoint_items(training):
        command.extend(("--checkpoint", f"{name}:{path}"))
    for task in TASKS:
        command.extend(("--task", f"{task}:{config_path(architecture, task)}"))
    if resume:
        command.append("--resume")
    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", choices=ALL_RUN_IDS, required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--python", type=Path, default=PYTHON)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    command = build_command(args.run_id, gpu=args.gpu, python=args.python, resume=args.resume)
    print("[QEWC v4 formal ID] " + " ".join(command), flush=True)
    if not args.dry_run:
        subprocess.run(command, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
