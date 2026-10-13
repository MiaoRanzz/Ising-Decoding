#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pair all final V4 methods from one architecture on the same frozen ID shots."""

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
    config_path,
    final_checkpoint,
    load_training,
    run_ids,
)


EVALUATOR = REPO_ROOT / "code" / "scripts" / "experiments" / "qewc_v3" / "evaluate_realistic_matrix.py"
PYTHON = Path("/root/miniconda3/envs/ising-decoding/bin/python")


def build_command(architecture: str, *, gpu: str, resume: bool) -> list[str]:
    spec = ARCHITECTURES[architecture]
    command = [
        str(PYTHON), "-u", str(EVALUATOR),
        "--track", f"formal_id_final_comparison_{architecture}",
        "--sample-root", str(FINAL_STREAM_ROOT),
        "--num-samples", "65536", "--latency-num-samples", "1000",
        "--batch-size", "2048", "--model-id", str(spec.model_id),
        "--seed", "20260916", "--gpus", gpu, "--parallelism", "1",
        "--python", str(PYTHON),
        "--output-dir", str(REPO_ROOT / "outputs" / "qewc_v4" / "evaluations" / "id_final_comparison" / architecture),
    ]
    for run_id in run_ids(architecture):
        command.extend(("--checkpoint", f"{run_id}:{final_checkpoint(load_training(run_id))}"))
    for task in TASKS:
        command.extend(("--task", f"{task}:{config_path(architecture, task)}"))
    if resume:
        command.append("--resume")
    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", choices=tuple(ARCHITECTURES), required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    command = build_command(args.architecture, gpu=args.gpu, resume=args.resume)
    print("[QEWC v4 final ID comparison] " + " ".join(command), flush=True)
    if not args.dry_run:
        subprocess.run(command, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
