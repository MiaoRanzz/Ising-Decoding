#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run frozen synthetic-OOD or Willow evaluation for one V4 architecture."""

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
    config_path,
    final_checkpoint,
    load_training,
    run_ids,
)


PYTHON = Path("/root/miniconda3/envs/ising-decoding/bin/python")
OOD = REPO_ROOT / "code" / "examples" / "infer_ood.py"
WILLOW = REPO_ROOT / "code" / "examples" / "infer_willow.py"
WILLOW_ROOT = Path(
    "/mnt/data2/Ising-Decoding/benchmarks/google_qec/"
    "google_105Q_surface_code_d3_d5_d7/google_105Q_surface_code_d3_d5_d7"
)


def model_arguments(architecture: str) -> list[str]:
    spec = ARCHITECTURES[architecture]
    arguments = []
    for run_id in run_ids(architecture):
        checkpoint = final_checkpoint(load_training(run_id))
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        arguments.extend(("--model", f"{run_id}:{spec.model_id}:{checkpoint}"))
    return arguments


def build_command(target: str, architecture: str, *, gpu: str, resume: bool) -> list[str]:
    spec = ARCHITECTURES[architecture]
    output = REPO_ROOT / "outputs" / "qewc_v4" / "evaluations" / target / architecture
    if target == "ood":
        generated = REPO_ROOT / "outputs" / "qewc_v4" / "generated_configs" / "ood" / architecture
        command = [
            str(PYTHON), "-u", str(OOD),
            *model_arguments(architecture),
            "--base-config", str(config_path(architecture, "T0")),
            "--distances", "7,9", "--n-rounds", "9",
            "--multipliers", "1.2,1.5,2.0,2.5,3.0",
            "--generated-config-dir", str(generated),
            "--manifest", str(generated / "manifest.json"),
            "--num-samples", "262144", "--latency-num-samples", "10000",
            "--batch-size", "2048", "--num-workers", "0", "--basis", "both",
            "--seed", "20260916", "--gpus", gpu, "--parallelism", "1",
            "--python", str(PYTHON), "--output-dir", str(output),
        ]
    elif target == "willow":
        if not WILLOW_ROOT.is_dir():
            raise FileNotFoundError(WILLOW_ROOT)
        command = [
            str(PYTHON), "-u", str(WILLOW),
            *model_arguments(architecture),
            "--benchmark-root", str(WILLOW_ROOT),
            "--config-name", spec.base_config,
            "--num-samples", "0", "--latency-num-samples", "10000",
            "--batch-size", "2048", "--basis", "both", "--gpus", gpu,
            "--python", str(PYTHON), "--output-dir", str(output),
        ]
    else:
        raise ValueError(f"unknown external target: {target}")
    if resume:
        command.append("--resume")
    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=("ood", "willow"), required=True)
    parser.add_argument("--architecture", choices=tuple(ARCHITECTURES), required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    command = build_command(args.target, args.architecture, gpu=args.gpu, resume=args.resume)
    if args.dry_run:
        command.append("--dry-run")
    print(f"[QEWC v4 {args.target}] " + " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
