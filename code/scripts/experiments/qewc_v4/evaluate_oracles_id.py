#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluate C0 and four single-task V4 Oracles on the frozen ID streams."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[4]
CODE_ROOT = REPO_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from scripts.experiments.qewc_v4.evaluation_common import (  # noqa: E402
    ARCHITECTURES,
    FINAL_STREAM_ROOT,
    TASKS,
    config_path,
)


EVALUATOR = REPO_ROOT / "code" / "scripts" / "experiments" / "qewc_v3" / "evaluate_realistic_matrix.py"
MANIFEST = REPO_ROOT / "outputs" / "qewc_v4" / "frozen_experiment_manifest.yaml"
PYTHON = Path("/root/miniconda3/envs/ising-decoding/bin/python")


def oracle_checkpoint(architecture: str, task: str) -> Path:
    path = REPO_ROOT / "outputs" / "qewc_v4" / "oracles" / architecture / task / "run_manifest.json"
    training = json.loads(path.read_text(encoding="utf-8"))
    return Path(training["endpoints"]["C1"]["path"])


def build_command(
    architecture: str,
    *,
    gpu: str,
    python: Path = PYTHON,
    manifest_path: Path = MANIFEST,
    resume: bool,
) -> list[str]:
    spec = ARCHITECTURES[architecture]
    experiment = OmegaConf.load(manifest_path)
    command = [
        str(python),
        "-u",
        str(EVALUATOR),
        "--track",
        f"formal_id_oracles_{architecture}",
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
        str(REPO_ROOT / "outputs" / "qewc_v4" / "evaluations" / "id_oracles" / architecture),
        "--checkpoint",
        f"C0:{REPO_ROOT / str(experiment.stage0[architecture].endpoint.path)}",
    ]
    for task in TASKS[1:]:
        command.extend(("--checkpoint", f"O_{task}:{oracle_checkpoint(architecture, task)}"))
    for task in TASKS:
        command.extend(("--task", f"{task}:{config_path(architecture, task)}"))
    if resume:
        command.append("--resume")
    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", choices=tuple(ARCHITECTURES), required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--python", type=Path, default=PYTHON)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    command = build_command(
        args.architecture,
        gpu=args.gpu,
        python=args.python,
        manifest_path=args.manifest,
        resume=args.resume,
    )
    print("[QEWC v4 Oracle ID] " + " ".join(command), flush=True)
    if not args.dry_run:
        subprocess.run(command, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
