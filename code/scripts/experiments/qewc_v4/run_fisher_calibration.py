#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compute and select the QEWC-v4 T0 Fisher sample count for one architecture."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[4]
COMPUTE = REPO_ROOT / "code" / "scripts" / "compute_ewc_fisher.py"
SELECT = REPO_ROOT / "code" / "scripts" / "experiments" / "qewc_v4" / "select_fisher_samples.py"
MANIFEST = REPO_ROOT / "experiment" / "qewc-v4" / "experiment_manifest.yaml"
EXPECTED_PREFIX = Path("/root/miniconda3/envs/ising-decoding")
ARCHITECTURES = {
    "htnet_a101": "experiments/qewc_v4/htnet/T0",
    "ising_fast": "experiments/qewc_v4/ising/t0_base",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", choices=tuple(ARCHITECTURES), required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/qewc_v4/calibration"))
    parser.add_argument("--python", type=Path, default=EXPECTED_PREFIX / "bin" / "python")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if Path(sys.prefix).resolve() != EXPECTED_PREFIX.resolve():
        raise RuntimeError(f"wrong Python environment: {sys.prefix}")
    experiment = OmegaConf.load(args.manifest)
    samples = [int(value) for value in experiment.ewc_calibration.fisher_sample_candidates]
    if samples != [4096, 16384, 65536]:
        raise RuntimeError(f"unexpected frozen Fisher sample grid: {samples}")
    checkpoint = REPO_ROOT / str(experiment.stage0[args.architecture].endpoint.path)
    root = args.output_root if args.output_root.is_absolute() else REPO_ROOT / args.output_root
    root = root / args.architecture / "fisher"
    root.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["PYTHONPATH"] = str(REPO_ROOT / "code")
    timings = []
    paths = []
    for count in samples:
        output = root / f"T0_n{count}.pt"
        paths.append((count, output))
        command = [
            str(args.python), "-u", str(COMPUTE),
            "--config-name", ARCHITECTURES[args.architecture],
            "--distance", "9", "--n-rounds", "9",
            "--checkpoint", str(checkpoint),
            "--output", str(output), "--task-name", "T0",
            "--num-samples", str(count), "--batch-size", "2048",
            "--per-sample-chunk-size", "16", "--positive-weight", "1.0",
            "--seed", "20260912", "--device", "cuda:0",
        ]
        print("[QEWC v4 Fisher] " + " ".join(command), flush=True)
        started = time.monotonic()
        if not args.dry_run and not (args.resume and output.is_file()):
            subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)
        timings.append({"samples": count, "wall_seconds": 0.0 if args.dry_run else time.monotonic() - started})
    selection = root / "selection.json"
    command = [
        str(args.python), "-u", str(SELECT),
        "--reference-samples", "65536",
        "--cosine-minimum", str(experiment.ewc_calibration.fisher_cosine_minimum),
        "--total-ratio-tolerance", str(experiment.ewc_calibration.fisher_total_ratio_tolerance),
        "--output", str(selection),
    ]
    for count, path in paths:
        command.extend(("--state", f"{count}:{path}"))
    print("[QEWC v4 Fisher] " + " ".join(command), flush=True)
    if args.dry_run:
        return
    subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)
    result = json.loads(selection.read_text(encoding="utf-8"))
    payload = {
        "architecture": args.architecture,
        "checkpoint": str(checkpoint),
        "sample_grid": samples,
        "timings": timings,
        "selection": result,
    }
    (root / "run_manifest.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
