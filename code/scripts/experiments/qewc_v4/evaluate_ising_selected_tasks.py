#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluate the Stage-0 checkpoint on the four QEWC v4 frozen Stress tasks on Ising-Fast."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
CODE_ROOT = REPO_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from scripts.qadapt_example_utils import InferenceJob, parse_gpus, run_jobs  # noqa: E402


PAIRED_SCRIPT = REPO_ROOT / "code" / "scripts" / "paired_inference_compare.py"
STRESS_DIR = REPO_ROOT / "conf" / "experiments" / "qewc_v4" / "ising"


def build_job(args: argparse.Namespace, config: Path) -> InferenceJob:
    name = config.stem
    output = args.output_dir / f"{name}.json"
    command = (
        str(args.python), "-u", str(PAIRED_SCRIPT),
        "--config-file", str(config), "--distance", "9", "--n-rounds", "9",
        "--num-samples", str(args.num_samples),
        "--latency-num-samples", str(args.latency_num_samples),
        "--batch-size", str(args.batch_size), "--num-workers", "0",
        "--seed", str(args.seed), "--basis", "both", "--device", "cuda:0",
        "--output", str(output),
        "--model", f"C0:1:{args.checkpoint}:{config}",
    )
    return InferenceJob(name, command, output)


def write_summary(args: argparse.Namespace, configs: list[Path]) -> None:
    candidates = {}
    for config in configs:
        payload = json.loads((args.output_dir / f"{config.stem}.json").read_text(encoding="utf-8"))
        summary = {row["method"]: row for row in payload["summary"]}["C0"]
        candidates[config.stem] = {
            "config": str(config),
            "stage0_ler": float(summary["ler_avg"]),
            "input_syndrome_density": float(summary["input_syndrome_density"]),
            "residual_syndrome_density": float(summary["residual_syndrome_density"]),
            "density_reduction_fraction": float(summary["density_reduction_fraction"]),
        }
    result = {
        "checkpoint": str(args.checkpoint),
        "seed": args.seed,
        "samples_per_basis": args.num_samples,
        "candidates": candidates,
    }
    (args.output_dir / "stress_ler_manifest.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/qewc_v4/ising_diagnostic/ler"))
    parser.add_argument("--num-samples", type=int, default=32768)
    parser.add_argument("--latency-num-samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--parallelism", type=int, default=1)
    parser.add_argument("--python", type=Path, default=Path("/root/miniconda3/envs/ising-decoding/bin/python"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.checkpoint = args.checkpoint if args.checkpoint.is_absolute() else REPO_ROOT / args.checkpoint
    args.output_dir = args.output_dir if args.output_dir.is_absolute() else REPO_ROOT / args.output_dir
    return args


def main() -> None:
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    configs = sorted(STRESS_DIR.glob("T*.yaml"))
    if len(configs) != 4:
        raise RuntimeError(f"Expected 4 frozen Stress configs, found {len(configs)}")
    run_jobs(
        [build_job(args, config) for config in configs],
        gpus=parse_gpus(args.gpus),
        parallelism=args.parallelism,
        resume=args.resume,
        dry_run=args.dry_run,
    )
    if not args.dry_run:
        write_summary(args, configs)
        print(f"[QEWC v4 Ising diagnostic] stress LER manifest: {args.output_dir / 'stress_ler_manifest.json'}")


if __name__ == "__main__":
    main()
