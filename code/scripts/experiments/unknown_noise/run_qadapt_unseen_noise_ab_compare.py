#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run sharded paired EWC/no-EWC inference for the unseen-noise A/B design."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


CODE_ROOT = Path(__file__).resolve().parents[3]
REPO_ROOT = CODE_ROOT.parent
DEFAULT_MANIFEST = "outputs/analysis/qadapt_unseen_noise_ab_manifest.json"
DEFAULT_OUTPUT_DIR = "outputs/paired_inference_compare/qadapt_unseen_noise_ab"
DEFAULT_EWC_CHECKPOINT = (
    "outputs/ising_domestic_fast_opt_stfusion_r9_x_seq_ewc/"
    "models/PreDecoderSTFusion_v2.0.100.pt"
)
DEFAULT_NOEWC_CHECKPOINT = (
    "outputs/ising_domestic_fast_opt_stfusion_r9_x_seq_noewc/"
    "models/PreDecoderSTFusion_v2.0.100.pt"
)


def rel(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def result_path(output_dir: Path, environment: dict[str, Any], distance: int) -> Path:
    return (
        output_dir
        / environment["family"]
        / f"d{distance}"
        / f"{Path(environment['config_filename']).stem}_d{distance}.json"
    )


def build_tasks(
    manifest: dict[str, Any],
    *,
    families: set[str],
    distances: list[int],
) -> list[tuple[dict[str, Any], int]]:
    tasks = []
    for environment in manifest["environments"]:
        if environment["family_key"] not in families:
            continue
        for distance in distances:
            tasks.append((environment, int(distance)))
    return tasks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--families", default="A,B")
    parser.add_argument("--distances", default="7,9")
    parser.add_argument("--n-rounds", type=int, default=9)
    parser.add_argument("--num-samples", type=int, default=262144)
    parser.add_argument("--latency-num-samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--ewc-checkpoint", default=DEFAULT_EWC_CHECKPOINT)
    parser.add_argument("--noewc-checkpoint", default=DEFAULT_NOEWC_CHECKPOINT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("require num_shards > 0 and 0 <= shard_index < num_shards")

    manifest_path = rel(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    families = {item.strip() for item in args.families.split(",") if item.strip()}
    manifest_families = {
        str(environment["family_key"])
        for environment in manifest["environments"]
    }
    unknown_families = families - manifest_families
    if unknown_families:
        raise ValueError(f"unknown family keys: {sorted(unknown_families)}")
    distances = [int(item.strip()) for item in args.distances.split(",") if item.strip()]
    tasks = build_tasks(manifest, families=families, distances=distances)
    tasks = [
        task for task_index, task in enumerate(tasks)
        if task_index % args.num_shards == args.shard_index
    ]
    if args.limit is not None:
        tasks = tasks[: args.limit]

    output_dir = rel(args.output_dir)
    ewc_checkpoint = rel(args.ewc_checkpoint)
    noewc_checkpoint = rel(args.noewc_checkpoint)
    for checkpoint in (ewc_checkpoint, noewc_checkpoint):
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)

    print(
        f"[design] sha256={manifest['design_sha256']} tasks={len(tasks)} "
        f"shard={args.shard_index}/{args.num_shards}"
    )
    failures = []
    for task_index, (environment, distance) in enumerate(tasks, start=1):
        output = result_path(output_dir, environment, distance)
        if output.exists() and not args.overwrite:
            print(f"[skip {task_index}/{len(tasks)}] {output}")
            continue
        output.parent.mkdir(parents=True, exist_ok=True)
        config_path = rel(environment["config_path"])
        log_path = output.with_suffix(".log")
        command = [
            sys.executable,
            str(CODE_ROOT / "scripts" / "paired_inference_compare.py"),
            "--config-file",
            str(config_path),
            "--distance",
            str(distance),
            "--n-rounds",
            str(args.n_rounds),
            "--num-samples",
            str(args.num_samples),
            "--latency-num-samples",
            str(args.latency_num_samples),
            "--batch-size",
            str(args.batch_size),
            "--seed",
            str(args.seed),
            "--device",
            args.device,
            "--basis",
            "both",
            "--model",
            f"r9x_seq_ewc_e100:111:{ewc_checkpoint}",
            "--model",
            f"r9x_seq_noewc_e100:111:{noewc_checkpoint}",
            "--paired-comparison",
            "r9x_seq_ewc_e100:r9x_seq_noewc_e100",
            "--output",
            str(output),
        ]
        print(
            f"[run {task_index}/{len(tasks)}] family={environment['family_key']} "
            f"env={environment['env_key']} d={distance} device={args.device}"
        )
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        log_path.write_text(completed.stdout, encoding="utf-8")
        if completed.returncode != 0:
            failures.append(
                {
                    "family": environment["family_key"],
                    "env_key": environment["env_key"],
                    "distance": distance,
                    "returncode": completed.returncode,
                    "log": str(log_path),
                }
            )
            print(f"[fail] {log_path}")
        else:
            print(f"[done] {output}")

    if failures:
        failure_path = output_dir / f"failures_shard_{args.shard_index:02d}.json"
        failure_path.write_text(
            json.dumps(failures, indent=2, sort_keys=True), encoding="utf-8"
        )
        raise SystemExit(f"{len(failures)} tasks failed; see {failure_path}")


if __name__ == "__main__":
    main()
