#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate immutable, seeded Stim sample files for a QEWC-v3 task track."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[4]
CODE_ROOT = REPO_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from export.generate_test_data import generate_test_data  # noqa: E402
from workflows.config_validator import (  # noqa: E402
    apply_public_defaults_and_model,
    validate_public_config,
)


@dataclass(frozen=True)
class Task:
    name: str
    config: Path


def parse_task(value: str) -> Task:
    name, separator, raw_path = value.partition(":")
    if not separator or not name or not raw_path:
        raise argparse.ArgumentTypeError("--task must be NAME:CONFIG_PATH")
    path = Path(raw_path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return Task(name, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generate_task(task: Task, output: Path, num_samples: int, seed: int) -> dict:
    raw = OmegaConf.load(task.config)
    spec = validate_public_config(raw)
    cfg = apply_public_defaults_and_model(raw, spec)
    noise_model = OmegaConf.to_container(cfg.data.noise_model, resolve=True)
    if not isinstance(noise_model, dict):
        raise ValueError(f"{task.name} must define one explicit noise model")
    output.mkdir(parents=True, exist_ok=True)
    bases = {}
    for basis_index, basis in enumerate(("X", "Z")):
        basis_seed = seed + basis_index
        generate_test_data(
            distance=int(cfg.distance),
            n_rounds=int(cfg.n_rounds),
            basis=basis,
            p_error=float(cfg.test.p_error),
            code_rotation=str(cfg.data.code_rotation),
            noise_model_params=noise_model,
            num_samples=num_samples,
            seed=basis_seed,
            output_dir=str(output),
            write_stim_artifacts=True,
            write_cudaq_artifacts=False,
        )
        samples = output / f"samples_{basis}.dets"
        metadata = output / f"metadata_{basis}.json"
        bases[basis] = {
            "seed": basis_seed,
            "samples_path": str(samples),
            "samples_sha256": sha256(samples),
            "samples_bytes": samples.stat().st_size,
            "metadata_path": str(metadata),
            "metadata_sha256": sha256(metadata),
        }
    return {
        "config_path": str(task.config),
        "config_sha256": sha256(task.config),
        "output_dir": str(output),
        "bases": bases,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", required=True)
    parser.add_argument("--task", action="append", type=parse_task, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=65536)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    if args.num_samples <= 0:
        parser.error("--num-samples must be positive")
    names = [task.name for task in args.task]
    if len(names) != len(set(names)):
        parser.error("task names must be unique")
    if not args.output_root.is_absolute():
        args.output_root = REPO_ROOT / args.output_root
    return args


def main() -> None:
    args = parse_args()
    for task in args.task:
        if not task.config.is_file():
            raise FileNotFoundError(task.config)
    args.output_root.mkdir(parents=True, exist_ok=True)
    tasks = {
        task.name: generate_task(
            task, args.output_root / task.name, args.num_samples, args.seed
        )
        for task in args.task
    }
    manifest = {
        "track": args.track,
        "seed_X": args.seed,
        "seed_Z": args.seed + 1,
        "samples_per_basis": args.num_samples,
        "tasks": tasks,
    }
    manifest_path = args.output_root / "stream_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[QEWC v3] fixed evaluation streams: {manifest_path}")


if __name__ == "__main__":
    main()
