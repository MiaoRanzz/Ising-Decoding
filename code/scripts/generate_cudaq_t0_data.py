#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate and validate a CUDA-Q binary data package from the QAdapt T0 config."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

from omegaconf import OmegaConf

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from export.generate_test_data import generate_test_data  # noqa: E402
from scripts.config_paths import config_path  # noqa: E402


REQUIRED_FILES = (
    "detectors.bin",
    "observables.bin",
    "pymatching_predictions.bin",
    "H_csr.bin",
    "O_csr.bin",
    "priors.bin",
    "metadata.txt",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-name", default="examples/qadapt/config_qadapt_t0_base"
    )
    parser.add_argument("--distance", type=int, default=9)
    parser.add_argument("--n-rounds", type=int, default=9)
    parser.add_argument("--basis", choices=("X", "Z"), required=True)
    parser.add_argument("--num-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dense_shape(path: Path) -> tuple[int, int]:
    with path.open("rb") as stream:
        header = stream.read(8)
    if len(header) != 8:
        raise ValueError(f"truncated dense header: {path}")
    return struct.unpack("<II", header)


def csr_shape(path: Path) -> tuple[int, int, int]:
    with path.open("rb") as stream:
        header = stream.read(12)
    if len(header) != 12:
        raise ValueError(f"truncated CSR header: {path}")
    return struct.unpack("<III", header)


def main() -> None:
    args = parse_args()
    cfg = OmegaConf.load(config_path(args.config_name))
    noise_model = OmegaConf.to_container(cfg.data.noise_model, resolve=True)
    generate_test_data(
        distance=args.distance,
        n_rounds=args.n_rounds,
        basis=args.basis,
        code_rotation=str(cfg.data.code_rotation),
        noise_model_params=noise_model,
        num_samples=args.num_samples,
        seed=args.seed,
        output_dir=str(args.output_dir),
        write_stim_artifacts=False,
        write_cudaq_artifacts=True,
    )

    missing = [name for name in REQUIRED_FILES if not (args.output_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"CUDA-Q package is missing files: {missing}")
    detector_shape = dense_shape(args.output_dir / "detectors.bin")
    observable_shape = dense_shape(args.output_dir / "observables.bin")
    prediction_shape = dense_shape(args.output_dir / "pymatching_predictions.bin")
    if detector_shape[0] != args.num_samples:
        raise AssertionError(f"detector shots mismatch: {detector_shape}")
    if observable_shape != prediction_shape or observable_shape[0] != args.num_samples:
        raise AssertionError(
            f"observable/prediction shapes mismatch: {observable_shape}, {prediction_shape}"
        )

    report = {
        "status": "PASS",
        "basis": args.basis,
        "distance": args.distance,
        "n_rounds": args.n_rounds,
        "samples": args.num_samples,
        "seed": args.seed,
        "noise_config": args.config_name,
        "detectors_shape": detector_shape,
        "observables_shape": observable_shape,
        "H_shape_nnz": csr_shape(args.output_dir / "H_csr.bin"),
        "O_shape_nnz": csr_shape(args.output_dir / "O_csr.bin"),
        "files": {
            name: {
                "size_bytes": (args.output_dir / name).stat().st_size,
                "sha256": sha256(args.output_dir / name),
            }
            for name in REQUIRED_FILES
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
