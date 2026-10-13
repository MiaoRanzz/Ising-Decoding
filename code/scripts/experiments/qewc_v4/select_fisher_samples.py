#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Select the smallest stable Fisher sample count for QEWC-v4."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[4]
CODE_ROOT = REPO_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from training.ewc import load_ewc_state  # noqa: E402


@dataclass(frozen=True)
class StateSpec:
    samples: int
    path: Path


def parse_state(value: str) -> StateSpec:
    raw_samples, separator, raw_path = value.partition(":")
    if not separator or not raw_samples or not raw_path:
        raise argparse.ArgumentTypeError("--state must be SAMPLES:PATH")
    samples = int(raw_samples)
    if samples <= 0:
        raise argparse.ArgumentTypeError("Fisher sample count must be positive")
    path = Path(raw_path)
    return StateSpec(samples=samples, path=path if path.is_absolute() else REPO_ROOT / path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compare_fisher_dicts(candidate: dict[str, torch.Tensor], reference: dict[str, torch.Tensor]) -> dict:
    if set(candidate) != set(reference):
        raise ValueError("candidate and reference Fisher parameter names differ")
    dot = 0.0
    candidate_norm_sq = 0.0
    reference_norm_sq = 0.0
    candidate_total = 0.0
    reference_total = 0.0
    for name in sorted(reference):
        left = candidate[name].detach().cpu().double().reshape(-1)
        right = reference[name].detach().cpu().double().reshape(-1)
        if left.shape != right.shape:
            raise ValueError(f"Fisher shape differs for {name}: {left.shape} != {right.shape}")
        dot += float(torch.dot(left, right))
        candidate_norm_sq += float(torch.dot(left, left))
        reference_norm_sq += float(torch.dot(right, right))
        candidate_total += float(left.sum())
        reference_total += float(right.sum())
    denominator = math.sqrt(candidate_norm_sq * reference_norm_sq)
    cosine = dot / denominator if denominator > 0 else float("nan")
    total_ratio = candidate_total / reference_total if reference_total > 0 else float("nan")
    return {
        "cosine_to_reference": cosine,
        "fisher_total": candidate_total,
        "reference_total": reference_total,
        "total_ratio_to_reference": total_ratio,
        "total_ratio_relative_error": abs(total_ratio - 1.0),
    }


def select(records: list[dict], *, cosine_minimum: float, total_tolerance: float) -> dict:
    eligible = [
        row for row in sorted(records, key=lambda item: int(item["samples"]))
        if float(row["cosine_to_reference"]) >= cosine_minimum
        and float(row["total_ratio_relative_error"]) <= total_tolerance
    ]
    if not eligible:
        raise ValueError("no Fisher sample count meets the stability thresholds")
    return {
        "selected_fisher_samples": int(eligible[0]["samples"]),
        "cosine_minimum": cosine_minimum,
        "total_ratio_tolerance": total_tolerance,
        "records": sorted(records, key=lambda item: int(item["samples"])),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", action="append", type=parse_state, required=True)
    parser.add_argument("--reference-samples", type=int, default=65536)
    parser.add_argument("--cosine-minimum", type=float, default=0.99)
    parser.add_argument("--total-ratio-tolerance", type=float, default=0.05)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    by_samples = {spec.samples: spec for spec in args.state}
    if len(by_samples) != len(args.state):
        raise ValueError("duplicate Fisher sample count")
    if args.reference_samples not in by_samples:
        raise ValueError("reference sample count is missing")
    states = {samples: load_ewc_state(spec.path) for samples, spec in by_samples.items()}
    reference = states[args.reference_samples]
    records = []
    for samples, state in states.items():
        metrics = compare_fisher_dicts(state.fisher, reference.fisher)
        path = by_samples[samples].path
        records.append({
            "samples": samples,
            "path": str(path),
            "sha256": sha256(path),
            "serialized_bytes": path.stat().st_size,
            **metrics,
        })
    result = select(
        records,
        cosine_minimum=args.cosine_minimum,
        total_tolerance=args.total_ratio_tolerance,
    )
    output = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"[QEWC v4] selected Fisher samples={result['selected_fisher_samples']}: {output}")


if __name__ == "__main__":
    main()
