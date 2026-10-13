# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared statistics and artifact helpers for the T0 quantization benchmark."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist
from typing import Any, Iterable, Mapping

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def paired_noninferiority(
    candidate_errors: np.ndarray,
    baseline_errors: np.ndarray,
    *,
    absolute_margin: float = 1e-3,
    relative_margin: float = 0.03,
    alpha: float = 0.05,
) -> dict[str, float | int | bool]:
    """Test whether candidate LER is below the pre-registered paired margin.

    The one-sided null hypothesis is delta >= margin.  A normal approximation is
    used for the paired per-shot differences, matching the existing paired LER
    implementation while fixing its test direction.
    """

    candidate = np.asarray(candidate_errors, dtype=np.bool_).reshape(-1)
    baseline = np.asarray(baseline_errors, dtype=np.bool_).reshape(-1)
    if candidate.shape != baseline.shape:
        raise ValueError(
            f"paired error masks must have identical shapes: {candidate.shape} != {baseline.shape}"
        )
    if candidate.size == 0:
        raise ValueError("paired error masks must not be empty")
    if absolute_margin <= 0 or relative_margin <= 0:
        raise ValueError("noninferiority margins must be positive")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one")

    samples = int(candidate.size)
    candidate_only = int(np.count_nonzero(candidate & ~baseline))
    baseline_only = int(np.count_nonzero(~candidate & baseline))
    both = int(np.count_nonzero(candidate & baseline))
    neither = samples - candidate_only - baseline_only - both
    candidate_ler = float(candidate.mean())
    baseline_ler = float(baseline.mean())
    delta = candidate_ler - baseline_ler
    margin = min(float(absolute_margin), float(relative_margin) * baseline_ler)
    differences = candidate.astype(np.int8) - baseline.astype(np.int8)
    standard_error = (
        float(differences.std(ddof=1) / math.sqrt(samples)) if samples > 1 else 0.0
    )
    critical_value = NormalDist().inv_cdf(1.0 - alpha)
    upper_bound = delta + critical_value * standard_error
    if standard_error == 0.0:
        p_value = 0.0 if delta < margin else 1.0
    else:
        p_value = NormalDist().cdf((delta - margin) / standard_error)

    return {
        "samples": samples,
        "candidate_logical_errors": int(candidate.sum()),
        "baseline_logical_errors": int(baseline.sum()),
        "candidate_ler": candidate_ler,
        "baseline_ler": baseline_ler,
        "candidate_only_errors": candidate_only,
        "baseline_only_errors": baseline_only,
        "both_errors": both,
        "neither_errors": neither,
        "delta_ler": delta,
        "absolute_margin": float(absolute_margin),
        "relative_margin": float(relative_margin),
        "effective_margin": margin,
        "standard_error": standard_error,
        "alpha": float(alpha),
        "one_sided_upper_bound": upper_bound,
        "noninferiority_p_value": p_value,
        "noninferiority_passed": bool(upper_bound <= margin),
    }


def holm_adjust(p_values: Mapping[str, float], *, alpha: float = 0.05) -> dict[str, dict[str, float | bool]]:
    if not p_values:
        raise ValueError("p_values must not be empty")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one")
    for name, value in p_values.items():
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"invalid p-value for {name}: {value}")

    ordered = sorted(p_values, key=lambda name: float(p_values[name]))
    adjusted: dict[str, dict[str, float | bool]] = {}
    running = 0.0
    total = len(ordered)
    for rank, name in enumerate(ordered):
        corrected = min(1.0, max(running, (total - rank) * float(p_values[name])))
        running = corrected
        adjusted[name] = {
            "raw_p_value": float(p_values[name]),
            "holm_adjusted_p_value": corrected,
            "holm_passed": bool(corrected < alpha),
        }
    return adjusted


def combine_basis_results(
    results: Mapping[str, Mapping[str, Any]], *, alpha: float = 0.05
) -> dict[str, Any]:
    expected = {"X", "Z"}
    if set(results) != expected:
        raise ValueError(f"basis results must contain exactly X and Z: {sorted(results)}")
    adjusted = holm_adjust(
        {
            basis: float(result["noninferiority_p_value"])
            for basis, result in results.items()
        },
        alpha=alpha,
    )
    combined = {
        basis: {**dict(results[basis]), **adjusted[basis]}
        for basis in ("X", "Z")
    }
    return {
        "status": (
            "VALID_PASS"
            if all(bool(combined[basis]["holm_passed"]) for basis in ("X", "Z"))
            else "ACCURACY_FAIL"
        ),
        "alpha": float(alpha),
        "basis_results": combined,
    }


def latency_summary(values_ms: Iterable[float]) -> dict[str, float | int]:
    measured = np.asarray(list(values_ms), dtype=np.float64)
    if measured.size == 0:
        raise ValueError("latency values must not be empty")
    if np.any(~np.isfinite(measured)) or np.any(measured < 0):
        raise ValueError("latency values must be finite and non-negative")
    return {
        "count": int(measured.size),
        "mean_ms": float(measured.mean()),
        "std_ms": float(measured.std(ddof=1)) if measured.size > 1 else 0.0,
        "p50_ms": float(np.percentile(measured, 50)),
        "p95_ms": float(np.percentile(measured, 95)),
        "p99_ms": float(np.percentile(measured, 99)),
        "min_ms": float(measured.min()),
        "max_ms": float(measured.max()),
    }


def batch_size_grid(max_batch: int) -> list[int]:
    if max_batch < 1:
        raise ValueError("max_batch must be positive")
    values = []
    value = 1
    while value <= max_batch:
        values.append(value)
        value *= 2
    if values[-1] != max_batch and max_batch & (max_batch - 1):
        values.append(max_batch)
    return values
