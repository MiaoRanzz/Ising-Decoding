#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluate learned 25-parameter noise models on held-out Google QEC data.

The script compares Google's supplied SI1000 circuit against one or more
learned effective noise models on the same hardware shots.  It reports both
syndrome-statistics calibration and downstream PyMatching logical error rate
(LER), including paired confidence intervals and an exact McNemar test.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Mapping, Sequence

import numpy as np
import pymatching
from scipy.stats import binomtest
import stim
import yaml

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from noise_learning.google_qec import (  # noqa: E402
    GoogleQECDataset,
    GoogleQECExperiment,
    _metadata_qubits,
    build_moments,
    inject_noise,
    predict_moments,
)
from qec.noise_model import NoiseModel  # noqa: E402


@dataclass(frozen=True)
class ModelSpec:
    name: str
    noise_model: NoiseModel | None
    source: str


def wilson_interval(errors: int, shots: int, z: float = 1.96) -> tuple[float, float]:
    """Return a two-sided Wilson score interval for a binomial rate."""

    if shots <= 0:
        return float("nan"), float("nan")
    p = float(errors) / float(shots)
    denominator = 1.0 + z * z / shots
    center = (p + z * z / (2.0 * shots)) / denominator
    half_width = (
        z
        * math.sqrt((p * (1.0 - p) + z * z / (4.0 * shots)) / shots)
        / denominator
    )
    return max(0.0, center - half_width), min(1.0, center + half_width)


def paired_statistics(
    candidate_errors: np.ndarray,
    baseline_errors: np.ndarray,
) -> dict[str, int | float]:
    """Return paired LER difference statistics on shared hardware shots."""

    candidate = np.asarray(candidate_errors, dtype=np.bool_).reshape(-1)
    baseline = np.asarray(baseline_errors, dtype=np.bool_).reshape(-1)
    if candidate.shape != baseline.shape:
        raise ValueError(f"paired shapes differ: {candidate.shape} != {baseline.shape}")
    candidate_only = int(np.count_nonzero(candidate & ~baseline))
    baseline_only = int(np.count_nonzero(~candidate & baseline))
    both = int(np.count_nonzero(candidate & baseline))
    neither = int(candidate.size - candidate_only - baseline_only - both)
    samples = int(candidate.size)
    delta = float((candidate_only - baseline_only) / samples)
    if samples > 1:
        difference_square_sum = candidate_only + baseline_only
        variance = max(
            0.0,
            (difference_square_sum - samples * delta * delta) / (samples - 1),
        )
        standard_error = math.sqrt(variance / samples)
    else:
        standard_error = 0.0 if samples == 1 else float("nan")
    margin = 1.96 * standard_error
    discordant = candidate_only + baseline_only
    mcnemar_p = (
        float(binomtest(candidate_only, discordant, 0.5).pvalue)
        if discordant
        else 1.0
    )
    return {
        "samples": samples,
        "candidate_only_errors": candidate_only,
        "baseline_only_errors": baseline_only,
        "both_errors": both,
        "neither_errors": neither,
        "delta_errors": candidate_only - baseline_only,
        "delta_ler": delta,
        "standard_error": standard_error,
        "ci95_low": max(-1.0, delta - margin),
        "ci95_high": min(1.0, delta + margin),
        "mcnemar_exact_pvalue": mcnemar_p,
    }


def _load_noise_model(path: Path) -> NoiseModel:
    payload = yaml.safe_load(path.read_text())
    if not isinstance(payload, Mapping) or "noise_model" not in payload:
        raise ValueError(f"{path} does not contain a noise_model mapping")
    return NoiseModel.from_config_dict(dict(payload["noise_model"]))


def parse_model_spec(value: str) -> ModelSpec:
    """Parse NAME:PATH learned-model CLI values."""

    if ":" not in value:
        raise argparse.ArgumentTypeError("--model must be NAME:/path/to/result.yaml")
    name, raw_path = value.split(":", 1)
    if not name or not raw_path:
        raise argparse.ArgumentTypeError("--model name and path must be non-empty")
    path = Path(raw_path).expanduser().resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"noise result does not exist: {path}")
    try:
        model = _load_noise_model(path)
    except Exception as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    return ModelSpec(name=name, noise_model=model, source=str(path))


def _moment_kind(indices: tuple[int, ...], num_detectors: int) -> str:
    if len(indices) == 1:
        return "logical_single" if indices[0] >= num_detectors else "detector_single"
    if any(index >= num_detectors for index in indices):
        return "detector_logical_pair"
    return "detector_pair"


def moment_metrics(
    experiment: GoogleQECExperiment,
    noisy_circuit: stim.Circuit,
    *,
    max_pair_moments: int,
) -> dict[str, object]:
    """Compare observed and predicted parity moments."""

    moments = build_moments(experiment, max_pair_moments=max_pair_moments)
    predicted = predict_moments(noisy_circuit, moments)
    observed = np.asarray([moment.observed for moment in moments], dtype=np.float64)
    sigma = np.asarray([moment.sigma for moment in moments], dtype=np.float64)
    residual = predicted - observed
    normalized = residual / sigma

    def summarize(mask: np.ndarray) -> dict[str, int | float]:
        values = residual[mask]
        z_values = normalized[mask]
        return {
            "count": int(mask.sum()),
            "mae": float(np.mean(np.abs(values))),
            "rmse": float(np.sqrt(np.mean(np.square(values)))),
            "weighted_rmse": float(np.sqrt(np.mean(np.square(z_values)))),
            "cost": float(0.5 * np.dot(z_values, z_values)),
        }

    kinds = np.asarray(
        [_moment_kind(moment.detectors, experiment.circuit.num_detectors) for moment in moments]
    )
    result: dict[str, object] = summarize(np.ones(len(moments), dtype=np.bool_))
    result["breakdown"] = {
        kind: summarize(kinds == kind)
        for kind in sorted(set(kinds.tolist()))
    }
    result["mean_observed"] = float(np.mean(observed))
    result["mean_predicted"] = float(np.mean(predicted))
    return result


def _matcher(noisy_circuit: stim.Circuit) -> pymatching.Matching:
    dem = noisy_circuit.detector_error_model(
        decompose_errors=True,
        flatten_loops=True,
        allow_gauge_detectors=True,
        approximate_disjoint_errors=True,
    )
    return pymatching.Matching.from_detector_error_model(dem)


def ler_metrics(
    experiment: GoogleQECExperiment,
    noisy_circuit: stim.Circuit,
) -> tuple[dict[str, int | float], np.ndarray]:
    """Decode hardware detector samples and compare logical observables."""

    matcher = _matcher(noisy_circuit)
    predictions = np.asarray(
        matcher.decode_batch(
            np.ascontiguousarray(experiment.detection_events, dtype=np.uint8)
        ),
        dtype=np.uint8,
    )
    if predictions.ndim == 1:
        predictions = predictions.reshape(-1, 1)
    actual = np.asarray(experiment.observable_flips, dtype=np.uint8)
    if predictions.shape != actual.shape:
        raise ValueError(f"prediction shape {predictions.shape} != {actual.shape}")
    error_mask = np.any(predictions != actual, axis=1)
    errors = int(error_mask.sum())
    shots = int(error_mask.size)
    low, high = wilson_interval(errors, shots)
    return (
        {
            "shots": shots,
            "logical_errors": errors,
            "ler": float(errors / shots),
            "ci95_low": low,
            "ci95_high": high,
            "matching_edges": int(matcher.num_edges),
        },
        error_mask,
    )


def _learned_circuit(
    experiment: GoogleQECExperiment,
    model: NoiseModel,
) -> stim.Circuit:
    return inject_noise(
        experiment.circuit,
        model,
        basis=str(experiment.metadata["basis"]),
        data_qubits=_metadata_qubits(
            experiment.metadata, "data_qubit_coords", experiment.circuit
        ),
        measurement_qubits=_metadata_qubits(
            experiment.metadata, "meas_qubit_coords", experiment.circuit
        ),
    )


def _google_si1000_circuit(source: Path, key: str) -> stim.Circuit:
    if source.is_file():
        raise ValueError(
            "official Google SI1000 baseline requires an extracted benchmark directory"
        )
    path = source / key / "circuit_noisy_si1000.stim"
    if not path.is_file():
        raise FileNotFoundError(path)
    return stim.Circuit.from_file(path)


def _aggregate_ler(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    by_model: dict[str, dict[str, int | float]] = {}
    for row in rows:
        name = str(row["model"])
        entry = by_model.setdefault(name, {"cases": 0, "shots": 0, "logical_errors": 0})
        entry["cases"] = int(entry["cases"]) + 1
        entry["shots"] = int(entry["shots"]) + int(row["shots"])
        entry["logical_errors"] = int(entry["logical_errors"]) + int(
            row["logical_errors"]
        )
    for entry in by_model.values():
        shots = int(entry["shots"])
        errors = int(entry["logical_errors"])
        low, high = wilson_interval(errors, shots)
        entry.update(ler=errors / shots, ci95_low=low, ci95_high=high)
    return by_model


def _aggregate_moments(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    by_model: dict[str, dict[str, float | int]] = {}
    for row in rows:
        name = str(row["model"])
        entry = by_model.setdefault(name, {"cases": 0, "count": 0, "cost": 0.0})
        entry["cases"] = int(entry["cases"]) + 1
        entry["count"] = int(entry["count"]) + int(row["count"])
        entry["cost"] = float(entry["cost"]) + float(row["cost"])
    for entry in by_model.values():
        entry["weighted_rmse"] = math.sqrt(
            2.0 * float(entry["cost"]) / int(entry["count"])
        )
    return by_model


def _aggregate_paired(
    case_masks: Sequence[Mapping[str, np.ndarray]],
    *,
    candidate: str,
    baseline: str,
) -> dict[str, int | float | str]:
    candidate_values = np.concatenate([masks[candidate] for masks in case_masks])
    baseline_values = np.concatenate([masks[baseline] for masks in case_masks])
    return {
        "candidate": candidate,
        "baseline": baseline,
        **paired_statistics(candidate_values, baseline_values),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Extracted Google benchmark root.")
    parser.add_argument("--experiment-key", action="append", required=True)
    parser.add_argument(
        "--model",
        action="append",
        type=parse_model_spec,
        required=True,
        help="Learned model as NAME:/path/to/result.yaml; may be repeated.",
    )
    parser.add_argument("--max-shots", type=int, default=0, help="0 uses all shots.")
    parser.add_argument("--max-pair-moments", type=int, default=96)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.max_shots < 0:
        parser.error("--max-shots must be >= 0")
    if args.max_pair_moments < 0:
        parser.error("--max-pair-moments must be >= 0")
    names = [model.name for model in args.model]
    if len(set(names)) != len(names) or "google_si1000" in names:
        parser.error("model names must be unique and cannot be google_si1000")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    source = args.source.resolve()
    dataset = GoogleQECDataset(source)
    specs = [ModelSpec("google_si1000", None, "circuit_noisy_si1000.stim"), *args.model]
    moment_rows: list[dict[str, object]] = []
    ler_rows: list[dict[str, object]] = []
    paired_rows: list[dict[str, object]] = []
    case_masks: list[dict[str, np.ndarray]] = []

    for key in args.experiment_key:
        experiment = dataset.load(key, max_shots=args.max_shots or None)
        circuits = {
            spec.name: (
                _google_si1000_circuit(source, key)
                if spec.noise_model is None
                else _learned_circuit(experiment, spec.noise_model)
            )
            for spec in specs
        }
        masks: dict[str, np.ndarray] = {}
        print(
            f"[case] {key} shots={experiment.shots} "
            f"detectors={experiment.circuit.num_detectors}"
        )
        for spec in specs:
            moments = moment_metrics(
                experiment,
                circuits[spec.name],
                max_pair_moments=args.max_pair_moments,
            )
            ler, mask = ler_metrics(experiment, circuits[spec.name])
            moment_rows.append({"experiment_key": key, "model": spec.name, **moments})
            ler_rows.append({"experiment_key": key, "model": spec.name, **ler})
            masks[spec.name] = mask
            print(
                f"  {spec.name}: moment_cost={float(moments['cost']):.6g} "
                f"weighted_rmse={float(moments['weighted_rmse']):.6g} "
                f"LER={float(ler['ler']):.6g} "
                f"({int(ler['logical_errors'])}/{int(ler['shots'])})"
            )
        for candidate in (spec.name for spec in specs if spec.name != "google_si1000"):
            paired_rows.append(
                {
                    "experiment_key": key,
                    "candidate": candidate,
                    "baseline": "google_si1000",
                    **paired_statistics(masks[candidate], masks["google_si1000"]),
                }
            )
        if len(args.model) >= 2:
            paired_rows.append(
                {
                    "experiment_key": key,
                    "candidate": args.model[-1].name,
                    "baseline": args.model[0].name,
                    **paired_statistics(masks[args.model[-1].name], masks[args.model[0].name]),
                }
            )
        case_masks.append(masks)

    aggregate_paired = []
    for candidate in (spec.name for spec in specs if spec.name != "google_si1000"):
        aggregate_paired.append(
            _aggregate_paired(
                case_masks, candidate=candidate, baseline="google_si1000"
            )
        )
    if len(args.model) >= 2:
        aggregate_paired.append(
            _aggregate_paired(
                case_masks,
                candidate=args.model[-1].name,
                baseline=args.model[0].name,
            )
        )

    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(source),
        "experiment_keys": list(args.experiment_key),
        "max_shots": args.max_shots,
        "max_pair_moments": args.max_pair_moments,
        "models": [
            {
                "name": spec.name,
                "source": spec.source,
                "noise_model_sha256": (
                    spec.noise_model.sha256() if spec.noise_model is not None else None
                ),
            }
            for spec in specs
        ],
        "moment_rows": moment_rows,
        "ler_rows": ler_rows,
        "paired_rows": paired_rows,
        "aggregate": {
            "moments": _aggregate_moments(moment_rows),
            "ler": _aggregate_ler(ler_rows),
            "paired": aggregate_paired,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="") as stream:
        fields = [
            "experiment_key",
            "model",
            "count",
            "cost",
            "weighted_rmse",
            "mae",
            "rmse",
            "shots",
            "logical_errors",
            "ler",
            "ci95_low",
            "ci95_high",
            "matching_edges",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        moments_by_key = {
            (row["experiment_key"], row["model"]): row for row in moment_rows
        }
        for ler in ler_rows:
            moment = moments_by_key[(ler["experiment_key"], ler["model"])]
            writer.writerow({field: moment.get(field, ler.get(field)) for field in fields})
    print(f"[output] JSON={args.output}")
    print(f"[output] CSV={csv_path}")
    for row in aggregate_paired:
        print(
            f"[paired] {row['candidate']} vs {row['baseline']}: "
            f"delta_LER={float(row['delta_ler']):+.6g} "
            f"CI95=[{float(row['ci95_low']):+.6g}, {float(row['ci95_high']):+.6g}] "
            f"p={float(row['mcnemar_exact_pvalue']):.6g}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
