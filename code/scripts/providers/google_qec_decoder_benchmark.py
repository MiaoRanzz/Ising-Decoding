#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark PyMatching and QAdapt seq+EWC on Google QEC hardware data."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pymatching
import stim
import torch
from omegaconf import OmegaConf

CODE_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = CODE_ROOT.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from evaluation.logical_error_rate import (  # noqa: E402
    PreDecoderMemoryEvalModule,
    _build_stab_maps,
)
from qec.surface_code.memory_circuit import SurfaceCode  # noqa: E402
from scripts.config_paths import config_path  # noqa: E402
from scripts.paired_inference_compare import (  # noqa: E402
    SyndromeDensityAccumulator,
    model_density_statistics,
)
from workflows.config_validator import (  # noqa: E402
    apply_public_defaults_and_model,
    validate_public_config,
)
from workflows.run import _load_model  # noqa: E402


DEFAULT_BENCHMARK_ROOT = (
    REPO_ROOT / "benchmarks/google_qec/google_105Q_surface_code_d3_d5_d7"
)


@dataclass(frozen=True)
class BenchmarkModel:
    name: str
    model_id: int
    checkpoint: Path


DEFAULT_MODELS = {
    "qadapt_seq_ewc": BenchmarkModel(
        "qadapt_seq_ewc",
        111,
        REPO_ROOT
        / (
            "outputs/qadapt_seq_ewc/models/"
            "HTnet.0.100.pt"
        ),
    ),
}


def maybe_compile_model(
    model: torch.nn.Module,
    *,
    enabled: bool,
    mode: str = "default",
) -> torch.nn.Module:
    """Optionally compile one cached model with dynamic detector dimensions."""

    return torch.compile(model, mode=mode, dynamic=True) if enabled else model


@dataclass(frozen=True)
class GoogleQECCase:
    path: Path
    patch: str
    distance: int
    basis: str
    rounds: int
    shots: int


REQUIRED_CASE_FILES = (
    "circuit_ideal.stim",
    "circuit_noisy_si1000.stim",
    "detection_events.b8",
    "obs_flips_actual.b8",
)


def discover_cases(
    root: Path,
    *,
    distances: set[int] | None = None,
    rounds: set[int] | None = None,
    bases: set[str] | None = None,
    patches: set[str] | None = None,
) -> list[GoogleQECCase]:
    """Discover complete Google benchmark cases selected by metadata."""

    root = Path(root)
    cases = []
    for metadata_path in root.glob("d*_at_q*/[XZ]/r*/metadata.json"):
        metadata = json.loads(metadata_path.read_text())
        case_dir = metadata_path.parent
        patch = case_dir.parents[1].name
        distance = int(metadata["distance"])
        basis = str(metadata["basis"]).upper()
        n_rounds = int(metadata["rounds"])
        if distances is not None and distance not in distances:
            continue
        if rounds is not None and n_rounds not in rounds:
            continue
        if bases is not None and basis not in bases:
            continue
        if patches is not None and patch not in patches:
            continue
        missing = [name for name in REQUIRED_CASE_FILES if not (case_dir / name).is_file()]
        if missing:
            raise FileNotFoundError(f"Incomplete Google QEC case {case_dir}: missing {missing}")
        cases.append(
            GoogleQECCase(
                path=case_dir,
                patch=patch,
                distance=distance,
                basis=basis,
                rounds=n_rounds,
                shots=int(metadata["shots"]),
            )
        )
    return sorted(cases, key=lambda case: (case.distance, case.patch, case.basis, case.rounds))


def _google_to_xv_coordinate(
    coordinate: Sequence[float],
    *,
    min_difference: int,
    min_sum: int,
) -> tuple[int, int]:
    if len(coordinate) < 2:
        raise ValueError(f"Google coordinate must contain x and y, got {coordinate!r}")
    x = float(coordinate[0])
    y = float(coordinate[1])
    if not x.is_integer() or not y.is_integer():
        raise ValueError(f"Google coordinate must be integral, got {coordinate!r}")
    x_int = int(x)
    y_int = int(y)
    return (
        x_int - y_int - int(min_difference) + 1,
        x_int + y_int - int(min_sum) + 1,
    )


def build_detector_permutation(
    circuit: stim.Circuit,
    metadata: Mapping[str, Any],
) -> np.ndarray:
    """Return indices that map Google detector columns to the model's XV order.

    Google emits each bulk round in physical measurement-qubit order. The
    predecoder consumes initial-boundary, X-block, Z-block, ..., final-boundary
    order, with stabilizers indexed by the repository's XV patch convention.
    """

    distance = int(metadata["distance"])
    rounds = int(metadata["rounds"])
    basis = str(metadata["basis"]).upper()
    if basis not in {"X", "Z"}:
        raise ValueError(f"basis must be X or Z, got {basis!r}")
    if distance < 3 or distance % 2 == 0:
        raise ValueError(f"distance must be an odd integer >= 3, got {distance}")
    if rounds < 1:
        raise ValueError(f"rounds must be positive, got {rounds}")

    half = (distance * distance - 1) // 2
    expected_detectors = 2 * rounds * half
    if int(circuit.num_detectors) != expected_detectors:
        raise ValueError(
            "detector count mismatch: "
            f"circuit has {circuit.num_detectors}, expected {expected_detectors} "
            f"for d={distance}, rounds={rounds}"
        )

    data_coordinates = [tuple(item) for item in metadata["data_qubit_coords"]]
    if len(data_coordinates) != distance * distance:
        raise ValueError(
            f"data coordinate count mismatch: {len(data_coordinates)} != {distance * distance}"
        )
    min_difference = min(int(x) - int(y) for x, y in data_coordinates)
    min_sum = min(int(x) + int(y) for x, y in data_coordinates)
    transformed_data = {
        _google_to_xv_coordinate(
            coordinate,
            min_difference=min_difference,
            min_sum=min_sum,
        )
        for coordinate in data_coordinates
    }
    odd_coordinates = range(1, 2 * distance, 2)
    expected_data = {(x, y) for x in odd_coordinates for y in odd_coordinates}
    if transformed_data != expected_data:
        raise ValueError("Google data-qubit coordinates do not form the expected rotated patch")

    code = SurfaceCode(distance, first_bulk_syndrome_type="X", rotated_type="V")
    x_indices = {
        tuple(map(int, code.xcheck_qubits_dict[int(qubit)]["coord"])): index
        for index, qubit in enumerate(code.xcheck_qubits)
    }
    z_indices = {
        tuple(map(int, code.zcheck_qubits_dict[int(qubit)]["coord"])): index
        for index, qubit in enumerate(code.zcheck_qubits)
    }
    detector_coordinates = circuit.get_detector_coordinates()
    if len(detector_coordinates) != expected_detectors:
        raise ValueError(
            "detector coordinate count mismatch: "
            f"{len(detector_coordinates)} != {expected_detectors}"
        )

    canonical_to_source = np.full(expected_detectors, -1, dtype=np.int64)
    boundary_start = expected_detectors - half
    for source_index in range(expected_detectors):
        raw_coordinate = detector_coordinates[source_index]
        if len(raw_coordinate) < 3:
            raise ValueError(f"detector {source_index} has no spatial/time coordinate")
        # Initial and bulk detectors end in their stabilizer coordinate. Google
        # final-boundary detectors list data coordinates first and the previous
        # ancilla/stabilizer coordinate last, so the last coordinate triple is
        # the uniform choice for every phase.
        model_coordinate = _google_to_xv_coordinate(
            raw_coordinate[-3:-1],
            min_difference=min_difference,
            min_sum=min_sum,
        )
        if model_coordinate in x_indices:
            stabilizer_type = "X"
            stabilizer_index = x_indices[model_coordinate]
        elif model_coordinate in z_indices:
            stabilizer_type = "Z"
            stabilizer_index = z_indices[model_coordinate]
        else:
            raise ValueError(
                f"detector {source_index} coordinate {raw_coordinate!r} maps to "
                f"unknown XV stabilizer {model_coordinate}"
            )

        if source_index < half:
            if stabilizer_type != basis:
                raise ValueError(
                    f"initial detector {source_index} is {stabilizer_type}, expected {basis}"
                )
            canonical_index = stabilizer_index
        elif source_index >= boundary_start:
            if stabilizer_type != basis:
                raise ValueError(
                    f"boundary detector {source_index} is {stabilizer_type}, expected {basis}"
                )
            canonical_index = boundary_start + stabilizer_index
        else:
            bulk_offset = source_index - half
            bulk_round = bulk_offset // (2 * half)
            type_offset = 0 if stabilizer_type == "X" else half
            canonical_index = half + bulk_round * 2 * half + type_offset + stabilizer_index

        if canonical_to_source[canonical_index] != -1:
            raise ValueError(
                f"duplicate detector mapping for canonical index {canonical_index}"
            )
        canonical_to_source[canonical_index] = source_index

    if np.any(canonical_to_source < 0):
        missing = np.flatnonzero(canonical_to_source < 0).tolist()
        raise ValueError(f"incomplete detector mapping; missing canonical indices {missing}")
    return canonical_to_source


def google_to_canonical(data: np.ndarray, canonical_to_source: np.ndarray) -> np.ndarray:
    rows = np.asarray(data)
    permutation = np.asarray(canonical_to_source, dtype=np.int64)
    if rows.ndim != 2 or rows.shape[1] != permutation.size:
        raise ValueError(
            f"Google detector shape {rows.shape} is incompatible with permutation "
            f"width {permutation.size}"
        )
    return np.ascontiguousarray(rows[:, permutation])


def canonical_to_google(data: np.ndarray, canonical_to_source: np.ndarray) -> np.ndarray:
    rows = np.asarray(data)
    permutation = np.asarray(canonical_to_source, dtype=np.int64)
    if rows.ndim != 2 or rows.shape[1] != permutation.size:
        raise ValueError(
            f"canonical detector shape {rows.shape} is incompatible with permutation "
            f"width {permutation.size}"
        )
    restored = np.empty_like(rows)
    restored[:, permutation] = rows
    return np.ascontiguousarray(restored)


def verify_bulk_data_fault_equivalence(
    circuit: stim.Circuit,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare all inter-cycle physical X/Y/Z faults with CSS signatures."""

    distance = int(metadata["distance"])
    basis = str(metadata["basis"]).upper()
    if basis not in {"X", "Z"}:
        raise ValueError(f"basis must be X or Z, got {basis!r}")

    data_coordinates = [tuple(map(int, item)) for item in metadata["data_qubit_coords"]]
    if len(data_coordinates) != distance * distance:
        raise ValueError(
            f"data coordinate count mismatch: {len(data_coordinates)} != {distance * distance}"
        )
    qubit_coordinates = {
        int(qubit): tuple(map(int, coordinate))
        for qubit, coordinate in circuit.get_final_qubit_coordinates().items()
    }
    coordinate_to_qubit = {coordinate: qubit for qubit, coordinate in qubit_coordinates.items()}
    missing_qubits = [coordinate for coordinate in data_coordinates if coordinate not in coordinate_to_qubit]
    if missing_qubits:
        raise ValueError(f"data coordinates missing from circuit: {missing_qubits}")
    data_qubits = {coordinate_to_qubit[coordinate] for coordinate in data_coordinates}

    cycle_boundaries = []
    for instruction_index in range(len(circuit)):
        instruction = circuit[instruction_index]
        if instruction.name != "Y":
            continue
        targets = {
            int(target.value)
            for target in instruction.targets_copy()
            if target.is_qubit_target
        }
        if targets == data_qubits:
            cycle_boundaries.append(instruction_index)
    expected_boundaries = int(metadata["rounds"]) - 1
    if len(cycle_boundaries) != expected_boundaries:
        raise ValueError(
            "inter-cycle boundary count mismatch: "
            f"{len(cycle_boundaries)} != {expected_boundaries}"
        )

    permutation = build_detector_permutation(circuit, metadata)
    maps = _build_stab_maps(distance, "XV")
    hx = maps["Hx_i32"].to(torch.uint8).cpu().numpy()
    hz = maps["Hz_i32"].to(torch.uint8).cpu().numpy()
    half = (distance * distance - 1) // 2
    min_difference = min(x - y for x, y in data_coordinates)
    min_sum = min(x + y for x, y in data_coordinates)
    mismatches = []
    error_names = {"X": "X_ERROR", "Y": "Y_ERROR", "Z": "Z_ERROR"}

    for pair_index, boundary_index in enumerate(cycle_boundaries):
        insertion_index = boundary_index + 1
        pair_start = half + pair_index * 2 * half
        for coordinate in data_coordinates:
            qubit = coordinate_to_qubit[coordinate]
            model_x, model_y = _google_to_xv_coordinate(
                coordinate,
                min_difference=min_difference,
                min_sum=min_sum,
            )
            row = (model_x - 1) // 2
            column = (model_y - 1) // 2
            data_index = row * distance + column
            has_local_hadamard = (row + column) % 2 == 1

            for physical_pauli, error_name in error_names.items():
                if physical_pauli == "Y":
                    css_components = {"x", "z"}
                elif physical_pauli == "X":
                    css_components = {"z" if has_local_hadamard else "x"}
                else:
                    css_components = {"x" if has_local_hadamard else "z"}

                faulty = circuit[:insertion_index]
                faulty.append(error_name, [qubit], 1.0)
                faulty += circuit[insertion_index:]
                google_detectors, observables = faulty.compile_detector_sampler().sample(
                    shots=1,
                    separate_observables=True,
                )
                actual_detectors = google_to_canonical(
                    np.asarray(google_detectors, dtype=np.uint8),
                    permutation,
                )[0]
                actual_observable = int(np.asarray(observables, dtype=np.uint8)[0, 0])

                expected_detectors = np.zeros(int(circuit.num_detectors), dtype=np.uint8)
                if "z" in css_components:
                    expected_detectors[pair_start : pair_start + half] ^= hx[:, data_index]
                if "x" in css_components:
                    expected_detectors[pair_start + half : pair_start + 2 * half] ^= hz[:, data_index]
                expected_observable = int(
                    (basis == "X" and "z" in css_components and row == 0)
                    or (basis == "Z" and "x" in css_components and column == 0)
                )
                if not np.array_equal(actual_detectors, expected_detectors) or (
                    actual_observable != expected_observable
                ):
                    mismatches.append(
                        {
                            "bulk_pair_index": pair_index,
                            "coordinate": list(coordinate),
                            "qubit": qubit,
                            "physical_pauli": physical_pauli,
                            "local_hadamard": has_local_hadamard,
                            "css_components": sorted(css_components),
                            "actual_detector_indices": np.flatnonzero(actual_detectors).tolist(),
                            "expected_detector_indices": np.flatnonzero(expected_detectors).tolist(),
                            "actual_observable": actual_observable,
                            "expected_observable": expected_observable,
                        }
                    )

    return {
        "distance": distance,
        "basis": basis,
        "bulk_pair_indices": list(range(len(cycle_boundaries))),
        "faults_checked": 3 * len(data_coordinates) * len(cycle_boundaries),
        "mismatches": mismatches,
    }



def verify_final_data_fault_equivalence(
    circuit: stim.Circuit,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare Google final-measurement fault signatures with CSS-frame signatures.

    An X immediately before the final data-qubit measurement flips exactly one
    physical measurement result.  For every data qubit this checks that the
    resulting Google detector/observable signature, after canonicalization,
    equals the CSS parity-check column and logical-string parity used by the
    predecoder.
    """

    distance = int(metadata["distance"])
    basis = str(metadata["basis"]).upper()
    if basis not in {"X", "Z"}:
        raise ValueError(f"basis must be X or Z, got {basis!r}")

    data_coordinates = [tuple(map(int, item)) for item in metadata["data_qubit_coords"]]
    if len(data_coordinates) != distance * distance:
        raise ValueError(
            f"data coordinate count mismatch: {len(data_coordinates)} != {distance * distance}"
        )
    qubit_coordinates = {
        int(qubit): tuple(map(int, coordinate))
        for qubit, coordinate in circuit.get_final_qubit_coordinates().items()
    }
    coordinate_to_qubit = {coordinate: qubit for qubit, coordinate in qubit_coordinates.items()}
    missing_qubits = [coordinate for coordinate in data_coordinates if coordinate not in coordinate_to_qubit]
    if missing_qubits:
        raise ValueError(f"data coordinates missing from circuit: {missing_qubits}")
    data_qubits = {coordinate_to_qubit[coordinate] for coordinate in data_coordinates}

    final_measurement_index = None
    for instruction_index in range(len(circuit) - 1, -1, -1):
        instruction = circuit[instruction_index]
        if instruction.name not in {"M", "MX", "MY"}:
            continue
        measured_qubits = {
            int(target.value)
            for target in instruction.targets_copy()
            if target.is_qubit_target
        }
        if measured_qubits == data_qubits:
            final_measurement_index = instruction_index
            break
    if final_measurement_index is None:
        raise ValueError("could not find the final all-data-qubit measurement")

    permutation = build_detector_permutation(circuit, metadata)
    maps = _build_stab_maps(distance, "XV")
    parity_matrix = (
        maps["Hx_i32"] if basis == "X" else maps["Hz_i32"]
    ).to(torch.uint8).cpu().numpy()
    half = (distance * distance - 1) // 2
    boundary_start = int(circuit.num_detectors) - half
    min_difference = min(x - y for x, y in data_coordinates)
    min_sum = min(x + y for x, y in data_coordinates)
    mismatches = []

    for coordinate in data_coordinates:
        qubit = coordinate_to_qubit[coordinate]
        model_x, model_y = _google_to_xv_coordinate(
            coordinate,
            min_difference=min_difference,
            min_sum=min_sum,
        )
        row = (model_x - 1) // 2
        column = (model_y - 1) // 2
        data_index = row * distance + column

        faulty = circuit[:final_measurement_index]
        faulty.append("X_ERROR", [qubit], 1.0)
        faulty += circuit[final_measurement_index:]
        google_detectors, observables = faulty.compile_detector_sampler().sample(
            shots=1,
            separate_observables=True,
        )
        actual_detectors = google_to_canonical(
            np.asarray(google_detectors, dtype=np.uint8),
            permutation,
        )[0]
        actual_observable = int(np.asarray(observables, dtype=np.uint8)[0, 0])

        expected_detectors = np.zeros(int(circuit.num_detectors), dtype=np.uint8)
        expected_detectors[boundary_start:] = parity_matrix[:, data_index] % 2
        expected_observable = int(row == 0) if basis == "X" else int(column == 0)
        if not np.array_equal(actual_detectors, expected_detectors) or (
            actual_observable != expected_observable
        ):
            mismatches.append(
                {
                    "coordinate": list(coordinate),
                    "qubit": qubit,
                    "model_data_index": data_index,
                    "actual_detector_indices": np.flatnonzero(actual_detectors).tolist(),
                    "expected_detector_indices": np.flatnonzero(expected_detectors).tolist(),
                    "actual_observable": actual_observable,
                    "expected_observable": expected_observable,
                }
            )

    return {
        "distance": distance,
        "basis": basis,
        "faults_checked": len(data_coordinates),
        "mismatches": mismatches,
    }


def wilson_interval(errors: int, shots: int, z: float = 1.96) -> tuple[float, float]:
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


def paired_error_counts(
    candidate_errors: np.ndarray,
    baseline_errors: np.ndarray,
) -> dict[str, int | float]:
    candidate = np.asarray(candidate_errors, dtype=np.bool_).reshape(-1)
    baseline = np.asarray(baseline_errors, dtype=np.bool_).reshape(-1)
    if candidate.shape != baseline.shape:
        raise ValueError(
            f"paired error shape mismatch: {candidate.shape} != {baseline.shape}"
        )
    candidate_only = int(np.count_nonzero(candidate & ~baseline))
    baseline_only = int(np.count_nonzero(~candidate & baseline))
    both = int(np.count_nonzero(candidate & baseline))
    neither = int(candidate.size - candidate_only - baseline_only - both)
    result = _paired_statistics_from_counts(
        samples=int(candidate.size),
        candidate_only=candidate_only,
        baseline_only=baseline_only,
        both=both,
        neither=neither,
    )
    # Kept for backward compatibility with existing candidate-vs-PyMatching rows.
    result["delta_ler_vs_pymatching"] = result["delta_ler"]
    return result


def _paired_statistics_from_counts(
    *,
    samples: int,
    candidate_only: int,
    baseline_only: int,
    both: int,
    neither: int,
) -> dict[str, int | float]:
    if samples < 0 or min(candidate_only, baseline_only, both, neither) < 0:
        raise ValueError("paired counts must be non-negative")
    if candidate_only + baseline_only + both + neither != samples:
        raise ValueError("paired outcome counts must sum to samples")
    delta_errors = candidate_only - baseline_only
    delta_ler = float(delta_errors / samples) if samples else float("nan")
    if samples > 1:
        difference_square_sum = candidate_only + baseline_only
        variance = max(
            0.0,
            (difference_square_sum - samples * delta_ler * delta_ler)
            / (samples - 1),
        )
        standard_error = math.sqrt(variance / samples)
    else:
        standard_error = 0.0 if samples == 1 else float("nan")
    margin = 1.96 * standard_error
    return {
        "samples": samples,
        "candidate_only_errors": candidate_only,
        "baseline_only_errors": baseline_only,
        "both_errors": both,
        "neither_errors": neither,
        "delta_logical_errors": delta_errors,
        "delta_ler": delta_ler,
        "standard_error": standard_error,
        "ci95_low": max(-1.0, delta_ler - margin),
        "ci95_high": min(1.0, delta_ler + margin),
    }


MODEL_PAIRWISE_PRIORITY = (
    "qadapt_seq_ewc",
)


def build_model_pairwise_rows(
    error_masks: Mapping[str, np.ndarray],
    case_fields: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Build pairwise rows when more than one neural model is selected."""

    known = [name for name in MODEL_PAIRWISE_PRIORITY if name in error_masks]
    extras = sorted(set(error_masks) - set(known) - {"pymatching"})
    methods = known + extras
    rows: list[dict[str, Any]] = []
    for candidate_index, candidate in enumerate(methods):
        for baseline in methods[candidate_index + 1 :]:
            rows.append(
                {
                    **dict(case_fields),
                    "candidate": candidate,
                    "baseline": baseline,
                    **paired_error_counts(
                        error_masks[candidate],
                        error_masks[baseline],
                    ),
                }
            )
            rows[-1].pop("delta_ler_vs_pymatching", None)
    return rows


def aggregate_paired_rows(
    rows: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Pool case-level paired outcomes without treating cases as independent CIs."""

    totals: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["candidate"]), str(row["baseline"]))
        entry = totals.setdefault(
            key,
            {
                "candidate": key[0],
                "baseline": key[1],
                "cases": 0,
                "samples": 0,
                "candidate_only_errors": 0,
                "baseline_only_errors": 0,
                "both_errors": 0,
                "neither_errors": 0,
            },
        )
        entry["cases"] += 1
        for field in (
            "samples",
            "candidate_only_errors",
            "baseline_only_errors",
            "both_errors",
            "neither_errors",
        ):
            entry[field] += int(row[field])

    results = []
    for entry in totals.values():
        stats = _paired_statistics_from_counts(
            samples=int(entry["samples"]),
            candidate_only=int(entry["candidate_only_errors"]),
            baseline_only=int(entry["baseline_only_errors"]),
            both=int(entry["both_errors"]),
            neither=int(entry["neither_errors"]),
        )
        results.append(
            {
                "candidate": entry["candidate"],
                "baseline": entry["baseline"],
                "cases": entry["cases"],
                **stats,
            }
        )
    return sorted(results, key=lambda row: (row["candidate"], row["baseline"]))


def aggregate_rows(rows: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    totals: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("status", "ok") != "ok":
            continue
        method = str(row["method"])
        entry = totals.setdefault(
            method,
            {"method": method, "cases": 0, "shots": 0, "logical_errors": 0},
        )
        entry["cases"] += 1
        entry["shots"] += int(row["shots"])
        entry["logical_errors"] += int(row["logical_errors"])
    for entry in totals.values():
        shots = int(entry["shots"])
        errors = int(entry["logical_errors"])
        low, high = wilson_interval(errors, shots)
        entry.update(
            ler=float(errors / shots) if shots else float("nan"),
            ci95_low=low,
            ci95_high=high,
        )
    return totals


def _read_b8(
    path: Path,
    *,
    num_detectors: int,
    num_observables: int,
) -> np.ndarray:
    data = stim.read_shot_data_file(
        path=str(path),
        format="b8",
        num_detectors=int(num_detectors),
        num_observables=int(num_observables),
    )
    return np.asarray(data, dtype=np.uint8)


def load_case_data(
    case: GoogleQECCase,
    *,
    max_shots: int = 0,
) -> tuple[stim.Circuit, stim.Circuit, dict[str, Any], np.ndarray, np.ndarray]:
    metadata = json.loads((case.path / "metadata.json").read_text())
    ideal = stim.Circuit.from_file(case.path / "circuit_ideal.stim")
    noisy = stim.Circuit.from_file(case.path / "circuit_noisy_si1000.stim")
    if ideal.num_detectors != noisy.num_detectors:
        raise ValueError(f"ideal/noisy detector mismatch in {case.path}")
    if ideal.num_observables != noisy.num_observables:
        raise ValueError(f"ideal/noisy observable mismatch in {case.path}")
    detectors = _read_b8(
        case.path / "detection_events.b8",
        num_detectors=int(ideal.num_detectors),
        num_observables=0,
    )
    observables = _read_b8(
        case.path / "obs_flips_actual.b8",
        num_detectors=0,
        num_observables=int(ideal.num_observables),
    )
    if detectors.shape[0] != observables.shape[0]:
        raise ValueError(
            f"detector/observable shot mismatch in {case.path}: "
            f"{detectors.shape[0]} != {observables.shape[0]}"
        )
    if detectors.shape[0] != int(metadata["shots"]):
        raise ValueError(
            f"metadata shot mismatch in {case.path}: "
            f"{detectors.shape[0]} != {metadata['shots']}"
        )
    limit = int(max_shots)
    if limit > 0:
        detectors = detectors[:limit]
        observables = observables[:limit]
    return ideal, noisy, metadata, detectors, observables


def build_matcher(noisy_circuit: stim.Circuit) -> pymatching.Matching:
    dem = noisy_circuit.detector_error_model(decompose_errors=True)
    return pymatching.Matching.from_detector_error_model(dem)


def _decode_batch(matcher: pymatching.Matching, detectors: np.ndarray) -> np.ndarray:
    predictions = np.asarray(
        matcher.decode_batch(np.ascontiguousarray(detectors, dtype=np.uint8)),
        dtype=np.uint8,
    )
    if predictions.ndim == 1:
        predictions = predictions.reshape(-1, 1)
    return predictions


def time_single_shot(
    matcher: pymatching.Matching,
    detectors: np.ndarray,
    *,
    rounds: int,
) -> float:
    rows = np.asarray(detectors, dtype=np.uint8)
    if len(rows) == 0:
        return float("nan")
    for row in rows[: min(20, len(rows))]:
        matcher.decode(row)
    timings = []
    for row in rows:
        start = time.perf_counter()
        matcher.decode(row)
        timings.append(time.perf_counter() - start)
    return float(np.mean(timings) * 1e6 / max(1, int(rounds)))


def _error_metrics(predictions: np.ndarray, observables: np.ndarray) -> tuple[dict[str, Any], np.ndarray]:
    predicted = np.asarray(predictions, dtype=np.uint8)
    actual = np.asarray(observables, dtype=np.uint8)
    if predicted.shape != actual.shape:
        raise ValueError(f"prediction/observable shape mismatch: {predicted.shape} != {actual.shape}")
    error_mask = np.any(predicted != actual, axis=1)
    errors = int(error_mask.sum())
    shots = int(len(error_mask))
    low, high = wilson_interval(errors, shots)
    return (
        {
            "logical_errors": errors,
            "shots": shots,
            "ler": float(errors / shots) if shots else float("nan"),
            "ci95_low": low,
            "ci95_high": high,
        },
        error_mask,
    )


def evaluate_pymatching(
    matcher: pymatching.Matching,
    detectors: np.ndarray,
    observables: np.ndarray,
    *,
    rounds: int,
    latency_shots: int,
) -> tuple[dict[str, Any], np.ndarray]:
    start = time.perf_counter()
    predictions = _decode_batch(matcher, detectors)
    batch_seconds = time.perf_counter() - start
    metrics, error_mask = _error_metrics(predictions, observables)
    latency_rows = detectors[: min(int(latency_shots), len(detectors))]
    input_density = SyndromeDensityAccumulator()
    input_density.update(detectors)
    metrics.update(
        {
            "method": "pymatching",
            "decoder": "uncorrelated_pymatching_si1000_prior",
            "batch_decode_us_per_shot": float(batch_seconds * 1e6 / max(1, len(detectors))),
            "pymatching_latency_us_per_round": time_single_shot(
                matcher,
                latency_rows,
                rounds=rounds,
            ),
            **input_density.statistics("input"),
        }
    )
    return metrics, error_mask


def build_model_cfg(
    spec: BenchmarkModel,
    case: GoogleQECCase,
    *,
    config_name: str,
    batch_size: int,
    latency_shots: int,
) -> Any:
    cfg = OmegaConf.load(config_path(config_name))
    cfg.model_id = int(spec.model_id)
    cfg.distance = int(case.distance)
    cfg.n_rounds = int(case.rounds)
    cfg.workflow.task = "inference"
    public_spec = validate_public_config(cfg)
    cfg = apply_public_defaults_and_model(cfg, public_spec)
    cfg.model_checkpoint_file = str(spec.checkpoint)
    cfg.test.meas_basis_test = str(case.basis)
    cfg.test.num_samples = int(case.shots)
    cfg.test.latency_num_samples = int(latency_shots)
    cfg.test.batch_size = int(batch_size)
    cfg.test.dataloader_num_workers = 0
    return cfg


def evaluate_predecoder(
    model: torch.nn.Module,
    cfg: Any,
    matcher: pymatching.Matching,
    google_detectors: np.ndarray,
    canonical_detectors: np.ndarray,
    observables: np.ndarray,
    canonical_to_source: np.ndarray,
    *,
    device: torch.device,
    rounds: int,
    batch_size: int,
    latency_shots: int,
) -> tuple[dict[str, Any], np.ndarray]:
    maps = _build_stab_maps(int(cfg.distance), str(cfg.data.code_rotation))
    module = PreDecoderMemoryEvalModule(model, cfg, maps, device).to(device).eval()
    predictions = []
    residual_google_rows = []
    model_seconds = 0.0
    residual_matching_seconds = 0.0

    input_density = SyndromeDensityAccumulator()
    residual_density = SyndromeDensityAccumulator()
    input_density.update(google_detectors)
    def synchronize() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    with torch.inference_mode():
        for start_index in range(0, len(canonical_detectors), int(batch_size)):
            canonical_batch = canonical_detectors[
                start_index : start_index + int(batch_size)
            ]
            tensor = torch.from_numpy(canonical_batch).to(
                device=device,
                dtype=torch.uint8,
            )
            synchronize()
            started = time.perf_counter()
            output = module(tensor)
            synchronize()
            model_seconds += time.perf_counter() - started

            pre_logical = output[:, :1].to(torch.uint8).cpu().numpy()
            canonical_residual = output[:, 1:].to(torch.uint8).cpu().numpy()
            google_residual = canonical_to_google(
                canonical_residual,
                canonical_to_source,
            )
            started = time.perf_counter()
            residual_prediction = _decode_batch(matcher, google_residual)
            residual_density.update(google_residual)
            residual_matching_seconds += time.perf_counter() - started
            predictions.append((pre_logical + residual_prediction) % 2)
            residual_google_rows.append(google_residual)

    final_predictions = np.concatenate(predictions, axis=0)
    residual_google = np.concatenate(residual_google_rows, axis=0)
    metrics, error_mask = _error_metrics(final_predictions, observables)
    latency_rows = residual_google[: min(int(latency_shots), len(residual_google))]
    residual_latency = time_single_shot(matcher, latency_rows, rounds=rounds)
    density_statistics = model_density_statistics(input_density, residual_density)
    shots = max(1, len(google_detectors))
    metrics.update(
        {
            "model_latency_us_per_shot": float(model_seconds * 1e6 / shots),
            "residual_pymatching_batch_us_per_shot": float(
                residual_matching_seconds * 1e6 / shots
            ),
            "end_to_end_batch_us_per_shot": float(
                (model_seconds + residual_matching_seconds) * 1e6 / shots
            ),
            "pymatching_latency_us_per_round": residual_latency,
            **density_statistics,
            "syndrome_reduction": float(density_statistics["density_reduction_fraction"]),
        }
    )
    return metrics, error_mask


def _case_fields(case: GoogleQECCase) -> dict[str, Any]:
    return {
        "patch": case.patch,
        "distance": case.distance,
        "basis": case.basis,
        "rounds": case.rounds,
    }


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.benchmark_root).resolve()
    selected_models = [DEFAULT_MODELS[name] for name in args.models]
    missing_checkpoints = [
        str(spec.checkpoint) for spec in selected_models if not spec.checkpoint.is_file()
    ]
    if missing_checkpoints:
        raise FileNotFoundError(f"Missing model checkpoint(s): {missing_checkpoints}")
    cases = discover_cases(
        root,
        distances=set(args.distances),
        rounds=set(args.rounds),
        bases={basis.upper() for basis in args.bases},
        patches=set(args.patches) if args.patches else None,
    )
    if not cases:
        raise RuntimeError("No Google QEC benchmark cases match the selected filters")
    if args.list_cases:
        for case in cases:
            print(case.path.relative_to(root))
        return {"cases": [str(case.path.relative_to(root)) for case in cases]}

    device = torch.device(
        args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    )
    print(f"[google-qec] device={device} cases={len(cases)}")
    model_cache: dict[str, torch.nn.Module] = {}
    rows: list[dict[str, Any]] = []
    paired_comparisons: list[dict[str, Any]] = []

    for case_index, case in enumerate(cases, start=1):
        print(
            f"[google-qec] case {case_index}/{len(cases)} "
            f"{case.patch}/{case.basis}/r{case.rounds}"
        )
        ideal, noisy, metadata, detectors, observables = load_case_data(
            case,
            max_shots=int(args.max_shots),
        )
        matcher = build_matcher(noisy)
        permutation = build_detector_permutation(ideal, metadata)
        canonical_detectors = google_to_canonical(detectors, permutation)
        baseline, baseline_errors = evaluate_pymatching(
            matcher,
            detectors,
            observables,
            rounds=case.rounds,
            latency_shots=int(args.latency_shots),
        )
        baseline.update(_case_fields(case), status="ok")
        rows.append(baseline)
        print(
            f"  pymatching: LER={baseline['ler']:.6g} "
            f"({baseline['logical_errors']}/{baseline['shots']})"
        )

        if case.rounds < 2:
            for spec in selected_models:
                rows.append(
                    {
                        **_case_fields(case),
                        "method": spec.name,
                        "status": "unsupported",
                        "reason": "predecoder requires rounds >= 2",
                        "shots": int(len(detectors)),
                    }
                )
            print("  neural predecoders skipped: rounds=1 is unsupported")
            continue

        model_error_masks: dict[str, np.ndarray] = {}
        for spec in selected_models:
            cfg = build_model_cfg(
                spec,
                case,
                config_name=args.config_name,
                batch_size=int(args.batch_size),
                latency_shots=int(args.latency_shots),
            )
            if spec.name not in model_cache:
                distributed = SimpleNamespace(rank=0, device=device)
                loaded_model = _load_model(cfg, distributed).to(device).eval()
                model_cache[spec.name] = maybe_compile_model(
                    loaded_model,
                    enabled=bool(args.torch_compile),
                    mode=str(args.torch_compile_mode),
                )
                if args.torch_compile:
                    print(f"  {spec.name}: torch.compile mode={args.torch_compile_mode}")
            metrics, error_mask = evaluate_predecoder(
                model_cache[spec.name],
                cfg,
                matcher,
                detectors,
                canonical_detectors,
                observables,
                permutation,
                device=device,
                rounds=case.rounds,
                batch_size=int(args.batch_size),
                latency_shots=int(args.latency_shots),
            )
            model_error_masks[spec.name] = error_mask
            metrics.update(
                _case_fields(case),
                method=spec.name,
                checkpoint=str(spec.checkpoint),
                status="ok",
            )
            paired_vs_pymatching = paired_error_counts(error_mask, baseline_errors)
            for field in (
                "candidate_only_errors",
                "baseline_only_errors",
                "both_errors",
                "neither_errors",
                "delta_logical_errors",
                "delta_ler_vs_pymatching",
            ):
                metrics[field] = paired_vs_pymatching[field]
            metrics.update(
                paired_samples_vs_pymatching=paired_vs_pymatching["samples"],
                paired_standard_error_vs_pymatching=paired_vs_pymatching["standard_error"],
                paired_ci95_low_vs_pymatching=paired_vs_pymatching["ci95_low"],
                paired_ci95_high_vs_pymatching=paired_vs_pymatching["ci95_high"],
            )
            baseline_latency = float(baseline["pymatching_latency_us_per_round"])
            residual_latency = float(metrics["pymatching_latency_us_per_round"])
            metrics["pymatching_speedup"] = (
                baseline_latency / residual_latency
                if residual_latency > 0 and math.isfinite(residual_latency)
                else float("nan")
            )
            rows.append(metrics)
            print(
                f"  {spec.name}: LER={metrics['ler']:.6g} "
                f"delta={metrics['delta_ler_vs_pymatching']:+.6g} "
                f"syndrome_reduction={metrics['syndrome_reduction']:.3f}"
            )

        paired_comparisons.extend(
            build_model_pairwise_rows(model_error_masks, _case_fields(case))
        )
    payload = {
        "schema_version": 2,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "benchmark_root": str(root),
        "decoder_prior": "Google circuit_noisy_si1000.stim DEM",
        "detector_mapping": "Google physical order <-> repository XV canonical order",
        "device": str(device),
        "filters": {
            "distances": list(args.distances),
            "rounds": list(args.rounds),
            "bases": list(args.bases),
            "patches": list(args.patches or []),
            "max_shots": int(args.max_shots),
            "batch_size": int(args.batch_size),
            "latency_shots": int(args.latency_shots),
            "torch_compile": bool(args.torch_compile),
            "torch_compile_mode": str(args.torch_compile_mode),
        },
        "models": {
            spec.name: {
                "model_id": spec.model_id,
                "checkpoint": str(spec.checkpoint),
            }
            for spec in selected_models
        },
        "rows": rows,
        "aggregate": aggregate_rows(rows),
        "paired_comparisons": paired_comparisons,
        "paired_aggregate": aggregate_paired_rows(paired_comparisons),
    }
    return payload



def merge_benchmark_payloads(
    payloads: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Merge disjoint benchmark shards and recompute all pooled statistics."""

    if not payloads:
        raise ValueError("at least one benchmark payload is required")
    reference = payloads[0]
    for index, payload in enumerate(payloads):
        if int(payload.get("schema_version", 0)) != 2:
            raise ValueError(f"benchmark shard {index} is not schema_version=2")
        for field in (
            "benchmark_root",
            "decoder_prior",
            "detector_mapping",
            "models",
        ):
            if payload.get(field) != reference.get(field):
                raise ValueError(f"benchmark shard {index} disagrees on {field}")

    rows = [dict(row) for payload in payloads for row in payload.get("rows", [])]
    paired = [
        dict(row)
        for payload in payloads
        for row in payload.get("paired_comparisons", [])
    ]
    row_keys = [
        (
            str(row.get("patch")),
            int(row.get("distance", 0)),
            str(row.get("basis")),
            int(row.get("rounds", 0)),
            str(row.get("method")),
        )
        for row in rows
    ]
    if len(row_keys) != len(set(row_keys)):
        raise ValueError("benchmark shards contain duplicate case/method rows")
    paired_keys = [
        (
            str(row.get("patch")),
            int(row.get("distance", 0)),
            str(row.get("basis")),
            int(row.get("rounds", 0)),
            str(row.get("candidate")),
            str(row.get("baseline")),
        )
        for row in paired
    ]
    if len(paired_keys) != len(set(paired_keys)):
        raise ValueError("benchmark shards contain duplicate paired comparisons")

    rows.sort(
        key=lambda row: (
            int(row.get("distance", 0)),
            str(row.get("patch")),
            str(row.get("basis")),
            int(row.get("rounds", 0)),
            str(row.get("method")),
        )
    )
    paired.sort(
        key=lambda row: (
            int(row.get("distance", 0)),
            str(row.get("patch")),
            str(row.get("basis")),
            int(row.get("rounds", 0)),
            str(row.get("candidate")),
            str(row.get("baseline")),
        )
    )
    max_shots = {
        int(payload.get("filters", {}).get("max_shots", 0)) for payload in payloads
    }
    if len(max_shots) != 1:
        raise ValueError("benchmark shards disagree on max_shots")
    execution_filters = {}
    for field in (
        "batch_size",
        "latency_shots",
        "torch_compile",
        "torch_compile_mode",
    ):
        values = {payload.get("filters", {}).get(field) for payload in payloads}
        if len(values) != 1:
            raise ValueError(f"benchmark shards disagree on {field}")
        execution_filters[field] = values.pop()
    return {
        "schema_version": 2,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "benchmark_root": reference["benchmark_root"],
        "decoder_prior": reference["decoder_prior"],
        "detector_mapping": reference["detector_mapping"],
        "device": "merged_shards",
        "filters": {
            "distances": sorted({int(row["distance"]) for row in rows}),
            "rounds": sorted({int(row["rounds"]) for row in rows}),
            "bases": sorted({str(row["basis"]) for row in rows}),
            "patches": sorted({str(row["patch"]) for row in rows}),
            "max_shots": max_shots.pop(),
            **execution_filters,
        },
        "models": reference["models"],
        "rows": rows,
        "aggregate": aggregate_rows(rows),
        "paired_comparisons": paired,
        "paired_aggregate": aggregate_paired_rows(paired),
    }

def write_results(payload: Mapping[str, Any], output_path: Path) -> tuple[Path, Path]:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    csv_path = output_path.with_suffix(".csv")
    rows = list(payload.get("rows", []))
    fieldnames = sorted({str(key) for row in rows for key in row})
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    paired_rows = list(payload.get("paired_comparisons", []))
    paired_csv_path = output_path.with_name(
        f"{output_path.stem}_paired.csv"
    )
    paired_fields = sorted({str(key) for row in paired_rows for key in row})
    with paired_csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=paired_fields)
        if paired_fields:
            writer.writeheader()
            writer.writerows(paired_rows)
    return output_path, csv_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate PyMatching and QAdapt sequential + EWC on Google QEC "
            "hardware samples."
        )
    )
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--distances", nargs="+", type=int, default=[3, 5, 7])
    parser.add_argument(
        "--rounds",
        nargs="+",
        type=int,
        default=[13],
        help="Google cycle counts. The default r13 is the calibration slice.",
    )
    parser.add_argument("--bases", nargs="+", choices=("X", "Z"), default=["X", "Z"])
    parser.add_argument(
        "--patches",
        nargs="+",
        default=None,
        help="Optional exact patch directory names, for example d7_at_q6_7.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=tuple(DEFAULT_MODELS),
        default=list(DEFAULT_MODELS),
    )
    parser.add_argument("--config-name", default="examples/qadapt/config_qadapt_t0_base")
    parser.add_argument("--max-shots", type=int, default=0, help="0 uses all shots.")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--latency-shots", type=int, default=512)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--torch-compile",
        action="store_true",
        help="Compile each neural model once with dynamic input shapes.",
    )
    parser.add_argument(
        "--torch-compile-mode",
        choices=(
            "default",
            "reduce-overhead",
            "max-autotune",
            "max-autotune-no-cudagraphs",
        ),
        default="default",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--merge-inputs",
        nargs="+",
        type=Path,
        default=None,
        help="Merge disjoint schema-v2 benchmark JSON shards instead of running inference.",
    )
    parser.add_argument("--list-cases", action="store_true")
    args = parser.parse_args(argv)
    if args.max_shots < 0:
        parser.error("--max-shots must be >= 0")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.latency_shots <= 0:
        parser.error("--latency-shots must be positive")
    if args.output is None:
        args.output = Path(args.benchmark_root) / "ising_decoder_results/results.json"
    if args.merge_inputs and args.list_cases:
        parser.error("--merge-inputs cannot be combined with --list-cases")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.merge_inputs:
        payload = merge_benchmark_payloads(
            [json.loads(Path(path).read_text(encoding="utf-8")) for path in args.merge_inputs]
        )
        payload["merged_inputs"] = [str(Path(path).resolve()) for path in args.merge_inputs]
        print(f"[google-qec] merged {len(args.merge_inputs)} shards")
    else:
        payload = run_benchmark(args)
    if args.list_cases:
        return 0
    json_path, csv_path = write_results(payload, args.output)
    print(f"[google-qec] JSON: {json_path}")
    print(f"[google-qec] CSV:  {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
