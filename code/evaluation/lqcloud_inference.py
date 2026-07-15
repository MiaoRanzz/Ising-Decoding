# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Decode LQCloud hardware measurements with the NVIDIA pre-decoder pipeline."""

from __future__ import annotations

import ast
import importlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Sequence


_REPO_ROOT = Path(__file__).resolve().parents[2]
_ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def _candidate_memories(value: Any) -> List[List[str]]:
    if isinstance(value, dict):
        value = value.get("memory")
    if not isinstance(value, (list, tuple)) or not value:
        return []
    if all(isinstance(item, str) and set(item.strip()) <= {"0", "1"} for item in value):
        return [[item.strip() for item in value]]
    return []


def parse_measurement_log(
    path: str | Path,
    *,
    expected_width: int,
    bit_order: str = "as_returned",
    max_shots: int = 0,
) -> List[List[int]]:
    """Extract the largest printed ``get_memory()`` list from an LQCloud log."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"LQCloud measurement file does not exist: {path}")

    text = _ANSI_ESCAPE.sub("", path.read_text(encoding="utf-8", errors="replace"))
    candidates: List[List[str]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or not (line.startswith("[") or line.startswith("{")):
            continue
        try:
            candidates.extend(_candidate_memories(ast.literal_eval(line)))
        except (SyntaxError, ValueError):
            continue

    if not candidates:
        raise ValueError(
            f"No printed LQCloud memory list was found in {path}. "
            "Expected a line like ['0101...', '1100...']."
        )

    memory = max(candidates, key=len)
    mode = str(bit_order).strip().lower()
    if mode not in {"as_returned", "reverse"}:
        raise ValueError("lqcloud.bit_order must be 'as_returned' or 'reverse'")

    parsed: List[List[int]] = []
    for shot_index, bit_string in enumerate(memory):
        if len(bit_string) != int(expected_width):
            raise ValueError(
                f"Shot {shot_index} in {path} has {len(bit_string)} bits; "
                f"expected {expected_width}."
            )
        if mode == "reverse":
            bit_string = bit_string[::-1]
        parsed.append([int(bit) for bit in bit_string])

    if int(max_shots) > 0:
        parsed = parsed[:int(max_shots)]
    if not parsed:
        raise ValueError(f"No hardware shots remain after applying max_shots={max_shots}")
    return parsed


def _import_lqcloud_circuits():
    repo_path = str(_REPO_ROOT)
    added = repo_path not in sys.path
    if added:
        sys.path.insert(0, repo_path)
    try:
        return importlib.import_module("my_file.lqcloud.lqcloud_d3_surface_code.circuits")
    finally:
        if added:
            sys.path.remove(repo_path)


def _row_supports(matrix: Any) -> List[frozenset[int]]:
    return [
        frozenset(index for index, value in enumerate(row) if int(value) != 0)
        for row in matrix
    ]


def _ordered_lq_qubits(
    nvidia_supports: Sequence[frozenset[int]],
    lq_qubits: Iterable[int],
    lq_supports: dict[int, Iterable[int]],
    *,
    stabilizer_type: str,
    code_rotation: str,
) -> List[int]:
    available = {int(q): frozenset(int(v) for v in lq_supports[int(q)]) for q in lq_qubits}
    ordered: List[int] = []
    for row_index, support in enumerate(nvidia_supports):
        matches = [q for q, candidate in available.items() if candidate == support]
        if len(matches) != 1:
            raise ValueError(
                f"LQCloud {stabilizer_type}-stabilizer support {sorted(support)} "
                f"(NVIDIA row {row_index}) has {len(matches)} matches for rotation "
                f"{code_rotation}. The lqcloud d=3 circuit is expected to use XH/O2."
            )
        ordered.append(matches[0])
    if len(set(ordered)) != len(available):
        raise ValueError(f"Incomplete {stabilizer_type}-stabilizer mapping")
    return ordered


def build_model_detector_permutation(
    *,
    distance: int,
    n_rounds: int,
    basis: str,
    code_rotation: str,
    lq_circuits: Any,
    surface_code: Any,
) -> List[int]:
    """Map LQCloud detector emission order to NVIDIA's model input order.

    The returned permutation satisfies ``model_dets = lq_dets[:, permutation]``.
    NVIDIA expects each bulk time step as X-stabilizers followed by
    Z-stabilizers; LQCloud emits ancillas in physical qubit-index order.
    """
    D = int(distance)
    T = int(n_rounds)
    basis = str(basis).upper()
    if basis not in {"X", "Z"}:
        raise ValueError(f"Unsupported memory basis: {basis!r}")

    qubit_coords = lq_circuits.generate_qubit_coords(D, mirror=True)
    lq_z, lq_x = lq_circuits.generate_z_x_measure_qubits(D, mirror=True)
    _, lq_supports = lq_circuits.generate_cz_pattern_and_stabilizer_qubits(
        qubit_coords=qubit_coords,
        z_measure_qubits=lq_z,
        x_measure_qubits=lq_x,
        mirror=True,
    )
    lq_x = [int(q) for q in lq_x]
    lq_z = [int(q) for q in lq_z]
    measure_qubits = list(range(D * D, 2 * D * D - 1))

    ordered_x = _ordered_lq_qubits(
        _row_supports(surface_code.hx),
        lq_x,
        lq_supports,
        stabilizer_type="X",
        code_rotation=code_rotation,
    )
    ordered_z = _ordered_lq_qubits(
        _row_supports(surface_code.hz),
        lq_z,
        lq_supports,
        stabilizer_type="Z",
        code_rotation=code_rotation,
    )
    half = len(ordered_x)
    if len(ordered_z) != half or len(measure_qubits) != 2 * half:
        raise ValueError("Unexpected surface-code stabilizer count")

    boundary_lq = lq_x if basis == "X" else lq_z
    boundary_model = ordered_x if basis == "X" else ordered_z
    boundary_positions = [boundary_lq.index(q) for q in boundary_model]

    permutation = list(boundary_positions)
    for round_index in range(T - 1):
        block_start = half + round_index * 2 * half
        for q in ordered_x + ordered_z:
            permutation.append(block_start + measure_qubits.index(q))
    final_start = half + (T - 1) * 2 * half
    permutation.extend(final_start + position for position in boundary_positions)

    expected = 2 * T * half
    if len(permutation) != expected or sorted(permutation) != list(range(expected)):
        raise ValueError("Detector permutation is not a complete one-to-one mapping")
    return permutation


def _resolve_repo_path(value: str | Path) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else (_REPO_ROOT / path).resolve()


@dataclass
class HardwareSamples:
    circuit: Any
    measurements: Any
    lq_dets: Any
    model_dets: Any
    observables: Any
    model_to_lq: List[int]
    basis: str


def load_lqcloud_hardware_samples(cfg) -> HardwareSamples:
    import numpy as np
    from qec.surface_code.memory_circuit import SurfaceCode

    D = int(cfg.distance)
    T = int(cfg.n_rounds)
    if D != 3:
        raise ValueError(f"The current LQCloud hardware layout supports distance=3, got {D}")

    lq_cfg = cfg.lqcloud
    circuit_type = str(getattr(lq_cfg, "circuit_type", "memory_z")).strip().lower()
    if circuit_type not in {"memory_x", "memory_z"}:
        raise ValueError("lqcloud.circuit_type must be memory_x or memory_z")
    basis = circuit_type[-1].upper()
    reset = bool(getattr(lq_cfg, "reset", False))
    mirror = bool(getattr(lq_cfg, "mirror", True))
    if reset or not mirror:
        raise ValueError("The QZ01 integration currently requires reset=false and mirror=true")

    initial_state = [int(v) for v in getattr(lq_cfg, "initial_state", [0] * (D * D))]
    if len(initial_state) != D * D or any(v not in (0, 1) for v in initial_state):
        raise ValueError(f"lqcloud.initial_state must contain exactly {D * D} binary values")

    expected_width = T * (D * D - 1) + D * D
    measurements = np.asarray(
        parse_measurement_log(
            _resolve_repo_path(lq_cfg.measurement_file),
            expected_width=expected_width,
            bit_order=str(getattr(lq_cfg, "bit_order", "as_returned")),
            max_shots=int(getattr(lq_cfg, "max_shots", 0)),
        ),
        dtype=np.bool_,
    )

    lq_circuits = _import_lqcloud_circuits()
    noise_cfg = getattr(lq_cfg, "stim_noise", None)

    def noise_value(name: str, default: float) -> float:
        return float(getattr(noise_cfg, name, default)) if noise_cfg is not None else default

    circuit = lq_circuits.build_stim_circuit(
        distance=D,
        ini_state=initial_state,
        cycle=T,
        circuit_type=circuit_type,
        reset=reset,
        mirror=mirror,
        sq_error=noise_value("sq_error", 0.0011),
        cz_error=noise_value("cz_error", 0.0064),
        measure_error=noise_value("measure_error", 0.015),
        idle_z_error=noise_value("idle_z_error", 0.0),
        idle_dep_error=noise_value("idle_dep_error", 0.019),
    )
    lq_dets, observables = circuit.compile_m2d_converter().convert(
        measurements=measurements,
        separate_observables=True,
    )
    lq_dets = np.asarray(lq_dets, dtype=np.uint8)
    observables = np.asarray(observables, dtype=np.uint8)
    if observables.ndim == 1:
        observables = observables.reshape(-1, 1)
    if observables.ndim != 2 or observables.shape[1] != 1:
        raise ValueError(
            f"Expected one logical observable per shot, got shape {observables.shape}"
        )

    rotation = str(cfg.data.code_rotation).upper()
    surface_code = SurfaceCode(
        D,
        first_bulk_syndrome_type=rotation[0],
        rotated_type=rotation[1],
    )
    model_to_lq = build_model_detector_permutation(
        distance=D,
        n_rounds=T,
        basis=basis,
        code_rotation=rotation,
        lq_circuits=lq_circuits,
        surface_code=surface_code,
    )
    if lq_dets.shape[1] != len(model_to_lq):
        raise ValueError(
            f"LQCloud circuit produced {lq_dets.shape[1]} detectors; "
            f"the NVIDIA model expects {len(model_to_lq)}"
        )
    model_dets = lq_dets[:, model_to_lq]
    return HardwareSamples(
        circuit=circuit,
        measurements=measurements,
        lq_dets=lq_dets,
        model_dets=model_dets,
        observables=observables,
        model_to_lq=model_to_lq,
        basis=basis,
    )


def _count_observable_mismatches(predictions, observables) -> int:
    import numpy as np

    predictions = np.asarray(predictions, dtype=np.uint8).reshape(observables.shape)
    return int(np.any(predictions != observables, axis=1).sum())


def run_inference_modified(model, device, dist, cfg):
    """Run Ising pre-decoding plus PyMatching on one LQCloud hardware log."""
    import numpy as np
    import pymatching
    import torch

    from evaluation.logical_error_rate import PreDecoderMemoryEvalModule, _build_stab_maps

    if int(getattr(dist, "world_size", 1)) != 1:
        raise ValueError("integrate_to_nvidia must run with one process/GPU")
    if str(getattr(cfg.test, "decode_mode", "")).strip().lower() == "pymatching_only":
        raise ValueError("integrate_to_nvidia requires the Ising pre-decoder model")

    samples = load_lqcloud_hardware_samples(cfg)
    cfg.test.meas_basis_test = samples.basis
    cfg.test.n_rounds = int(cfg.n_rounds)
    cfg.test.num_samples = int(samples.model_dets.shape[0])

    dem = samples.circuit.detector_error_model(
        decompose_errors=True,
        approximate_disjoint_errors=True,
    )
    matcher = pymatching.Matching.from_detector_error_model(dem)
    baseline_predictions = matcher.decode_batch(samples.lq_dets)
    baseline_errors = _count_observable_mismatches(
        baseline_predictions,
        samples.observables,
    )

    model.eval()
    try:
        model = model.to(memory_format=torch.channels_last_3d)
    except Exception:
        pass
    maps = _build_stab_maps(int(cfg.distance), str(cfg.data.code_rotation))
    pipeline = PreDecoderMemoryEvalModule(model, cfg, maps, device).to(device)
    pipeline.eval()

    batch_size = max(int(getattr(cfg.lqcloud, "batch_size", 256)), 1)
    final_predictions = []
    residual_lq_rows = []
    with torch.inference_mode():
        for start in range(0, samples.model_dets.shape[0], batch_size):
            stop = min(start + batch_size, samples.model_dets.shape[0])
            dets = torch.as_tensor(
                samples.model_dets[start:stop],
                dtype=torch.uint8,
                device=device,
            )
            output = pipeline(dets)
            pre_logical = output[:, 0].to("cpu").numpy().astype(np.uint8, copy=False)
            residual_model = output[:, 1:].to("cpu").numpy().astype(np.uint8, copy=False)

            residual_lq = np.empty_like(residual_model)
            residual_lq[:, samples.model_to_lq] = residual_model
            residual_prediction = np.asarray(
                matcher.decode_batch(residual_lq),
                dtype=np.uint8,
            ).reshape(-1)
            final_predictions.append((pre_logical.reshape(-1) ^ residual_prediction).reshape(-1, 1))
            residual_lq_rows.append(residual_lq)

    final_predictions_arr = np.concatenate(final_predictions, axis=0)
    residual_lq_arr = np.concatenate(residual_lq_rows, axis=0)
    corrected_errors = _count_observable_mismatches(
        final_predictions_arr,
        samples.observables,
    )
    raw_errors = int(np.any(samples.observables != 0, axis=1).sum())
    shots = int(samples.observables.shape[0])

    result = {
        "shots": shots,
        "basis": samples.basis,
        "distance": int(cfg.distance),
        "n_rounds": int(cfg.n_rounds),
        "raw_logical_errors": raw_errors,
        "raw_logical_error_rate": raw_errors / shots,
        "pymatching_logical_errors": baseline_errors,
        "pymatching_logical_error_rate": baseline_errors / shots,
        "ising_plus_pymatching_logical_errors": corrected_errors,
        "ising_plus_pymatching_logical_error_rate": corrected_errors / shots,
        "input_detector_density": float(samples.lq_dets.mean()),
        "residual_detector_density": float(residual_lq_arr.mean()),
    }

    if dist.rank == 0:
        print(
            f"\n[LQCloud Hardware] d={result['distance']}, rounds={result['n_rounds']}, "
            f"basis={result['basis']}, shots={shots}"
        )
        print(f"  {'Decoder':<30}{'Logical errors':>16}{'LER':>12}")
        print(
            f"  {'Raw (no decoder)':<30}{raw_errors:>16}"
            f"{result['raw_logical_error_rate']:>12.6f}"
        )
        print(
            f"  {'PyMatching':<30}{baseline_errors:>16}"
            f"{result['pymatching_logical_error_rate']:>12.6f}"
        )
        print(
            f"  {'Ising pre-decoder + PyMatching':<30}{corrected_errors:>16}"
            f"{result['ising_plus_pymatching_logical_error_rate']:>12.6f}"
        )
        print(
            f"  Detector density: {result['input_detector_density']:.6f} -> "
            f"{result['residual_detector_density']:.6f}"
        )
        print("[LQCloud Summary] " + json.dumps(result, sort_keys=True))
    return result


__all__ = [
    "HardwareSamples",
    "build_model_detector_permutation",
    "load_lqcloud_hardware_samples",
    "parse_measurement_log",
    "run_inference_modified",
]
