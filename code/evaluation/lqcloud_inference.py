# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Decode LQCloud hardware measurements with the NVIDIA pre-decoder pipeline."""

from __future__ import annotations

import importlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Sequence


_REPO_ROOT = Path(__file__).resolve().parents[2]
def normalize_measurement_memory(
    memory: Sequence[str],
    *,
    expected_width: int,
    bit_order: str = "as_returned",
    max_shots: int = 0,
) -> List[List[int]]:
    """Validate and convert ``Result.get_memory()`` strings to binary rows."""
    if not isinstance(memory, (list, tuple)) or not memory:
        raise ValueError("LQCloud result.get_memory() returned no measurement shots")

    mode = str(bit_order).strip().lower()
    if mode not in {"as_returned", "reverse"}:
        raise ValueError("lqcloud.bit_order must be 'as_returned' or 'reverse'")

    parsed: List[List[int]] = []
    for shot_index, raw_bit_string in enumerate(memory):
        if not isinstance(raw_bit_string, str):
            raise ValueError(
                f"Shot {shot_index} is {type(raw_bit_string).__name__}, expected a bit string"
            )
        bit_string = raw_bit_string.strip()
        if len(bit_string) != int(expected_width):
            raise ValueError(
                f"Shot {shot_index} has {len(bit_string)} bits; expected {expected_width}."
            )
        if set(bit_string) - {"0", "1"}:
            raise ValueError(f"Shot {shot_index} contains values other than 0 and 1")
        if mode == "reverse":
            bit_string = bit_string[::-1]
        parsed.append([int(bit) for bit in bit_string])

    if int(max_shots) > 0:
        parsed = parsed[:int(max_shots)]
    if not parsed:
        raise ValueError(f"No hardware shots remain after applying max_shots={max_shots}")
    return parsed


def load_measurement_file(
    path: str | Path,
    *,
    expected_width: int,
    bit_order: str = "as_returned",
    max_shots: int = 0,
) -> List[List[int]]:
    """Load a JSON file containing the list returned by ``result.get_memory()``."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"LQCloud measurement file does not exist: {path}")

    try:
        with path.open("r", encoding="utf-8") as file:
            memory = json.load(file)
        return normalize_measurement_memory(
            memory,
            expected_width=expected_width,
            bit_order=bit_order,
            max_shots=max_shots,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"Invalid measurement data in {path}: {exc}") from exc


def _measurement_files(source: str | Path) -> List[Path]:
    """Return the JSON measurement files represented by a file or directory source."""
    source = Path(source)
    if source.is_file():
        return [source]
    if source.is_dir():
        files = sorted(
            path for path in source.iterdir()
            if path.is_file() and path.suffix.lower() == ".json"
        )
        if files:
            return files
        raise ValueError(f"LQCloud measurement directory contains no JSON files: {source}")
    raise FileNotFoundError(f"LQCloud measurement source does not exist: {source}")


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


def _import_lqcloud_main():
    repo_path = str(_REPO_ROOT)
    added = repo_path not in sys.path
    if added:
        sys.path.insert(0, repo_path)
    try:
        return importlib.import_module("my_file.lqcloud.lqcloud_d3_surface_code.main")
    finally:
        if added:
            sys.path.remove(repo_path)


def collect_hardware_measurement_memory(
    cfg,
    *,
    initial_state: Sequence[int],
    hardware_runner=None,
) -> Sequence[str]:
    """Execute the configured LQCloud job and return ``result.get_memory()``."""
    lq_cfg = cfg.lqcloud
    if hardware_runner is None:
        lq_main = _import_lqcloud_main()
        hardware_runner = lq_main.run_hardware_experiment
    result = hardware_runner(
        distance=int(cfg.distance),
        ini_state=initial_state,
        cycle=int(cfg.n_rounds),
        shots=int(getattr(lq_cfg, "shots", 100)),
        backend_name=str(getattr(lq_cfg, "backend_name", "QZ01-surface_code")),
        circuit_type=str(getattr(lq_cfg, "circuit_type", "memory_z")),
    )
    memory = result.get_memory()
    if memory is None:
        raise ValueError("LQCloud result.get_memory() returned None")
    return memory


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


def load_lqcloud_hardware_samples(
    cfg,
    *,
    measurement_memory: Sequence[str] | None = None,
    measurement_path: str | Path | None = None,
    max_shots: int | None = None,
) -> HardwareSamples:
    import numpy as np
    from qec.surface_code.memory_circuit import SurfaceCode

    D = int(cfg.distance)
    T = int(cfg.n_rounds)

    lq_cfg = cfg.lqcloud
    circuit_type = str(getattr(lq_cfg, "circuit_type", "memory_z")).strip().lower()
    if circuit_type not in {"memory_x", "memory_z"}:
        raise ValueError("lqcloud.circuit_type must be memory_x or memory_z")
    basis = circuit_type[-1].upper()
    reset = bool(getattr(lq_cfg, "reset", False))
    mirror = bool(getattr(lq_cfg, "mirror", True))
    if reset or not mirror:
        raise ValueError("The QZ01 integration currently requires reset=false and mirror=true")

    configured_initial_state = getattr(lq_cfg, "initial_state", None)
    if configured_initial_state is None:
        initial_state = [0] * (D * D)
    else:
        initial_state = [int(v) for v in configured_initial_state]

    if len(initial_state) != D * D or any(v not in (0, 1) for v in initial_state):
        raise ValueError(
            f"lqcloud.initial_state must contain exactly {D * D} binary values"
        )

    expected_width = T * (D * D - 1) + D * D
    source = str(getattr(lq_cfg, "source", "hardware")).strip().lower()
    shot_limit = int(getattr(lq_cfg, "max_shots", 0)) if max_shots is None else int(max_shots)
    measurements_list = None
    if measurement_memory is None:
        if source == "hardware":
            measurement_memory = collect_hardware_measurement_memory(
                cfg,
                initial_state=initial_state,
            )
        elif source == "file":
            path = _resolve_repo_path(
                measurement_path if measurement_path is not None else lq_cfg.measurement_source
            )
            if not path.is_file():
                raise ValueError(
                    f"Expected one JSON measurement file, got {path}. "
                    "Directories are processed one file at a time by run_inference_modified."
                )
            measurements_list = load_measurement_file(
                path,
                expected_width=expected_width,
                bit_order=str(getattr(lq_cfg, "bit_order", "as_returned")),
                max_shots=shot_limit,
            )
            measurement_memory = None
        else:
            raise ValueError("lqcloud.source must be 'hardware' or 'file'")
    if measurement_memory is not None:
        measurements_list = normalize_measurement_memory(
            measurement_memory,
            expected_width=expected_width,
            bit_order=str(getattr(lq_cfg, "bit_order", "as_returned")),
            max_shots=shot_limit,
        )
    if measurements_list is None:
        raise RuntimeError("LQCloud measurement source produced no data")
    measurements = np.asarray(measurements_list, dtype=np.bool_)

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


def _time_lqcloud_pymatching_latency(
    matcher,
    baseline_syndromes,
    residual_syndromes,
    *,
    n_rounds: int,
    num_samples: int,
    warmup_iterations: int,
    timer,
) -> tuple[float, float, int]:
    """Time the available hardware shots using NVIDIA's latency definition."""
    available = min(len(baseline_syndromes), len(residual_syndromes))
    sample_count = min(max(int(num_samples), 0), available)
    if sample_count == 0:
        return float("nan"), float("nan"), 0

    baseline_us, predecoder_us = timer(
        matcher=matcher,
        baseline_syndromes=baseline_syndromes[:sample_count],
        residual_syndromes=residual_syndromes[:sample_count],
        n_rounds=int(n_rounds),
        warmup_iterations=max(int(warmup_iterations), 0),
    )
    return float(baseline_us), float(predecoder_us), sample_count


def _decode_lqcloud_samples(
    samples: HardwareSamples,
    pipeline,
    matcher,
    unionfind_decoder,
    *,
    device,
    batch_size: int,
    n_rounds: int,
    latency_num_samples: int | None,
    latency_warmup_iterations: int,
    timer,
) -> dict:
    """Decode one measurement file without retaining its batch outputs."""
    import numpy as np
    import torch

    shots = int(samples.observables.shape[0])
    baseline_predictions = matcher.decode_batch(samples.lq_dets)
    baseline_errors = _count_observable_mismatches(baseline_predictions, samples.observables)
    unionfind_baseline_predictions = unionfind_decoder.decode_batch(samples.lq_dets)
    unionfind_baseline_errors = _count_observable_mismatches(
        unionfind_baseline_predictions,
        samples.observables,
    )
    raw_errors = int(np.any(samples.observables != 0, axis=1).sum())

    corrected_errors = 0
    unionfind_corrected_errors = 0
    residual_detector_ones = 0
    requested_latency_samples = shots if latency_num_samples is None else int(latency_num_samples)
    latency_limit = min(max(requested_latency_samples, 0), shots)
    residual_latency_rows = []
    latency_rows = 0
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
            ).reshape(-1, 1)
            unionfind_residual_prediction = np.asarray(
                unionfind_decoder.decode_batch(residual_lq),
                dtype=np.uint8,
            ).reshape(-1, 1)
            final_prediction = pre_logical.reshape(-1, 1) ^ residual_prediction
            unionfind_final_prediction = pre_logical.reshape(-1, 1) ^ unionfind_residual_prediction
            observables = samples.observables[start:stop]
            corrected_errors += _count_observable_mismatches(final_prediction, observables)
            unionfind_corrected_errors += _count_observable_mismatches(
                unionfind_final_prediction,
                observables,
            )
            residual_detector_ones += int(residual_lq.sum())

            remaining_latency_rows = latency_limit - latency_rows
            if remaining_latency_rows > 0:
                take = min(remaining_latency_rows, len(residual_lq))
                residual_latency_rows.append(residual_lq[:take].copy())
                latency_rows += take

    if residual_latency_rows:
        residual_lq_arr = np.concatenate(residual_latency_rows, axis=0)
    else:
        residual_lq_arr = np.empty((0, samples.lq_dets.shape[1]), dtype=np.uint8)
    baseline_us_per_round, predecoder_us_per_round, latency_samples = (
        _time_lqcloud_pymatching_latency(
            matcher,
            samples.lq_dets,
            residual_lq_arr,
            n_rounds=n_rounds,
            num_samples=requested_latency_samples,
            warmup_iterations=latency_warmup_iterations,
            timer=timer,
        )
    )
    unionfind_baseline_us_per_round, unionfind_predecoder_us_per_round, _ = (
        _time_lqcloud_pymatching_latency(
            unionfind_decoder,
            samples.lq_dets,
            residual_lq_arr,
            n_rounds=n_rounds,
            num_samples=requested_latency_samples,
            warmup_iterations=latency_warmup_iterations,
            timer=timer,
        )
    )
    return {
        "shots": shots,
        "basis": samples.basis,
        "raw_logical_errors": raw_errors,
        "raw_logical_error_rate": raw_errors / shots,
        "pymatching_logical_errors": baseline_errors,
        "pymatching_logical_error_rate": baseline_errors / shots,
        "ising_plus_pymatching_logical_errors": corrected_errors,
        "ising_plus_pymatching_logical_error_rate": corrected_errors / shots,
        "unionfind_logical_errors": unionfind_baseline_errors,
        "unionfind_logical_error_rate": unionfind_baseline_errors / shots,
        "ising_plus_unionfind_logical_errors": unionfind_corrected_errors,
        "ising_plus_unionfind_logical_error_rate": unionfind_corrected_errors / shots,
        "input_detector_density": float(samples.lq_dets.mean()),
        "residual_detector_density": residual_detector_ones / samples.lq_dets.size,
        "latency_samples": latency_samples,
        "pymatch latency (baseline µs/round)": baseline_us_per_round,
        "pymatch latency (after predecoder µs/round)": predecoder_us_per_round,
        "unionfind latency (baseline µs/round)": unionfind_baseline_us_per_round,
        "unionfind latency (after predecoder µs/round)": unionfind_predecoder_us_per_round,
    }


def _aggregate_lqcloud_file_results(file_results: List[dict], *, distance: int, n_rounds: int) -> dict:
    """Combine per-file metrics without treating unequal shot counts equally."""
    if not file_results:
        raise RuntimeError("LQCloud measurement source produced no data")

    shots = sum(int(file_result["shots"]) for file_result in file_results)

    def total(name: str) -> int:
        return sum(int(file_result[name]) for file_result in file_results)

    def weighted_by_shots(name: str) -> float:
        return sum(
            float(file_result[name]) * int(file_result["shots"])
            for file_result in file_results
        ) / shots

    latency_samples = sum(int(file_result["latency_samples"]) for file_result in file_results)

    def weighted_by_latency_samples(name: str) -> float:
        if latency_samples == 0:
            return float("nan")
        return sum(
            float(file_result[name]) * int(file_result["latency_samples"])
            for file_result in file_results
            if int(file_result["latency_samples"]) > 0
        ) / latency_samples

    raw_errors = total("raw_logical_errors")
    pymatching_errors = total("pymatching_logical_errors")
    ising_pymatching_errors = total("ising_plus_pymatching_logical_errors")
    unionfind_errors = total("unionfind_logical_errors")
    ising_unionfind_errors = total("ising_plus_unionfind_logical_errors")
    return {
        "shots": shots,
        "files": len(file_results),
        "basis": file_results[0]["basis"],
        "distance": distance,
        "n_rounds": n_rounds,
        "raw_logical_errors": raw_errors,
        "raw_logical_error_rate": raw_errors / shots,
        "pymatching_logical_errors": pymatching_errors,
        "pymatching_logical_error_rate": pymatching_errors / shots,
        "ising_plus_pymatching_logical_errors": ising_pymatching_errors,
        "ising_plus_pymatching_logical_error_rate": ising_pymatching_errors / shots,
        "unionfind_logical_errors": unionfind_errors,
        "unionfind_logical_error_rate": unionfind_errors / shots,
        "ising_plus_unionfind_logical_errors": ising_unionfind_errors,
        "ising_plus_unionfind_logical_error_rate": ising_unionfind_errors / shots,
        "input_detector_density": weighted_by_shots("input_detector_density"),
        "residual_detector_density": weighted_by_shots("residual_detector_density"),
        "latency_samples": latency_samples,
        "pymatch latency (baseline µs/round)": weighted_by_latency_samples(
            "pymatch latency (baseline µs/round)"
        ),
        "pymatch latency (after predecoder µs/round)": weighted_by_latency_samples(
            "pymatch latency (after predecoder µs/round)"
        ),
        "unionfind latency (baseline µs/round)": weighted_by_latency_samples(
            "unionfind latency (baseline µs/round)"
        ),
        "unionfind latency (after predecoder µs/round)": weighted_by_latency_samples(
            "unionfind latency (after predecoder µs/round)"
        ),
        "file_results": file_results,
    }


def run_inference_modified(model, device, dist, cfg):
    """Decode LQCloud samples with both global decoders, with/without Ising."""
    import torch

    from evaluation.decoder_backends import PYMATCHING, UNIONFIND, build_decoder
    from evaluation.logical_error_rate import (
        PreDecoderMemoryEvalModule,
        _build_stab_maps,
        _time_single_shot_latency_stim,
    )

    if int(getattr(dist, "world_size", 1)) != 1:
        raise ValueError("integrate_to_nvidia must run with one process/GPU")
    if str(getattr(cfg.test, "decode_mode", "")).strip().lower() == "pymatching_only":
        raise ValueError("integrate_to_nvidia requires the Ising pre-decoder model")

    model.eval()
    try:
        model = model.to(memory_format=torch.channels_last_3d)
    except Exception:
        pass
    batch_size = max(int(getattr(cfg.lqcloud, "batch_size", 256)), 1)
    latency_num_samples = getattr(cfg.lqcloud, "latency_num_samples", None)
    latency_warmup_iterations = int(
        getattr(cfg.lqcloud, "latency_warmup_iterations", 50)
    )
    source = str(getattr(cfg.lqcloud, "source", "hardware")).strip().lower()
    if source == "hardware":
        measurement_paths = [None]
        remaining_shots = None
    elif source == "file":
        measurement_paths = _measurement_files(
            _resolve_repo_path(cfg.lqcloud.measurement_source)
        )
        configured_max_shots = int(getattr(cfg.lqcloud, "max_shots", 0))
        remaining_shots = configured_max_shots if configured_max_shots > 0 else None
    else:
        raise ValueError("lqcloud.source must be 'hardware' or 'file'")

    pipeline = None
    matcher = None
    unionfind_decoder = None
    file_results = []
    for measurement_path in measurement_paths:
        if remaining_shots is not None and remaining_shots <= 0:
            break
        samples = load_lqcloud_hardware_samples(
            cfg,
            measurement_path=measurement_path,
            max_shots=remaining_shots,
        )
        if pipeline is None:
            cfg.test.meas_basis_test = samples.basis
            cfg.test.n_rounds = int(cfg.n_rounds)
            maps = _build_stab_maps(int(cfg.distance), str(cfg.data.code_rotation))
            pipeline = PreDecoderMemoryEvalModule(model, cfg, maps, device).to(device)
            pipeline.eval()
            dem = samples.circuit.detector_error_model(
                decompose_errors=True,
                approximate_disjoint_errors=True,
            )
            matcher = build_decoder(dem, PYMATCHING)
            unionfind_decoder = build_decoder(dem, UNIONFIND)
        elif samples.basis != cfg.test.meas_basis_test:
            raise ValueError("All LQCloud measurement files must use the same circuit basis")

        file_result = _decode_lqcloud_samples(
            samples,
            pipeline,
            matcher,
            unionfind_decoder,
            device=device,
            batch_size=batch_size,
            n_rounds=int(cfg.n_rounds),
            latency_num_samples=latency_num_samples,
            latency_warmup_iterations=latency_warmup_iterations,
            timer=_time_single_shot_latency_stim,
        )
        file_result["measurement_path"] = (
            "hardware" if measurement_path is None else str(measurement_path)
        )
        file_results.append(file_result)
        if remaining_shots is not None:
            remaining_shots -= int(file_result["shots"])
        del samples

    result = _aggregate_lqcloud_file_results(
        file_results,
        distance=int(cfg.distance),
        n_rounds=int(cfg.n_rounds),
    )
    cfg.test.num_samples = int(result["shots"])

    if dist.rank == 0:
        print(
            f"\n[LQCloud Hardware] d={result['distance']}, rounds={result['n_rounds']}, "
            f"basis={result['basis']}, shots={result['shots']}, files={result['files']}"
        )
        print(f"  {'Decoder':<30}{'Logical errors':>16}{'LER':>12}")
        print(
            f"  {'Raw (no decoder)':<30}{result['raw_logical_errors']:>16}"
            f"{result['raw_logical_error_rate']:>12.6f}"
        )
        print(
            f"  {'PyMatching':<30}{result['pymatching_logical_errors']:>16}"
            f"{result['pymatching_logical_error_rate']:>12.6f}"
        )
        print(
            f"  {'Ising pre-decoder + PyMatching':<30}"
            f"{result['ising_plus_pymatching_logical_errors']:>16}"
            f"{result['ising_plus_pymatching_logical_error_rate']:>12.6f}"
        )
        print(
            f"  {'Union-Find':<30}{result['unionfind_logical_errors']:>16}"
            f"{result['unionfind_logical_error_rate']:>12.6f}"
        )
        print(
            f"  {'Ising pre-decoder + Union-Find':<30}"
            f"{result['ising_plus_unionfind_logical_errors']:>16}"
            f"{result['ising_plus_unionfind_logical_error_rate']:>12.6f}"
        )
        print(
            f"  Detector density: {result['input_detector_density']:.6f} -> "
            f"{result['residual_detector_density']:.6f}"
        )
        print(
            f"  PyMatching latency ({result['latency_samples']} single-shot samples):\n"
            f"    Baseline:         {result['pymatch latency (baseline µs/round)']:.6f} us/round\n"
            f"    After predecoder: {result['pymatch latency (after predecoder µs/round)']:.6f} us/round"
        )
        print(
            f"  Union-Find latency ({result['latency_samples']} single-shot samples):\n"
            f"    Baseline:         {result['unionfind latency (baseline µs/round)']:.6f} us/round\n"
            f"    After predecoder: {result['unionfind latency (after predecoder µs/round)']:.6f} us/round"
        )
        print("[LQCloud Summary] " + json.dumps(result, sort_keys=True))
    return result


__all__ = [
    "HardwareSamples",
    "build_model_detector_permutation",
    "collect_hardware_measurement_memory",
    "load_measurement_file",
    "load_lqcloud_hardware_samples",
    "normalize_measurement_memory",
    "run_inference_modified",
]
