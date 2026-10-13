# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Map Google QEC detector shots to the paper (B, 4, 2, D, D) input."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .google_qec import GoogleQECExperiment


@dataclass(frozen=True)
class PaperInputMapping:
    distance: int
    x_type: np.ndarray
    z_type: np.ndarray
    detector_type: tuple[str, ...]
    detector_grid_index: tuple[int, ...]
    detector_round: tuple[int, ...]


def _coordinate_key(values) -> tuple[float, float]:
    return float(values[0]), float(values[1])


def _qubits_from_metadata(experiment: GoogleQECExperiment, key: str) -> set[int]:
    wanted = {tuple(float(v) for v in coord) for coord in experiment.metadata[key]}
    return {
        int(q)
        for q, coord in experiment.circuit.get_final_qubit_coordinates().items()
        if tuple(float(v) for v in coord) in wanted
    }


def build_paper_input_mapping(experiment: GoogleQECExperiment) -> PaperInputMapping:
    """Derive check type and D-grid placement from circuit geometry/support."""

    distance = int(experiment.metadata["distance"])
    data_qubits = _qubits_from_metadata(experiment, "data_qubit_coords")
    meas_qubits = _qubits_from_metadata(experiment, "meas_qubit_coords")
    qcoords = {
        int(q): tuple(float(v) for v in coord)
        for q, coord in experiment.circuit.get_final_qubit_coordinates().items()
    }

    transformed = {
        q: ((qcoords[q][0] + qcoords[q][1]) / 2, (qcoords[q][0] - qcoords[q][1]) / 2)
        for q in data_qubits
    }
    min_u = min(u for u, _ in transformed.values())
    min_v = min(v for _, v in transformed.values())
    data_grid = {
        q: (int(round(u - min_u)), int(round(v - min_v)))
        for q, (u, v) in transformed.items()
    }
    if set(data_grid.values()) != {
        (row, col) for row in range(distance) for col in range(distance)
    }:
        raise ValueError("Google data-qubit coordinates do not form a D x D grid")

    supports: dict[int, set[int]] = {q: set() for q in meas_qubits}
    for instruction in experiment.circuit:
        if instruction.name not in {"CZ", "CX", "CNOT"}:
            continue
        targets = instruction.targets_copy()
        if any(not target.is_qubit_target for target in targets):
            continue
        qubits = [target.value for target in targets]
        for offset in range(0, len(qubits), 2):
            left, right = qubits[offset : offset + 2]
            if left in meas_qubits and right in data_qubits:
                supports[left].add(right)
            elif right in meas_qubits and left in data_qubits:
                supports[right].add(left)

    detector_coords = experiment.circuit.get_detector_coordinates()
    first_round_xy = {
        _coordinate_key(coord)
        for coord in detector_coords.values()
        if len(coord) >= 3 and int(round(coord[2])) == 0
    }
    basis = str(experiment.metadata["basis"]).upper()
    coordinate_to_meas = {_coordinate_key(qcoords[q]): q for q in meas_qubits}

    def check_type(xy: tuple[float, float]) -> str:
        same_as_basis = xy in first_round_xy
        if basis == "X":
            return "X" if same_as_basis else "Z"
        return "Z" if same_as_basis else "X"

    def mapped_position(meas: int, kind: str) -> tuple[int, int, float]:
        positions = sorted(data_grid[q] for q in supports[meas])
        if len(positions) not in (2, 4):
            raise ValueError(f"check qubit {meas} has support size {len(positions)}")
        if len(positions) == 4:
            row = min(r for r, _ in positions)
            columns = [c for r, c in positions if r == row]
            col = min(columns) if kind == "X" else max(columns)
            return row, col, 1.0
        rows = {r for r, _ in positions}
        columns = {c for _, c in positions}
        if kind == "X":
            chosen = min(positions, key=(lambda rc: rc[1]) if len(rows) == 1 else (lambda rc: rc[0]))
        else:
            chosen = min(positions, key=(lambda rc: rc[0]) if len(columns) == 1 else (lambda rc: -rc[1]))
        return chosen[0], chosen[1], 0.5

    x_type = np.zeros((distance, distance), dtype=np.float32)
    z_type = np.zeros((distance, distance), dtype=np.float32)
    per_meas: dict[int, tuple[str, int]] = {}
    for xy, meas in coordinate_to_meas.items():
        kind = check_type(xy)
        row, col, weight = mapped_position(meas, kind)
        grid_index = row * distance + col
        per_meas[meas] = kind, grid_index
        target = x_type if kind == "X" else z_type
        if target[row, col] != 0:
            raise ValueError(f"duplicate {kind}-check grid position {(row, col)}")
        target[row, col] = weight

    kinds: list[str] = []
    grid_indices: list[int] = []
    rounds: list[int] = []
    for detector in range(experiment.circuit.num_detectors):
        coord = detector_coords[detector]
        xy = _coordinate_key(coord)
        if xy in coordinate_to_meas:
            meas = coordinate_to_meas[xy]
            kind, grid_index = per_meas[meas]
        else:
            # Boundary detectors can be placed between check coordinates and
            # are intentionally excluded from the two-round bulk input.
            kind, grid_index = "", -1
        kinds.append(kind)
        grid_indices.append(grid_index)
        rounds.append(int(round(coord[2])))
    return PaperInputMapping(
        distance=distance,
        x_type=x_type,
        z_type=z_type,
        detector_type=tuple(kinds),
        detector_grid_index=tuple(grid_indices),
        detector_round=tuple(rounds),
    )


def google_experiment_to_paper_tensor(
    experiment: GoogleQECExperiment,
    *,
    first_round: int | None = None,
) -> torch.Tensor:
    """Return paper inputs using two consecutive non-boundary rounds."""

    mapping = build_paper_input_mapping(experiment)
    available = sorted(set(mapping.detector_round))
    bulk = [value for value in available if value > min(available) and value < max(available)]
    consecutive = [(left, left + 1) for left in bulk if left + 1 in bulk]
    if not consecutive:
        raise ValueError("experiment has no consecutive bulk detector rounds")
    if first_round is None:
        pair = consecutive[len(consecutive) // 2]
    else:
        pair = (int(first_round), int(first_round) + 1)
        if pair not in consecutive:
            raise ValueError(f"requested non-bulk round pair {pair}")

    shots = experiment.detection_events.shape[0]
    distance = mapping.distance
    result = np.zeros((shots, 4, 2, distance, distance), dtype=np.float32)
    result[:, 0, :, :, :] = mapping.x_type[None, None, :, :]
    result[:, 1, :, :, :] = mapping.z_type[None, None, :, :]
    for detector, (kind, grid_index, round_index) in enumerate(
        zip(
            mapping.detector_type,
            mapping.detector_grid_index,
            mapping.detector_round,
            strict=True,
        )
    ):
        if round_index not in pair or kind not in {"X", "Z"} or grid_index < 0:
            continue
        time_index = pair.index(round_index)
        row, col = divmod(grid_index, distance)
        channel = 2 if kind == "X" else 3
        result[:, channel, time_index, row, col] = experiment.detection_events[:, detector]
    return torch.from_numpy(result)
