# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fit Ising-Decoding's 25-parameter Pauli model to Google QEC data.

The fitter uses detector-parity moments.  For a proposed noise vector it
injects the corresponding channels into Google's ideal Stim circuit, builds a
detector error model, and evaluates detector and detector-pair parity
probabilities exactly.  These predictions are fitted to moments measured from
the hardware ``detection_events.b8`` files.

The 25 parameters are not always all identifiable from one code/basis.  Fits
therefore combine multiple experiments and use a weak log-space prior.  The
reported Jacobian rank and condition number make this limitation explicit.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path, PurePosixPath
from typing import Iterable, Mapping, Sequence
import zipfile

import numpy as np
import stim

from qec.noise_model import CNOT_ERROR_TYPES, NoiseModel


PARAMETER_NAMES = tuple(NoiseModel().canonical_parameters())


@dataclass(frozen=True)
class GoogleQECExperiment:
    """One basis/distance/round-count experiment from the Google archive."""

    key: str
    metadata: Mapping[str, object]
    circuit: stim.Circuit
    detection_events: np.ndarray
    observable_flips: np.ndarray

    @property
    def shots(self) -> int:
        return int(self.detection_events.shape[0])


class GoogleQECDataset:
    """Read selected experiments directly from an extracted tree or zip file."""

    def __init__(self, source: str | Path):
        self.source = Path(source)
        if not self.source.exists():
            raise FileNotFoundError(self.source)

    def keys(self) -> tuple[str, ...]:
        suffix = "/metadata.json"
        if self.source.is_file():
            with zipfile.ZipFile(self.source) as archive:
                names = (n for n in archive.namelist() if n.endswith(suffix))
                return tuple(sorted(n[: -len(suffix)] for n in names))
        return tuple(
            sorted(
                str(p.parent.relative_to(self.source))
                for p in self.source.rglob("metadata.json")
            )
        )

    def select(
        self,
        *,
        distances: Sequence[int] | None = None,
        bases: Sequence[str] | None = None,
        rounds: Sequence[int] | None = None,
        max_experiments: int | None = None,
    ) -> tuple[str, ...]:
        distance_set = None if distances is None else {int(v) for v in distances}
        basis_set = None if bases is None else {str(v).upper() for v in bases}
        rounds_set = None if rounds is None else {int(v) for v in rounds}
        selected: list[str] = []
        for key in self.keys():
            meta = self._read_json(f"{key}/metadata.json")
            if distance_set is not None and int(meta["distance"]) not in distance_set:
                continue
            if basis_set is not None and str(meta["basis"]).upper() not in basis_set:
                continue
            if rounds_set is not None and int(meta["rounds"]) not in rounds_set:
                continue
            selected.append(key)
            if max_experiments is not None and len(selected) >= max_experiments:
                break
        return tuple(selected)

    def load(self, key: str, *, max_shots: int | None = None) -> GoogleQECExperiment:
        metadata = self._read_json(f"{key}/metadata.json")
        circuit = stim.Circuit(self._read_bytes(f"{key}/circuit_ideal.stim").decode("utf-8"))
        shots = int(metadata["shots"])
        raw = self._read_bytes(f"{key}/detection_events.b8")
        dets = unpack_b8(raw, shots=shots, bits_per_shot=circuit.num_detectors)
        raw_obs = self._read_bytes(f"{key}/obs_flips_actual.b8")
        observables = unpack_b8(
            raw_obs, shots=shots, bits_per_shot=circuit.num_observables
        )
        if max_shots is not None:
            dets = dets[: int(max_shots)]
            observables = observables[: int(max_shots)]
        return GoogleQECExperiment(key, metadata, circuit, dets, observables)

    def _read_bytes(self, relative: str) -> bytes:
        if self.source.is_file():
            with zipfile.ZipFile(self.source) as archive:
                return archive.read(str(PurePosixPath(relative)))
        return (self.source / relative).read_bytes()

    def _read_json(self, relative: str) -> Mapping[str, object]:
        return json.loads(self._read_bytes(relative))


def unpack_b8(data: bytes, *, shots: int, bits_per_shot: int) -> np.ndarray:
    """Decode Stim's shot-major, little-endian ``b8`` format."""

    bytes_per_shot = (int(bits_per_shot) + 7) // 8
    expected = int(shots) * bytes_per_shot
    if len(data) != expected:
        raise ValueError(f"b8 size mismatch: got {len(data)} bytes, expected {expected}")
    packed = np.frombuffer(data, dtype=np.uint8).reshape(int(shots), bytes_per_shot)
    return np.unpackbits(packed, axis=1, bitorder="little")[:, :bits_per_shot].astype(bool)


def _append_channel_1(out: stim.Circuit, probs: Sequence[float], targets: Iterable[int]) -> None:
    targets = tuple(int(q) for q in targets)
    if targets and sum(probs) > 0:
        out.append("PAULI_CHANNEL_1", targets, [float(v) for v in probs])


def _append_basis_flip(
    out: stim.Circuit, probability: float, targets: Iterable[int], *, basis: str
) -> None:
    targets = tuple(int(q) for q in targets)
    if targets and probability > 0:
        out.append("Z_ERROR" if basis.upper() == "X" else "X_ERROR", targets, probability)


def _qubit_targets(instruction: stim.CircuitInstruction) -> tuple[int, ...]:
    return tuple(t.value for t in instruction.targets_copy() if t.is_qubit_target)


def inject_noise(
    circuit: stim.Circuit,
    noise_model: NoiseModel,
    *,
    basis: str,
    data_qubits: Sequence[int],
    measurement_qubits: Sequence[int],
) -> stim.Circuit:
    """Inject the repository's 25 channel families into a Google ideal circuit.

    Google uses CZ entanglers while Ising-Decoding's native circuit uses CX.
    ``PAULI_CHANNEL_2`` is attached after CZ in the listed target order; its
    Pauli labels therefore retain the repository's first/second-qubit
    convention even though the physical entangler differs.
    """

    out = stim.Circuit()
    data = frozenset(int(q) for q in data_qubits)
    measurement = frozenset(int(q) for q in measurement_qubits)
    all_active = data | measurement
    initial_prep_done = False

    for instruction in circuit:
        name = instruction.name
        targets = _qubit_targets(instruction)

        if name == "TICK" and not initial_prep_done:
            # Google's initial reset is implicit.  State-preparation faults
            # commute through the following sweep-bit logical randomisation,
            # so attach the basis flip immediately before the first tick.
            prep_p = noise_model.p_prep_X if basis.upper() == "X" else noise_model.p_prep_Z
            _append_basis_flip(out, prep_p, data, basis=basis)
            initial_prep_done = True

        if name in {"M", "MZ", "MX"}:
            meas_targets = [q for q in targets if q in measurement]
            data_targets = [q for q in targets if q in data]
            physical_flip = "Z_ERROR" if name == "MX" else "X_ERROR"
            if meas_targets and noise_model.p_meas_X > 0:
                out.append(physical_flip, meas_targets, noise_model.p_meas_X)
            data_p = noise_model.p_meas_X if basis.upper() == "X" else noise_model.p_meas_Z
            if data_targets and data_p > 0:
                out.append(physical_flip, data_targets, data_p)

        out.append(instruction)

        if name in {"R", "RZ", "RX"}:
            reset_measurement = [q for q in targets if q in measurement]
            # Google prepares stabilizer ancillas as |+>: an R reset followed
            # by H.  An X fault after R is equivalent to the basis-flipping Z
            # fault after H, hence this is the X-basis preparation parameter.
            _append_basis_flip(
                out, noise_model.p_prep_X, reset_measurement, basis="Z"
            )
            _append_channel_1(
                out,
                noise_model.to_stim_pauli_channel_1_args_spam(),
                data,
            )
        elif name in {"CZ", "CX", "CNOT"}:
            # Initial logical-state randomisation uses CX sweep[k] q.  This is
            # classical feed-forward, not a physical two-qubit gate.
            raw_targets = instruction.targets_copy()
            if any(not target.is_qubit_target for target in raw_targets):
                continue
            if len(targets) % 2:
                raise ValueError(f"{name} has an odd number of qubit targets")
            probs = noise_model.to_stim_pauli_channel_2_args()
            for k in range(0, len(targets), 2):
                if sum(probs) > 0:
                    out.append("PAULI_CHANNEL_2", targets[k : k + 2], probs)
            _append_channel_1(
                out,
                noise_model.to_stim_pauli_channel_1_args_cnot(),
                all_active.difference(targets),
            )
    return out


@dataclass(frozen=True)
class _Moment:
    detectors: tuple[int, ...]
    observed: float
    sigma: float


def _detector_coordinates(circuit: stim.Circuit) -> dict[int, tuple[float, ...]]:
    return {
        int(k): tuple(float(v) for v in values)
        for k, values in circuit.get_detector_coordinates().items()
    }


def build_moments(
    experiment: GoogleQECExperiment,
    *,
    max_pair_moments: int = 512,
) -> tuple[_Moment, ...]:
    """Create single-detector and local pair-parity fit targets."""

    dets = experiment.detection_events
    n = dets.shape[0]
    moments: list[_Moment] = []

    def add(indices: tuple[int, ...], values: np.ndarray) -> None:
        p = float(np.mean(values))
        sigma = max(math.sqrt(max(p * (1 - p), 1e-6) / n), 1 / n)
        moments.append(_Moment(indices, p, sigma))

    num_detectors = dets.shape[1]
    for index in range(num_detectors):
        add((index,), dets[:, index])
    for observable in range(experiment.observable_flips.shape[1]):
        add(
            (num_detectors + observable,),
            experiment.observable_flips[:, observable],
        )

    # Logical-observable correlations can distinguish fault channels with the
    # same detector signature.  Reserve one quarter of the pair budget for
    # detector/logical parities, then use the remainder for local detector pairs.
    logical_pair_count = min(num_detectors, max_pair_moments // 4)
    if experiment.observable_flips.shape[1]:
        logical = experiment.observable_flips[:, 0]
        for detector in range(logical_pair_count):
            add(
                (detector, num_detectors),
                np.logical_xor(dets[:, detector], logical),
            )

    coordinates = _detector_coordinates(experiment.circuit)
    candidates: list[tuple[float, int, int]] = []
    ids = sorted(coordinates)
    for pos, left in enumerate(ids):
        a = coordinates[left]
        if len(a) < 3:
            continue
        for right in ids[pos + 1 :]:
            b = coordinates[right]
            if len(b) < 3 or abs(a[2] - b[2]) > 1.01:
                continue
            distance = sum(abs(a[k] - b[k]) for k in range(3))
            if distance <= 3.01:
                candidates.append((distance, left, right))
    candidates.sort()
    detector_pair_budget = max(0, max_pair_moments - logical_pair_count)
    for _, left, right in candidates[:detector_pair_budget]:
        add((left, right), np.logical_xor(dets[:, left], dets[:, right]))
    return tuple(moments)


def predict_moments(circuit: stim.Circuit, moments: Sequence[_Moment]) -> np.ndarray:
    """Evaluate detector-parity probabilities from a detector error model."""

    dem = circuit.detector_error_model(
        decompose_errors=False,
        flatten_loops=True,
        allow_gauge_detectors=True,
        approximate_disjoint_errors=True,
    )
    token_count = circuit.num_detectors + circuit.num_observables
    membership = np.zeros((token_count, len(moments)), dtype=np.bool_)
    for moment_index, moment in enumerate(moments):
        membership[list(moment.detectors), moment_index] = True
    log_products = np.zeros(len(moments), dtype=np.float64)
    for instruction in dem.flattened():
        if instruction.type != "error":
            continue
        p = float(instruction.args_copy()[0])
        if p <= 0:
            continue
        affected_values = [
            t.val for t in instruction.targets_copy() if t.is_relative_detector_id()
        ]
        affected_values.extend(
            circuit.num_detectors + t.val
            for t in instruction.targets_copy()
            if t.is_logical_observable_id()
        )
        affected = tuple(set(affected_values))
        if not affected:
            continue
        contribution = math.log(max(1 - 2 * min(p, 0.499999999999), 1e-15))
        flips_moment = np.bitwise_xor.reduce(membership[list(affected)], axis=0)
        log_products[flips_moment] += contribution
    return 0.5 * (1 - np.exp(log_products))


@dataclass(frozen=True)
class NoiseLearningResult:
    noise_model: NoiseModel
    cost: float
    initial_cost: float
    optimality: float
    iterations: int
    success: bool
    status: int
    message: str
    jacobian_rank: int
    jacobian_condition: float
    experiment_keys: tuple[str, ...]
    moment_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "noise_model": self.noise_model.canonical_parameters(),
            "noise_model_sha256": self.noise_model.sha256(),
            "fit": {
                "cost": self.cost,
                "initial_cost": self.initial_cost,
                "cost_reduction_fraction": (
                    1.0 - self.cost / self.initial_cost if self.initial_cost > 0 else 0.0
                ),
                "optimality": self.optimality,
                "iterations": self.iterations,
                "success": self.success,
                "status": self.status,
                "message": self.message,
                "jacobian_rank": self.jacobian_rank,
                "jacobian_condition": self.jacobian_condition,
                "experiment_keys": list(self.experiment_keys),
                "moment_count": self.moment_count,
            },
        }


def fit_noise_model(
    experiments: Sequence[GoogleQECExperiment],
    *,
    initial: NoiseModel | None = None,
    max_nfev: int = 80,
    prior_strength: float = 0.05,
    min_probability: float = 1e-5,
    max_probability: float = 3e-2,
    max_pair_moments: int = 512,
) -> NoiseLearningResult:
    """Fit all 25 probabilities using bounded nonlinear least squares."""

    from scipy.optimize import least_squares

    if not experiments:
        raise ValueError("at least one experiment is required")
    initial = initial or NoiseModel.from_si1000(1e-3)
    initial_values = np.array(
        [initial.canonical_parameters()[name] for name in PARAMETER_NAMES], dtype=np.float64
    )
    if not (0 < min_probability < max_probability):
        raise ValueError("expected 0 < min_probability < max_probability")
    x0 = np.log(np.clip(initial_values, min_probability, max_probability))
    lower = np.full_like(x0, math.log(min_probability))
    upper = np.full_like(x0, math.log(max_probability))
    all_moments = [build_moments(e, max_pair_moments=max_pair_moments) for e in experiments]

    def decode(x: np.ndarray) -> NoiseModel:
        return NoiseModel.from_config_dict(dict(zip(PARAMETER_NAMES, np.exp(x), strict=True)))

    def residual(x: np.ndarray) -> np.ndarray:
        model = decode(x)
        chunks: list[np.ndarray] = []
        for experiment, moments in zip(experiments, all_moments, strict=True):
            noisy = inject_noise(
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
            predicted = predict_moments(noisy, moments)
            observed = np.array([m.observed for m in moments])
            sigma = np.array([m.sigma for m in moments])
            chunks.append((predicted - observed) / sigma)
        if prior_strength > 0:
            chunks.append(math.sqrt(prior_strength) * (x - x0))
        return np.concatenate(chunks)

    initial_residual = residual(x0)
    initial_cost = 0.5 * float(np.dot(initial_residual, initial_residual))
    solution = least_squares(
        residual,
        x0,
        bounds=(lower, upper),
        max_nfev=int(max_nfev),
        x_scale="jac",
        verbose=0,
    )
    # The log-space prior contributes an identity block and would make the
    # reported rank trivially full.  Diagnose identifiability from hardware
    # moments only, excluding regularization rows.
    data_moment_count = sum(len(v) for v in all_moments)
    data_jacobian = solution.jac[:data_moment_count, :]
    singular_values = np.linalg.svd(data_jacobian, compute_uv=False)
    tolerance = np.finfo(float).eps * max(data_jacobian.shape) * singular_values[0]
    rank = int(np.sum(singular_values > tolerance))
    condition = (
        float(singular_values[0] / singular_values[-1])
        if singular_values[-1] > 0
        else float("inf")
    )
    return NoiseLearningResult(
        noise_model=decode(solution.x),
        cost=float(solution.cost),
        initial_cost=initial_cost,
        optimality=float(solution.optimality),
        iterations=int(solution.nfev),
        success=bool(solution.success),
        status=int(solution.status),
        message=str(solution.message),
        jacobian_rank=rank,
        jacobian_condition=condition,
        experiment_keys=tuple(e.key for e in experiments),
        moment_count=data_moment_count,
    )


def _metadata_qubits(
    metadata: Mapping[str, object],
    key: str,
    circuit: stim.Circuit,
) -> tuple[int, ...]:
    """Map coordinate lists in metadata back to Stim qubit ids."""

    wanted = {tuple(float(v) for v in coord) for coord in metadata[key]}  # type: ignore[index]
    coordinates = circuit.get_final_qubit_coordinates()
    return tuple(
        sorted(
            int(qubit)
            for qubit, coordinate in coordinates.items()
            if tuple(float(v) for v in coordinate) in wanted
        )
    )
