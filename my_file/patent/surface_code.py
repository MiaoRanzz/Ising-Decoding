"""Surface-code action-space construction for the patent gate.

The builder probes the repository's production ``PreDecoderMemoryEvalModule``
with unit actions.  This produces an extended detector map and logical map
whose column ordering is exactly the ordering used by the current Ising-fast
post-processing path, including time-boundary conventions.
"""
from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
CODE_ROOT = REPO_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from data.predecoder_transform import dets_to_predecoder_inputs
from evaluation.logical_error_rate import PreDecoderMemoryEvalModule, _build_stab_maps

try:
    from .classical_gate import Action, ActionSpace
except ImportError:  # Direct script/test execution from this directory.
    from classical_gate import Action, ActionSpace


FAMILIES = ("data_z", "data_x", "meas_x", "meas_z")


class FixedActionModel(torch.nn.Module):
    """Expose a supplied action tensor as threshold-compatible logits."""

    def __init__(self) -> None:
        super().__init__()
        self._device_anchor = torch.nn.Parameter(torch.empty(0), requires_grad=False)
        self.actions: torch.Tensor | None = None

    def set_actions(self, actions: torch.Tensor) -> None:
        self.actions = actions

    def forward(self, train_x: torch.Tensor) -> torch.Tensor:
        if self.actions is None or self.actions.shape[0] != train_x.shape[0]:
            raise RuntimeError("fixed actions were not set for this batch")
        return self.actions.to(device=train_x.device, dtype=torch.float32).mul(2.0).sub(1.0)


def _detector_geometry(
    distance: int,
    n_rounds: int,
    rotation: str,
) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, ...]]:
    """Return detector coordinates, boundary mask, and local graph."""

    maps = _build_stab_maps(distance, rotation)
    x_sites = np.asarray(maps["stab_x"], dtype=np.int64)
    z_sites = np.asarray(maps["stab_z"], dtype=np.int64)
    half = (distance * distance - 1) // 2
    if x_sites.size != half or z_sites.size != half:
        raise ValueError("surface-code stabilizer maps have an unexpected size")

    coordinates = np.zeros((2 * n_rounds * half, 4), dtype=np.int32)
    for timeline in range(2 * n_rounds):
        detector_type = timeline & 1
        time_index = timeline // 2
        sites = x_sites if detector_type == 0 else z_sites
        start = timeline * half
        coordinates[start : start + half, 0] = time_index
        coordinates[start : start + half, 1] = sites // distance
        coordinates[start : start + half, 2] = sites % distance
        coordinates[start : start + half, 3] = detector_type

    boundary = np.zeros(len(coordinates), dtype=bool)
    for detector_type in (0, 1):
        selected = coordinates[:, 3] == detector_type
        rows = coordinates[selected, 1]
        columns = coordinates[selected, 2]
        indices = np.flatnonzero(selected)
        boundary[indices] = (
            (rows == rows.min())
            | (rows == rows.max())
            | (columns == columns.min())
            | (columns == columns.max())
            | (coordinates[indices, 0] == 0)
            | (coordinates[indices, 0] == n_rounds - 1)
        )

    neighbor_sets = [set() for _ in range(len(coordinates))]
    for left in range(len(coordinates)):
        lt, lr, lc, ltype = coordinates[left]
        for right in range(left + 1, len(coordinates)):
            rt, rr, rc, rtype = coordinates[right]
            if ltype != rtype:
                continue
            temporal = lr == rr and lc == rc and abs(int(lt) - int(rt)) == 1
            spatial = lt == rt and 0 < abs(int(lr) - int(rr)) + abs(int(lc) - int(rc)) <= 2
            if temporal or spatial:
                neighbor_sets[left].add(right)
                neighbor_sets[right].add(left)
    neighbors = tuple(np.asarray(sorted(values), dtype=np.int32) for values in neighbor_sets)
    return coordinates, boundary, neighbors


def _expanded_detector_neighborhoods(
    neighbors: tuple[np.ndarray, ...], radius: int
) -> tuple[np.ndarray, ...]:
    if radius < 0:
        raise ValueError("interaction_radius must be non-negative")
    expanded: list[np.ndarray] = []
    for seed in range(len(neighbors)):
        reached = {seed}
        frontier = {seed}
        for _ in range(radius):
            next_frontier: set[int] = set()
            for node in frontier:
                next_frontier.update(int(value) for value in neighbors[node])
            next_frontier -= reached
            reached.update(next_frontier)
            frontier = next_frontier
            if not frontier:
                break
        expanded.append(np.asarray(sorted(reached), dtype=np.int32))
    return tuple(expanded)


def _action_neighbors(
    h: np.ndarray,
    action_locations: list[tuple[int, int, int, int]],
    detector_neighbors: tuple[np.ndarray, ...],
    interaction_radius: int,
) -> tuple[np.ndarray, ...]:
    detector_to_actions = [np.flatnonzero(h[row]).astype(np.int32) for row in range(h.shape[0])]
    expanded = _expanded_detector_neighborhoods(detector_neighbors, interaction_radius)
    neighbor_sets = [set() for _ in range(h.shape[1])]
    for action_index in range(h.shape[1]):
        support = np.flatnonzero(h[:, action_index])
        for detector in support:
            for nearby in expanded[int(detector)]:
                neighbor_sets[action_index].update(int(value) for value in detector_to_actions[int(nearby)])

    # X and Z at the same data site form a Y-capable packet even though their
    # detector supports belong to different stabilizer families.
    by_location: dict[tuple[int, int, int], list[int]] = {}
    for action_index, (channel, time_index, row, column) in enumerate(action_locations):
        if channel < 2:
            by_location.setdefault((time_index, row, column), []).append(action_index)
    for values in by_location.values():
        for left in values:
            neighbor_sets[left].update(values)

    for action_index, values in enumerate(neighbor_sets):
        values.discard(action_index)
    return tuple(np.asarray(sorted(values), dtype=np.int32) for values in neighbor_sets)


def build_surface_action_space(
    cfg: Any,
    *,
    distance: int,
    n_rounds: int,
    basis: str,
    rotation: str = "XV",
    device: str | torch.device = "cpu",
    probe_batch_size: int = 256,
    interaction_radius: int = 1,
) -> ActionSpace:
    """Build ``H~``/``L~`` and topology for the current surface-code endpoint."""

    distance = int(distance)
    n_rounds = int(n_rounds)
    basis = str(basis).upper()
    rotation = str(rotation).upper()
    if basis not in ("X", "Z"):
        raise ValueError("basis must be X or Z")
    if probe_batch_size <= 0:
        raise ValueError("probe_batch_size must be positive")
    device = torch.device(device)
    cfg.distance = distance
    cfg.n_rounds = n_rounds
    cfg.data.code_rotation = rotation
    cfg.test.meas_basis_test = basis
    cfg.test.sampling_mode = "threshold"
    cfg.test.th_data = 0.0
    cfg.test.th_syn = 0.0

    half = (distance * distance - 1) // 2
    num_detectors = 2 * n_rounds * half
    zero_detector = torch.zeros((1, num_detectors), dtype=torch.uint8)
    train_x, _, _ = dets_to_predecoder_inputs(
        zero_detector,
        distance=distance,
        n_rounds=n_rounds,
        basis=basis,
        code_rotation=rotation,
    )
    valid = np.ones((4, n_rounds, distance, distance), dtype=bool)
    valid[2] = train_x[0, 2].cpu().numpy() > 0
    valid[3] = train_x[0, 3].cpu().numpy() > 0

    locations: list[tuple[int, int, int, int]] = []
    for channel in range(4):
        for time_index in range(n_rounds):
            for row in range(distance):
                for column in range(distance):
                    if valid[channel, time_index, row, column]:
                        locations.append((channel, time_index, row, column))

    fixed_model = FixedActionModel().to(device).eval()
    maps = _build_stab_maps(distance, rotation)
    endpoint = PreDecoderMemoryEvalModule(fixed_model, cfg, maps, device).to(device).eval()
    h = np.zeros((num_detectors, len(locations)), dtype=np.uint8)
    l = np.zeros((1, len(locations)), dtype=np.uint8)
    with torch.no_grad():
        for start in range(0, len(locations), probe_batch_size):
            batch_locations = locations[start : start + probe_batch_size]
            actions = torch.zeros(
                (len(batch_locations), 4, n_rounds, distance, distance),
                dtype=torch.uint8,
                device=device,
            )
            for row_index, (channel, time_index, row, column) in enumerate(batch_locations):
                actions[row_index, channel, time_index, row, column] = 1
            fixed_model.set_actions(actions)
            detector_batch = torch.zeros(
                (len(batch_locations), num_detectors), dtype=torch.uint8, device=device
            )
            output = endpoint(detector_batch).to(torch.uint8).cpu().numpy()
            l[:, start : start + len(batch_locations)] = output[:, 0].reshape(1, -1)
            h[:, start : start + len(batch_locations)] = output[:, 1:].T

    _, detector_boundary, detector_neighbors = _detector_geometry(
        distance, n_rounds, rotation
    )
    action_neighbors = _action_neighbors(
        h, locations, detector_neighbors, interaction_radius
    )
    actions: list[Action] = []
    for action_index, (channel, time_index, row, column) in enumerate(locations):
        support = tuple(int(value) for value in np.flatnonzero(h[:, action_index]))
        logical = tuple(int(value) for value in l[:, action_index])
        if support:
            boundary_score = float(detector_boundary[list(support)].mean())
        else:
            boundary_score = float(
                time_index in (0, n_rounds - 1)
                or row in (0, distance - 1)
                or column in (0, distance - 1)
            )
        actions.append(
            Action(
                index=action_index,
                family=FAMILIES[channel],
                channel=channel,
                time=time_index,
                row=row,
                column=column,
                detector_support=support,
                logical_effect=logical,
                boundary_score=boundary_score,
            )
        )

    return ActionSpace(
        h=h,
        l=l,
        actions=tuple(actions),
        detector_neighbors=detector_neighbors,
        action_neighbors=action_neighbors,
        detector_boundary=detector_boundary,
        code_distance=distance,
        n_rounds=n_rounds,
        dense_shape=(4, n_rounds, distance, distance),
    )


def apply_dense_actions(
    space: ActionSpace,
    syndromes: np.ndarray,
    dense_actions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized endpoint effect for a batch of dense four-channel actions."""

    syndromes = np.asarray(syndromes, dtype=np.uint8)
    dense_actions = np.asarray(dense_actions, dtype=np.uint8)
    if syndromes.ndim != 2 or syndromes.shape[1] != space.num_detectors:
        raise ValueError("syndromes have the wrong shape")
    if dense_actions.shape != (syndromes.shape[0], *space.dense_shape):
        raise ValueError("dense actions have the wrong shape")
    masks = np.stack([space.mask_from_dense(row) for row in dense_actions]).astype(np.uint8)
    detector_delta = np.remainder(
        masks.astype(np.int64) @ space.h.T.astype(np.int64), 2
    ).astype(np.uint8)
    logical_delta = np.remainder(
        masks.astype(np.int64) @ space.l.T.astype(np.int64), 2
    ).astype(np.uint8)
    return np.bitwise_xor(syndromes, detector_delta), logical_delta, masks


def dense_probabilities(space: ActionSpace, logits: np.ndarray) -> np.ndarray:
    """Gather dense sigmoid probabilities into the action-space column order."""

    try:
        from .classical_gate import sigmoid
    except ImportError:
        from classical_gate import sigmoid

    logits = np.asarray(logits)
    if logits.ndim != 5 or tuple(logits.shape[1:]) != space.dense_shape:
        raise ValueError("logits have the wrong shape")
    probabilities = sigmoid(logits)
    return np.stack([space.probabilities_from_dense(row) for row in probabilities])


__all__ = [
    "FixedActionModel",
    "apply_dense_actions",
    "build_surface_action_space",
    "dense_probabilities",
]
