# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Reference implementation of topology-residual utility gating v2.

The implementation deliberately operates on the explicit binary action maps
described by the patent draft:

* ``extended_h[d, e]`` says whether action ``e`` flips detector ``d``;
* ``extended_l[l, e]`` says whether action ``e`` flips logical frame ``l``;
* action probabilities are converted into typed *candidates*, not committed
  corrections;
* interacting candidates are clustered, legal combinations are generated,
  and every retained combination is evaluated against the current residual;
* at most one combination is selected per cluster and later cluster scores are
  recomputed after every state update.

This is a deterministic NumPy reference path intended for correctness and
validation.  It is not the optimized CUDA/TensorRT deployment path.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from itertools import combinations
import math
from typing import Iterable, Mapping, Sequence

import numpy as np


DATA_Z = "data_z"
DATA_X = "data_x"
MEASUREMENT_X = "measurement_x"
MEASUREMENT_Z = "measurement_z"
ACTION_TYPES = (DATA_Z, DATA_X, MEASUREMENT_X, MEASUREMENT_Z)


@dataclass(frozen=True)
class GateConfig:
    """Search, workload, and safety parameters for one gating run."""

    data_threshold: float = 0.7
    measurement_threshold: float = 0.7
    interaction_radius: int = 1
    exact_cluster_size: int = 10
    max_combination_actions: int = 4
    combination_budget: int = 256
    max_total_actions: int = 64
    max_candidates: int = 32
    acceptance_threshold: float = 0.0
    max_workload_increase: float = 0.0
    max_logical_risk: float = 1.0
    workload_active_weight: float = 1.0
    workload_component_weight: float = 0.25
    workload_pair_weight: float = 0.1
    workload_mode: str = "legacy"
    workload_largest_weight: float = 0.0
    workload_pair_radius: int = 1
    pointwise_guard_enabled: bool = False
    pointwise_guard_relative_margin: float = 0.0
    uncertainty_weight: float = 0.1
    logical_risk_weight: float = 1.0
    gate_cost_weight: float = 0.0

    def __post_init__(self) -> None:
        for name in ("data_threshold", "measurement_threshold"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        for name in (
            "interaction_radius",
            "exact_cluster_size",
            "max_combination_actions",
            "combination_budget",
            "max_total_actions",
            "max_candidates",
            "workload_pair_radius",
        ):
            if int(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.combination_budget < 1:
            raise ValueError("combination_budget must include at least the empty combination")
        if self.workload_pair_radius < 1:
            raise ValueError("workload_pair_radius must be positive")
        if self.workload_mode not in ("legacy", "topology_v3"):
            raise ValueError("workload_mode must be legacy or topology_v3")
        if not 0.0 <= self.pointwise_guard_relative_margin < 1.0:
            raise ValueError("pointwise_guard_relative_margin must be in [0, 1)")
        for name in (
            "workload_active_weight",
            "workload_component_weight",
            "workload_pair_weight",
            "workload_largest_weight",
        ):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be non-negative")


@dataclass(frozen=True)
class WorkloadFeatures:
    active_count: int
    active_density: float
    component_count: int
    largest_component: int
    component_square_sum: int
    local_pair_count: int
    radius_pair_count: int = 0


@dataclass(frozen=True)
class WorkloadGraph:
    adjacency: tuple[frozenset[int], ...]
    radius_adjacency: tuple[frozenset[int], ...]
    pair_radius: int


@dataclass(frozen=True)
class CombinationEvaluation:
    actions: tuple[int, ...]
    utility: float
    uncertainty: float
    logical_risk: float
    gate_cost_proxy: float
    trial_residual: np.ndarray
    logical_delta: np.ndarray
    workload_before: WorkloadFeatures
    workload_after: WorkloadFeatures
    search_exact: bool
    combinations_evaluated: int


@dataclass(frozen=True)
class GateDecision:
    cluster: tuple[int, ...]
    selected_actions: tuple[int, ...]
    accepted: bool
    reason: str
    utility: float
    uncertainty: float
    logical_risk: float
    workload_before: WorkloadFeatures
    workload_after: WorkloadFeatures
    search_exact: bool
    combinations_evaluated: int


@dataclass(frozen=True)
class GateResult:
    accepted_actions: np.ndarray
    residual: np.ndarray
    local_logical_frame: np.ndarray
    candidate_count: int
    decisions: tuple[GateDecision, ...]
    selection_source: str = "native"
    combination_workload_score: float | None = None
    pointwise_workload_score: float | None = None

    @property
    def accepted_count(self) -> int:
        return int(self.accepted_actions.sum())

    @property
    def combinations_evaluated(self) -> int:
        return sum(item.combinations_evaluated for item in self.decisions)

    @property
    def exact_search(self) -> bool:
        return all(item.search_exact for item in self.decisions)


def _binary_vector(value: np.ndarray | Sequence[int], *, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.uint8)
    if result.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {result.shape}")
    if np.any(result > 1):
        raise ValueError(f"{name} must contain only binary values")
    return result


def _binary_matrix(value: np.ndarray, *, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.uint8)
    if result.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional, got shape {result.shape}")
    if np.any(result > 1):
        raise ValueError(f"{name} must contain only binary values")
    return result


def _logical_matrix(value: np.ndarray, action_count: int) -> np.ndarray:
    result = np.asarray(value, dtype=np.uint8)
    if result.ndim == 1:
        result = result.reshape(1, -1)
    result = _binary_matrix(result, name="extended_l")
    if result.shape[1] != action_count:
        raise ValueError(
            f"extended_l has {result.shape[1]} actions, expected {action_count}"
        )
    return result


def _normalise_adjacency(
    adjacency: Sequence[Iterable[int]], detector_count: int
) -> tuple[frozenset[int], ...]:
    if len(adjacency) != detector_count:
        raise ValueError(
            f"detector_adjacency has {len(adjacency)} rows, expected {detector_count}"
        )
    rows: list[set[int]] = [set() for _ in range(detector_count)]
    for left, neighbours in enumerate(adjacency):
        for right_value in neighbours:
            right = int(right_value)
            if right < 0 or right >= detector_count:
                raise ValueError(f"detector adjacency index {right} is out of range")
            if right == left:
                continue
            rows[left].add(right)
            rows[right].add(left)
    return tuple(frozenset(row) for row in rows)


def build_workload_graph(
    detector_adjacency: Sequence[Iterable[int]], pair_radius: int = 1
) -> WorkloadGraph:
    """Precompute immediate and finite-radius detector neighbourhoods."""

    radius = int(pair_radius)
    if radius < 1:
        raise ValueError("pair_radius must be positive")
    adjacency = _normalise_adjacency(
        detector_adjacency, len(detector_adjacency)
    )
    radius_rows: list[frozenset[int]] = []
    for root in range(len(adjacency)):
        visited = {root}
        frontier = {root}
        for _ in range(radius):
            following = {
                neighbour
                for node in frontier
                for neighbour in adjacency[node]
                if neighbour not in visited
            }
            if not following:
                break
            visited.update(following)
            frontier = following
        visited.remove(root)
        radius_rows.append(frozenset(visited))
    return WorkloadGraph(adjacency, tuple(radius_rows), radius)


def _normalise_incompatible(
    incompatible_pairs: Iterable[tuple[int, int]], action_count: int
) -> frozenset[tuple[int, int]]:
    pairs = set()
    for left_value, right_value in incompatible_pairs:
        left, right = sorted((int(left_value), int(right_value)))
        if left == right:
            raise ValueError("an action cannot be incompatible with itself")
        if left < 0 or right >= action_count:
            raise ValueError(f"incompatible pair {(left, right)} is out of range")
        pairs.add((left, right))
    return frozenset(pairs)


def typed_candidates(
    probabilities: np.ndarray,
    action_types: Sequence[str],
    *,
    data_threshold: float,
    measurement_threshold: float,
    valid_actions: np.ndarray | None = None,
    max_candidates: int | None = None,
) -> np.ndarray:
    """Return deterministic candidate indices using type-specific thresholds."""

    probability = np.asarray(probabilities, dtype=np.float64)
    if probability.ndim != 1:
        raise ValueError("probabilities must be one-dimensional")
    if len(action_types) != probability.size:
        raise ValueError("action_types and probabilities must have equal length")
    if not np.all(np.isfinite(probability)) or np.any((probability < 0) | (probability > 1)):
        raise ValueError("probabilities must be finite values in [0, 1]")
    unknown = sorted(set(action_types) - set(ACTION_TYPES))
    if unknown:
        raise ValueError(f"unknown action types: {unknown}")
    valid = (
        np.ones(probability.size, dtype=bool)
        if valid_actions is None
        else np.asarray(valid_actions, dtype=bool)
    )
    if valid.shape != probability.shape:
        raise ValueError("valid_actions must match probabilities")
    thresholds = np.asarray(
        [
            measurement_threshold if item in (MEASUREMENT_X, MEASUREMENT_Z) else data_threshold
            for item in action_types
        ],
        dtype=np.float64,
    )
    selected = np.flatnonzero(valid & (probability >= thresholds)).astype(np.int64)
    if max_candidates is not None and selected.size > int(max_candidates):
        if int(max_candidates) < 0:
            raise ValueError("max_candidates must be non-negative")
        selected = np.asarray(
            sorted(selected.tolist(), key=lambda item: (-probability[item], item))[
                : int(max_candidates)
            ],
            dtype=np.int64,
        )
    return selected


def detector_neighbourhoods(
    adjacency: Sequence[Iterable[int]], radius: int
) -> tuple[frozenset[int], ...]:
    """Return each detector's inclusive graph neighbourhood up to ``radius``."""

    normalised = _normalise_adjacency(adjacency, len(adjacency))
    radius = int(radius)
    if radius < 0:
        raise ValueError("interaction radius must be non-negative")
    output: list[frozenset[int]] = []
    for root in range(len(normalised)):
        reached = {root}
        frontier = {root}
        for _ in range(radius):
            frontier = {node for item in frontier for node in normalised[item]} - reached
            if not frontier:
                break
            reached.update(frontier)
        output.append(frozenset(reached))
    return tuple(output)


def interaction_clusters(
    candidates: Sequence[int],
    extended_h: np.ndarray,
    detector_adjacency: Sequence[Iterable[int]],
    *,
    radius: int,
    incompatible_pairs: Iterable[tuple[int, int]] = (),
) -> list[tuple[int, ...]]:
    """Build connected candidate interaction domains.

    Actions interact when their detector support overlaps, is within the
    configured detector-graph radius, or has an explicit compatibility edge.
    """

    h = _binary_matrix(extended_h, name="extended_h")
    candidate_list = sorted({int(item) for item in candidates})
    if any(item < 0 or item >= h.shape[1] for item in candidate_list):
        raise ValueError("candidate action index is out of range")
    if not candidate_list:
        return []
    neighbourhood = detector_neighbourhoods(detector_adjacency, radius)
    incompatible = _normalise_incompatible(incompatible_pairs, h.shape[1])
    supports = {item: set(np.flatnonzero(h[:, item]).tolist()) for item in candidate_list}
    graph = {item: set() for item in candidate_list}
    for position, left in enumerate(candidate_list):
        left_reach = {
            detector
            for support_detector in supports[left]
            for detector in neighbourhood[support_detector]
        }
        for right in candidate_list[position + 1 :]:
            pair = (left, right)
            if pair in incompatible or bool(left_reach & supports[right]):
                graph[left].add(right)
                graph[right].add(left)
    remaining = set(candidate_list)
    clusters: list[tuple[int, ...]] = []
    while remaining:
        root = min(remaining)
        remaining.remove(root)
        queue = deque([root])
        cluster = []
        while queue:
            node = queue.popleft()
            cluster.append(node)
            new_nodes = sorted(graph[node] & remaining)
            remaining.difference_update(new_nodes)
            queue.extend(new_nodes)
        clusters.append(tuple(sorted(cluster)))
    return clusters


def _legal(actions: tuple[int, ...], incompatible: frozenset[tuple[int, int]]) -> bool:
    return all(tuple(sorted(pair)) not in incompatible for pair in combinations(actions, 2))


def _combination_priority(actions: tuple[int, ...], probabilities: np.ndarray) -> float:
    if not actions:
        return 0.0
    clipped = np.clip(probabilities[list(actions)], 1e-9, 1 - 1e-9)
    return float(np.log(clipped / (1 - clipped)).sum())


def legal_combinations(
    cluster: Sequence[int],
    probabilities: np.ndarray,
    *,
    incompatible_pairs: Iterable[tuple[int, int]] = (),
    exact_cluster_size: int,
    max_actions: int,
    budget: int,
) -> tuple[list[tuple[int, ...]], bool]:
    """Generate legal combinations, always including the empty combination.

    Small clusters are exhaustively enumerated.  Larger clusters use a
    deterministic probability-prioritised beam capped by ``budget``.
    """

    actions = tuple(sorted({int(item) for item in cluster}))
    probability = np.asarray(probabilities, dtype=np.float64)
    if any(item < 0 or item >= probability.size for item in actions):
        raise ValueError("cluster action index is out of range")
    action_count = probability.size
    incompatible = _normalise_incompatible(incompatible_pairs, action_count)
    max_size = min(len(actions), max(0, int(max_actions)))
    budget = int(budget)
    if budget < 1:
        raise ValueError("combination budget must be positive")
    theoretical = sum(math.comb(len(actions), size) for size in range(max_size + 1))
    exact = len(actions) <= int(exact_cluster_size) and theoretical <= budget
    if exact:
        values = [
            item
            for size in range(max_size + 1)
            for item in combinations(actions, size)
            if _legal(item, incompatible)
        ]
        return values, True

    beam: set[tuple[int, ...]] = {()}
    ordered = sorted(actions, key=lambda item: (-probability[item], item))
    for action in ordered:
        expanded = set(beam)
        for current in beam:
            if len(current) >= max_size:
                continue
            proposed = tuple(sorted((*current, action)))
            if _legal(proposed, incompatible):
                expanded.add(proposed)
        ranked = sorted(
            expanded,
            key=lambda item: (
                -_combination_priority(item, probability),
                len(item),
                item,
            ),
        )
        beam = set(ranked[:budget])
    values = sorted(beam, key=lambda item: (len(item), item))
    if () not in values:
        values = [()] + values[: budget - 1]
    return values[:budget], False


def workload_features(
    syndrome: np.ndarray | Sequence[int],
    detector_adjacency: Sequence[Iterable[int]] | WorkloadGraph,
    *,
    pair_radius: int = 1,
) -> WorkloadFeatures:
    """Compute state-dependent active-detector workload proxies."""

    bits = _binary_vector(syndrome, name="syndrome")
    graph = (
        detector_adjacency
        if isinstance(detector_adjacency, WorkloadGraph)
        else build_workload_graph(detector_adjacency, pair_radius)
    )
    if len(graph.adjacency) != bits.size:
        raise ValueError(
            f"workload graph has {len(graph.adjacency)} rows, expected {bits.size}"
        )
    return _workload_features_normalized(
        bits, graph.adjacency, graph.radius_adjacency
    )


def _workload_features_normalized(
    syndrome: np.ndarray | Sequence[int],
    adjacency: tuple[frozenset[int], ...],
    radius_adjacency: tuple[frozenset[int], ...] | None = None,
) -> WorkloadFeatures:
    """Compute state-dependent active-detector workload proxies."""

    bits = _binary_vector(syndrome, name="syndrome")
    active = set(np.flatnonzero(bits).tolist())
    component_sizes: list[int] = []
    remaining = set(active)
    while remaining:
        root = min(remaining)
        remaining.remove(root)
        queue = [root]
        size = 0
        while queue:
            node = queue.pop()
            size += 1
            neighbours = adjacency[node] & remaining
            remaining.difference_update(neighbours)
            queue.extend(neighbours)
        component_sizes.append(size)
    local_pairs = sum(
        1
        for left in active
        for right in adjacency[left]
        if right in active and left < right
    )
    pair_rows = adjacency if radius_adjacency is None else radius_adjacency
    radius_pairs = sum(
        1
        for left in active
        for right in pair_rows[left]
        if right in active and left < right
    )
    active_count = len(active)
    return WorkloadFeatures(
        active_count=active_count,
        active_density=active_count / bits.size if bits.size else 0.0,
        component_count=len(component_sizes),
        largest_component=max(component_sizes, default=0),
        component_square_sum=sum(size * size for size in component_sizes),
        local_pair_count=local_pairs,
        radius_pair_count=radius_pairs,
    )


def workload_score(
    features: WorkloadFeatures, detector_count: int, config: GateConfig
) -> float:
    detector_count = max(1, int(detector_count))
    if config.workload_mode == "legacy":
        pair_scale = max(1, detector_count * (detector_count - 1) // 2)
        return (
            config.workload_active_weight * features.active_density
            + config.workload_component_weight
            * features.component_square_sum
            / (detector_count * detector_count)
            + config.workload_pair_weight * features.local_pair_count / pair_scale
        )

    active_count = int(features.active_count)
    if active_count <= 1:
        component_concentration = 0.0
        radius_pair_concentration = 0.0
    else:
        active_pair_scale = active_count * (active_count - 1)
        component_concentration = (
            features.component_square_sum - active_count
        ) / active_pair_scale
        radius_pair_concentration = (
            2.0 * features.radius_pair_count / active_pair_scale
        )
    return (
        config.workload_active_weight * features.active_density
        + config.workload_component_weight
        * features.active_density
        * component_concentration
        + config.workload_largest_weight
        * features.largest_component
        / detector_count
        + config.workload_pair_weight
        * features.active_density
        * radius_pair_concentration
    )


def _xor_columns(matrix: np.ndarray, actions: tuple[int, ...]) -> np.ndarray:
    if not actions:
        return np.zeros(matrix.shape[0], dtype=np.uint8)
    return np.bitwise_xor.reduce(matrix[:, actions], axis=1).astype(np.uint8, copy=False)


def _uncertainty(probabilities: np.ndarray, actions: tuple[int, ...]) -> float:
    if not actions:
        return 0.0
    selected = np.clip(probabilities[list(actions)], 1e-9, 1 - 1e-9)
    entropy = -selected * np.log(selected) - (1 - selected) * np.log(1 - selected)
    return float(entropy.mean())


def evaluate_cluster(
    syndrome: np.ndarray,
    cluster: Sequence[int],
    probabilities: np.ndarray,
    extended_h: np.ndarray,
    extended_l: np.ndarray,
    detector_adjacency: Sequence[Iterable[int]],
    config: GateConfig,
    *,
    incompatible_pairs: Iterable[tuple[int, int]] = (),
    boundary_risk: np.ndarray | None = None,
    nontrivial_risk: np.ndarray | None = None,
    workload_graph: WorkloadGraph | None = None,
) -> CombinationEvaluation:
    """Select the highest-utility legal combination for one cluster."""

    state = _binary_vector(syndrome, name="syndrome")
    h = _binary_matrix(extended_h, name="extended_h")
    logical = _logical_matrix(extended_l, h.shape[1])
    if h.shape[0] != state.size:
        raise ValueError("extended_h detector count does not match syndrome")
    probability = np.asarray(probabilities, dtype=np.float64)
    if probability.shape != (h.shape[1],):
        raise ValueError("probabilities do not match extended_h actions")
    boundary = (
        np.zeros(h.shape[1], dtype=np.float64)
        if boundary_risk is None
        else np.asarray(boundary_risk, dtype=np.float64)
    )
    nontrivial = (
        np.zeros(h.shape[1], dtype=np.float64)
        if nontrivial_risk is None
        else np.asarray(nontrivial_risk, dtype=np.float64)
    )
    if boundary.shape != probability.shape or nontrivial.shape != probability.shape:
        raise ValueError("risk vectors must match probabilities")
    choices, exact = legal_combinations(
        cluster,
        probability,
        incompatible_pairs=incompatible_pairs,
        exact_cluster_size=config.exact_cluster_size,
        max_actions=config.max_combination_actions,
        budget=config.combination_budget,
    )
    graph = workload_graph or build_workload_graph(
        detector_adjacency, config.workload_pair_radius
    )
    if len(graph.adjacency) != state.size:
        raise ValueError("workload graph detector count does not match syndrome")
    adjacency = graph.adjacency
    before = _workload_features_normalized(
        state, adjacency, graph.radius_adjacency
    )
    before_score = workload_score(before, state.size, config)
    evaluations: list[CombinationEvaluation] = []
    for actions in choices:
        detector_delta = _xor_columns(h, actions)
        trial = state ^ detector_delta
        logical_delta = _xor_columns(logical, actions)
        after = _workload_features_normalized(
            trial, adjacency, graph.radius_adjacency
        )
        after_score = workload_score(after, state.size, config)
        uncertainty = _uncertainty(probability, actions)
        logical_risk = 0.0
        gate_cost = 0.0
        if actions:
            logical_risk = (
                uncertainty * float(logical_delta.sum())
                + float(boundary[list(actions)].mean())
                + float(nontrivial[list(actions)].mean())
            )
            gate_cost = len(actions) / max(1, config.max_combination_actions)
        utility = (
            before_score
            - after_score
            - config.uncertainty_weight * uncertainty
            - config.logical_risk_weight * logical_risk
            - config.gate_cost_weight * gate_cost
        )
        if after_score > before_score + config.max_workload_increase:
            utility = float("-inf")
        if logical_risk > config.max_logical_risk:
            utility = float("-inf")
        evaluations.append(
            CombinationEvaluation(
                actions=actions,
                utility=float(utility),
                uncertainty=uncertainty,
                logical_risk=logical_risk,
                gate_cost_proxy=gate_cost,
                trial_residual=trial,
                logical_delta=logical_delta,
                workload_before=before,
                workload_after=after,
                search_exact=exact,
                combinations_evaluated=len(choices),
            )
        )
    return min(
        evaluations,
        key=lambda item: (
            -item.utility,
            item.logical_risk,
            len(item.actions),
            item.actions,
        ),
    )


def gate_actions(
    syndrome: np.ndarray,
    probabilities: np.ndarray,
    action_types: Sequence[str],
    extended_h: np.ndarray,
    extended_l: np.ndarray,
    detector_adjacency: Sequence[Iterable[int]],
    config: GateConfig,
    *,
    valid_actions: np.ndarray | None = None,
    incompatible_pairs: Iterable[tuple[int, int]] = (),
    boundary_risk: np.ndarray | None = None,
    nontrivial_risk: np.ndarray | None = None,
    workload_graph: WorkloadGraph | None = None,
) -> GateResult:
    """Run the complete typed-candidate, combination, and update loop."""

    state = _binary_vector(syndrome, name="syndrome").copy()
    h = _binary_matrix(extended_h, name="extended_h")
    if h.shape[0] != state.size:
        raise ValueError("extended_h detector count does not match syndrome")
    logical = _logical_matrix(extended_l, h.shape[1])
    graph = workload_graph or build_workload_graph(
        detector_adjacency, config.workload_pair_radius
    )
    probability = np.asarray(probabilities, dtype=np.float64)
    candidates = typed_candidates(
        probability,
        action_types,
        data_threshold=config.data_threshold,
        measurement_threshold=config.measurement_threshold,
        valid_actions=valid_actions,
        max_candidates=config.max_candidates,
    )
    incompatible = _normalise_incompatible(incompatible_pairs, h.shape[1])
    pending = interaction_clusters(
        candidates,
        h,
        graph.adjacency,
        radius=config.interaction_radius,
        incompatible_pairs=incompatible,
    )
    accepted = np.zeros(h.shape[1], dtype=np.uint8)
    local_frame = np.zeros(logical.shape[0], dtype=np.uint8)
    decisions: list[GateDecision] = []
    while pending:
        evaluated = [
            (
                cluster,
                evaluate_cluster(
                    state,
                    cluster,
                    probability,
                    h,
                    logical,
                    detector_adjacency,
                    config,
                    incompatible_pairs=incompatible,
                    boundary_risk=boundary_risk,
                    nontrivial_risk=nontrivial_risk,
                    workload_graph=graph,
                ),
            )
            for cluster in pending
        ]
        cluster, best = min(
            evaluated,
            key=lambda item: (
                -item[1].utility,
                item[1].logical_risk,
                len(item[1].actions),
                item[0],
            ),
        )
        has_budget = accepted.sum() + len(best.actions) <= config.max_total_actions
        accepted_now = bool(
            best.actions
            and has_budget
            and best.utility >= config.acceptance_threshold
            and np.isfinite(best.utility)
        )
        if accepted_now:
            state = best.trial_residual.copy()
            local_frame ^= best.logical_delta
            accepted[list(best.actions)] = 1
            reason = "accepted"
        elif not best.actions:
            reason = "empty_combination_selected"
        elif not has_budget:
            reason = "global_action_budget"
        elif not np.isfinite(best.utility):
            reason = "hard_constraint"
        else:
            reason = "below_acceptance_threshold"
        decisions.append(
            GateDecision(
                cluster=cluster,
                selected_actions=best.actions,
                accepted=accepted_now,
                reason=reason,
                utility=best.utility,
                uncertainty=best.uncertainty,
                logical_risk=best.logical_risk,
                workload_before=best.workload_before,
                workload_after=best.workload_after,
                search_exact=best.search_exact,
                combinations_evaluated=best.combinations_evaluated,
            )
        )
        pending.remove(cluster)
    return GateResult(
        accepted_actions=accepted,
        residual=state,
        local_logical_frame=local_frame,
        candidate_count=int(candidates.size),
        decisions=tuple(decisions),
    )


def pointwise_actions(
    probabilities: np.ndarray,
    action_types: Sequence[str],
    config: GateConfig,
    *,
    valid_actions: np.ndarray | None = None,
) -> np.ndarray:
    """Return the direct-threshold baseline correction vector."""

    candidates = typed_candidates(
        probabilities,
        action_types,
        data_threshold=config.data_threshold,
        measurement_threshold=config.measurement_threshold,
        valid_actions=valid_actions,
        max_candidates=config.max_candidates,
    )
    result = np.zeros(len(action_types), dtype=np.uint8)
    result[candidates] = 1
    return result


def gate_actions_latency_guarded(
    syndrome: np.ndarray,
    probabilities: np.ndarray,
    action_types: Sequence[str],
    extended_h: np.ndarray,
    extended_l: np.ndarray,
    detector_adjacency: Sequence[Iterable[int]],
    config: GateConfig,
    *,
    valid_actions: np.ndarray | None = None,
    incompatible_pairs: Iterable[tuple[int, int]] = (),
    boundary_risk: np.ndarray | None = None,
    nontrivial_risk: np.ndarray | None = None,
    workload_graph: WorkloadGraph | None = None,
) -> GateResult:
    """Use combination gating only when it beats the pointwise S07 reference."""

    graph = workload_graph or build_workload_graph(
        detector_adjacency, config.workload_pair_radius
    )
    combination = gate_actions(
        syndrome,
        probabilities,
        action_types,
        extended_h,
        extended_l,
        detector_adjacency,
        config,
        valid_actions=valid_actions,
        incompatible_pairs=incompatible_pairs,
        boundary_risk=boundary_risk,
        nontrivial_risk=nontrivial_risk,
        workload_graph=graph,
    )
    pointwise = pointwise_actions(
        probabilities, action_types, config, valid_actions=valid_actions
    )
    pointwise_residual, pointwise_frame = apply_actions(
        syndrome, pointwise, extended_h, extended_l
    )
    combination_score = workload_score(
        workload_features(combination.residual, graph),
        combination.residual.size,
        config,
    )
    pointwise_score = workload_score(
        workload_features(pointwise_residual, graph),
        pointwise_residual.size,
        config,
    )
    risk_legal = all(
        (not decision.accepted)
        or decision.logical_risk <= config.max_logical_risk
        for decision in combination.decisions
    )
    budget_legal = combination.accepted_count <= config.max_total_actions
    guard_passed = combination_score <= (
        1.0 - config.pointwise_guard_relative_margin
    ) * pointwise_score
    use_combination = (
        not config.pointwise_guard_enabled
        or (guard_passed and risk_legal and budget_legal)
    )
    if use_combination:
        return GateResult(
            accepted_actions=combination.accepted_actions,
            residual=combination.residual,
            local_logical_frame=combination.local_logical_frame,
            candidate_count=combination.candidate_count,
            decisions=combination.decisions,
            selection_source="combination",
            combination_workload_score=combination_score,
            pointwise_workload_score=pointwise_score,
        )
    return GateResult(
        accepted_actions=pointwise,
        residual=pointwise_residual,
        local_logical_frame=pointwise_frame,
        candidate_count=int(pointwise.sum()),
        decisions=combination.decisions,
        selection_source="pointwise_fallback",
        combination_workload_score=combination_score,
        pointwise_workload_score=pointwise_score,
    )


def apply_actions(
    syndrome: np.ndarray,
    correction: np.ndarray,
    extended_h: np.ndarray,
    extended_l: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply an action vector over GF(2) and return residual and local frame."""

    state = _binary_vector(syndrome, name="syndrome")
    selected = _binary_vector(correction, name="correction")
    h = _binary_matrix(extended_h, name="extended_h")
    logical = _logical_matrix(extended_l, h.shape[1])
    if h.shape != (state.size, selected.size):
        raise ValueError("syndrome/correction dimensions do not match extended_h")
    detector_delta = np.remainder(h.astype(np.int64) @ selected.astype(np.int64), 2)
    logical_delta = np.remainder(logical.astype(np.int64) @ selected.astype(np.int64), 2)
    return (state ^ detector_delta.astype(np.uint8), logical_delta.astype(np.uint8))


def config_from_mapping(value: Mapping[str, object]) -> GateConfig:
    """Build :class:`GateConfig` from a YAML-compatible mapping."""

    known = set(GateConfig.__dataclass_fields__)
    unknown = sorted(set(value) - known)
    if unknown:
        raise ValueError(f"unknown topology-gating config fields: {unknown}")
    return GateConfig(**value)
