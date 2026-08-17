"""Classical topology-residual utility gate from the patent disclosure.

The gate is deliberately model-agnostic.  It consumes calibrated action
probabilities and the current detector vector, while ``ActionSpace`` supplies
the circuit/code metadata that is fixed offline:

* ``H[:, i]`` is the detector delta caused by candidate action ``i``;
* ``L[:, i]`` is the tracked logical-frame increment caused by action ``i``.

``L`` does *not* reveal whether a correction is logically correct.  Ground
truth observables are consequently absent from this module and are used only
by the separate evaluation script.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from itertools import combinations
import math
import time
from typing import Mapping, Sequence

import numpy as np


DATA_FAMILIES = frozenset({"data_z", "data_x"})
MEASUREMENT_FAMILIES = frozenset({"meas_x", "meas_z"})
ALL_FAMILIES = DATA_FAMILIES | MEASUREMENT_FAMILIES


@dataclass(frozen=True)
class Action:
    """One typed correction action at a space-time location."""

    index: int
    family: str
    channel: int
    time: int
    row: int
    column: int
    detector_support: tuple[int, ...]
    logical_effect: tuple[int, ...]
    boundary_score: float = 0.0

    def __post_init__(self) -> None:
        if self.family not in ALL_FAMILIES:
            raise ValueError(f"unsupported action family {self.family!r}")
        if not 0.0 <= float(self.boundary_score) <= 1.0:
            raise ValueError("boundary_score must be in [0, 1]")

    @property
    def is_measurement(self) -> bool:
        return self.family in MEASUREMENT_FAMILIES


@dataclass
class ActionSpace:
    """Offline action metadata and sparse topology used by the online gate."""

    h: np.ndarray
    l: np.ndarray
    actions: tuple[Action, ...]
    detector_neighbors: tuple[np.ndarray, ...]
    action_neighbors: tuple[np.ndarray, ...]
    detector_boundary: np.ndarray
    code_distance: int
    n_rounds: int
    dense_shape: tuple[int, int, int, int]

    def __post_init__(self) -> None:
        self.h = np.ascontiguousarray(self.h, dtype=np.uint8)
        self.l = np.ascontiguousarray(self.l, dtype=np.uint8)
        self.detector_boundary = np.asarray(self.detector_boundary, dtype=bool)
        if self.h.ndim != 2 or self.l.ndim != 2:
            raise ValueError("H and L must be two-dimensional")
        if self.h.shape[1] != len(self.actions) or self.l.shape[1] != len(self.actions):
            raise ValueError("H/L column counts must equal the number of actions")
        if self.h.shape[0] != len(self.detector_neighbors):
            raise ValueError("detector topology size does not match H")
        if len(self.action_neighbors) != len(self.actions):
            raise ValueError("action topology size does not match actions")
        if self.detector_boundary.shape != (self.h.shape[0],):
            raise ValueError("detector_boundary must have one entry per detector")
        if self.dense_shape[0] != 4:
            raise ValueError("dense action shape must be (4, T, D, D)")
        for expected, action in enumerate(self.actions):
            if action.index != expected:
                raise ValueError("actions must use contiguous indices")

    @property
    def num_detectors(self) -> int:
        return int(self.h.shape[0])

    @property
    def num_actions(self) -> int:
        return int(self.h.shape[1])

    @property
    def num_logicals(self) -> int:
        return int(self.l.shape[0])

    def probabilities_from_dense(self, dense: np.ndarray) -> np.ndarray:
        dense = np.asarray(dense)
        if dense.shape != self.dense_shape:
            raise ValueError(f"dense tensor has shape {dense.shape}, expected {self.dense_shape}")
        return np.asarray(
            [dense[a.channel, a.time, a.row, a.column] for a in self.actions],
            dtype=np.float64,
        )

    def mask_from_dense(self, dense: np.ndarray) -> np.ndarray:
        return self.probabilities_from_dense(dense).astype(bool)

    def dense_from_mask(self, mask: np.ndarray) -> np.ndarray:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (self.num_actions,):
            raise ValueError("action mask has the wrong shape")
        dense = np.zeros(self.dense_shape, dtype=np.uint8)
        for action in self.actions:
            if mask[action.index]:
                dense[action.channel, action.time, action.row, action.column] = 1
        return dense

    def apply_mask(self, syndrome: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        syndrome = np.asarray(syndrome, dtype=np.uint8).reshape(-1)
        mask = np.asarray(mask, dtype=np.uint8).reshape(-1)
        if syndrome.shape != (self.num_detectors,) or mask.shape != (self.num_actions,):
            raise ValueError("syndrome or action mask has the wrong shape")
        selected = np.flatnonzero(mask)
        if selected.size:
            detector_delta = np.bitwise_xor.reduce(self.h[:, selected], axis=1)
            logical_delta = np.bitwise_xor.reduce(self.l[:, selected], axis=1)
        else:
            detector_delta = np.zeros(self.num_detectors, dtype=np.uint8)
            logical_delta = np.zeros(self.num_logicals, dtype=np.uint8)
        return np.bitwise_xor(syndrome, detector_delta), logical_delta


@dataclass(frozen=True)
class WorkloadWeights:
    """Weights for the normalized residual-topology workload proxy."""

    active_count: float = 1.0
    component_count: float = 0.10
    largest_component: float = 0.25
    squared_component_sum: float = 0.50
    local_pair_count: float = 0.10
    boundary_active_count: float = 0.10

    @classmethod
    def from_mapping(cls, values: Mapping[str, float] | None) -> "WorkloadWeights":
        return cls(**({} if values is None else dict(values)))


@dataclass(frozen=True)
class WorkloadFeatures:
    active_count: int
    component_count: int
    largest_component: int
    squared_component_sum: int
    local_pair_count: int
    boundary_active_count: int
    normalized_score: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True)
class GateConfig:
    """Search, risk, and acceptance parameters for the online gate."""

    data_probability_threshold: float = 0.50
    measurement_probability_threshold: float = 0.50
    max_candidates: int = 256
    max_cluster_size: int = 8
    max_combination_actions: int = 4
    max_combinations_per_cluster: int = 128
    max_accepted_actions: int = 128
    max_iterations: int = 128
    utility_threshold: float = 0.0
    workload_increase_tolerance: float = 0.0
    max_logical_risk: float = 1.0
    uncertainty_weight: float = 0.10
    logical_risk_weight: float = 0.25
    gate_cost_weight: float = 0.0
    risk_uncertain_logical: float = 1.0
    risk_boundary: float = 0.10
    risk_nontrivial: float = 0.10
    gate_cost_per_action: float = 1.0
    gate_cost_per_evaluation: float = 0.01
    probability_epsilon: float = 1e-7
    workload: WorkloadWeights = field(default_factory=WorkloadWeights)

    def __post_init__(self) -> None:
        for value, name in (
            (self.data_probability_threshold, "data_probability_threshold"),
            (self.measurement_probability_threshold, "measurement_probability_threshold"),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        for value, name in (
            (self.max_candidates, "max_candidates"),
            (self.max_cluster_size, "max_cluster_size"),
            (self.max_combination_actions, "max_combination_actions"),
            (self.max_combinations_per_cluster, "max_combinations_per_cluster"),
            (self.max_accepted_actions, "max_accepted_actions"),
            (self.max_iterations, "max_iterations"),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")

    @classmethod
    def from_mapping(cls, values: Mapping[str, object] | None) -> "GateConfig":
        raw = {} if values is None else dict(values)
        workload = WorkloadWeights.from_mapping(raw.pop("workload", None))
        return cls(workload=workload, **raw)


@dataclass(frozen=True)
class GateDecision:
    iteration: int
    cluster_size: int
    action_indices: tuple[int, ...]
    utility: float
    workload_before: float
    workload_after: float
    uncertainty: float
    logical_effect: tuple[int, ...]
    logical_risk: float
    gate_cost: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class GateResult:
    residual_syndrome: np.ndarray
    local_logical_frame: np.ndarray
    accepted_mask: np.ndarray
    candidate_mask: np.ndarray
    decisions: tuple[GateDecision, ...]
    initial_workload: WorkloadFeatures
    final_workload: WorkloadFeatures
    evaluated_combinations: int
    elapsed_seconds: float

    @property
    def accepted_count(self) -> int:
        return int(self.accepted_mask.sum())

    @property
    def candidate_count(self) -> int:
        return int(self.candidate_mask.sum())

    def summary(self) -> dict[str, object]:
        return {
            "candidate_count": self.candidate_count,
            "accepted_count": self.accepted_count,
            "evaluated_combinations": self.evaluated_combinations,
            "elapsed_seconds": self.elapsed_seconds,
            "initial_workload": self.initial_workload.to_dict(),
            "final_workload": self.final_workload.to_dict(),
            "decisions": [decision.to_dict() for decision in self.decisions],
        }


@dataclass(frozen=True)
class _Proposal:
    cluster_size: int
    action_indices: tuple[int, ...]
    utility: float
    workload_before: WorkloadFeatures
    workload_after: WorkloadFeatures
    uncertainty: float
    logical_effect: tuple[int, ...]
    logical_risk: float
    gate_cost: float
    residual: np.ndarray
    evaluated: int


class TopologyResidualGate:
    """State-dependent candidate-cluster-combination utility gate."""

    def __init__(self, action_space: ActionSpace, config: GateConfig | None = None):
        self.space = action_space
        self.config = config or GateConfig()

    def workload(self, syndrome: np.ndarray) -> WorkloadFeatures:
        syndrome = np.asarray(syndrome, dtype=np.uint8).reshape(-1)
        if syndrome.shape != (self.space.num_detectors,):
            raise ValueError("syndrome has the wrong width")
        active = syndrome.astype(bool)
        active_indices = np.flatnonzero(active)
        visited = np.zeros(self.space.num_detectors, dtype=bool)
        component_sizes: list[int] = []
        local_pairs = 0
        for node in active_indices:
            local_pairs += int(active[self.space.detector_neighbors[node]].sum())
            if visited[node]:
                continue
            stack = [int(node)]
            visited[node] = True
            size = 0
            while stack:
                current = stack.pop()
                size += 1
                for neighbor in self.space.detector_neighbors[current]:
                    neighbor = int(neighbor)
                    if active[neighbor] and not visited[neighbor]:
                        visited[neighbor] = True
                        stack.append(neighbor)
            component_sizes.append(size)
        local_pairs //= 2
        active_count = int(active_indices.size)
        component_count = len(component_sizes)
        largest = max(component_sizes, default=0)
        squared_sum = sum(size * size for size in component_sizes)
        boundary_count = int(np.logical_and(active, self.space.detector_boundary).sum())
        nd = max(1, self.space.num_detectors)
        possible_pairs = max(1, sum(len(n) for n in self.space.detector_neighbors) // 2)
        w = self.config.workload
        score = (
            w.active_count * active_count / nd
            + w.component_count * component_count / nd
            + w.largest_component * largest / nd
            + w.squared_component_sum * squared_sum / (nd * nd)
            + w.local_pair_count * local_pairs / possible_pairs
            + w.boundary_active_count * boundary_count / nd
        )
        return WorkloadFeatures(
            active_count=active_count,
            component_count=component_count,
            largest_component=largest,
            squared_component_sum=squared_sum,
            local_pair_count=local_pairs,
            boundary_active_count=boundary_count,
            normalized_score=float(score),
        )

    def _candidate_mask(self, probabilities: np.ndarray) -> np.ndarray:
        probabilities = np.asarray(probabilities, dtype=np.float64).reshape(-1)
        if probabilities.shape != (self.space.num_actions,):
            raise ValueError("probabilities have the wrong shape")
        if not np.all(np.isfinite(probabilities)):
            raise ValueError("probabilities must be finite")
        probabilities = np.clip(probabilities, 0.0, 1.0)
        thresholds = np.asarray(
            [
                self.config.measurement_probability_threshold
                if action.is_measurement
                else self.config.data_probability_threshold
                for action in self.space.actions
            ],
            dtype=np.float64,
        )
        mask = probabilities >= thresholds
        selected = np.flatnonzero(mask)
        if selected.size > self.config.max_candidates:
            keep = selected[np.argsort(-probabilities[selected], kind="stable")[: self.config.max_candidates]]
            mask[:] = False
            mask[keep] = True
        return mask

    def _clusters(self, remaining: np.ndarray, probabilities: np.ndarray) -> list[tuple[int, ...]]:
        active = np.zeros(self.space.num_actions, dtype=bool)
        active[remaining] = True
        visited = np.zeros(self.space.num_actions, dtype=bool)
        components: list[list[int]] = []
        order = remaining[np.argsort(-probabilities[remaining], kind="stable")]
        for seed in order:
            seed = int(seed)
            if visited[seed]:
                continue
            component: list[int] = []
            stack = [seed]
            visited[seed] = True
            while stack:
                current = stack.pop()
                component.append(current)
                neighbors = self.space.action_neighbors[current]
                for neighbor in neighbors:
                    neighbor = int(neighbor)
                    if active[neighbor] and not visited[neighbor]:
                        visited[neighbor] = True
                        stack.append(neighbor)
            components.append(component)

        result: list[tuple[int, ...]] = []
        limit = self.config.max_cluster_size
        for component in components:
            unassigned = set(component)
            # Split an oversized component into bounded connected windows.
            # Each window grows from its highest-confidence remaining seed,
            # preserving local interactions instead of globally chunking by p.
            while unassigned:
                seed = min(unassigned, key=lambda idx: (-probabilities[idx], idx))
                queue = [seed]
                queued = {seed}
                window: list[int] = []
                while queue and len(window) < limit:
                    current = queue.pop(0)
                    queued.discard(current)
                    if current not in unassigned:
                        continue
                    unassigned.remove(current)
                    window.append(current)
                    local = [
                        int(value)
                        for value in self.space.action_neighbors[current]
                        if int(value) in unassigned and int(value) not in queued
                    ]
                    local.sort(key=lambda idx: (-probabilities[idx], idx))
                    queue.extend(local)
                    queued.update(local)
                result.append(tuple(window))
        return result

    def _ranked_combinations(
        self, cluster: Sequence[int], probabilities: np.ndarray
    ) -> list[tuple[int, ...]]:
        eps = self.config.probability_epsilon
        cluster = tuple(int(i) for i in cluster)
        p = np.clip(probabilities[list(cluster)], eps, 1.0 - eps)
        entries: list[tuple[float, tuple[int, ...]]] = []
        max_size = min(len(cluster), self.config.max_combination_actions)
        for size in range(1, max_size + 1):
            for chosen in combinations(range(len(cluster)), size):
                chosen_set = set(chosen)
                log_likelihood = sum(
                    math.log(p[pos]) if pos in chosen_set else math.log1p(-p[pos])
                    for pos in range(len(cluster))
                )
                entries.append((log_likelihood, tuple(cluster[pos] for pos in chosen)))
        entries.sort(key=lambda item: (-item[0], len(item[1]), item[1]))
        return [entry[1] for entry in entries[: self.config.max_combinations_per_cluster]]

    def _uncertainty(self, chosen: Sequence[int], probabilities: np.ndarray) -> float:
        eps = self.config.probability_epsilon
        p = np.clip(probabilities[list(chosen)], eps, 1.0 - eps)
        entropy = -(p * np.log(p) + (1.0 - p) * np.log1p(-p)) / math.log(2.0)
        return float(entropy.mean())

    def _risk(
        self,
        chosen: Sequence[int],
        uncertainty: float,
        logical_effect: np.ndarray,
    ) -> float:
        actions = [self.space.actions[index] for index in chosen]
        logical_weight = int(np.asarray(logical_effect, dtype=bool).sum())
        boundary = float(np.mean([action.boundary_score for action in actions]))
        rows = [action.row for action in actions]
        columns = [action.column for action in actions]
        times = [action.time for action in actions]
        spatial_span = (
            (max(rows) - min(rows)) + (max(columns) - min(columns))
        ) / max(1.0, 2.0 * (self.space.code_distance - 1))
        temporal_span = (max(times) - min(times)) / max(1.0, self.space.n_rounds - 1)
        nontrivial = min(1.0, max(spatial_span, temporal_span))
        return float(
            self.config.risk_uncertain_logical * uncertainty * logical_weight
            + self.config.risk_boundary * boundary
            + self.config.risk_nontrivial * nontrivial
        )

    def _best_proposal(
        self,
        cluster: Sequence[int],
        syndrome: np.ndarray,
        before: WorkloadFeatures,
        probabilities: np.ndarray,
    ) -> _Proposal | None:
        best: _Proposal | None = None
        evaluated = 0
        for chosen in self._ranked_combinations(cluster, probabilities):
            evaluated += 1
            mask = np.zeros(self.space.num_actions, dtype=np.uint8)
            mask[list(chosen)] = 1
            trial, logical = self.space.apply_mask(syndrome, mask)
            after = self.workload(trial)
            uncertainty = self._uncertainty(chosen, probabilities)
            risk = self._risk(chosen, uncertainty, logical)
            gate_cost = (
                self.config.gate_cost_per_action * len(chosen)
                + self.config.gate_cost_per_evaluation * evaluated
            )
            utility = (
                before.normalized_score
                - after.normalized_score
                - self.config.uncertainty_weight * uncertainty
                - self.config.logical_risk_weight * risk
                - self.config.gate_cost_weight * gate_cost
            )
            if after.normalized_score > before.normalized_score + self.config.workload_increase_tolerance:
                continue
            if risk > self.config.max_logical_risk:
                continue
            proposal = _Proposal(
                cluster_size=len(cluster),
                action_indices=tuple(chosen),
                utility=float(utility),
                workload_before=before,
                workload_after=after,
                uncertainty=uncertainty,
                logical_effect=tuple(int(v) for v in logical),
                logical_risk=risk,
                gate_cost=float(gate_cost),
                residual=trial,
                evaluated=evaluated,
            )
            if best is None or (proposal.utility, -len(chosen), tuple(-i for i in chosen)) > (
                best.utility,
                -len(best.action_indices),
                tuple(-i for i in best.action_indices),
            ):
                best = proposal
        if best is None:
            return None
        # Preserve the actual number of combinations examined, including
        # infeasible ones after the last feasible proposal.
        return _Proposal(**{**best.__dict__, "evaluated": evaluated})

    def run(self, syndrome: np.ndarray, probabilities: np.ndarray) -> GateResult:
        started = time.perf_counter()
        syndrome = np.asarray(syndrome, dtype=np.uint8).reshape(-1)
        probabilities = np.asarray(probabilities, dtype=np.float64).reshape(-1)
        if syndrome.shape != (self.space.num_detectors,):
            raise ValueError("syndrome has the wrong width")
        candidate_mask = self._candidate_mask(probabilities)
        accepted = np.zeros(self.space.num_actions, dtype=bool)
        residual = syndrome.copy()
        local_frame = np.zeros(self.space.num_logicals, dtype=np.uint8)
        initial_workload = self.workload(residual)
        decisions: list[GateDecision] = []
        evaluated_total = 0

        for iteration in range(self.config.max_iterations):
            remaining = np.flatnonzero(candidate_mask & ~accepted)
            if remaining.size == 0 or int(accepted.sum()) >= self.config.max_accepted_actions:
                break
            before = self.workload(residual)
            proposals: list[_Proposal] = []
            for cluster in self._clusters(remaining, probabilities):
                remaining_budget = self.config.max_accepted_actions - int(accepted.sum())
                if remaining_budget <= 0:
                    break
                proposal = self._best_proposal(cluster, residual, before, probabilities)
                if proposal is not None and len(proposal.action_indices) <= remaining_budget:
                    proposals.append(proposal)
                evaluated_total += (
                    proposal.evaluated
                    if proposal is not None
                    else min(
                        self.config.max_combinations_per_cluster,
                        sum(
                            math.comb(len(cluster), size)
                            for size in range(
                                1,
                                min(len(cluster), self.config.max_combination_actions) + 1,
                            )
                        ),
                    )
                )
            # The empty combination has utility zero.  Prefer that safe no-op
            # on a tie, even when gamma_a is configured as zero or negative.
            acceptable = [
                p
                for p in proposals
                if p.utility > 0.0 and p.utility >= self.config.utility_threshold
            ]
            if not acceptable:
                break
            chosen = max(
                acceptable,
                key=lambda p: (p.utility, -p.logical_risk, -len(p.action_indices), tuple(-i for i in p.action_indices)),
            )
            accepted[list(chosen.action_indices)] = True
            residual = chosen.residual
            local_frame ^= np.asarray(chosen.logical_effect, dtype=np.uint8)
            decisions.append(
                GateDecision(
                    iteration=iteration,
                    cluster_size=chosen.cluster_size,
                    action_indices=chosen.action_indices,
                    utility=chosen.utility,
                    workload_before=chosen.workload_before.normalized_score,
                    workload_after=chosen.workload_after.normalized_score,
                    uncertainty=chosen.uncertainty,
                    logical_effect=chosen.logical_effect,
                    logical_risk=chosen.logical_risk,
                    gate_cost=chosen.gate_cost,
                )
            )

        return GateResult(
            residual_syndrome=residual,
            local_logical_frame=local_frame,
            accepted_mask=accepted,
            candidate_mask=candidate_mask,
            decisions=tuple(decisions),
            initial_workload=initial_workload,
            final_workload=self.workload(residual),
            evaluated_combinations=evaluated_total,
            elapsed_seconds=time.perf_counter() - started,
        )


def sigmoid(values: np.ndarray) -> np.ndarray:
    """Stable NumPy sigmoid used by the evaluation path."""

    values = np.asarray(values, dtype=np.float64)
    output = np.empty_like(values)
    positive = values >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    output[~positive] = exp_values / (1.0 + exp_values)
    return output


__all__ = [
    "Action",
    "ActionSpace",
    "GateConfig",
    "GateDecision",
    "GateResult",
    "TopologyResidualGate",
    "WorkloadFeatures",
    "WorkloadWeights",
    "sigmoid",
]
