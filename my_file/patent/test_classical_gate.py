"""Tests for the patent gate and its production surface-code mapping."""
from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from .classical_gate import Action, ActionSpace, GateConfig, TopologyResidualGate, WorkloadWeights

try:
    import torch
    from .surface_code import FixedActionModel, apply_dense_actions, build_surface_action_space
    from evaluation.logical_error_rate import PreDecoderMemoryEvalModule, _build_stab_maps
    SURFACE_DEPENDENCIES_AVAILABLE = True
    SURFACE_SKIP_REASON = ""
except ModuleNotFoundError as exc:
    SURFACE_DEPENDENCIES_AVAILABLE = False
    SURFACE_SKIP_REASON = f"surface-code runtime dependency unavailable: {exc}"


def _synthetic_space(h: np.ndarray, l: np.ndarray, action_neighbors) -> ActionSpace:
    h = np.asarray(h, dtype=np.uint8)
    l = np.asarray(l, dtype=np.uint8)
    actions = tuple(
        Action(
            index=index,
            family="data_z",
            channel=0,
            time=0,
            row=0,
            column=index,
            detector_support=tuple(int(v) for v in np.flatnonzero(h[:, index])),
            logical_effect=tuple(int(v) for v in l[:, index]),
        )
        for index in range(h.shape[1])
    )
    detector_neighbors = tuple(np.empty(0, dtype=np.int32) for _ in range(h.shape[0]))
    return ActionSpace(
        h=h,
        l=l,
        actions=actions,
        detector_neighbors=detector_neighbors,
        action_neighbors=tuple(np.asarray(values, dtype=np.int32) for values in action_neighbors),
        detector_boundary=np.zeros(h.shape[0], dtype=bool),
        code_distance=max(3, h.shape[1]),
        n_rounds=1,
        dense_shape=(4, 1, max(1, h.shape[1]), max(1, h.shape[1])),
    )


def _simple_config(**overrides) -> GateConfig:
    values = dict(
        data_probability_threshold=0.5,
        measurement_probability_threshold=0.5,
        max_candidates=16,
        max_cluster_size=8,
        max_combination_actions=4,
        max_combinations_per_cluster=64,
        max_accepted_actions=16,
        max_iterations=16,
        utility_threshold=0.0,
        uncertainty_weight=0.0,
        logical_risk_weight=0.0,
        gate_cost_weight=0.0,
        risk_boundary=0.0,
        risk_nontrivial=0.0,
        workload=WorkloadWeights(
            active_count=1.0,
            component_count=0.0,
            largest_component=0.0,
            squared_component_sum=0.0,
            local_pair_count=0.0,
            boundary_active_count=0.0,
        ),
    )
    values.update(overrides)
    return GateConfig(**values)


class ClassicalGateTests(unittest.TestCase):
    def test_gate_accepts_only_the_action_that_reduces_residual(self):
        space = _synthetic_space(np.eye(2, dtype=np.uint8), np.zeros((1, 2), dtype=np.uint8), [[], []])
        result = TopologyResidualGate(space, _simple_config()).run(
            np.asarray([1, 0], dtype=np.uint8), np.asarray([0.9, 0.9])
        )
        np.testing.assert_array_equal(result.residual_syndrome, [0, 0])
        np.testing.assert_array_equal(result.accepted_mask, [True, False])

    def test_cluster_can_select_a_joint_combination(self):
        space = _synthetic_space(
            np.eye(2, dtype=np.uint8),
            np.zeros((1, 2), dtype=np.uint8),
            [[1], [0]],
        )
        result = TopologyResidualGate(space, _simple_config()).run(
            np.asarray([1, 1], dtype=np.uint8), np.asarray([0.9, 0.9])
        )
        np.testing.assert_array_equal(result.residual_syndrome, [0, 0])
        self.assertEqual(result.accepted_count, 2)
        self.assertEqual(len(result.decisions), 1)

    def test_uncertain_logical_flip_is_rejected_by_hard_risk_limit(self):
        space = _synthetic_space(
            np.asarray([[1]], dtype=np.uint8),
            np.asarray([[1]], dtype=np.uint8),
            [[]],
        )
        config = _simple_config(
            logical_risk_weight=0.0,
            max_logical_risk=0.2,
            risk_uncertain_logical=1.0,
        )
        result = TopologyResidualGate(space, config).run(
            np.asarray([1], dtype=np.uint8), np.asarray([0.5])
        )
        self.assertEqual(result.accepted_count, 0)
        np.testing.assert_array_equal(result.local_logical_frame, [0])

    def test_empty_combination_wins_zero_utility_tie(self):
        space = _synthetic_space(
            np.asarray([[0]], dtype=np.uint8),
            np.asarray([[0]], dtype=np.uint8),
            [[]],
        )
        result = TopologyResidualGate(space, _simple_config()).run(
            np.asarray([0], dtype=np.uint8), np.asarray([1.0])
        )
        self.assertEqual(result.accepted_count, 0)

    def test_cluster_budget_prioritizes_uncertain_logical_cluster(self):
        space = _synthetic_space(
            np.eye(2, dtype=np.uint8),
            np.asarray([[0, 1]], dtype=np.uint8),
            [[], []],
        )
        config = _simple_config(
            max_iterations=1,
            max_clusters_per_iteration=1,
            cluster_priority_risk_weight=1.0,
            cluster_priority_workload_weight=0.0,
        )
        result = TopologyResidualGate(space, config).run(
            np.asarray([1, 1], dtype=np.uint8), np.asarray([0.9, 0.5])
        )
        np.testing.assert_array_equal(result.accepted_mask, [False, True])
        self.assertEqual(result.available_cluster_evaluations, 2)
        self.assertEqual(result.processed_cluster_evaluations, 1)
        self.assertGreater(result.decisions[0].cluster_risk, 0.0)

    def test_cluster_budget_can_prioritize_backend_workload(self):
        space = _synthetic_space(
            np.eye(2, dtype=np.uint8),
            np.zeros((1, 2), dtype=np.uint8),
            [[], []],
        )
        config = _simple_config(
            max_iterations=1,
            max_clusters_per_iteration=1,
            cluster_priority_risk_weight=0.0,
            cluster_priority_workload_weight=1.0,
        )
        result = TopologyResidualGate(space, config).run(
            np.asarray([0, 1], dtype=np.uint8), np.asarray([0.9, 0.9])
        )
        np.testing.assert_array_equal(result.accepted_mask, [False, True])
        self.assertEqual(result.available_cluster_evaluations, 2)
        self.assertEqual(result.processed_cluster_evaluations, 1)
        self.assertEqual(result.decisions[0].cluster_workload_share, 1.0)

    def test_harmful_veto_starts_from_accept_all_and_removes_harmful_action(self):
        space = _synthetic_space(
            np.eye(2, dtype=np.uint8),
            np.zeros((1, 2), dtype=np.uint8),
            [[], []],
        )
        config = _simple_config(selection_mode="harmful_veto")
        result = TopologyResidualGate(space, config).run(
            np.asarray([0, 1], dtype=np.uint8), np.asarray([0.9, 0.9])
        )
        np.testing.assert_array_equal(result.accepted_mask, [False, True])
        np.testing.assert_array_equal(result.vetoed_mask, [True, False])
        np.testing.assert_array_equal(result.residual_syndrome, [0, 0])
        self.assertEqual(result.decisions[0].decision_kind, "veto")

    def test_harmful_veto_does_not_remove_useful_action_only_for_risk(self):
        space = _synthetic_space(
            np.asarray([[1]], dtype=np.uint8),
            np.asarray([[1]], dtype=np.uint8),
            [[]],
        )
        config = _simple_config(
            selection_mode="harmful_veto",
            uncertainty_weight=1.0,
            logical_risk_weight=1.0,
        )
        result = TopologyResidualGate(space, config).run(
            np.asarray([1], dtype=np.uint8), np.asarray([0.5])
        )
        np.testing.assert_array_equal(result.accepted_mask, [True])
        np.testing.assert_array_equal(result.vetoed_mask, [False])
        np.testing.assert_array_equal(result.residual_syndrome, [0])

    def test_harmful_veto_ranks_lower_confidence_action_for_removal(self):
        space = _synthetic_space(
            np.eye(2, dtype=np.uint8),
            np.zeros((1, 2), dtype=np.uint8),
            [[1], [0]],
        )
        gate = TopologyResidualGate(space, _simple_config(selection_mode="harmful_veto"))
        ranked = gate._ranked_combinations(
            (0, 1), np.asarray([0.51, 0.99]), veto=True
        )
        self.assertEqual(ranked[0], (0,))


@unittest.skipUnless(SURFACE_DEPENDENCIES_AVAILABLE, SURFACE_SKIP_REASON)
class SurfaceMappingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.distance = 3
        cls.n_rounds = 3
        cls.basis = "X"
        cls.rotation = "XV"
        cls.cfg = SimpleNamespace(
            distance=cls.distance,
            n_rounds=cls.n_rounds,
            enable_fp16=False,
            data=SimpleNamespace(code_rotation=cls.rotation),
            test=SimpleNamespace(
                meas_basis_test=cls.basis,
                n_rounds=cls.n_rounds,
                th_data=0.0,
                th_syn=0.0,
                sampling_mode="threshold",
                temperature=1.0,
            ),
        )
        cls.space = build_surface_action_space(
            cls.cfg,
            distance=cls.distance,
            n_rounds=cls.n_rounds,
            basis=cls.basis,
            rotation=cls.rotation,
            device="cpu",
            probe_batch_size=64,
        )

    def test_measurement_actions_have_zero_direct_logical_frame(self):
        for action in self.space.actions:
            if action.is_measurement:
                self.assertEqual(action.logical_effect, (0,))

    def test_unit_probed_maps_match_production_endpoint_for_random_combinations(self):
        rng = np.random.default_rng(20260817)
        batch = 8
        masks = rng.random((batch, self.space.num_actions)) < 0.08
        dense = np.stack([self.space.dense_from_mask(mask) for mask in masks])
        detectors = rng.integers(
            0, 2, size=(batch, self.space.num_detectors), dtype=np.uint8
        )
        expected_residual, expected_l, _ = apply_dense_actions(self.space, detectors, dense)

        fixed = FixedActionModel().eval()
        fixed.set_actions(torch.as_tensor(dense, dtype=torch.uint8))
        endpoint = PreDecoderMemoryEvalModule(
            fixed,
            self.cfg,
            _build_stab_maps(self.distance, self.rotation),
            torch.device("cpu"),
        ).eval()
        with torch.no_grad():
            actual = endpoint(torch.as_tensor(detectors, dtype=torch.uint8)).cpu().numpy()
        np.testing.assert_array_equal(actual[:, 0:1], expected_l)
        np.testing.assert_array_equal(actual[:, 1:], expected_residual)


if __name__ == "__main__":
    unittest.main()
