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

from __future__ import annotations

import unittest

import numpy as np

from evaluation.topology_gating_v2 import (
    DATA_X,
    DATA_Z,
    MEASUREMENT_X,
    MEASUREMENT_Z,
    GateConfig,
    WorkloadFeatures,
    apply_actions,
    build_workload_graph,
    gate_actions,
    gate_actions_latency_guarded,
    interaction_clusters,
    legal_combinations,
    typed_candidates,
    workload_features,
    workload_score,
)


def line_adjacency(size: int) -> list[set[int]]:
    result = [set() for _ in range(size)]
    for index in range(size - 1):
        result[index].add(index + 1)
        result[index + 1].add(index)
    return result


class TopologyGatingV2Test(unittest.TestCase):
    def test_typed_candidates_use_distinct_data_and_measurement_thresholds(self):
        candidates = typed_candidates(
            np.array([0.75, 0.65, 0.65, 0.85]),
            [DATA_Z, DATA_X, MEASUREMENT_X, MEASUREMENT_Z],
            data_threshold=0.7,
            measurement_threshold=0.8,
        )
        np.testing.assert_array_equal(candidates, np.array([0, 3]))

    def test_candidate_budget_keeps_highest_probabilities(self):
        candidates = typed_candidates(
            np.array([0.8, 0.95, 0.7, 0.9]),
            [DATA_Z, DATA_X, MEASUREMENT_X, MEASUREMENT_Z],
            data_threshold=0.5,
            measurement_threshold=0.5,
            max_candidates=2,
        )
        np.testing.assert_array_equal(candidates, np.array([1, 3]))


    def test_interaction_clusters_include_adjacent_detector_support(self):
        h = np.array(
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [0, 0, 0],
            ],
            dtype=np.uint8,
        )
        clusters = interaction_clusters(
            [0, 1, 2], h, line_adjacency(4), radius=1
        )
        self.assertEqual(clusters, [(0, 1, 2)])

    def test_legal_combinations_keep_empty_and_apply_conflicts(self):
        values, exact = legal_combinations(
            [0, 1, 2],
            np.array([0.9, 0.8, 0.7]),
            incompatible_pairs={(0, 1)},
            exact_cluster_size=4,
            max_actions=3,
            budget=16,
        )
        self.assertTrue(exact)
        self.assertIn((), values)
        self.assertIn((0, 2), values)
        self.assertNotIn((0, 1), values)
        self.assertNotIn((0, 1, 2), values)

    def test_gate_selects_best_subset_instead_of_committing_whole_cluster(self):
        syndrome = np.array([1, 1, 0], dtype=np.uint8)
        h = np.array(
            [
                [1, 0, 1],
                [0, 1, 1],
                [0, 0, 1],
            ],
            dtype=np.uint8,
        )
        result = gate_actions(
            syndrome,
            np.array([0.9, 0.9, 0.95]),
            [DATA_Z, DATA_X, MEASUREMENT_X],
            h,
            np.zeros((1, 3), dtype=np.uint8),
            line_adjacency(3),
            GateConfig(
                data_threshold=0.5,
                measurement_threshold=0.5,
                max_combination_actions=3,
                workload_component_weight=0.0,
                workload_pair_weight=0.0,
                uncertainty_weight=0.0,
                logical_risk_weight=0.0,
                max_workload_increase=1.0,
            ),
        )
        np.testing.assert_array_equal(result.accepted_actions, np.array([1, 1, 0]))
        np.testing.assert_array_equal(result.residual, np.zeros(3, dtype=np.uint8))
        self.assertEqual(result.decisions[0].selected_actions, (0, 1))

    def test_logical_risk_separates_detector_degenerate_actions(self):
        result = gate_actions(
            np.array([1], dtype=np.uint8),
            np.array([0.9, 0.9]),
            [DATA_Z, DATA_Z],
            np.array([[1, 1]], dtype=np.uint8),
            np.array([[0, 1]], dtype=np.uint8),
            [set()],
            GateConfig(
                data_threshold=0.5,
                measurement_threshold=0.5,
                workload_component_weight=0.0,
                workload_pair_weight=0.0,
                uncertainty_weight=0.0,
                logical_risk_weight=10.0,
                max_workload_increase=1.0,
            ),
            incompatible_pairs={(0, 1)},
        )
        self.assertEqual(result.decisions[0].selected_actions, (0,))
        np.testing.assert_array_equal(result.local_logical_frame, np.array([0]))

    def test_remaining_clusters_are_recomputed_after_state_update(self):
        result = gate_actions(
            np.array([1, 0, 1], dtype=np.uint8),
            np.array([0.9, 0.9]),
            [DATA_Z, MEASUREMENT_Z],
            np.array([[1, 0], [0, 0], [0, 1]], dtype=np.uint8),
            np.zeros((1, 2), dtype=np.uint8),
            [set(), set(), set()],
            GateConfig(
                data_threshold=0.5,
                measurement_threshold=0.5,
                interaction_radius=0,
                workload_component_weight=0.0,
                workload_pair_weight=0.0,
                uncertainty_weight=0.0,
                logical_risk_weight=0.0,
                max_workload_increase=1.0,
            ),
        )
        self.assertEqual(len(result.decisions), 2)
        self.assertEqual(result.decisions[0].workload_before.active_count, 2)
        self.assertEqual(result.decisions[1].workload_before.active_count, 1)
        np.testing.assert_array_equal(result.residual, np.zeros(3, dtype=np.uint8))

    def test_legacy_workload_score_is_bitwise_compatible(self):
        features = WorkloadFeatures(
            active_count=3,
            active_density=0.3,
            component_count=2,
            largest_component=2,
            component_square_sum=5,
            local_pair_count=1,
            radius_pair_count=3,
        )
        config = GateConfig(
            workload_mode="legacy",
            workload_active_weight=1.25,
            workload_component_weight=0.75,
            workload_pair_weight=0.2,
            workload_largest_weight=99.0,
            workload_pair_radius=2,
        )
        expected = 1.25 * 0.3 + 0.75 * 5 / 100 + 0.2 * 1 / 45
        self.assertEqual(workload_score(features, 10, config), expected)

    def test_topology_v3_zero_and_single_active_are_finite(self):
        config = GateConfig(
            workload_mode="topology_v3",
            workload_component_weight=1.0,
            workload_largest_weight=0.5,
            workload_pair_weight=1.0,
        )
        graph = build_workload_graph(line_adjacency(3), pair_radius=2)
        for syndrome in (
            np.zeros(3, dtype=np.uint8),
            np.array([0, 1, 0], dtype=np.uint8),
        ):
            score = workload_score(workload_features(syndrome, graph), 3, config)
            self.assertTrue(np.isfinite(score))

    def test_topology_v3_terms_are_monotone(self):
        config = GateConfig(
            workload_mode="topology_v3",
            workload_component_weight=1.0,
            workload_largest_weight=1.0,
            workload_pair_weight=1.0,
        )
        base = WorkloadFeatures(3, 0.3, 3, 1, 3, 0, 0)
        larger = WorkloadFeatures(3, 0.3, 2, 2, 5, 1, 1)
        more_pairs = WorkloadFeatures(3, 0.3, 2, 2, 5, 1, 2)
        more_active = WorkloadFeatures(4, 0.4, 2, 2, 8, 2, 3)
        self.assertLess(workload_score(base, 10, config), workload_score(larger, 10, config))
        self.assertLess(
            workload_score(larger, 10, config),
            workload_score(more_pairs, 10, config),
        )
        self.assertLess(
            workload_score(more_pairs, 10, config),
            workload_score(more_active, 10, config),
        )

    def test_radius_cache_matches_direct_line_graph_distance(self):
        graph = build_workload_graph(line_adjacency(5), pair_radius=2)
        self.assertEqual(graph.radius_adjacency[0], frozenset((1, 2)))
        self.assertEqual(graph.radius_adjacency[2], frozenset((0, 1, 3, 4)))
        features = workload_features(
            np.array([1, 0, 1, 0, 1], dtype=np.uint8), graph
        )
        self.assertEqual(features.radius_pair_count, 2)

    def test_pointwise_guard_fallback_is_atomic(self):
        syndrome = np.array([1], dtype=np.uint8)
        probabilities = np.array([0.9])
        action_types = [DATA_Z]
        h = np.array([[1]], dtype=np.uint8)
        logical = np.array([[1]], dtype=np.uint8)
        config = GateConfig(
            data_threshold=0.5,
            measurement_threshold=0.5,
            workload_mode="topology_v3",
            workload_component_weight=0.0,
            workload_largest_weight=0.0,
            workload_pair_weight=0.0,
            uncertainty_weight=0.0,
            logical_risk_weight=0.0,
            max_workload_increase=1.0,
            acceptance_threshold=100.0,
            pointwise_guard_enabled=True,
        )
        result = gate_actions_latency_guarded(
            syndrome,
            probabilities,
            action_types,
            h,
            logical,
            [set()],
            config,
        )
        expected_actions = np.ones(1, dtype=np.uint8)
        expected_residual, expected_frame = apply_actions(
            syndrome, expected_actions, h, logical
        )
        self.assertEqual(result.selection_source, "pointwise_fallback")
        np.testing.assert_array_equal(result.accepted_actions, expected_actions)
        np.testing.assert_array_equal(result.residual, expected_residual)
        np.testing.assert_array_equal(result.local_logical_frame, expected_frame)

    def test_invalid_v3_configuration_is_rejected(self):
        with self.assertRaises(ValueError):
            GateConfig(workload_mode="unknown")
        with self.assertRaises(ValueError):
            GateConfig(workload_pair_radius=0)
        with self.assertRaises(ValueError):
            GateConfig(pointwise_guard_relative_margin=1.0)

    def test_apply_actions_uses_gf2_for_detector_and_logical_maps(self):
        residual, logical = apply_actions(
            np.array([1, 0], dtype=np.uint8),
            np.array([1, 1], dtype=np.uint8),
            np.array([[1, 1], [0, 1]], dtype=np.uint8),
            np.array([[1, 0]], dtype=np.uint8),
        )
        np.testing.assert_array_equal(residual, np.array([1, 1]))
        np.testing.assert_array_equal(logical, np.array([1]))


if __name__ == "__main__":
    unittest.main()
