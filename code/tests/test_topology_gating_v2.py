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
    apply_actions,
    gate_actions,
    interaction_clusters,
    legal_combinations,
    typed_candidates,
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
