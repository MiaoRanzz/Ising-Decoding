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
import torch

from evaluation.surface_topology_adapter import build_surface_action_adapter
from qec.surface_code.data_mapping import (
    compute_stabX_to_data_index_map,
    compute_stabZ_to_data_index_map,
    construct_X_stab_Parity_check_Mat,
    construct_Z_stab_Parity_check_Mat,
)
from qec.surface_code.detector_input import SurfaceDetectorInputTransform
from training.topology_loss import (
    TopologyLossWeights,
    soft_odd_probability,
    topology_joint_loss,
)


def reference_detector_delta(actions: torch.Tensor, basis: str) -> torch.Tensor:
    """Apply the production residual recurrence to one four-channel action tensor."""

    _, rounds, distance, _ = actions.shape
    half = (distance * distance - 1) // 2
    hx = construct_X_stab_Parity_check_Mat(distance).to(torch.int64)
    hz = construct_Z_stab_Parity_check_Mat(distance).to(torch.int64)
    x_indices = compute_stabX_to_data_index_map(distance, "XV").to(torch.long)
    z_indices = compute_stabZ_to_data_index_map(distance, "XV").to(torch.long)
    z_data = actions[0].reshape(rounds, -1).to(torch.int64)
    x_data = actions[1].reshape(rounds, -1).to(torch.int64)
    syn_x = actions[2].reshape(rounds, -1).index_select(1, x_indices).to(torch.int64)
    syn_z = actions[3].reshape(rounds, -1).index_select(1, z_indices).to(torch.int64)
    induced_x = torch.remainder(z_data @ hx.t(), 2)
    induced_z = torch.remainder(x_data @ hz.t(), 2)
    residual_x = torch.zeros((rounds, half), dtype=torch.int64)
    residual_z = torch.zeros((rounds, half), dtype=torch.int64)
    residual_x[0] = syn_x[0] ^ induced_x[0]
    residual_z[0] = syn_z[0] ^ induced_z[0]
    residual_x[1:] = syn_x[1:] ^ syn_x[:-1] ^ induced_x[1:]
    residual_z[1:] = syn_z[1:] ^ syn_z[:-1] ^ induced_z[1:]
    first = residual_x[0] if basis == "X" else residual_z[0]
    rest = torch.stack((residual_x[1:], residual_z[1:]), dim=1).reshape(-1)
    boundary = torch.zeros(half, dtype=torch.int64)
    return torch.cat((first, rest, boundary)).to(torch.uint8)


class SurfaceTopologyAdapterTest(unittest.TestCase):
    def test_four_channel_maps_match_expected_geometry(self):
        adapter = build_surface_action_adapter(5, 5, "X", "XV")
        self.assertEqual(tuple(adapter.extended_h.shape), (120, 500))
        self.assertEqual(tuple(adapter.extended_l.shape), (1, 500))
        self.assertEqual(int(adapter.detector_training_mask.sum()), 108)
        self.assertEqual(int(adapter.valid_actions.sum()), 346)
        self.assertEqual(len(adapter.action_types), 500)

    def test_extended_h_matches_production_residual_recurrence(self):
        generator = torch.Generator().manual_seed(20260810)
        for basis in ("X", "Z"):
            adapter = build_surface_action_adapter(5, 5, basis, "XV")
            actions = torch.randint(
                0, 2, (4, 5, 5, 5), generator=generator, dtype=torch.uint8
            )
            actions = actions.reshape(-1) * adapter.valid_actions.to(torch.uint8)
            matrix_delta = torch.remainder(
                adapter.extended_h.to(torch.int64) @ actions.to(torch.int64), 2
            ).to(torch.uint8)
            reference = reference_detector_delta(actions.reshape(4, 5, 5, 5), basis)
            torch.testing.assert_close(matrix_delta, reference)

    def test_detector_state_round_trip_uses_main_rows_and_zero_boundary(self):
        generator = torch.Generator().manual_seed(7)
        for basis in ("X", "Z"):
            adapter = build_surface_action_adapter(5, 5, basis, "XV")
            transform = SurfaceDetectorInputTransform(
                distance=5, rounds=5, basis=basis, rotation="XV"
            )
            detectors = torch.randint(
                0,
                2,
                (3, transform.detector_width),
                generator=generator,
                dtype=torch.uint8,
            )
            train_x, _, _, _ = transform.build_train_x(detectors)
            recovered = adapter.detector_state_from_train_x(train_x).to(torch.uint8)
            expected = detectors[:, : transform.num_main_dets].clone()
            final_round = (
                transform.num_stabs
                + (transform.rounds - 2) * 2 * transform.num_stabs
            )
            if basis == "X":
                expected[
                    :,
                    final_round + transform.num_stabs : final_round + 2 * transform.num_stabs,
                ] = 0
            else:
                expected[:, final_round : final_round + transform.num_stabs] = 0
            torch.testing.assert_close(
                recovered[:, : transform.num_main_dets], expected
            )
            self.assertEqual(int(recovered[:, transform.num_main_dets :].sum()), 0)

    def test_measurement_action_has_two_time_adjacent_detector_effects(self):
        adapter = build_surface_action_adapter(5, 5, "X", "XV")
        d2 = 25
        grid_index = int(adapter.x_stabilizer_grid_indices[0])
        action = ((2 * 5 + 1) * d2) + grid_index
        rows = torch.nonzero(adapter.extended_h[:, action], as_tuple=False).reshape(-1)
        self.assertEqual(rows.tolist(), [12, 36])

    def test_logical_map_tracks_only_basis_anticommuting_data_channel(self):
        x_adapter = build_surface_action_adapter(5, 5, "X", "XV")
        z_adapter = build_surface_action_adapter(5, 5, "Z", "XV")
        self.assertEqual(int(x_adapter.extended_l.sum()), 25)
        self.assertEqual(int(z_adapter.extended_l.sum()), 25)
        self.assertEqual(int(x_adapter.extended_l[:, 125:].sum()), 0)
        self.assertEqual(int(z_adapter.extended_l[:, :125].sum()), 0)


class TopologyJointLossTest(unittest.TestCase):
    def test_soft_odd_probability_preserves_probabilities_above_half(self):
        probabilities = torch.tensor([[0.9, 0.8]], requires_grad=True)
        incidence = torch.tensor([[1.0, 1.0], [1.0, 0.0]])
        result = soft_odd_probability(probabilities, incidence)
        torch.testing.assert_close(result, torch.tensor([[0.26, 0.9]]))
        result.sum().backward()
        self.assertTrue(torch.isfinite(probabilities.grad).all())
        self.assertNotEqual(float(probabilities.grad[0, 0]), 0.0)

    def test_zero_topology_weights_are_exact_bce(self):
        adapter = build_surface_action_adapter(3, 3, "X", "XV")
        logits = torch.randn(2, 4, 3, 3, 3, requires_grad=True)
        targets = torch.randint(0, 2, logits.shape).float()
        train_x = torch.zeros(2, 4, 3, 3, 3)
        total, components = topology_joint_loss(
            logits,
            targets,
            train_x,
            adapter,
            weights=TopologyLossWeights(),
        )
        expected = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
        torch.testing.assert_close(total, expected)
        torch.testing.assert_close(components["topology"], torch.zeros(()))

    def test_joint_loss_is_finite_and_backpropagates(self):
        adapter = build_surface_action_adapter(3, 3, "Z", "XV")
        logits = torch.randn(2, 4, 3, 3, 3, requires_grad=True)
        targets = torch.randint(0, 2, logits.shape).float()
        transform = SurfaceDetectorInputTransform(
            distance=3, rounds=3, basis="Z", rotation="XV"
        )
        detectors = torch.randint(0, 2, (2, transform.detector_width)).float()
        train_x, _, _, _ = transform.build_train_x(detectors)
        total, components = topology_joint_loss(
            logits,
            targets,
            train_x,
            adapter,
            weights=TopologyLossWeights(0.2, 0.1, 0.05),
        )
        self.assertTrue(all(torch.isfinite(value) for value in components.values()))
        total.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())


if __name__ == "__main__":
    unittest.main()
