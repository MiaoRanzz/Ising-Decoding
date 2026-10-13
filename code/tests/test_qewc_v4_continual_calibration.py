# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

from scripts.experiments.qewc_v4.select_continual_calibration import (
    choose,
    make_record,
    next_ewc_step,
)


def matrix(offset=0.0):
    return [
        [0.10 + offset, 0.20, 0.20, 0.20, 0.20],
        [0.11 + offset, 0.10 + offset, 0.20, 0.20, 0.20],
        [0.12 + offset, 0.11 + offset, 0.20, 0.10 + offset, 0.20],
        [0.13 + offset, 0.12 + offset, 0.10 + offset, 0.11 + offset, 0.20],
        [0.14 + offset, 0.13 + offset, 0.11 + offset, 0.12 + offset, 0.10 + offset],
    ]


class TestQewcV4ContinualCalibration(unittest.TestCase):
    def test_ewc_final_ler_default_limit_is_two_percent(self):
        baseline = matrix()
        for relative, eligible in ((0.015, True), (0.01999, True), (0.02001, False)):
            candidate = matrix()
            candidate[4] = [value * (1 + relative) for value in baseline[4]]
            record = make_record(
                "L100", 100, candidate, baseline,
                kind="ewc", stable=True, gpu_hours=1.0,
            )
            self.assertEqual(record["eligible"], eligible)

    def test_ewc_rejects_more_than_five_percent_plasticity_loss(self):
        baseline = matrix()
        candidate = matrix()
        candidate[1][1] = 0.106
        record = make_record(
            "L10", 10, candidate, baseline, kind="ewc", stable=True, gpu_hours=1.0
        )
        self.assertFalse(record["eligible"])

    def test_ewc_choice_prioritizes_average_forgetting(self):
        records = [
            {"name": "a", "value": 1, "eligible": True, "average_forgetting": 0.02, "backward_transfer": -0.01, "final_average_ler": 0.1},
            {"name": "b", "value": 10, "eligible": True, "average_forgetting": 0.01, "backward_transfer": -0.02, "final_average_ler": 0.2},
        ]
        self.assertEqual(choose(records, "ewc")["name"], "b")

    def test_boundary_extends_one_decade(self):
        records = [{"value": value} for value in (0.1, 1, 10, 100, 1000)]
        step = next_ewc_step(records, {"value": 1000}, "coarse", False)
        self.assertEqual(step["action"], "extend_one_decade")
        self.assertEqual(step["candidate_values"], [10000])

    def test_interior_adds_geometric_midpoints(self):
        records = [{"value": value} for value in (0.1, 1, 10, 100, 1000)]
        step = next_ewc_step(records, {"value": 10}, "coarse", False)
        self.assertEqual(step["action"], "refine_geometric_midpoints")
        self.assertAlmostEqual(step["candidate_values"][0], 10 ** 0.5)
        self.assertAlmostEqual(step["candidate_values"][1], 10 ** 1.5)


if __name__ == "__main__":
    unittest.main()
