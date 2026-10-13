# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

from scripts.experiments.qewc_v3.summarize_matrix import summarize


class TestQewcV3Metrics(unittest.TestCase):

    def test_natural_order_metrics_follow_design_equations(self):
        ler = [
            [0.10, 0.50, 0.60, 0.70, 0.80],
            [0.11, 0.20, 0.55, 0.65, 0.75],
            [0.12, 0.22, 0.30, 0.60, 0.70],
            [0.13, 0.24, 0.32, 0.40, 0.65],
            [0.14, 0.26, 0.34, 0.42, 0.50],
        ]
        result = summarize(
            ler,
            ["T1", "T2", "T3", "T4"],
            {"T1": 0.19, "T2": 0.29, "T3": 0.39, "T4": 0.49},
        )
        self.assertAlmostEqual(result["final_average_ler"], 0.332)
        self.assertAlmostEqual(result["forgetting"]["T0"], 0.04)
        self.assertAlmostEqual(result["forgetting"]["T1"], 0.06)
        self.assertIsNone(result["forgetting"]["T4"])
        self.assertAlmostEqual(result["backward_transfer"], -0.04)
        self.assertAlmostEqual(result["forward_transfer"], 0.10)
        self.assertAlmostEqual(result["acquisition_gap"]["T3"], 0.01)

    def test_rejects_non_permutation_order(self):
        with self.assertRaisesRegex(ValueError, "permutation"):
            summarize([[0.0] * 5 for _ in range(5)], ["T1"] * 4)


if __name__ == "__main__":
    unittest.main()
