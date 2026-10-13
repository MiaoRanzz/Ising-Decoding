# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

from scripts.experiments.qewc_v3.aggregate_ood_results import (
    method_summary,
    paired_summary,
)


class TestQewcV3OodAggregate(unittest.TestCase):

    def test_method_summary_weights_ler_and_density_by_samples(self):
        rows = [
            {
                "method": "ewc",
                "samples": 10,
                "logical_errors": 1,
                "latency_us_per_round": 2.0,
                "speedup_vs_pymatching": 1.5,
                "input_syndrome_ones": 20,
                "input_syndrome_elements": 100,
                "residual_syndrome_ones": 5,
                "residual_syndrome_elements": 100,
            },
            {
                "method": "ewc",
                "samples": 30,
                "logical_errors": 9,
                "latency_us_per_round": 4.0,
                "speedup_vs_pymatching": 2.5,
                "input_syndrome_ones": 40,
                "input_syndrome_elements": 300,
                "residual_syndrome_ones": 15,
                "residual_syndrome_elements": 300,
            },
        ]
        result = method_summary(rows, configurations=1)
        self.assertAlmostEqual(result["ler"], 0.25)
        self.assertAlmostEqual(result["input_syndrome_density"], 0.15)
        self.assertAlmostEqual(result["residual_syndrome_density"], 0.05)
        self.assertAlmostEqual(result["density_reduction_fraction"], 2.0 / 3.0)

    def test_paired_summary_combines_contingency_counts(self):
        rows = [
            {
                "_configuration": "c0",
                "method_a": "ewc",
                "method_b": "seq",
                "basis": "both",
                "samples": 10,
                "a_only_error": 1,
                "b_only_error": 3,
                "both_error": 2,
                "neither_error": 4,
            },
            {
                "_configuration": "c1",
                "method_a": "ewc",
                "method_b": "seq",
                "basis": "both",
                "samples": 10,
                "a_only_error": 2,
                "b_only_error": 4,
                "both_error": 1,
                "neither_error": 3,
            },
        ]
        result = paired_summary(rows)
        self.assertEqual(result["configurations"], 2)
        self.assertEqual(result["samples"], 20)
        self.assertAlmostEqual(result["ler_delta_a_minus_b"], -0.2)
        self.assertLess(result["ler_delta_ci95_normal"][0], -0.2)
        self.assertGreater(result["ler_delta_ci95_normal"][1], -0.2)


if __name__ == "__main__":
    unittest.main()
