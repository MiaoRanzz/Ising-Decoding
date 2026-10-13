# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

from scripts.experiments.qewc_v4.aggregate_external_results import (
    aggregate_ood_pair,
    aggregate_rows,
    generalization_decision,
    normalize_willow_pair,
)


class TestQewcV4ExternalResults(unittest.TestCase):
    def test_aggregate_rows_reports_micro_macro_and_latency(self):
        rows = [
            {"samples": 100, "logical_errors": 10, "ler": 0.10, "model_latency_us_per_shot": 2.0},
            {"samples": 200, "logical_errors": 40, "ler": 0.20, "model_latency_us_per_shot": 4.0},
        ]
        result = aggregate_rows(rows)
        self.assertAlmostEqual(result["micro_ler"], 1 / 6)
        self.assertAlmostEqual(result["macro_ler"], 0.15)
        self.assertEqual(result["model_latency_us_per_shot_mean"], 3.0)

    def test_ood_pair_is_oriented_as_ewc_minus_seq(self):
        row = {
            "method_a": "H_SEQ",
            "method_b": "H_EWC",
            "samples": 1000,
            "a_only_error": 30,
            "b_only_error": 10,
            "both_error": 100,
            "neither_error": 860,
        }
        result = aggregate_ood_pair([row], "H_EWC", "H_SEQ")
        self.assertEqual(result["candidate_only_error"], 10)
        self.assertEqual(result["baseline_only_error"], 30)
        self.assertAlmostEqual(result["ler_delta_candidate_minus_baseline"], -0.02)

    def test_willow_reversed_pair_is_also_oriented_ewc_minus_seq(self):
        row = {
            "candidate": "I_SEQ",
            "baseline": "I_EWC",
            "samples": 1000,
            "candidate_only_errors": 25,
            "baseline_only_errors": 5,
            "both_errors": 100,
            "neither_errors": 870,
        }
        result = normalize_willow_pair(row, "I_EWC", "I_SEQ")
        self.assertAlmostEqual(result["ler_delta_candidate_minus_baseline"], -0.02)

    def test_generalization_requires_both_relative_and_paired_thresholds(self):
        passing = {
            "ewc_vs_seq": {
                "relative_ler_improvement": 0.02,
                "ler_delta_ci95_normal": [-0.02, -0.001],
            }
        }
        failing = {
            "ewc_vs_seq": {
                "relative_ler_improvement": 0.005,
                "ler_delta_ci95_normal": [-0.02, -0.001],
            }
        }
        self.assertTrue(generalization_decision(passing, passing)["passed_both"])
        self.assertFalse(generalization_decision(passing, failing)["passed_both"])


if __name__ == "__main__":
    unittest.main()
