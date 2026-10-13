# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

from scripts.experiments.qewc_v3.aggregate_formal_results import summary_row


class TestQewcV3AggregateResults(unittest.TestCase):

    def test_continual_metrics_and_acquisition_gap_are_preserved(self):
        result = {
            "run_id": "S2",
            "track": "stress",
            "method": "ewc",
            "kind": "continual_matrix",
            "summary": {
                "final_average_ler": 0.1,
                "average_incremental_performance": 0.9,
                "average_forgetting": 0.02,
                "backward_transfer": -0.01,
                "forward_transfer": 0.03,
                "acquisition_gap": {"T1": 0.01, "T2": 0.02, "T3": 0.03, "T4": 0.04},
            },
        }
        row = summary_row(result)
        self.assertAlmostEqual(row["mean_acquisition_gap"], 0.025)
        self.assertEqual(row["ag_T3"], 0.03)

    def test_mix_reports_only_final_average_ler(self):
        result = {
            "run_id": "S4",
            "track": "stress",
            "method": "mix_total",
            "kind": "joint_optimization_trajectory",
            "ler": [[0.1] * 5, [0.2] * 5],
        }
        row = summary_row(result)
        self.assertAlmostEqual(row["final_average_ler"], 0.2)
        self.assertIsNone(row["average_forgetting"])


if __name__ == "__main__":
    unittest.main()
