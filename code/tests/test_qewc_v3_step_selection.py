# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

from scripts.experiments.qewc_v3.select_optimizer_steps import Candidate, select


def comparison(candidate, baseline, delta, low, high):
    return {
        "basis": "both",
        "candidate": candidate,
        "baseline": baseline,
        "delta_ler": delta,
        "ci95_low": low,
        "ci95_high": high,
    }


class TestQewcV3StepSelection(unittest.TestCase):

    def payload(self, t1_high):
        return {
            "ler_rows": {
                "S1K": [0.1] * 5,
                "S5K": [0.09] * 5,
            },
            "paired_comparisons": {
                "T0": [comparison("S5K", "S1K", -0.01, -0.02, -0.001)],
                "T1": [comparison("S5K", "S1K", -0.01, -0.02, t1_high)],
            },
        }

    def candidates(self):
        return [
            Candidate("S1K", 1000, 0.1, 0.1, 600),
            Candidate("S5K", 5000, 0.09, 0.09, 3000),
        ]

    def test_selects_lowest_when_larger_gain_is_not_significant(self):
        result = select(self.payload(0.001), self.candidates(), {"S1K", "S5K"}, 2.4)
        self.assertEqual(result["selected"]["steps"], 1000)

    def test_selects_larger_when_current_task_gain_is_significant(self):
        result = select(self.payload(-0.001), self.candidates(), {"S1K", "S5K"}, 2.4)
        self.assertEqual(result["selected"]["steps"], 5000)


if __name__ == "__main__":
    unittest.main()
