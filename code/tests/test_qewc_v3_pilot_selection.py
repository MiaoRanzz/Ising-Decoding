# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

from scripts.experiments.qewc_v3.select_pilot_hyperparameter import (
    Candidate,
    TASKS,
    select,
)


def paired_rows(first, second, half_width=0.005):
    return {
        task: [
            {
                "basis": "both",
                "candidate": second,
                "baseline": first,
                "ci95_low": -half_width,
                "ci95_high": half_width,
            }
        ]
        for task in TASKS
    }


class TestQewcV3PilotSelection(unittest.TestCase):

    def test_replay_tie_break_selects_smaller_ratio(self):
        payload = {
            "ler_rows": {
                "SEQ": [0.10] * 5,
                "R10": [0.10] * 5,
                "R25": [0.099] * 5,
            },
            "paired_comparisons": paired_rows("R10", "R25"),
        }
        result = select(
            payload,
            "replay",
            "SEQ",
            [Candidate("R10", 0.10), Candidate("R25", 0.25)],
        )
        self.assertEqual(result["best_raw_primary"]["name"], "R25")
        self.assertEqual(result["selected"], {"name": "R10", "value": 0.10})

    def test_ewc_excludes_t1_failure_and_t0_t1_dominance(self):
        payload = {
            "ler_rows": {
                "SEQ": [0.10] * 5,
                "A": [0.10, 0.10, 0.10, 0.10, 0.10],
                "B": [0.09, 0.10, 0.10, 0.10, 0.10],
                "C": [0.08, 0.106, 0.10, 0.10, 0.10],
            },
            "paired_comparisons": paired_rows("A", "B"),
        }
        result = select(
            payload,
            "ewc",
            "SEQ",
            [Candidate("A", 1.0), Candidate("B", 2.0), Candidate("C", 3.0)],
        )
        by_name = {row["name"]: row for row in result["candidates"]}
        self.assertTrue(by_name["A"]["dominated_on_t0_t1"])
        self.assertFalse(by_name["C"]["eligible"])
        self.assertEqual(result["selected"]["name"], "B")


if __name__ == "__main__":
    unittest.main()
