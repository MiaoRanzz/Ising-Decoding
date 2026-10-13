# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

from scripts.experiments.qewc_v3.select_stress_tasks import FAMILIES, greedy_order, select


class TestQewcV3StressSelection(unittest.TestCase):

    def test_selection_keeps_difficulty_primary_and_returns_reverse(self):
        names = [f"{family}_{level}" for family in FAMILIES for level in range(4)]
        ler = {
            "candidates": {
                name: {"stage0_ler": 0.1 + 0.01 * int(name.rsplit("_", 1)[1])}
                for name in names
            }
        }
        cosine = {
            left: {right: (1.0 if left == right else 0.0) for right in names}
            for left in names
        }
        distance = {
            left: {right: (0.0 if left == right else 1.0) for right in names}
            for left in names
        }
        screen = {
            "gradient_cosine": cosine,
            "representation_distance": distance,
            "metrics": {name: {"gradient_norm": 1.0} for name in names},
        }
        result = select(ler, screen, difficulty_slack=0.0)
        self.assertEqual(len(result["selected_by_task"]), 4)
        self.assertEqual(result["reverse_order"], list(reversed(result["high_conflict_order"])))
        selected_ler = [item["stage0_ler"] for item in result["selected_by_task"].values()]
        self.assertEqual(max(selected_ler) - min(selected_ler), 0.0)

    def test_greedy_order_uses_conflict(self):
        names = ("a", "b", "c", "d")
        cosine = {
            left: {right: (1.0 if left == right else 0.5) for right in names}
            for left in names
        }
        cosine["a"]["b"] = cosine["b"]["a"] = -1.0
        order = greedy_order(names, cosine, {name: 1.0 for name in names})
        self.assertIn(order[0], ("a", "b"))
        self.assertIn(order[1], ("a", "b"))


if __name__ == "__main__":
    unittest.main()
