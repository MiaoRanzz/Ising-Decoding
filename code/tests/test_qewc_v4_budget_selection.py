# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

from scripts.experiments.qewc_v4.select_budget import select_task, sustained_decline


def record(epochs, ler, bce, *, declining=False):
    return {
        "legacy_epochs": epochs,
        "current_task_ler": ler,
        "validation_bce": bce,
        "stable": True,
        "train_loss_sustained_decline": None if epochs == 1 else declining,
        "validation_loss_sustained_decline": None if epochs == 1 else declining,
    }


class TestQewcV4BudgetSelection(unittest.TestCase):

    def test_selects_lowest_plateaued_candidate(self):
        result = select_task(
            [
                record(1, 0.12, 0.025),
                record(5, 0.1000, 0.0200),
                record(20, 0.0995, 0.0199),
                record(50, 0.0994, 0.01985),
            ]
        )
        self.assertEqual(result["selected_legacy_epochs"], 5)
        self.assertFalse(result["needs_extension_to_100"])
        self.assertTrue(result["comparisons"][1]["current_budget_is_sufficient"])

    def test_requires_extension_when_50_epochs_is_still_improving(self):
        result = select_task(
            [
                record(1, 0.12, 0.025),
                record(5, 0.11, 0.023, declining=True),
                record(20, 0.10, 0.021, declining=True),
                record(50, 0.09, 0.019, declining=True),
            ]
        )
        self.assertIsNone(result["selected_legacy_epochs"])
        self.assertTrue(result["needs_extension_to_100"])

    def test_sustained_decline_uses_last_two_fifths(self):
        self.assertIsNone(sustained_decline([1.0]))
        self.assertTrue(sustained_decline([1.0, 0.9, 0.8, 0.7, 0.6]))
        self.assertFalse(sustained_decline([1.0, 0.9, 0.8, 0.70, 0.699]))

    def test_100_epoch_cap_terminates_with_explicit_limitation(self):
        result = select_task(
            [
                record(1, 0.12, 0.025),
                record(5, 0.11, 0.023, declining=True),
                record(20, 0.10, 0.021, declining=True),
                record(50, 0.09, 0.019, declining=True),
                record(100, 0.08, 0.017, declining=True),
            ]
        )
        self.assertEqual(result["selected_legacy_epochs"], 100)
        self.assertFalse(result["needs_extension_to_100"])
        self.assertTrue(result["selected_at_cap_still_improving"])

    def test_only_one_task_may_have_the_100_epoch_extension(self):
        short = select_task(
            [
                record(1, 0.12, 0.025), record(5, 0.10, 0.020),
                record(20, 0.0995, 0.0199), record(50, 0.0994, 0.0198),
            ]
        )
        long = select_task(
            [
                record(1, 0.12, 0.025), record(5, 0.11, 0.023, declining=True),
                record(20, 0.10, 0.021, declining=True), record(50, 0.09, 0.019, declining=True),
                record(100, 0.08, 0.017, declining=True),
            ]
        )
        self.assertEqual(short["selected_legacy_epochs"], 5)
        self.assertEqual(long["selected_legacy_epochs"], 100)


if __name__ == "__main__":
    unittest.main()
