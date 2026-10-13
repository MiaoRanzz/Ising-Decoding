# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

from scripts.experiments.qewc_v3.build_balanced_replay import basis_quotas, task_quotas


class TestQewcV3BalancedReplay(unittest.TestCase):

    def test_quotas_are_exact_and_differ_by_at_most_one(self):
        quotas = task_quotas(594, ["T0", "T1", "T2", "T3"])
        self.assertEqual(sum(quotas.values()), 594)
        self.assertLessEqual(max(quotas.values()) - min(quotas.values()), 1)

    def test_basis_quotas_cover_task_and_differ_by_at_most_one(self):
        for capacity in (1, 2, 17, 149):
            quotas = basis_quotas(capacity)
            self.assertEqual(sum(quotas.values()), capacity)
            self.assertLessEqual(abs(quotas["X"] - quotas["Z"]), 1)


    def test_capacity_must_cover_each_task(self):
        with self.assertRaisesRegex(ValueError, "at least"):
            task_quotas(2, ["T0", "T1", "T2"])


if __name__ == "__main__":
    unittest.main()
