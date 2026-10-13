# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

from scripts.experiments.qewc_v4.calibrate_replay_capacity import estimated_capacity


class TestQewcV4ReplayCapacity(unittest.TestCase):
    def test_reference_capacity_round_trips(self):
        self.assertEqual(estimated_capacity(5229158), 594)

    def test_capacity_scales_with_architecture_state_bytes(self):
        self.assertEqual(estimated_capacity(10458316), 1188)

    def test_nonpositive_target_is_rejected(self):
        with self.assertRaises(ValueError):
            estimated_capacity(0)


if __name__ == "__main__":
    unittest.main()
