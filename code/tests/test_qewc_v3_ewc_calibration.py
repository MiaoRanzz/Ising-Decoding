# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

from scripts.experiments.qewc_v3.calibrate_ewc_lambda import lambda_candidates


class TestQewcV3EwcCalibration(unittest.TestCase):

    def test_candidate_lambda_hits_requested_ratio(self):
        data_loss = 20.0
        penalty = 0.5
        targets = [0.001, 0.01, 0.1, 1.0]
        candidates = lambda_candidates(data_loss, penalty, targets)
        for target in targets:
            value = candidates[f"{target:g}"]
            self.assertAlmostEqual(value * penalty / data_loss, target)

    def test_rejects_zero_penalty(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            lambda_candidates(1.0, 0.0, [0.1])


if __name__ == "__main__":
    unittest.main()
