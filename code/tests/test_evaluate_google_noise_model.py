# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.evaluate_google_noise_model import paired_statistics, wilson_interval


class TestEvaluateGoogleNoiseModel(unittest.TestCase):
    def test_wilson_interval_contains_empirical_rate(self):
        low, high = wilson_interval(10, 100)
        self.assertLess(low, 0.1)
        self.assertGreater(high, 0.1)

    def test_paired_statistics_uses_shared_shots(self):
        candidate = np.array([0, 1, 0, 1, 1, 0], dtype=bool)
        baseline = np.array([0, 0, 1, 1, 0, 0], dtype=bool)
        result = paired_statistics(candidate, baseline)
        self.assertEqual(result["candidate_only_errors"], 2)
        self.assertEqual(result["baseline_only_errors"], 1)
        self.assertEqual(result["both_errors"], 1)
        self.assertEqual(result["neither_errors"], 2)
        self.assertAlmostEqual(result["delta_ler"], 1 / 6)

    def test_identical_masks_have_zero_delta_and_unit_pvalue(self):
        values = np.array([0, 1, 0, 1], dtype=bool)
        result = paired_statistics(values, values)
        self.assertEqual(result["delta_ler"], 0.0)
        self.assertEqual(result["mcnemar_exact_pvalue"], 1.0)


if __name__ == "__main__":
    unittest.main()
