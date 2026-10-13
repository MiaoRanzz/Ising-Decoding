# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

import torch

from scripts.experiments.qewc_v4.select_fisher_samples import compare_fisher_dicts, select


class TestQewcV4FisherSelection(unittest.TestCase):
    def test_comparison_reports_cosine_and_total_ratio(self):
        result = compare_fisher_dicts(
            {"a": torch.tensor([1.0, 2.0])},
            {"a": torch.tensor([1.0, 2.0])},
        )
        self.assertAlmostEqual(result["cosine_to_reference"], 1.0)
        self.assertAlmostEqual(result["total_ratio_to_reference"], 1.0)

    def test_selects_smallest_eligible_sample_count(self):
        records = [
            {"samples": 4096, "cosine_to_reference": 0.98, "total_ratio_relative_error": 0.01},
            {"samples": 16384, "cosine_to_reference": 0.995, "total_ratio_relative_error": 0.04},
            {"samples": 65536, "cosine_to_reference": 1.0, "total_ratio_relative_error": 0.0},
        ]
        result = select(records, cosine_minimum=0.99, total_tolerance=0.05)
        self.assertEqual(result["selected_fisher_samples"], 16384)

    def test_parameter_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "names differ"):
            compare_fisher_dicts({"a": torch.ones(1)}, {"b": torch.ones(1)})


if __name__ == "__main__":
    unittest.main()
