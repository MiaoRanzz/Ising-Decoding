# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from benchmarks.quant_t0_common import (
    batch_size_grid,
    combine_basis_results,
    holm_adjust,
    latency_summary,
    paired_noninferiority,
)


class TestPairedNoninferiority(unittest.TestCase):

    def test_identical_masks_pass(self):
        baseline = np.zeros(10000, dtype=np.bool_)
        baseline[:400] = True
        result = paired_noninferiority(baseline.copy(), baseline)
        self.assertTrue(result["noninferiority_passed"])
        self.assertEqual(result["delta_ler"], 0.0)
        self.assertAlmostEqual(result["effective_margin"], 0.001)

    def test_degraded_candidate_fails(self):
        baseline = np.zeros(10000, dtype=np.bool_)
        baseline[:400] = True
        candidate = baseline.copy()
        candidate[1000:1050] = True
        result = paired_noninferiority(candidate, baseline)
        self.assertFalse(result["noninferiority_passed"])
        self.assertGreater(result["delta_ler"], result["effective_margin"])

    def test_relative_margin_controls_at_low_ler(self):
        baseline = np.zeros(100000, dtype=np.bool_)
        baseline[:100] = True
        result = paired_noninferiority(baseline.copy(), baseline)
        self.assertAlmostEqual(result["effective_margin"], 0.00003)

    def test_requires_paired_shapes(self):
        with self.assertRaises(ValueError):
            paired_noninferiority(np.zeros(2), np.zeros(3))


class TestHolmAndSummaries(unittest.TestCase):

    def test_holm_is_monotone_in_sorted_order(self):
        adjusted = holm_adjust({"X": 0.01, "Z": 0.04})
        self.assertAlmostEqual(adjusted["X"]["holm_adjusted_p_value"], 0.02)
        self.assertAlmostEqual(adjusted["Z"]["holm_adjusted_p_value"], 0.04)
        self.assertTrue(adjusted["X"]["holm_passed"])
        self.assertTrue(adjusted["Z"]["holm_passed"])

    def test_combine_requires_both_bases(self):
        basis = {
            "X": {"noninferiority_p_value": 0.01},
            "Z": {"noninferiority_p_value": 0.1},
        }
        self.assertEqual(combine_basis_results(basis)["status"], "ACCURACY_FAIL")

    def test_latency_summary_and_batch_grid(self):
        summary = latency_summary([1.0, 2.0, 3.0])
        self.assertEqual(summary["count"], 3)
        self.assertEqual(summary["p50_ms"], 2.0)
        self.assertEqual(batch_size_grid(10), [1, 2, 4, 8, 10])


if __name__ == "__main__":
    unittest.main()
