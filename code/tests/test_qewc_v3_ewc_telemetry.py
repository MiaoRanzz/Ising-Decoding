# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

from scripts.experiments.qewc_v3.extract_ewc_telemetry import summarize_values


class TestQewcV3EwcTelemetry(unittest.TestCase):

    def test_scalar_summary_preserves_endpoints_and_distribution(self):
        result = summarize_values([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(result["count"], 4)
        self.assertEqual(result["first"], 1.0)
        self.assertEqual(result["last"], 4.0)
        self.assertEqual(result["mean"], 2.5)
        self.assertEqual(result["max"], 4.0)

    def test_empty_series_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            summarize_values([])


if __name__ == "__main__":
    unittest.main()
