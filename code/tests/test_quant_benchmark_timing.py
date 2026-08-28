# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

from benchmarks.bench_quantized_predecoder import parse_batch_sizes


class TestQuantBenchmarkTiming(unittest.TestCase):

    def test_parse_batch_sizes_preserves_order_and_deduplicates(self):
        self.assertEqual(parse_batch_sizes("1,8,8,32"), [1, 8, 32])

    def test_parse_batch_sizes_rejects_nonpositive(self):
        with self.assertRaises(Exception):
            parse_batch_sizes("1,0,8")


if __name__ == "__main__":
    unittest.main()
