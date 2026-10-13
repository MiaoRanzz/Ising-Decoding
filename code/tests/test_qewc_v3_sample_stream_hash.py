# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import unittest

import numpy as np

from scripts.paired_inference_compare import array_sha256


class TestQewcV3SampleStreamHash(unittest.TestCase):

    def test_hashes_exact_contiguous_bytes(self):
        samples = np.arange(24, dtype=np.uint8).reshape(4, 6)[:, ::2]
        expected = hashlib.sha256(np.ascontiguousarray(samples).tobytes()).hexdigest()
        self.assertEqual(array_sha256(samples), expected)
        changed = samples.copy()
        changed[-1, -1] ^= 1
        self.assertNotEqual(array_sha256(changed), expected)


if __name__ == "__main__":
    unittest.main()
