# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.experiments.quant_t0_compare.summarize_seeded import unpack_error_mask


class PackedErrorMaskTest(unittest.TestCase):
    def test_unpack_ignores_padding_and_validates_sample_count(self):
        expected = np.asarray([1, 0, 1, 1, 0, 0, 1, 0, 1], dtype=np.bool_)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "errors.npz"
            np.savez_compressed(
                path,
                samples=np.asarray([len(expected)], dtype=np.int64),
                tensorrt=np.packbits(expected),
            )
            np.testing.assert_array_equal(expected, unpack_error_mask(path, "tensorrt", 9))
            with self.assertRaisesRegex(ValueError, "sample count mismatch"):
                unpack_error_mask(path, "tensorrt", 8)


if __name__ == "__main__":
    unittest.main()
