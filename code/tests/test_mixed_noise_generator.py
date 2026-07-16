# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.mixed_noise_generator import MixedNoiseBatchGenerator


class _RecordingGenerator:

    def __init__(self, name):
        self.name = name
        self.calls = []

    def generate_batch(self, **kwargs):
        self.calls.append(kwargs)
        return self.name, kwargs


class TestMixedNoiseBatchGenerator(unittest.TestCase):

    def test_round_robin_forwards_upstream_generation_options(self):
        first = _RecordingGenerator("first")
        second = _RecordingGenerator("second")
        generator = MixedNoiseBatchGenerator([first, second], ["a", "b"])

        result = generator.generate_batch(
            step=1,
            batch_size=8,
            return_timing=True,
            profile_generator_subphases=True,
        )

        self.assertEqual(result[0], "second")
        self.assertEqual(first.calls, [])
        self.assertEqual(second.calls[0]["batch_size"], 8)
        self.assertTrue(second.calls[0]["return_timing"])
        self.assertTrue(second.calls[0]["profile_generator_subphases"])


if __name__ == "__main__":
    unittest.main()
