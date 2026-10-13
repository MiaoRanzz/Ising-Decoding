# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
import unittest

import torch

from scripts.experiments.qewc_v3.screen_stress_tasks import cosine, pairwise


class TestQewcV3StressScreen(unittest.TestCase):

    def test_pairwise_cosine_and_distance(self):
        vectors = {
            "a": torch.tensor([1.0, 0.0]),
            "b": torch.tensor([-1.0, 0.0]),
            "c": torch.tensor([0.0, 2.0]),
        }
        cosines = pairwise(list(vectors), vectors, "cosine")
        distances = pairwise(list(vectors), vectors, "distance")
        self.assertAlmostEqual(cosines["a"]["b"], -1.0)
        self.assertAlmostEqual(cosines["a"]["c"], 0.0)
        self.assertAlmostEqual(distances["a"]["b"], 2.0)
        self.assertAlmostEqual(distances["a"]["a"], 0.0)
        self.assertTrue(math.isnan(cosine(torch.zeros(2), vectors["a"])))


if __name__ == "__main__":
    unittest.main()
