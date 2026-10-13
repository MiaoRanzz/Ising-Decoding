# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

import torch

from data.datapipe_stim import QCDataPipePreDecoder_Memory_inference


def make_pipe(seed):
    return QCDataPipePreDecoder_Memory_inference(
        distance=3,
        n_rounds=3,
        num_samples=128,
        error_mode="circuit_level_surface_custom",
        p_error=0.01,
        measure_basis="X",
        code_rotation="XV",
        seed=seed,
    )


class TestStimInferenceSeed(unittest.TestCase):

    def test_same_seed_is_exact_and_different_seed_changes_stream(self):
        first = make_pipe(123)
        second = make_pipe(123)
        different = make_pipe(124)
        self.assertTrue(torch.equal(first.dets_and_obs, second.dets_and_obs))
        self.assertFalse(torch.equal(first.dets_and_obs, different.dets_and_obs))


if __name__ == "__main__":
    unittest.main()
