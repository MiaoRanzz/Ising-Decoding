# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest
from types import SimpleNamespace

import numpy as np

from benchmarks.bench_quantized_predecoder_seeded import _SAMPLE_METADATA, seed_dataset


class _Sampler:
    def __init__(self, seed):
        self.seed = seed

    def sample(self, shots):
        rng = np.random.default_rng(self.seed)
        return rng.integers(0, 2, size=(shots, 4), dtype=np.uint8)


class _Converter:
    def convert(self, *, measurements, append_observables):
        if not append_observables:
            raise AssertionError("observable must be appended")
        return measurements[:, :3]


class _Circuit:
    def __init__(self):
        self.received_seed = None

    def compile_sampler(self, *, seed):
        self.received_seed = seed
        return _Sampler(seed)

    def compile_m2d_converter(self):
        return _Converter()


class SeededStimDatasetTest(unittest.TestCase):
    def test_same_seed_produces_same_hash_and_different_seed_changes_it(self):
        circuit = _Circuit()
        dataset = SimpleNamespace(circ=SimpleNamespace(stim_circuit=circuit))
        seed_dataset(dataset, 128, 45678)
        first = _SAMPLE_METADATA["dataset_sample_sha256"]
        seed_dataset(dataset, 128, 45678)
        self.assertEqual(first, _SAMPLE_METADATA["dataset_sample_sha256"])
        seed_dataset(dataset, 128, 45679)
        self.assertNotEqual(first, _SAMPLE_METADATA["dataset_sample_sha256"])
        self.assertEqual(45679, circuit.received_seed)
        self.assertTrue(_SAMPLE_METADATA["stim_sampler_seeded"])


if __name__ == "__main__":
    unittest.main()
