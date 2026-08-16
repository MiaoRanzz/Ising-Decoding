"""CPU-only regression tests for persistent, explicitly seeded DEM streams."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

import qec.dem_sampling as dem_mod
from data.generator_torch import QCDataGeneratorTorch, _normalized_sampler_seed


class _FakeOptions:
    def __init__(self, device_id=0):
        self.device_id = device_id


class _FakeBitMatrixSampler:
    """Small NumPy stand-in with the cuST constructor/sample interface."""

    def __init__(self, H_transposed, probabilities, max_shots, **kwargs):
        del max_shots
        self.H_transposed = np.asarray(H_transposed, dtype=np.uint8)
        self.probabilities = np.asarray(probabilities, dtype=np.float64)
        self.rng = np.random.default_rng(kwargs.get("seed", 918273645))
        self.outcomes = None

    def sample(self, shots):
        errors = self.rng.random((int(shots), self.probabilities.size)) < self.probabilities
        self.outcomes = (
            errors.astype(np.uint8) @ self.H_transposed.astype(np.uint8)
        ) % 2

    def get_outcomes(self, bit_packed=False):
        if bit_packed:
            raise AssertionError("test stand-in only supports unpacked outcomes")
        return self.outcomes


class TestPersistentDemStreams(unittest.TestCase):
    def setUp(self):
        dem_mod._reset_sampler_cache()
        self.backend = patch.multiple(
            dem_mod,
            BitMatrixSampler=_FakeBitMatrixSampler,
            Options=_FakeOptions,
            _CUPY_AVAILABLE=False,
        )
        self.backend.start()

    def tearDown(self):
        dem_mod._reset_sampler_cache()
        self.backend.stop()

    @staticmethod
    def _inputs(flipped=False):
        H = torch.tensor(
            [[1, 0, 1, 0, 1], [0, 1, 1, 0, 0], [0, 0, 0, 1, 1]],
            dtype=torch.uint8,
        )
        if flipped:
            H = H.flip(0).clone()
        p = torch.tensor([0.11, 0.29, 0.47, 0.68, 0.83], dtype=torch.float32)
        return H, p

    def test_interleaving_validation_does_not_reset_training_stream(self):
        train_H, train_p = self._inputs()
        val_H, val_p = self._inputs(flipped=True)

        train_first = dem_mod.dem_sampling(train_H, train_p, 64, seed=101)
        dem_mod.dem_sampling(val_H, val_p, 64, seed=202)
        train_second_after_val = dem_mod.dem_sampling(train_H, train_p, 64)

        dem_mod._reset_sampler_cache()
        train_first_reference = dem_mod.dem_sampling(train_H, train_p, 64, seed=101)
        train_second_reference = dem_mod.dem_sampling(train_H, train_p, 64)

        self.assertTrue(torch.equal(train_first, train_first_reference))
        self.assertTrue(torch.equal(train_second_after_val, train_second_reference))
        self.assertEqual(len(dem_mod._sampler_cache), 1)

    def test_same_plan_and_seed_replays_both_batches(self):
        H, p = self._inputs()
        first_a = dem_mod.dem_sampling(H, p, 37, seed=777)
        second_a = dem_mod.dem_sampling(H, p, 37)
        dem_mod._reset_sampler_cache()
        first_b = dem_mod.dem_sampling(H, p, 37, seed=777)
        second_b = dem_mod.dem_sampling(H, p, 37)
        self.assertTrue(torch.equal(first_a, first_b))
        self.assertTrue(torch.equal(second_a, second_b))
        self.assertFalse(torch.equal(first_a, second_a))

    def test_stream_and_basis_seed_offsets_are_distinct(self):
        base = 1_900_000_000
        seeds = {
            _normalized_sampler_seed(base, 0, stage_offset + stream_offset, basis)
            for stage_offset in (0, 400_000_000, 800_000_000)
            for stream_offset in (0, 100_000_000, 200_000_000)
            for basis in ("X", "Z")
        }
        self.assertEqual(len(seeds), 18)


class _FakeMemoryCircuitTorch:
    instances = []

    def __init__(self, **kwargs):
        self.basis = kwargs["basis"]
        self.sampler_seed = kwargs["sampler_seed"]
        self.__class__.instances.append(self)


class TestGeneratorSeedWiring(unittest.TestCase):
    def setUp(self):
        _FakeMemoryCircuitTorch.instances.clear()

    def test_generator_wires_seed_without_mutating_global_torch_rng(self):
        artifacts = {
            "H": torch.zeros((6, 5), dtype=torch.uint8),
            "p": torch.full((5,), 0.2, dtype=torch.float32),
            "A": None,
        }
        torch.manual_seed(314159)
        state_before = torch.get_rng_state().clone()
        with patch(
            "qec.precompute_dem.precompute_dem_bundle_surface_code",
            return_value=artifacts,
        ), patch(
            "qec.surface_code.memory_circuit_torch.MemoryCircuitTorch",
            _FakeMemoryCircuitTorch,
        ):
            generator = QCDataGeneratorTorch(
                distance=3,
                n_rounds=3,
                p_error=0.01,
                measure_basis="both",
                base_seed=1234,
                seed_offset=100_000_000,
                device=torch.device("cpu"),
            )
        state_after = torch.get_rng_state()

        self.assertTrue(torch.equal(state_before, state_after))
        self.assertEqual(
            {item.basis: item.sampler_seed for item in _FakeMemoryCircuitTorch.instances},
            generator.sampler_seeds,
        )
        self.assertNotEqual(generator.sampler_seeds["X"], generator.sampler_seeds["Z"])


if __name__ == "__main__":
    unittest.main()
