# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
import unittest
from unittest import mock

import numpy as np
import torch

from ising_decoder import DecoderConfig, SurfaceDecoderCore


class _FakePipeline(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, detectors: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        predecoder = torch.ones((1, 1), dtype=torch.uint8, device=detectors.device)
        return torch.cat([predecoder, detectors.to(torch.uint8)], dim=1)


class _FakeMatcher:
    def __init__(self):
        self.calls = 0

    def decode(self, residual: np.ndarray) -> np.ndarray:
        self.calls += 1
        return np.asarray([1], dtype=np.uint8)


def _config(device: str = "cpu") -> DecoderConfig:
    return DecoderConfig(
        checkpoint_path=Path("unused.pt"),
        geometry_path=Path("unused.npz"),
        dem_path=Path("unused.dem"),
        basis="X",
        device=device,
    )


class SurfaceDecoderCoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self.pipeline = _FakePipeline()
        self.matcher = _FakeMatcher()
        self.decoder = SurfaceDecoderCore(
            _config(),
            pipeline=self.pipeline,
            matcher=self.matcher,
        )

    def test_decode_xors_predecoder_and_pymatching(self) -> None:
        detectors = np.zeros(720, dtype=np.uint8)
        result = self.decoder.decode_detectors(detectors)
        self.assertEqual(result.predecoder_flip, 1)
        self.assertEqual(result.pymatching_flip, 1)
        self.assertEqual(result.logical_flip, 0)
        self.assertEqual(result.status, "ok")
        self.assertIsNone(result.residual)
        self.assertEqual(self.pipeline.calls, 1)
        self.assertEqual(self.matcher.calls, 1)

    def test_residual_is_opt_in(self) -> None:
        detectors = np.zeros(720, dtype=np.uint8)
        result = self.decoder.decode_detectors(detectors, return_residual=True)
        np.testing.assert_array_equal(result.residual, detectors)

    def test_rejects_wrong_dtype(self) -> None:
        with self.assertRaisesRegex(TypeError, "dtype uint8"):
            self.decoder.decode_detectors(np.zeros(720, dtype=np.int32))

    def test_rejects_wrong_shape(self) -> None:
        with self.assertRaisesRegex(ValueError, "shape"):
            self.decoder.decode_detectors(np.zeros((1, 720), dtype=np.uint8))

    def test_rejects_non_binary_values(self) -> None:
        detectors = np.zeros(720, dtype=np.uint8)
        detectors[3] = 2
        with self.assertRaisesRegex(ValueError, "binary values"):
            self.decoder.decode_detectors(detectors)

    def test_cuda_request_does_not_silently_fall_back(self) -> None:
        with mock.patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "CUDA is unavailable"):
                SurfaceDecoderCore(
                    _config(device="cuda"),
                    pipeline=_FakePipeline(),
                    matcher=_FakeMatcher(),
                )


if __name__ == "__main__":
    unittest.main()
