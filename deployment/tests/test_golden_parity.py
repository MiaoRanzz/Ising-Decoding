# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np
import torch

from ising_decoder import DecoderConfig, SurfaceDecoderCore


DEPLOYMENT_ROOT = Path(__file__).resolve().parents[1]


class GoldenParityTest(unittest.TestCase):
    @classmethod
    def _decoder(cls, basis: str) -> SurfaceDecoderCore:
        return SurfaceDecoderCore(
            DecoderConfig(
                checkpoint_path=DEPLOYMENT_ROOT / "artifacts/HTnet.0.82.pt",
                geometry_path=DEPLOYMENT_ROOT / "artifacts/geometry_d9_xv.npz",
                dem_path=DEPLOYMENT_ROOT / f"artifacts/surface_d9_t9_xv_t0_{basis}.dem",
                basis=basis,
                device="cpu",
            )
        )

    def test_x_and_z_match_authoritative_golden_vectors(self) -> None:
        for basis in ("X", "Z"):
            with self.subTest(basis=basis):
                with np.load(
                    DEPLOYMENT_ROOT / f"tests/fixtures/golden_{basis}.npz",
                    allow_pickle=False,
                ) as golden:
                    detectors = np.ascontiguousarray(golden["detectors"][:2], dtype=np.uint8)
                    decoder = self._decoder(basis)
                    with torch.inference_mode():
                        output = decoder.pipeline(torch.from_numpy(detectors))
                    predecoder = output[:, 0].numpy()
                    residual = output[:, 1:].numpy()
                    np.testing.assert_array_equal(predecoder, golden["predecoder_flip"][:2])
                    np.testing.assert_array_equal(residual, golden["residual"][:2])
                    pymatching_flip = np.asarray(
                        decoder.matcher.decode_batch(residual), dtype=np.uint8
                    ).reshape(-1)
                    np.testing.assert_array_equal(
                        pymatching_flip, golden["pymatching_flip"][:2]
                    )
                    final = np.bitwise_xor(predecoder, pymatching_flip)
                    np.testing.assert_array_equal(final, golden["logical_flip"][:2])
                    single = decoder.decode_detectors(detectors[0], return_residual=True)
                    self.assertEqual(single.logical_flip, int(golden["logical_flip"][0]))
                    np.testing.assert_array_equal(single.residual, golden["residual"][0])


if __name__ == "__main__":
    unittest.main()
