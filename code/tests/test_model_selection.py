# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from workflows.run import _resolve_inference_checkpoint, find_best_model


class TestModelSelection(unittest.TestCase):
    def test_custom_model_weights_win_over_training_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = Path(tmpdir) / "PreDecoderSTFusion_v2.0.1.pt"
            checkpoint = Path(tmpdir) / "checkpoint.0.1.pt"
            model.touch()
            checkpoint.touch()

            self.assertEqual(find_best_model(tmpdir, rank=1), str(model))

    def test_highest_epoch_wins_for_any_predecoder_architecture(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            older = Path(tmpdir) / "PreDecoderFactorized_v1.0.2.pt"
            newer = Path(tmpdir) / "PreDecoderFactorized_v1.0.10.pt"
            older.touch()
            newer.touch()

            self.assertEqual(find_best_model(tmpdir, rank=1), str(newer))

    def test_named_release_model_remains_supported(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            release = Path(tmpdir) / "Ising-Decoder-SurfaceCode-1-Fast.pt"
            release.touch()

            self.assertEqual(find_best_model(tmpdir, rank=1), str(release))

    def test_public_inference_checkpoint_uses_zero_for_best_model(self):
        self.assertEqual(
            _resolve_inference_checkpoint(SimpleNamespace(inf=SimpleNamespace(checkpoint=0))),
            -1,
        )
        self.assertEqual(
            _resolve_inference_checkpoint(SimpleNamespace(inf=SimpleNamespace(checkpoint=12))),
            12,
        )
        self.assertIsNone(_resolve_inference_checkpoint(SimpleNamespace()))

    def test_public_inference_checkpoint_rejects_invalid_values(self):
        for invalid in (-1, True, "12"):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    _resolve_inference_checkpoint(
                        SimpleNamespace(inf=SimpleNamespace(checkpoint=invalid))
                    )


if __name__ == "__main__":
    unittest.main()
