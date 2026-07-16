# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import tempfile
import unittest
from pathlib import Path

from workflows.run import find_best_model


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


if __name__ == "__main__":
    unittest.main()
