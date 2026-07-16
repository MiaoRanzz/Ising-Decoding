# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
import tempfile
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.compute_ewc_fisher import normalize_state_dict_keys, select_model_checkpoint


class TestEWCFisherScript(unittest.TestCase):

    def test_select_model_checkpoint_ignores_training_checkpoint_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "checkpoint.0.99.pt").write_text("skip")
            (root / "PreDecoderSTFusion_v2.0.5.pt").write_text("old")
            (root / "PreDecoderSTFusion_v2.0.20.pt").write_text("new")

            selected = select_model_checkpoint(root)

            self.assertEqual(selected.name, "PreDecoderSTFusion_v2.0.20.pt")

    def test_normalize_state_dict_keys_strips_ddp_and_compile_prefixes(self):
        state = {
            "module._orig_mod.stem.weight": torch.ones(1),
            "_orig_mod.module.head.bias": torch.zeros(1),
        }

        normalized = normalize_state_dict_keys(state)

        self.assertEqual(set(normalized), {"stem.weight", "head.bias"})


if __name__ == "__main__":
    unittest.main()
