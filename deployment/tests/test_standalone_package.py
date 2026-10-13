# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
from pathlib import Path
import unittest

from ising_decoder.cli import _compose_config, _read_samples
from ising_decoder.config import DecoderConfig


DEPLOYMENT_ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_IMPORTS = {"data", "evaluation", "model", "training", "workflows"}


class StandalonePackageTest(unittest.TestCase):
    def test_source_does_not_import_parent_project_modules(self) -> None:
        source_root = DEPLOYMENT_ROOT / "src/ising_decoder"
        imported_roots: set[str] = set()
        for path in source_root.glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    imported_roots.add(node.module.split(".", 1)[0])
        self.assertFalse(imported_roots & FORBIDDEN_IMPORTS)

    def test_hydra_config_converts_to_plain_dataclass(self) -> None:
        cfg = _compose_config(["basis=Z", "device=cpu"])
        decoder_cfg = DecoderConfig.from_omegaconf(cfg)
        self.assertEqual(decoder_cfg.basis, "Z")
        self.assertEqual(decoder_cfg.device, "cpu")
        self.assertEqual(decoder_cfg.num_detectors, 720)

    def test_golden_dets_file_uses_expected_layout(self) -> None:
        detectors, observables = _read_samples(
            DEPLOYMENT_ROOT / "tests/fixtures/golden_X.dets",
            num_detectors=720,
            num_observables=1,
        )
        self.assertEqual(detectors.shape, (32, 720))
        self.assertEqual(observables.shape, (32, 1))


if __name__ == "__main__":
    unittest.main()
