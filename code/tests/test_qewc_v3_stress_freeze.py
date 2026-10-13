# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import sys
import unittest
from pathlib import Path

from omegaconf import OmegaConf


CODE_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = CODE_ROOT.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from workflows.config_validator import (  # noqa: E402
    apply_public_defaults_and_model,
    validate_public_config,
)


class TestQewcV3StressFreeze(unittest.TestCase):

    def test_frozen_config_hashes_and_orders(self):
        manifest = OmegaConf.load(REPO_ROOT / "experiment" / "qewc-v3" / "tasks_stress.yaml")
        self.assertTrue(bool(manifest.frozen))
        self.assertEqual(list(manifest.high_conflict_order), ["T1", "T3", "T2", "T4"])
        self.assertEqual(list(manifest.reverse_order), ["T4", "T2", "T3", "T1"])
        for name in ("T1", "T2", "T3", "T4"):
            task = manifest.tasks[name]
            path = REPO_ROOT / str(task.config)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(digest, str(task.config_sha256))

    def test_stress_mix_exactly_matches_frozen_tasks(self):
        manifest = OmegaConf.load(REPO_ROOT / "experiment" / "qewc-v3" / "tasks_stress.yaml")
        mix = OmegaConf.load(
            REPO_ROOT / "conf" / "experiments" / "qewc_v3" / "mix" / "stress_selected.yaml"
        )
        spec = validate_public_config(mix)
        merged = apply_public_defaults_and_model(mix, spec)
        self.assertIsNone(merged.data.noise_model)
        actual = {
            str(item.name): OmegaConf.to_container(item.noise_model, resolve=True)
            for item in mix.data.noise_model_mixture
        }
        for name in ("T0", "T1", "T2", "T3", "T4"):
            task_cfg = OmegaConf.load(REPO_ROOT / str(manifest.tasks[name].config))
            expected = OmegaConf.to_container(task_cfg.data.noise_model, resolve=True)
            self.assertEqual(actual[name], expected)


if __name__ == "__main__":
    unittest.main()
