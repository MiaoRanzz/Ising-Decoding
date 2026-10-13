# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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


REALISTIC_DIR = REPO_ROOT / "conf" / "experiments" / "qewc_v3" / "realistic"


class TestQewcV3MixConfig(unittest.TestCase):

    def setUp(self):
        self.cfg = OmegaConf.load(REALISTIC_DIR / "mix_t0_t4.yaml")

    def test_mix_passes_public_validation_without_injecting_default_noise(self):
        spec = validate_public_config(self.cfg)
        merged = apply_public_defaults_and_model(self.cfg, spec)

        self.assertIsNone(merged.data.noise_model)
        self.assertEqual(
            [str(item.name) for item in merged.data.noise_model_mixture],
            ["T0", "T1", "T2", "T3", "T4"],
        )
        self.assertEqual(int(merged.distance), 9)
        self.assertEqual(int(merged.n_rounds), 9)
        self.assertEqual(str(merged.meas_basis), "both")
        self.assertEqual(str(merged.data.code_rotation), "XV")

    def test_mix_noise_models_exactly_match_individual_task_configs(self):
        mix_models = {
            str(item.name): OmegaConf.to_container(item.noise_model, resolve=True)
            for item in self.cfg.data.noise_model_mixture
        }
        individual_paths = {
            "T0": "t0_base.yaml",
            "T1": "t1_meas_1p5.yaml",
            "T2": "t2_cnot_1p5.yaml",
            "T3": "t3_idle_1p5.yaml",
            "T4": "t4_z_bias_1p5.yaml",
        }
        for name, filename in individual_paths.items():
            individual = OmegaConf.load(REALISTIC_DIR / filename)
            expected = OmegaConf.to_container(individual.data.noise_model, resolve=True)
            self.assertEqual(mix_models[name], expected)

    def test_mix_rejects_duplicate_names_and_single_noise_conflict(self):
        duplicate = OmegaConf.create(OmegaConf.to_container(self.cfg, resolve=True))
        duplicate.data.noise_model_mixture[1].name = "T0"
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            validate_public_config(duplicate)

        conflict = OmegaConf.create(OmegaConf.to_container(self.cfg, resolve=True))
        conflict.data.noise_model = {"p": 0.003}
        with self.assertRaisesRegex(ValueError, "either"):
            validate_public_config(conflict)


if __name__ == "__main__":
    unittest.main()
