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


CONFIG_DIR = REPO_ROOT / "conf" / "experiments" / "qewc_v3" / "realistic"
CONFIGS = {
    "T0": CONFIG_DIR / "t0_base.yaml",
    "T1": CONFIG_DIR / "t1_meas_1p5.yaml",
    "T2": CONFIG_DIR / "t2_cnot_1p5.yaml",
    "T3": CONFIG_DIR / "t3_idle_1p5.yaml",
    "T4": CONFIG_DIR / "t4_z_bias_1p5.yaml",
}


class TestQewcV3RealisticConfigs(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.configs = {name: OmegaConf.load(path) for name, path in CONFIGS.items()}

    def test_configs_pass_public_validator_and_keep_a101_geometry(self):
        for name, cfg in self.configs.items():
            with self.subTest(task=name):
                spec = validate_public_config(cfg)
                merged = apply_public_defaults_and_model(cfg, spec)
                self.assertEqual(int(merged.distance), 9)
                self.assertEqual(int(merged.n_rounds), 9)
                self.assertEqual(str(merged.meas_basis), "both")
                self.assertEqual(str(merged.data.code_rotation), "XV")
                self.assertTrue(bool(merged.data.skip_noise_upscaling))
                self.assertEqual(int(merged.seed), 20260907)
                self.assertTrue(bool(merged.htnet_ablation.branch_mixer))
                self.assertFalse(bool(merged.htnet_ablation.axis_channel_gate))
                self.assertTrue(bool(merged.htnet_ablation.input_skip_head))

    def test_realistic_tasks_only_change_declared_noise_components(self):
        base = OmegaConf.to_container(self.configs["T0"].data.noise_model, resolve=True)
        changed = {
            name: {
                key
                for key, value in OmegaConf.to_container(
                    cfg.data.noise_model, resolve=True
                ).items()
                if value != base[key]
            }
            for name, cfg in self.configs.items()
            if name != "T0"
        }
        self.assertEqual(changed["T1"], {"p_meas_X", "p_meas_Z"})
        self.assertEqual(
            changed["T2"], {key for key in base if key.startswith("p_cnot_")}
        )
        self.assertEqual(
            changed["T3"],
            {key for key in base if key.startswith(("p_idle_cnot_", "p_idle_spam_"))},
        )
        self.assertEqual(
            changed["T4"],
            {
                "p_prep_X",
                "p_meas_X",
                "p_idle_cnot_Z",
                "p_idle_spam_Z",
                "p_cnot_IZ",
                "p_cnot_XZ",
                "p_cnot_YZ",
                "p_cnot_ZI",
                "p_cnot_ZX",
                "p_cnot_ZY",
                "p_cnot_ZZ",
            },
        )


if __name__ == "__main__":
    unittest.main()
