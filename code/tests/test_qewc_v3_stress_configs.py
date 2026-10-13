# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

import yaml

from scripts.experiments.qewc_v3.generate_stress_configs import (
    BASE_CONFIG,
    FAMILIES,
    MULTIPLIERS,
    build_configs,
    label,
)


class TestQewcV3StressConfigs(unittest.TestCase):

    def test_each_candidate_only_scales_its_declared_components(self):
        base = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
        base_noise = base["data"]["noise_model"]
        configs = build_configs(base)
        self.assertEqual(len(configs), 16)
        for family, keys in FAMILIES.items():
            for multiplier in MULTIPLIERS:
                name = f"{family}_{label(multiplier)}x"
                noise = configs[name]["data"]["noise_model"]
                changed = {key for key in base_noise if noise[key] != base_noise[key]}
                self.assertEqual(changed, set(keys), name)
                for key in keys:
                    self.assertAlmostEqual(noise[key], base_noise[key] * multiplier)


if __name__ == "__main__":
    unittest.main()
