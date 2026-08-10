# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import unittest

from qec.noise_model import NoiseModel
from scripts.experiments.unknown_noise.build_qadapt_unseen_noise_ab_report import (
    build_summary,
)
from scripts.experiments.unknown_noise.generate_qadapt_unseen_noise_ab_configs import (
    AXES,
    DEFAULT_BASE_CONFIG,
    generate_asymmetric_axis_family,
    generate_independent_parameter_family,
    load_base_noise_model,
)


class QAdaptUnseenNoiseABGeneratorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.base_noise = load_base_noise_model(DEFAULT_BASE_CONFIG)

    def test_asymmetric_family_is_deterministic_unique_and_nonuniform(self) -> None:
        first = generate_asymmetric_axis_family(self.base_noise)
        second = generate_asymmetric_axis_family(self.base_noise)
        self.assertEqual(len(first), 55)
        self.assertEqual(
            [item["noise_model_sha256"] for item in first],
            [item["noise_model_sha256"] for item in second],
        )
        self.assertEqual(len({item["noise_model_sha256"] for item in first}), 55)
        for item in first:
            active_values = [
                item["axis_multipliers"][axis] for axis in item["active_axes"]
            ]
            self.assertGreater(len(set(active_values)), 1)
            NoiseModel.from_config_dict(item["noise_model"])

    def test_asymmetric_overlap_uses_maximum_axis_multiplier(self) -> None:
        items = generate_asymmetric_axis_family(self.base_noise)
        item = next(
            value
            for value in items
            if {"meas_all", "z_bias"}.issubset(value["active_axes"])
        )
        expected = max(
            item["axis_multipliers"]["meas_all"],
            item["axis_multipliers"]["z_bias"],
        )
        self.assertEqual(item["parameter_multipliers"]["p_meas_X"], expected)
        self.assertIn("p_meas_X", AXES["meas_all"])
        self.assertIn("p_meas_X", AXES["z_bias"])

    def test_independent_family_perturbs_all_25_parameters_in_range(self) -> None:
        items = generate_independent_parameter_family(self.base_noise)
        self.assertEqual(len(items), 55)
        self.assertEqual(len({item["noise_model_sha256"] for item in items}), 55)
        all_values = []
        for item in items:
            multipliers = item["parameter_multipliers"]
            self.assertEqual(set(multipliers), set(self.base_noise))
            self.assertEqual(len(multipliers), 25)
            self.assertTrue(all(0.5 <= value <= 3.0 for value in multipliers.values()))
            all_values.extend(multipliers.values())
            NoiseModel.from_config_dict(item["noise_model"])
        self.assertLess(min(all_values), 1.0)
        self.assertGreater(max(all_values), 1.0)


class QAdaptUnseenNoiseABSummaryTest(unittest.TestCase):
    def test_summary_uses_configuration_level_effects(self) -> None:
        rows = []
        for family_key, family in (("A", "A_asymmetric_axes"), ("B", "B_independent_25p")):
            for distance in (7, 9):
                for index, delta in enumerate((-0.02, -0.01, 0.01)):
                    rows.append(
                        {
                            "family_key": family_key,
                            "family": family,
                            "distance": distance,
                            "basis": "both",
                            "samples": 100,
                            "ewc_ler": 0.2 + delta,
                            "noewc_ler": 0.2,
                            "delta_ler": delta,
                            "ewc_only_errors": 5,
                            "noewc_only_errors": 5 - int(delta < 0),
                            "ewc_win": int(delta < 0),
                            "residual_density_delta": delta / 10,
                            "backend_latency_delta_us_per_round": delta,
                        }
                    )
        summary = build_summary(rows, seed=123)
        self.assertEqual(len(summary), 6)
        family_a_d7 = next(
            row
            for row in summary
            if row["family_key"] == "A" and row["distance"] == "7"
        )
        self.assertEqual(family_a_d7["config_count"], 3)
        self.assertAlmostEqual(family_a_d7["delta_macro_mean"], -0.02 / 3)
        self.assertEqual(family_a_d7["ewc_win_count"], 2)


if __name__ == "__main__":
    unittest.main()
