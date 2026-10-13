# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import unittest

from qec.noise_model import NoiseModel
from scripts.experiments.unknown_noise.generate_qadapt_unseen_noise_hierarchical_configs import (
    MIN_OUTSIDE_TRAINING_ENVELOPE,
    SCALAR_RANGES,
    TOTAL_RANGES,
    generate_hierarchical_family,
)


class QAdaptHierarchicalNoiseTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.first = generate_hierarchical_family()
        cls.second = generate_hierarchical_family()

    def test_is_deterministic_unique_and_valid(self) -> None:
        self.assertEqual(len(self.first), 55)
        self.assertEqual(
            [item["noise_model_sha256"] for item in self.first],
            [item["noise_model_sha256"] for item in self.second],
        )
        self.assertEqual(
            len({item["noise_model_sha256"] for item in self.first}), 55
        )
        for item in self.first:
            self.assertEqual(len(item["noise_model"]), 25)
            NoiseModel.from_config_dict(item["noise_model"])

    def test_absolute_scalars_and_channel_totals_are_in_range(self) -> None:
        for item in self.first:
            noise = item["noise_model"]
            for key, (low, high) in SCALAR_RANGES.items():
                self.assertLessEqual(low, noise[key])
                self.assertLessEqual(noise[key], high)
            totals = item["sampled_totals"]
            for key, (low, high) in TOTAL_RANGES.items():
                self.assertLessEqual(low, totals[key])
                self.assertLessEqual(totals[key], high)
            self.assertAlmostEqual(
                totals["idle_cnot_total"],
                sum(noise[f"p_idle_cnot_{p}"] for p in "XYZ"),
            )
            self.assertAlmostEqual(
                totals["idle_spam_total"],
                sum(noise[f"p_idle_spam_{p}"] for p in "XYZ"),
            )
            self.assertAlmostEqual(
                totals["cnot_total"],
                sum(
                    value
                    for key, value in noise.items()
                    if key.startswith("p_cnot_")
                ),
            )

    def test_every_config_is_outside_training_envelope(self) -> None:
        for item in self.first:
            self.assertGreaterEqual(
                item["outside_training_envelope_count"],
                MIN_OUTSIDE_TRAINING_ENVELOPE,
            )
            self.assertGreater(
                item["min_log10_rms_distance_to_training"], 0.0
            )


if __name__ == "__main__":
    unittest.main()
