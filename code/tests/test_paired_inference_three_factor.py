# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for paired HTNet three-factor ablation statistics."""

import sys
import unittest
from pathlib import Path

import numpy as np

_repo_code = Path(__file__).resolve().parent.parent
if str(_repo_code) not in sys.path:
    sys.path.insert(0, str(_repo_code))

from scripts.paired_inference_compare import (
    build_three_factor_effect_rows,
    build_three_factor_latency_effect_rows,
    parse_model_spec,
    parse_three_factor_cells,
)


class TestPairedInferenceThreeFactor(unittest.TestCase):

    def test_model_spec_accepts_model_specific_config(self):
        spec = parse_model_spec("a000:111:/tmp/model.pt:/tmp/a000.yaml")
        self.assertEqual(spec.name, "a000")
        self.assertEqual(spec.model_id, 111)
        self.assertEqual(spec.checkpoint, Path("/tmp/model.pt"))
        self.assertEqual(spec.config_file, Path("/tmp/a000.yaml"))

    def test_three_factor_cell_order_requires_eight_unique_methods(self):
        cells = parse_three_factor_cells(
            "a000,a001,a010,a011,a100,a101,a110,a111"
        )
        self.assertEqual(cells[0], "a000")
        self.assertEqual(cells[-1], "a111")
        with self.assertRaises(Exception):
            parse_three_factor_cells("a000,a001")

    def test_main_effects_and_holm_use_paired_shot_contrasts(self):
        cells = parse_three_factor_cells(
            "a000,a001,a010,a011,a100,a101,a110,a111"
        )
        masks = {}
        for cell in cells:
            mixer_enabled = cell[1] == "1"
            masks[cell] = np.full(16, mixer_enabled, dtype=np.bool_)

        rows = build_three_factor_effect_rows({"X": masks}, cells)
        main = {
            row["factors"][0]: row
            for row in rows
            if row["effect_kind"] == "main"
        }
        self.assertAlmostEqual(main["branch_mixer"]["effect_ler"], 1.0)
        self.assertAlmostEqual(main["axis_channel_gate"]["effect_ler"], 0.0)
        self.assertAlmostEqual(main["input_skip_head"]["effect_ler"], 0.0)
        self.assertTrue(main["branch_mixer"]["holm_significant_0p05"])
        self.assertFalse(main["axis_channel_gate"]["holm_significant_0p05"])

        interactions = [
            row for row in rows if row["effect_kind"] != "main"
        ]
        self.assertTrue(
            all(abs(row["effect_ler"]) < 1e-12 for row in interactions)
        )

    def test_latency_factor_effects_use_repeat_level_contrasts(self):
        cells = parse_three_factor_cells(
            "a000,a001,a010,a011,a100,a101,a110,a111"
        )
        common_noise = np.arange(5, dtype=np.float64)
        metric_rows = []
        for cell in cells:
            mixer, axis_gate, input_skip = (int(bit) for bit in cell[1:])
            values = (
                10.0
                + common_noise
                + 2.0 * mixer
                + 3.0 * axis_gate
                + 4.0 * input_skip
            )
            metric_rows.append(
                {
                    "basis": "X",
                    "method": cell,
                    "latency_us_per_round": float(np.median(values)),
                    "latency_us_per_round_values": values.tolist(),
                }
            )

        rows = build_three_factor_latency_effect_rows(metric_rows, cells)
        main = {
            row["factors"][0]: row
            for row in rows
            if row["effect_kind"] == "main"
        }
        self.assertAlmostEqual(
            main["branch_mixer"]["effect_latency_us_per_round"], 2.0
        )
        self.assertAlmostEqual(
            main["axis_channel_gate"]["effect_latency_us_per_round"], 3.0
        )
        self.assertAlmostEqual(
            main["input_skip_head"]["effect_latency_us_per_round"], 4.0
        )


if __name__ == "__main__":
    unittest.main()
