# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import csv
import sys
import tempfile
import unittest
from pathlib import Path

from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent))

from qec.noise_model import NoiseModel, _single_p_mapping
from scripts.generate_unknown_seqfav_zstress_v2_configs import (
    DEFAULT_GRID,
    generate_candidate_specs,
    generate_zstress_noise_models,
    write_candidate_configs,
    write_fixed_configs,
)
from scripts.run_unknown_seqfav_zstress_v2 import (
    select_confirmed_fixed_rows,
    select_named_confirmed_fixed_rows,
    select_top_candidates,
    write_scan_csv,
    write_zstress_markdown,
    zstress_success_status,
)


class TestUnknownSeqfavZStressV2Generation(unittest.TestCase):

    def _base_config_path(self, root: Path) -> Path:
        cfg = {
            "model_id": 111,
            "distance": 9,
            "n_rounds": 9,
            "workflow": {"task": "train"},
            "data": {
                "code_rotation": "O1",
                "noise_model": _single_p_mapping(0.001),
            },
        }
        path = root / "base_t0.yaml"
        OmegaConf.save(OmegaConf.create(cfg), path)
        return path

    def test_candidate_specs_are_combination_ood_and_exclude_full_axis_hard_cases(self):
        specs = generate_candidate_specs(DEFAULT_GRID)

        self.assertGreater(len(specs), 0)
        self.assertEqual([spec["candidate_index"] for spec in specs], list(range(len(specs))))
        for spec in specs:
            multipliers = spec["axis_multipliers"]
            active_axes = [axis for axis, value in multipliers.items() if float(value) != 1.0]
            self.assertGreaterEqual(len(active_axes), 2)
            self.assertLess(len(active_axes), 4)
            self.assertNotEqual(
                set(active_axes),
                {"meas_all", "cnot_all", "idle_all", "z_bias"},
            )

    def test_noise_generation_uses_max_multiplier_for_overlapping_axes(self):
        base_noise = _single_p_mapping(0.001)
        specs = [
            {
                "candidate_index": 0,
                "candidate_key": "candidate_000",
                "axis_multipliers": {
                    "meas_all": 1.30,
                    "cnot_all": 1.45,
                    "idle_all": 1.20,
                    "z_bias": 1.65,
                },
                "purpose": "overlap check",
            }
        ]

        generated = generate_zstress_noise_models(base_noise, specs)
        noise = generated[0]["noise_model"]

        self.assertAlmostEqual(noise["p_meas_X"], base_noise["p_meas_X"] * 1.65)
        self.assertAlmostEqual(noise["p_meas_Z"], base_noise["p_meas_Z"] * 1.30)
        self.assertAlmostEqual(noise["p_cnot_IZ"], base_noise["p_cnot_IZ"] * 1.65)
        self.assertAlmostEqual(noise["p_cnot_IX"], base_noise["p_cnot_IX"] * 1.45)
        self.assertAlmostEqual(noise["p_idle_cnot_Z"], base_noise["p_idle_cnot_Z"] * 1.65)
        self.assertAlmostEqual(noise["p_idle_cnot_X"], base_noise["p_idle_cnot_X"] * 1.20)
        self.assertAlmostEqual(noise["p_prep_X"], base_noise["p_prep_X"] * 1.65)
        self.assertAlmostEqual(noise["p_prep_Z"], base_noise["p_prep_Z"])
        NoiseModel.from_config_dict(noise)

    def test_candidate_and_fixed_configs_are_valid_and_manifest_records_design(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base_config = self._base_config_path(root)
            specs = generate_candidate_specs(
                {
                    "meas_all": [1.00, 1.10],
                    "cnot_all": [1.35],
                    "idle_all": [1.00, 1.20],
                    "z_bias": [1.00],
                }
            )

            candidate_paths, candidate_manifest = write_candidate_configs(
                base_config=base_config,
                output_dir=root / "conf" / "experiments" / "unknown_seqfav_zstress_v2",
                manifest=root / "candidate_manifest.json",
                candidate_specs=specs,
            )
            fixed_paths, fixed_manifest = write_fixed_configs(
                base_config=base_config,
                output_dir=root / "conf" / "experiments" / "unknown_seqfav_zstress_v2",
                manifest=root / "fixed_manifest.json",
                selected_specs=specs[:2],
            )

            self.assertEqual(candidate_manifest["design"], "Z-basis seq-favoring stress test")
            self.assertEqual(
                candidate_manifest["environments"][0]["config_name"],
                "experiments/unknown_seqfav_zstress_v2/config_unknown_seqfav_zstress_v2_candidate_000",
            )
            self.assertEqual(fixed_manifest["design"], "Z-basis seq-favoring stress test")
            self.assertEqual(len(candidate_paths), len(specs))
            self.assertEqual([path.name for path in fixed_paths], [
                "config_unknown_seqfav_zstress_v2_e00.yaml",
                "config_unknown_seqfav_zstress_v2_e01.yaml",
            ])
            for path in candidate_paths + fixed_paths:
                cfg = OmegaConf.load(path)
                noise = dict(OmegaConf.to_container(cfg.data.noise_model, resolve=True))
                NoiseModel.from_config_dict(noise)
                self.assertEqual(set(noise), set(_single_p_mapping(0.001)))


class TestUnknownSeqfavZStressV2Selection(unittest.TestCase):

    def test_selector_requires_strict_delta_threshold_and_marks_top_five(self):
        rows = [
            {"candidate_key": "a", "ler_delta": -0.0040, "axis_signature": "cnot+idle"},
            {"candidate_key": "b", "ler_delta": -0.0030, "axis_signature": "cnot+z_bias"},
            {"candidate_key": "c", "ler_delta": -0.0025, "axis_signature": "meas+cnot"},
            {"candidate_key": "d", "ler_delta": -0.0020, "axis_signature": "idle+z_bias"},
            {"candidate_key": "e", "ler_delta": -0.0016, "axis_signature": "meas+z_bias"},
            {"candidate_key": "f", "ler_delta": -0.0015, "axis_signature": "meas+cnot+idle"},
            {"candidate_key": "g", "ler_delta": -0.0100, "axis_signature": "cnot+idle"},
        ]

        selected = select_top_candidates(rows, count=5, threshold=-0.0015)

        self.assertEqual([row["candidate_key"] for row in selected], ["g", "a", "b", "c", "d"])
        self.assertNotIn("f", {row["candidate_key"] for row in selected})
        self.assertTrue(all(row["selected"] for row in selected))
        self.assertEqual([row["rank"] for row in selected], [1, 2, 3, 4, 5])

    def test_confirmed_fixed_selector_uses_only_independent_winners(self):
        rows = [
            {"candidate_key": "a", "ler_delta": -0.0004},
            {"candidate_key": "b", "ler_delta": 0.0001},
            {"candidate_key": "c", "ler_delta": -0.0012},
            {"candidate_key": "d", "ler_delta": -0.0001},
        ]

        selected = select_confirmed_fixed_rows(rows, count=2, threshold=0.0)

        self.assertEqual([row["candidate_key"] for row in selected], ["c", "a"])
        self.assertTrue(all(row["ler_delta"] < 0 for row in selected))

    def test_named_confirmed_fixed_selector_rejects_non_winners(self):
        rows = [
            {"candidate_key": "candidate_001", "ler_delta": -0.0004},
            {"candidate_key": "candidate_002", "ler_delta": 0.0001},
        ]

        selected = select_named_confirmed_fixed_rows(
            rows,
            candidate_keys=["candidate_001"],
            threshold=0.0,
        )

        self.assertEqual([row["candidate_key"] for row in selected], ["candidate_001"])
        with self.assertRaises(ValueError):
            select_named_confirmed_fixed_rows(
                rows,
                candidate_keys=["candidate_002"],
                threshold=0.0,
            )

    def test_final_gate_does_not_require_every_environment_to_win(self):
        rows = [
            {"domestic100_ler": 0.20, "seq_noewc_ler": 0.19, "ler_delta": -0.01},
            {"domestic100_ler": 0.18, "seq_noewc_ler": 0.17, "ler_delta": -0.01},
            {"domestic100_ler": 0.16, "seq_noewc_ler": 0.15, "ler_delta": -0.01},
            {"domestic100_ler": 0.14, "seq_noewc_ler": 0.13, "ler_delta": -0.01},
            {"domestic100_ler": 0.12, "seq_noewc_ler": 0.121, "ler_delta": 0.001},
        ]

        strict_status = zstress_success_status(rows)
        final_status = zstress_success_status(rows, require_all_individual_win=False)

        self.assertFalse(strict_status["passed"])
        self.assertFalse(final_status["all_individual_win"])
        self.assertEqual(final_status["individual_win_count"], 4)
        self.assertTrue(final_status["avg_beats_domestic100"])
        self.assertTrue(final_status["worst_beats_domestic100"])
        self.assertTrue(final_status["passed"])

    def test_scan_csv_and_markdown_include_gate_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [
                {
                    "candidate_key": "candidate_000",
                    "config_name": "config_unknown_seqfav_zstress_v2_candidate_000",
                    "axis_signature": "cnot_all+idle_all",
                    "axis_multipliers": '{"cnot_all": 1.35, "idle_all": 1.2, "meas_all": 1.0, "z_bias": 1.0}',
                    "domestic100_ler": 0.10,
                    "seq_noewc_ler": 0.08,
                    "ler_delta": -0.02,
                    "rank": 1,
                    "selected": True,
                },
                {
                    "candidate_key": "candidate_001",
                    "config_name": "config_unknown_seqfav_zstress_v2_candidate_001",
                    "axis_signature": "cnot_all+z_bias",
                    "axis_multipliers": '{"cnot_all": 1.35, "idle_all": 1.0, "meas_all": 1.0, "z_bias": 1.35}',
                    "domestic100_ler": 0.11,
                    "seq_noewc_ler": 0.09,
                    "ler_delta": -0.02,
                    "rank": 2,
                    "selected": True,
                },
            ]
            csv_path = root / "scan.csv"
            md_path = root / "report.md"

            write_scan_csv(csv_path, rows)
            status = zstress_success_status(rows)
            write_zstress_markdown(
                md_path,
                title="Z-basis seq-favoring stress test",
                rows=rows,
                selected_rows=rows,
                status=status,
                basis="Z",
                seed=22345,
                num_samples=65536,
                output_dir=root,
                scan_csv=csv_path,
            )

            loaded = list(csv.DictReader(csv_path.open()))
            self.assertIn("delta", loaded[0])
            self.assertIn("rank", loaded[0])
            self.assertIn("selected", loaded[0])
            text = md_path.read_text(encoding="utf-8")
            self.assertIn("Z-basis seq-favoring stress test", text)
            self.assertIn("domestic-only epoch 100", text)
            self.assertTrue(status["passed"])


if __name__ == "__main__":
    unittest.main()
