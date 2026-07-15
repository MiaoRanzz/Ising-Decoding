# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent))

from qec.noise_model import NoiseModel, _single_p_mapping
from scripts.generate_unknown_axismix_1p5_configs import (
    AXIS_ORDER,
    DEFAULT_ENV_SPECS,
    generate_axismix_noise_models,
    write_axismix_configs,
)
from scripts.run_unknown_axismix_1p5_compare import (
    aggregate_payloads,
    write_outputs,
)


class TestUnknownAxisMix1p5Generation(unittest.TestCase):

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

    def test_default_specs_are_all_two_three_and_four_axis_combinations(self):
        self.assertEqual(len(DEFAULT_ENV_SPECS), 11)
        sizes = [len(spec["active_axes"]) for spec in DEFAULT_ENV_SPECS]
        self.assertEqual(sizes.count(2), 6)
        self.assertEqual(sizes.count(3), 4)
        self.assertEqual(sizes.count(4), 1)
        self.assertNotIn(1, sizes)
        self.assertEqual(
            DEFAULT_ENV_SPECS[0]["active_axes"],
            ("meas_all", "cnot_all"),
        )
        self.assertEqual(
            DEFAULT_ENV_SPECS[-1]["active_axes"],
            AXIS_ORDER,
        )

    def test_noise_generation_uses_uniform_1p5_and_max_for_overlaps(self):
        base_noise = _single_p_mapping(0.001)
        generated = generate_axismix_noise_models(
            base_noise,
            [
                {
                    "env_index": 0,
                    "env_key": "e00",
                    "active_axes": ("meas_all", "z_bias"),
                    "purpose": "overlap check",
                }
            ],
        )
        item = generated[0]
        noise = item["noise_model"]
        param_multipliers = item["parameter_multipliers"]

        self.assertAlmostEqual(param_multipliers["p_meas_X"], 1.5)
        self.assertAlmostEqual(noise["p_meas_X"], base_noise["p_meas_X"] * 1.5)
        self.assertNotAlmostEqual(noise["p_meas_X"], base_noise["p_meas_X"] * 2.25)
        self.assertAlmostEqual(noise["p_meas_Z"], base_noise["p_meas_Z"] * 1.5)
        self.assertAlmostEqual(noise["p_prep_X"], base_noise["p_prep_X"] * 1.5)
        self.assertAlmostEqual(noise["p_prep_Z"], base_noise["p_prep_Z"])
        self.assertTrue(all(value in (1.0, 1.5) for value in param_multipliers.values()))
        NoiseModel.from_config_dict(noise)

    def test_write_configs_outputs_valid_25_parameter_configs_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base_config = self._base_config_path(root)

            paths, manifest = write_axismix_configs(
                base_config=base_config,
                output_dir=root / "conf" / "experiments" / "unknown_axismix_1p5",
                manifest=root / "manifest.json",
            )

            self.assertEqual(len(paths), 11)
            self.assertEqual(manifest["design"], "training-axis mixed 1.5x OOD composite test")
            self.assertEqual(manifest["num_envs"], 11)
            self.assertEqual(
                manifest["environments"][0]["config_name"],
                "experiments/unknown_axismix_1p5/config_unknown_axismix_1p5_e00",
            )
            self.assertEqual(manifest["environments"][0]["combination_size"], 2)
            self.assertEqual(manifest["environments"][-1]["combination_size"], 4)
            for path in paths:
                cfg = OmegaConf.load(path)
                noise = dict(OmegaConf.to_container(cfg.data.noise_model, resolve=True))
                NoiseModel.from_config_dict(noise)
                self.assertEqual(set(noise), set(_single_p_mapping(0.001)))


class TestUnknownAxisMix1p5Aggregation(unittest.TestCase):

    def _payload(self, config_name: str, env_index: int, rows: list[dict]) -> dict:
        return {
            "config_name": config_name,
            "env_index": env_index,
            "env_key": f"e{env_index:02d}",
            "rows": rows,
            "summary": [],
        }

    def test_aggregate_payloads_reports_per_basis_delta_and_groups(self):
        manifest = {
            "environments": [
                {
                    "env_index": 0,
                    "env_key": "e00",
                    "config_name": "experiments/unknown_axismix_1p5/config_unknown_axismix_1p5_e00",
                    "axis_signature": "meas_all+cnot_all",
                    "active_axes": ["meas_all", "cnot_all"],
                    "combination_size": 2,
                    "contains_z_bias": False,
                    "contains_cnot_z_bias": False,
                },
                {
                    "env_index": 1,
                    "env_key": "e01",
                    "config_name": "experiments/unknown_axismix_1p5/config_unknown_axismix_1p5_e01",
                    "axis_signature": "cnot_all+z_bias",
                    "active_axes": ["cnot_all", "z_bias"],
                    "combination_size": 2,
                    "contains_z_bias": True,
                    "contains_cnot_z_bias": True,
                },
            ]
        }
        payloads = [
            self._payload(
                "config_unknown_axismix_1p5_e00",
                0,
                [
                    {"basis": "X", "method": "stfusion_domestic_e100", "ler": 0.10, "latency_us_per_round": 3.0, "speedup_vs_pymatching": 2.0},
                    {"basis": "X", "method": "stfusion_seq_noewc_e100", "ler": 0.09, "latency_us_per_round": 3.2, "speedup_vs_pymatching": 1.9},
                    {"basis": "Z", "method": "stfusion_domestic_e100", "ler": 0.20, "latency_us_per_round": 3.0, "speedup_vs_pymatching": 2.0},
                    {"basis": "Z", "method": "stfusion_seq_noewc_e100", "ler": 0.22, "latency_us_per_round": 3.2, "speedup_vs_pymatching": 1.9},
                ],
            ),
            self._payload(
                "config_unknown_axismix_1p5_e01",
                1,
                [
                    {"basis": "X", "method": "stfusion_domestic_e100", "ler": 0.30, "latency_us_per_round": 3.0, "speedup_vs_pymatching": 2.0},
                    {"basis": "X", "method": "stfusion_seq_noewc_e100", "ler": 0.27, "latency_us_per_round": 3.2, "speedup_vs_pymatching": 1.9},
                    {"basis": "Z", "method": "stfusion_domestic_e100", "ler": 0.40, "latency_us_per_round": 3.0, "speedup_vs_pymatching": 2.0},
                    {"basis": "Z", "method": "stfusion_seq_noewc_e100", "ler": 0.36, "latency_us_per_round": 3.2, "speedup_vs_pymatching": 1.9},
                ],
            ),
        ]

        env_rows, group_rows = aggregate_payloads(payloads, manifest)

        by_env_basis = {(row["env_key"], row["basis"]): row for row in env_rows}
        self.assertAlmostEqual(by_env_basis[("e00", "both")]["delta"], 0.005)
        self.assertAlmostEqual(by_env_basis[("e01", "Z")]["relative_delta_pct"], -10.0)
        self.assertIn("combination_size", env_rows[0])
        self.assertIn("contains_cnot_z_bias", env_rows[0])
        by_group = {(row["group_type"], row["group_value"], row["basis"]): row for row in group_rows}
        self.assertAlmostEqual(by_group[("combination_size", "2", "both")]["delta_avg"], -0.015)
        self.assertAlmostEqual(by_group[("contains_z_bias", "yes", "both")]["delta_avg"], -0.035)

    def test_write_outputs_includes_delta_fields_and_markdown_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env_rows = [
                {
                    "env_key": "e00",
                    "config_name": "experiments/unknown_axismix_1p5/config_unknown_axismix_1p5_e00",
                    "axis_signature": "meas_all+cnot_all",
                    "active_axes": "meas_all+cnot_all",
                    "combination_size": 2,
                    "contains_z_bias": False,
                    "contains_cnot_z_bias": False,
                    "basis": "both",
                    "domestic100_ler": 0.10,
                    "seq_noewc_ler": 0.09,
                    "delta": -0.01,
                    "relative_delta_pct": -10.0,
                    "winner": "seq_noewc_e100",
                }
            ]
            group_rows = [
                {
                    "group_type": "combination_size",
                    "group_value": "2",
                    "basis": "both",
                    "env_count": 1,
                    "domestic100_ler_avg": 0.10,
                    "seq_noewc_ler_avg": 0.09,
                    "delta_avg": -0.01,
                    "seq_win_count": 1,
                }
            ]
            csv_path = root / "summary.csv"
            md_path = root / "report.md"

            write_outputs(
                summary_csv=csv_path,
                summary_md=md_path,
                env_rows=env_rows,
                group_rows=group_rows,
                output_dirs=[root / "quick"],
                basis="both",
                num_samples=65536,
            )

            loaded = list(csv.DictReader(csv_path.open()))
            self.assertIn("delta", loaded[0])
            self.assertIn("combination_size", loaded[0])
            self.assertIn("contains_cnot_z_bias", loaded[0])
            report = md_path.read_text(encoding="utf-8")
            self.assertIn("training-axis mixed 1.5x OOD composite test", report)
            self.assertIn("Negative delta means seq no-EWC is better", report)


if __name__ == "__main__":
    unittest.main()
