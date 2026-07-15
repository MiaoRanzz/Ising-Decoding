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
from scripts.generate_unknown_parammix_u1p2_4p0_configs import (
    AXIS_ORDER,
    DEFAULT_ENV_SPECS,
    MAX_MULTIPLIER,
    MIN_MULTIPLIER,
    NUM_REPLICATES,
    generate_parammix_noise_models,
    write_parammix_configs,
)
from scripts.run_unknown_parammix_u1p2_4p0_compare import (
    aggregate_payloads,
    write_outputs,
)


class TestUnknownParamMixGeneration(unittest.TestCase):

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

    def test_generates_55_deterministic_unique_parameter_level_realizations(self):
        base_noise = _single_p_mapping(0.001)

        first = generate_parammix_noise_models(base_noise, seed=20260714)
        second = generate_parammix_noise_models(base_noise, seed=20260714)

        self.assertEqual(len(first), 11 * NUM_REPLICATES)
        self.assertEqual(
            [item["noise_model_sha256"] for item in first],
            [item["noise_model_sha256"] for item in second],
        )
        self.assertEqual(len({item["noise_model_sha256"] for item in first}), 55)
        self.assertEqual({item["replicate_index"] for item in first}, set(range(5)))

        env_sizes = {
            spec["env_index"]: len(spec["active_axes"])
            for spec in DEFAULT_ENV_SPECS
        }
        self.assertEqual(list(env_sizes.values()).count(2), 6)
        self.assertEqual(list(env_sizes.values()).count(3), 4)
        self.assertEqual(list(env_sizes.values()).count(4), 1)

    def test_samples_each_active_physical_parameter_once_and_preserves_inactive_parameters(self):
        base_noise = _single_p_mapping(0.001)
        item = generate_parammix_noise_models(
            base_noise,
            seed=20260714,
            env_specs=[
                {
                    "env_index": 0,
                    "env_key": "e00",
                    "active_axes": ("meas_all", "z_bias"),
                }
            ],
            num_replicates=1,
        )[0]

        active_parameters = item["active_parameters"]
        multipliers = item["parameter_multipliers"]
        self.assertEqual(len(active_parameters), len(set(active_parameters)))
        self.assertIn("p_meas_X", active_parameters)
        self.assertEqual(multipliers["p_prep_Z"], 1.0)
        self.assertEqual(item["noise_model"]["p_prep_Z"], base_noise["p_prep_Z"])
        self.assertTrue(
            all(
                MIN_MULTIPLIER <= multipliers[key] < MAX_MULTIPLIER
                for key in active_parameters
            )
        )
        self.assertGreater(len({multipliers[key] for key in active_parameters}), 1)
        NoiseModel.from_config_dict(item["noise_model"])

    def test_write_configs_outputs_valid_25_parameter_configs_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths, manifest = write_parammix_configs(
                base_config=self._base_config_path(root),
                output_dir=root / "conf" / "experiments" / "unknown_parammix_u1p2_4p0",
                manifest=root / "manifest.json",
            )

            self.assertEqual(len(paths), 55)
            self.assertEqual(paths[0].name, "config_unknown_parammix_u1p2_4p0_e00_r00.yaml")
            self.assertEqual(paths[-1].name, "config_unknown_parammix_u1p2_4p0_e10_r04.yaml")
            self.assertEqual(
                manifest["environments"][0]["config_name"],
                "experiments/unknown_parammix_u1p2_4p0/config_unknown_parammix_u1p2_4p0_e00_r00",
            )
            self.assertEqual(manifest["num_envs"], 11)
            self.assertEqual(manifest["num_realizations"], 55)
            self.assertEqual(manifest["num_replicates"], 5)
            self.assertEqual(manifest["axis_order"], list(AXIS_ORDER))
            for path in paths:
                cfg = OmegaConf.load(path)
                noise = dict(OmegaConf.to_container(cfg.data.noise_model, resolve=True))
                self.assertEqual(len(noise), 25)
                NoiseModel.from_config_dict(noise)


class TestUnknownParamMixAggregation(unittest.TestCase):

    def _manifest_and_payloads(self):
        environments = []
        payloads = []
        for spec in DEFAULT_ENV_SPECS:
            env_index = int(spec["env_index"])
            for replicate_index in range(NUM_REPLICATES):
                config_name = (
                    f"experiments/unknown_parammix_u1p2_4p0/config_unknown_parammix_u1p2_4p0_"
                    f"e{env_index:02d}_r{replicate_index:02d}"
                )
                environments.append(
                    {
                        "env_index": env_index,
                        "env_key": f"e{env_index:02d}",
                        "replicate_index": replicate_index,
                        "replicate_key": f"r{replicate_index:02d}",
                        "config_name": config_name,
                        "axis_signature": spec["axis_signature"],
                        "active_axes": list(spec["active_axes"]),
                        "combination_size": spec["combination_size"],
                        "contains_z_bias": spec["contains_z_bias"],
                        "contains_cnot_z_bias": spec["contains_cnot_z_bias"],
                    }
                )
                domestic_x = 0.10 + env_index * 0.001 + replicate_index * 0.0001
                domestic_z = domestic_x + 0.02
                seq_x = domestic_x - 0.002
                seq_z = domestic_z - 0.004
                payloads.append(
                    {
                        "config_name": config_name,
                        "run_phase": "full",
                        "path": f"/tmp/{config_name}.json",
                        "rows": [
                            {"basis": "X", "method": "stfusion_domestic_e100", "ler": domestic_x},
                            {"basis": "X", "method": "stfusion_seq_noewc_e100", "ler": seq_x},
                            {"basis": "Z", "method": "stfusion_domestic_e100", "ler": domestic_z},
                            {"basis": "Z", "method": "stfusion_seq_noewc_e100", "ler": seq_z},
                        ],
                    }
                )
        return {"environments": environments}, payloads

    def test_aggregation_outputs_55_by_3_details_and_11_by_3_env_summaries(self):
        manifest, payloads = self._manifest_and_payloads()
        for payload in payloads:
            payload["config_name"] = Path(str(payload["config_name"])).name

        detail_rows, summary_rows, group_rows = aggregate_payloads(payloads, manifest)

        self.assertEqual(len(detail_rows), 55 * 3)
        self.assertEqual(len(summary_rows), 11 * 3)
        self.assertTrue(group_rows)
        e00_both = next(
            row for row in summary_rows
            if row["env_key"] == "e00" and row["basis"] == "both"
        )
        self.assertEqual(e00_both["num_replicates"], 5)
        self.assertAlmostEqual(e00_both["delta_mean"], -0.003)
        self.assertAlmostEqual(e00_both["delta_min"], -0.003)
        self.assertAlmostEqual(e00_both["delta_max"], -0.003)
        self.assertEqual(e00_both["winner"], "seq_noewc_e100")

    def test_write_outputs_writes_detail_summary_and_parameter_random_label(self):
        manifest, payloads = self._manifest_and_payloads()
        detail_rows, summary_rows, group_rows = aggregate_payloads(payloads, manifest)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            details_path = root / "details.csv"
            summary_path = root / "summary.csv"
            report_path = root / "comparison.md"

            write_outputs(
                details_csv=details_path,
                summary_csv=summary_path,
                summary_md=report_path,
                detail_rows=detail_rows,
                summary_rows=summary_rows,
                group_rows=group_rows,
                output_dir=root / "full",
                basis="both",
                num_samples=262144,
            )

            self.assertEqual(len(list(csv.DictReader(details_path.open()))), 55 * 3)
            self.assertEqual(len(list(csv.DictReader(summary_path.open()))), 11 * 3)
            report = report_path.read_text(encoding="utf-8")
            self.assertIn("# 参数级随机混合噪声 OOD 对比报告", report)
            self.assertIn("## 结论摘要", report)
            self.assertIn("## 结果解读", report)
            self.assertIn("强噪声", report)
            self.assertIn("Uniform(1.2, 4.0)", report)
            self.assertIn("至少 8/11", report)


if __name__ == "__main__":
    unittest.main()
