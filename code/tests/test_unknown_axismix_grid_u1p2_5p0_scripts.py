# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import sys
import tempfile
import unittest
from pathlib import Path

from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent))

from qec.noise_model import NoiseModel, _single_p_mapping
from scripts.generate_unknown_axismix_grid_u1p2_5p0_configs import (
    GRID_MULTIPLIERS,
    generate_axismix_grid_noise_models,
    write_axismix_grid_configs,
)
from scripts.run_unknown_axismix_grid_u1p2_5p0_compare import (
    MODEL_SPECS,
    aggregate_payloads,
    build_tasks,
    generate_figures,
    parse_gpu_list,
    write_outputs,
    write_markdown,
)


class TestUnknownAxisMixGridGeneration(unittest.TestCase):

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

    def test_generates_11_by_9_grid_with_nonstacked_active_multipliers(self):
        base_noise = _single_p_mapping(0.001)
        generated = generate_axismix_grid_noise_models(base_noise)

        self.assertEqual(len(generated), 11 * len(GRID_MULTIPLIERS))
        self.assertEqual({item["env_index"] for item in generated}, set(range(11)))
        self.assertEqual({item["multiplier"] for item in generated}, set(GRID_MULTIPLIERS))

        e02_m5 = next(
            item for item in generated
            if item["env_index"] == 2 and item["multiplier_key"] == "m5p0"
        )
        self.assertEqual(e02_m5["axis_signature"], "meas_all+z_bias")
        multipliers = e02_m5["parameter_multipliers"]
        self.assertEqual(multipliers["p_meas_X"], 5.0)
        self.assertEqual(multipliers["p_prep_X"], 5.0)
        self.assertEqual(multipliers["p_prep_Z"], 1.0)
        self.assertEqual(e02_m5["noise_model"]["p_prep_Z"], base_noise["p_prep_Z"])
        NoiseModel.from_config_dict(e02_m5["noise_model"])

    def test_write_configs_outputs_99_valid_configs_and_manifest_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths, manifest = write_axismix_grid_configs(
                base_config=self._base_config_path(root),
                output_dir=root / "conf" / "experiments" / "unknown_axismix_grid_u1p2_5p0",
                manifest=root / "manifest.json",
            )

            self.assertEqual(len(paths), 99)
            self.assertEqual(paths[0].name, "config_unknown_axismix_grid_u1p2_5p0_e00_m1p2.yaml")
            self.assertEqual(paths[-1].name, "config_unknown_axismix_grid_u1p2_5p0_e10_m5p0.yaml")
            self.assertEqual(manifest["num_envs"], 11)
            self.assertEqual(manifest["num_configs"], 99)
            self.assertEqual(manifest["grid_multipliers"], list(GRID_MULTIPLIERS))
            self.assertEqual(
                manifest["environments"][0]["config_name"],
                "experiments/unknown_axismix_grid_u1p2_5p0/config_unknown_axismix_grid_u1p2_5p0_e00_m1p2",
            )
            for path in paths:
                cfg = OmegaConf.load(path)
                noise = dict(OmegaConf.to_container(cfg.data.noise_model, resolve=True))
                self.assertEqual(len(noise), 25)
                NoiseModel.from_config_dict(noise)


class TestUnknownAxisMixGridRunner(unittest.TestCase):

    def _manifest(self):
        envs = []
        for env_index, multiplier_key, multiplier in [(0, "m1p2", 1.2), (10, "m5p0", 5.0)]:
            envs.append(
                {
                    "env_index": env_index,
                    "env_key": f"e{env_index:02d}",
                    "multiplier_index": 0,
                    "multiplier_key": multiplier_key,
                    "multiplier": multiplier,
                    "config_name": (
                        "experiments/unknown_axismix_grid_u1p2_5p0/"
                        f"config_unknown_axismix_grid_u1p2_5p0_e{env_index:02d}_{multiplier_key}"
                    ),
                    "axis_signature": "meas_all+cnot_all" if env_index == 0 else "meas_all+cnot_all+idle_all+z_bias",
                    "active_axes": ["meas_all", "cnot_all"],
                    "combination_size": 2 if env_index == 0 else 4,
                    "contains_z_bias": env_index == 10,
                    "contains_cnot_z_bias": env_index == 10,
                }
            )
        return {"environments": envs}

    def _payload(self, config_name: str, domestic: float, mixed: float, seq: float) -> dict:
        rows = []
        for basis in ("X", "Z"):
            rows.extend(
                [
                    {"basis": basis, "method": "stfusion_domestic_e100", "ler": domestic, "latency_us_per_round": 3.0, "speedup_vs_pymatching": 2.0},
                    {"basis": basis, "method": "stfusion_mixed_e100", "ler": mixed, "latency_us_per_round": 3.1, "speedup_vs_pymatching": 1.9},
                    {"basis": basis, "method": "stfusion_seq_noewc_e100", "ler": seq, "latency_us_per_round": 3.2, "speedup_vs_pymatching": 1.8},
                ]
            )
        return {"config_name": config_name, "rows": rows, "summary": []}

    def test_build_tasks_assigns_gpus_and_includes_three_models(self):
        manifest = self._manifest()
        tasks = build_tasks(
            manifest,
            output_dir=Path("/tmp/out"),
            phase="quick",
            python="python",
            paired_script=Path("code/scripts/paired_inference_compare.py"),
            gpus=["0", "1"],
            distance=9,
            n_rounds=9,
            num_samples=4096,
            latency_num_samples=512,
            batch_size=2048,
            num_workers=0,
            seed=12345,
            basis="Z",
        )

        self.assertEqual(len(tasks), 2)
        self.assertEqual(tasks[0].gpu, "0")
        self.assertEqual(tasks[1].gpu, "1")
        self.assertIn("--device", tasks[0].cmd)
        self.assertIn("cuda:0", tasks[0].cmd)
        command = " ".join(tasks[0].cmd)
        self.assertIn("stfusion_domestic_e100", command)
        self.assertIn("stfusion_mixed_e100", command)
        self.assertIn("stfusion_seq_noewc_e100", command)
        self.assertEqual(len(MODEL_SPECS), 3)

    def test_parse_gpu_list_prefers_explicit_values(self):
        self.assertEqual(parse_gpu_list("0,2,7"), ["0", "2", "7"])

    def test_aggregate_payloads_computes_three_model_deltas(self):
        manifest = self._manifest()
        payloads = [
            self._payload(Path(env["config_name"]).name, domestic=0.20, mixed=0.18, seq=0.17)
            for env in manifest["environments"]
        ]

        details, summary, groups = aggregate_payloads(payloads, manifest)

        self.assertEqual(len(details), 2 * 3)
        first = next(row for row in details if row["basis"] == "both" and row["env_index"] == 0)
        self.assertAlmostEqual(first["delta_seq_vs_domestic"], -0.03)
        self.assertAlmostEqual(first["delta_seq_vs_mixed"], -0.01)
        self.assertAlmostEqual(first["delta_mixed_vs_domestic"], -0.02)
        self.assertTrue(summary)
        self.assertTrue(groups)

    def test_generate_figures_writes_expected_pngs(self):
        manifest = self._manifest()
        payloads = [
            self._payload(Path(env["config_name"]).name, domestic=0.20, mixed=0.18, seq=0.17)
            for env in manifest["environments"]
        ]
        details, summary, _ = aggregate_payloads(payloads, manifest)

        with tempfile.TemporaryDirectory() as tmp:
            paths = generate_figures(details, summary, Path(tmp) / "figures")

            self.assertEqual(
                {path.name for path in paths},
                {
                    "model_gap_vs_multiplier.png",
                    "delta_vs_multiplier.png",
                    "env_delta_heatmap_both.png",
                    "env_delta_seq_vs_mixed_heatmap_both.png",
                },
            )
            for path in paths:
                self.assertTrue(path.exists())
                self.assertGreater(path.stat().st_size, 0)

    def test_write_markdown_includes_detailed_chinese_sections_and_figures(self):
        manifest = self._manifest()
        payloads = [
            self._payload(Path(env["config_name"]).name, domestic=0.20, mixed=0.18, seq=0.17)
            for env in manifest["environments"]
        ]
        details, summary, _ = aggregate_payloads(payloads, manifest)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            figure_paths = [
                root / "figures" / "model_gap_vs_multiplier.png",
                root / "figures" / "delta_vs_multiplier.png",
                root / "figures" / "env_delta_heatmap_both.png",
                root / "figures" / "env_delta_seq_vs_mixed_heatmap_both.png",
            ]
            for figure_path in figure_paths:
                figure_path.parent.mkdir(parents=True, exist_ok=True)
                figure_path.write_bytes(b"png")
            report = root / "comparison.md"

            write_markdown(
                report,
                details_csv=root / "details.csv",
                summary_csv=root / "summary.csv",
                detail_rows=details,
                summary_rows=summary,
                output_dir=root / "per_config",
                basis="both",
                num_samples=262144,
                figure_paths=figure_paths,
            )

            text = report.read_text(encoding="utf-8")
            self.assertIn("## 实验目的与设计", text)
            self.assertIn("## 模型说明", text)
            self.assertIn("## 噪声构造解释", text)
            self.assertIn("## 图表总览", text)
            self.assertIn("## X/Z basis 差异", text)
            self.assertIn("## 与 mixed-noise 基线的关系", text)
            self.assertIn("## 速度与延迟分析", text)
            self.assertIn("负 delta 表示 seq no-EWC 更好", text)
            self.assertIn("不是随机未知噪声", text)
            self.assertIn("domestic-only e100", text)
            self.assertIn("重叠参数只乘一次", text)
            self.assertIn("model_gap_vs_multiplier.png", text)
            self.assertGreaterEqual(text.count("!["), 4)

    def test_write_markdown_uses_paths_relative_to_report_file(self):
        manifest = self._manifest()
        payloads = [
            self._payload(Path(env["config_name"]).name, domestic=0.20, mixed=0.18, seq=0.17)
            for env in manifest["environments"]
        ]
        details, summary, _ = aggregate_payloads(payloads, manifest)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = root / "outputs" / "analysis" / "comparison.md"
            figure_path = root / "outputs" / "analysis" / "figures" / "model_gap_vs_multiplier.png"
            figure_path.parent.mkdir(parents=True, exist_ok=True)
            figure_path.write_bytes(b"png")

            write_markdown(
                report,
                details_csv=root / "outputs" / "analysis" / "details.csv",
                summary_csv=root / "outputs" / "analysis" / "summary.csv",
                detail_rows=details,
                summary_rows=summary,
                output_dir=root / "outputs" / "paired",
                basis="both",
                num_samples=262144,
                figure_paths=[figure_path],
            )

            text = report.read_text(encoding="utf-8")
            self.assertIn("](figures/model_gap_vs_multiplier.png)", text)
            self.assertNotIn("](outputs/analysis/figures/model_gap_vs_multiplier.png)", text)
            self.assertTrue((report.parent / "figures" / "model_gap_vs_multiplier.png").exists())


    def test_write_outputs_uses_custom_figure_dir_and_records_distance(self):
        manifest = self._manifest()
        payloads = [
            self._payload(Path(env["config_name"]).name, domestic=0.20, mixed=0.18, seq=0.17)
            for env in manifest["environments"]
        ]
        details, summary, _ = aggregate_payloads(payloads, manifest)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = root / "analysis" / "d7_comparison.md"
            figure_dir = root / "analysis" / "d7_figures"

            write_outputs(
                details_csv=root / "analysis" / "d7_details.csv",
                summary_csv=root / "analysis" / "d7_summary.csv",
                summary_md=report,
                detail_rows=details,
                summary_rows=summary,
                output_dir=root / "paired" / "d7_full",
                basis="both",
                num_samples=262144,
                distance=7,
                n_rounds=7,
                figure_dir=figure_dir,
            )

            text = report.read_text(encoding="utf-8")
            self.assertIn("distance=`7`", text)
            self.assertIn("n_rounds=`7`", text)
            self.assertIn("](d7_figures/model_gap_vs_multiplier.png)", text)
            self.assertTrue((figure_dir / "model_gap_vs_multiplier.png").exists())


if __name__ == "__main__":
    unittest.main()
