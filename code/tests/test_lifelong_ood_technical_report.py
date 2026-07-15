# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.build_lifelong_ood_technical_report import (
    merge_grid_details_with_ewc,
    generate_ood_figures,
    summarize_ood_dimensions,
    summarize_t0_model_effectiveness,
    summarize_lifelong,
    write_technical_report,
)


def sample_grid_details():
    rows = []
    envs = [
        ("e00", "meas_all+cnot_all", False),
        ("e01", "meas_all+z_bias", True),
    ]
    for distance in (5, 7):
        for env_index, (env_key, axes, contains_z) in enumerate(envs):
            for multiplier_key, multiplier in (("m1p2", 1.2), ("m2p0", 2.0)):
                domestic = 0.20 + 0.01 * (distance - 5) + 0.005 * env_index
                seq = domestic - (0.01 if env_index == 0 else 0.02)
                mixed = domestic - 0.005
                rows.append(
                    {
                        "distance": distance,
                        "env_key": env_key,
                        "env_index": env_index,
                        "multiplier_key": multiplier_key,
                        "multiplier": multiplier,
                        "axis_signature": axes,
                        "active_axes": axes,
                        "combination_size": 2,
                        "contains_z_bias": contains_z,
                        "basis": "both",
                        "domestic100_ler": domestic,
                        "mixed100_ler": mixed,
                        "seq_noewc_ler": seq,
                    }
                )
    return rows


class TestLifelongOodTechnicalReport(unittest.TestCase):
    def test_summarize_t0_model_effectiveness_compares_domestic_models(self):
        rows = [
            {"task_key": "t0_base", "basis": "X", "method": "ising_domestic", "ler": "0.04", "latency_us_per_round": "2.4", "speedup_vs_pymatching": "2.0", "logical_errors": "40", "samples": "1000", "checkpoint": "ising.pt"},
            {"task_key": "t0_base", "basis": "Z", "method": "ising_domestic", "ler": "0.05", "latency_us_per_round": "2.6", "speedup_vs_pymatching": "1.9", "logical_errors": "50", "samples": "1000", "checkpoint": "ising.pt"},
            {"task_key": "t0_base", "basis": "X", "method": "stfusion_x_domestic", "ler": "0.03", "latency_us_per_round": "2.0", "speedup_vs_pymatching": "2.4", "logical_errors": "30", "samples": "1000", "checkpoint": "r9x.pt"},
            {"task_key": "t0_base", "basis": "Z", "method": "stfusion_x_domestic", "ler": "0.035", "latency_us_per_round": "2.2", "speedup_vs_pymatching": "2.3", "logical_errors": "35", "samples": "1000", "checkpoint": "r9x.pt"},
        ]

        summary = summarize_t0_model_effectiveness(rows)

        self.assertAlmostEqual(summary["ising"]["ler_mean"], 0.045)
        self.assertAlmostEqual(summary["r9x"]["ler_mean"], 0.0325)
        self.assertAlmostEqual(summary["ler_absolute_delta"], -0.0125)
        self.assertAlmostEqual(summary["ler_relative_reduction"], 0.0125 / 0.045)
        self.assertAlmostEqual(summary["latency_relative_reduction"], 0.4 / 2.5)
        self.assertEqual([row["basis"] for row in summary["basis_rows"]], ["X", "Z"])

    def test_merge_grid_details_with_ewc_adds_ewc_deltas(self):
        base = [
            {
                "distance": "9",
                "env_key": "e00",
                "multiplier_key": "m1p2",
                "basis": "both",
                "domestic100_ler": "0.20",
                "mixed100_ler": "0.18",
                "seq_noewc_ler": "0.17",
            }
        ]
        ewc = [
            {
                "distance": "9",
                "env_key": "e00",
                "multiplier_key": "m1p2",
                "basis": "both",
                "seq_ewc_ler": "0.19",
            }
        ]

        merged = merge_grid_details_with_ewc(base, ewc)

        self.assertEqual(len(merged), 1)
        self.assertAlmostEqual(merged[0]["delta_ewc_vs_domestic"], -0.01)
        self.assertAlmostEqual(merged[0]["delta_ewc_vs_noewc"], 0.02)
        self.assertFalse(merged[0]["ewc_beats_noewc"])

    def test_summarize_ood_dimensions_groups_multiplier_and_axes(self):
        dimensions = summarize_ood_dimensions(sample_grid_details())
        self.assertEqual(len(dimensions["multiplier_rows"]), 2)
        multiplier = next(row for row in dimensions["multiplier_rows"] if row["multiplier"] == 1.2)
        self.assertAlmostEqual(multiplier["delta_seq_vs_domestic_mean"], -0.015)
        self.assertEqual(len(dimensions["axis_rows"]), 2)
        meas = next(row for row in dimensions["axis_presence_rows"] if row["axis"] == "meas_all")
        self.assertEqual(meas["config_count"], 8)

    def test_generate_ood_figures_writes_three_png_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            figures = generate_ood_figures(sample_grid_details(), tmp)
            self.assertEqual(set(figures), {"multiplier", "axis_distance", "axis_multiplier"})
            for path in figures.values():
                self.assertTrue(path.exists())
                self.assertEqual(path.suffix, ".png")


    def test_write_technical_report_contains_required_sections(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "report.md"
            grid_overall = [
                {
                    "distance": 9,
                    "basis": "both",
                    "domestic100_ler_mean": 0.20,
                    "mixed100_ler_mean": 0.18,
                    "seq_noewc_ler_mean": 0.17,
                    "seq_ewc_ler_mean": 0.19,
                    "delta_noewc_vs_domestic_mean": -0.03,
                    "delta_ewc_vs_domestic_mean": -0.01,
                    "delta_ewc_vs_noewc_mean": 0.02,
                    "delta_ewc_vs_mixed_mean": 0.01,
                    "noewc_win_vs_domestic_count": 9,
                    "ewc_win_vs_domestic_count": 8,
                    "ewc_win_vs_mixed_count": 0,
                    "ewc_win_vs_noewc_count": 0,
                    "config_count": 9,
                }
            ]
            lifelong = {
                "domestic_reference_label": "domestic-only e20",
                "domestic_reference_avg": 0.064,
                "seq_noewc100_avg": 0.065,
                "seq_ewc100_avg": 0.067,
                "ewc_minus_noewc": 0.002,
                "final_task": {
                    "domestic_reference": {"t0_base": 0.064},
                    "seq_noewc100": {"t0_base": 0.065},
                    "seq_ewc100": {"t0_base": 0.067},
                },
                "trajectory": [
                    {
                        "model": "seq no-EWC",
                        "task": "t0_base",
                        "epoch20_ler": 0.066,
                        "epoch100_ler": 0.065,
                        "delta_e100_minus_e20": -0.001,
                    }
                ],
                "forgetting_summary": "EWC 未优于 no-EWC，未证明缓解灾难遗忘。",
            }

            model_effectiveness = {
                "ising": {"ler_mean": 0.0405, "latency_mean": 2.43, "speedup_mean": 1.98, "checkpoint": "ising.pt"},
                "r9x": {"ler_mean": 0.0336, "latency_mean": 2.16, "speedup_mean": 2.23, "checkpoint": "r9x.pt"},
                "basis_rows": [
                    {"basis": "X", "ising_ler": 0.0406, "r9x_ler": 0.0329, "ler_delta": -0.0077, "relative_reduction": 0.189},
                    {"basis": "Z", "ising_ler": 0.0404, "r9x_ler": 0.0342, "ler_delta": -0.0062, "relative_reduction": 0.154},
                ],
                "ler_absolute_delta": -0.0069,
                "ler_relative_reduction": 0.171,
                "latency_absolute_delta": -0.27,
                "latency_relative_reduction": 0.111,
                "parameter_relative_reduction": 0.287,
            }

            write_technical_report(
                output,
                model_effectiveness=model_effectiveness,
                grid_overall=grid_overall,
                lifelong=lifelong,
                grid_details=sample_grid_details(),
                figure_paths={
                    "multiplier": Path(tmp) / "figures" / "ood_delta_vs_multiplier_by_distance.png",
                    "axis_distance": Path(tmp) / "figures" / "ood_axis_combination_delta_heatmap.png",
                    "axis_multiplier": Path(tmp) / "figures" / "ood_axis_multiplier_heatmap.png",
                },
            )

            text = output.read_text(encoding="utf-8")
            self.assertIn("## 1. 问题背景", text)
            self.assertIn("## 2. 技术路线", text)
            self.assertIn("## 3. 实验设计", text)
            self.assertIn("## 4. 模型架构有效性：T0 同分布实验", text)
            self.assertIn("## 5. Lifelong 训练范式有效性", text)
            self.assertIn("R9-X + sequential no-EWC", text)
            self.assertIn("端到端组合方案", text)
            self.assertIn("OOD 泛化主证据", text)
            self.assertIn("#### 5.4.2 不同噪声倍率", text)
            self.assertIn("#### 5.4.3 不同噪声轴组合", text)
            self.assertIn("#### 5.4.4 轴组合与倍率的交互", text)
            self.assertIn("ood_delta_vs_multiplier_by_distance.png", text)
            self.assertIn("阴影", text)
            self.assertIn("不能由该实验单独分解架构贡献", text)
            self.assertIn("EWC 的 OOD 加权平均 LER", text)
            self.assertNotIn("整体最低的三距离加权平均 LER", text)
            self.assertNotIn("固定倍率复合 OOD 补充证据", text)
            self.assertIn("可持续进化", text)
            self.assertIn("seq + EWC", text)
            self.assertIn("灾难遗忘", text)
            self.assertIn("domestic-only e20", text)
            self.assertNotIn("domestic e100 ``", text)

    def test_summarize_lifelong_computes_stage_adaptation_and_forgetting(self):
        rows = []
        values = {
            20: {"t0_base": 0.09, "t1_meas_1p5": 0.12},
            40: {"t0_base": 0.08, "t1_meas_1p5": 0.10},
            100: {"t0_base": 0.085, "t1_meas_1p5": 0.095},
        }
        for epoch, tasks in values.items():
            for task, ler in tasks.items():
                for basis in ("X", "Z"):
                    rows.append({"task_key": task, "basis": basis, "group": "r9x_seq_noewc", "epoch": str(epoch), "ler": str(ler)})
                    rows.append({"task_key": task, "basis": basis, "group": "r9x_seq_ewc", "epoch": str(epoch), "ler": str(ler + 0.01)})
        for task in ("t0_base", "t1_meas_1p5"):
            for basis in ("X", "Z"):
                rows.append({"task_key": task, "basis": basis, "group": "r9x_domestic", "epoch": "20", "ler": "0.11"})

        summary = summarize_lifelong(rows)

        self.assertAlmostEqual(summary["forward_adaptation"][0]["delta_after_minus_before"], -0.02)
        self.assertAlmostEqual(summary["forgetting"][0]["forgetting"], 0.005)

    def test_summarize_lifelong_uses_domestic_e20_reference_when_e100_missing(self):
        rows = [
            {"task_key": "t0_base", "basis": "X", "group": "r9x_domestic", "epoch": "20", "ler": "0.10"},
            {"task_key": "t0_base", "basis": "Z", "group": "r9x_domestic", "epoch": "20", "ler": "0.12"},
            {"task_key": "t0_base", "basis": "X", "group": "r9x_seq_noewc", "epoch": "20", "ler": "0.09"},
            {"task_key": "t0_base", "basis": "Z", "group": "r9x_seq_noewc", "epoch": "20", "ler": "0.11"},
            {"task_key": "t0_base", "basis": "X", "group": "r9x_seq_noewc", "epoch": "100", "ler": "0.08"},
            {"task_key": "t0_base", "basis": "Z", "group": "r9x_seq_noewc", "epoch": "100", "ler": "0.10"},
            {"task_key": "t0_base", "basis": "X", "group": "r9x_seq_ewc", "epoch": "20", "ler": "0.095"},
            {"task_key": "t0_base", "basis": "Z", "group": "r9x_seq_ewc", "epoch": "20", "ler": "0.115"},
            {"task_key": "t0_base", "basis": "X", "group": "r9x_seq_ewc", "epoch": "100", "ler": "0.085"},
            {"task_key": "t0_base", "basis": "Z", "group": "r9x_seq_ewc", "epoch": "100", "ler": "0.105"},
        ]

        summary = summarize_lifelong(rows)

        self.assertEqual(summary["domestic_reference_label"], "domestic-only e20")
        self.assertAlmostEqual(summary["domestic_reference_avg"], 0.11)
        self.assertAlmostEqual(summary["final_task"]["domestic_reference"]["t0_base"], 0.11)
        self.assertNotIn("domestic100_avg", summary)


    def test_summarize_lifelong_uses_fair_budget_domestic_e100_when_available(self):
        rows = [
            {"task_key": "t0_base", "basis": "X", "group": "r9x_seq_noewc", "epoch": "20", "ler": "0.09"},
            {"task_key": "t0_base", "basis": "Z", "group": "r9x_seq_noewc", "epoch": "20", "ler": "0.11"},
            {"task_key": "t0_base", "basis": "X", "group": "r9x_seq_noewc", "epoch": "100", "ler": "0.08"},
            {"task_key": "t0_base", "basis": "Z", "group": "r9x_seq_noewc", "epoch": "100", "ler": "0.10"},
            {"task_key": "t0_base", "basis": "X", "group": "r9x_seq_ewc", "epoch": "100", "ler": "0.10"},
            {"task_key": "t0_base", "basis": "Z", "group": "r9x_seq_ewc", "epoch": "100", "ler": "0.12"},
        ]
        domestic = [{"epoch": "100", "ler_t0_base": "0.085", "ler_avg_5task": "0.085"}]

        summary = summarize_lifelong(rows, domestic_complete_rows=domestic)

        self.assertEqual(summary["domestic_reference_label"], "domestic-only e100")
        self.assertAlmostEqual(summary["domestic_reference_avg"], 0.085)
        self.assertAlmostEqual(summary["final_task"]["domestic_reference"]["t0_base"], 0.085)


if __name__ == "__main__":
    unittest.main()
