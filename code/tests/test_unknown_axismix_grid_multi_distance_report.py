# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.build_unknown_axismix_grid_multi_distance_report import (
    merge_distance_details,
    summarize_multi_distance,
    write_multi_distance_report,
)


class TestUnknownAxisMixGridMultiDistanceReport(unittest.TestCase):
    def _row(
        self,
        *,
        basis: str,
        env_key: str = "e00",
        multiplier_key: str = "m1p2",
        domestic: float = 0.20,
        mixed: float = 0.18,
        seq: float = 0.17,
    ) -> dict:
        return {
            "run_phase": "full",
            "env_key": env_key,
            "env_index": "0",
            "multiplier_key": multiplier_key,
            "multiplier": "1.2",
            "config_name": "experiments/unknown_axismix_grid_u1p2_5p0/config_unknown_axismix_grid_u1p2_5p0_e00_m1p2",
            "axis_signature": "meas_all+cnot_all",
            "active_axes": "meas_all+cnot_all",
            "combination_size": "2",
            "contains_z_bias": "False",
            "contains_cnot_z_bias": "False",
            "basis": basis,
            "domestic100_ler": str(domestic),
            "mixed100_ler": str(mixed),
            "seq_noewc_ler": str(seq),
            "delta_seq_vs_domestic": str(seq - domestic),
            "delta_seq_vs_mixed": str(seq - mixed),
            "delta_mixed_vs_domestic": str(mixed - domestic),
            "seq_beats_domestic": str(seq < domestic),
            "seq_beats_mixed": str(seq < mixed),
            "mixed_beats_domestic": str(mixed < domestic),
        }

    def test_merge_distance_details_adds_distance_metadata(self):
        merged = merge_distance_details(
            [
                (5, [self._row(basis="both", domestic=0.30, mixed=0.20, seq=0.22)]),
                (7, [self._row(basis="both", domestic=0.20, mixed=0.19, seq=0.18)]),
                (9, [self._row(basis="both", domestic=0.21, mixed=0.22, seq=0.20)]),
            ]
        )

        self.assertEqual([row["distance"] for row in merged], [5, 7, 9])
        self.assertEqual([row["distance_label"] for row in merged], ["d5", "d7", "d9"])
        self.assertAlmostEqual(merged[1]["delta_seq_vs_domestic"], -0.02)

    def test_summary_groups_by_distance_basis_multiplier_and_env(self):
        rows = merge_distance_details(
            [
                (
                    7,
                    [
                        self._row(basis="both", env_key="e00", multiplier_key="m1p2", domestic=0.20, mixed=0.19, seq=0.18),
                        self._row(basis="X", env_key="e00", multiplier_key="m1p2", domestic=0.21, mixed=0.19, seq=0.20),
                        self._row(basis="Z", env_key="e00", multiplier_key="m1p2", domestic=0.19, mixed=0.19, seq=0.16),
                    ],
                )
            ]
        )

        summary = summarize_multi_distance(rows)
        overall = next(row for row in summary if row["distance"] == 7 and row["group_type"] == "all" and row["basis"] == "both")
        multiplier = next(row for row in summary if row["group_type"] == "multiplier")

        self.assertEqual(overall["config_count"], 1)
        self.assertAlmostEqual(overall["delta_seq_vs_domestic_mean"], -0.02)
        self.assertEqual(multiplier["group_value"], "m1p2")

    def test_write_multi_distance_report_contains_three_d_conclusions(self):
        summary = [
            {
                "distance": distance,
                "distance_label": f"d{distance}",
                "group_type": "all",
                "group_value": "all",
                "basis": "both",
                "config_count": 99,
                "env_count": 11,
                "domestic100_ler_mean": 0.30,
                "mixed100_ler_mean": 0.29 if distance != 5 else 0.20,
                "seq_noewc_ler_mean": 0.28 if distance != 5 else 0.22,
                "delta_seq_vs_domestic_mean": -0.02,
                "delta_seq_vs_mixed_mean": -0.01 if distance != 5 else 0.02,
                "delta_mixed_vs_domestic_mean": -0.01,
                "seq_win_vs_domestic_count": 90,
                "seq_win_vs_mixed_count": 80 if distance != 5 else 2,
                "mixed_win_vs_domestic_count": 99,
            }
            for distance in (5, 7, 9)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "multi.md"
            write_multi_distance_report(path, summary_rows=summary)

            text = path.read_text(encoding="utf-8")
            self.assertIn("三距离固定倍率网格 OOD 融合分析报告", text)
            self.assertIn("d=5", text)
            self.assertIn("d=7", text)
            self.assertIn("d=9", text)
            self.assertIn("相对 domestic-only", text)
            self.assertIn("相对 mixed-noise", text)


if __name__ == "__main__":
    unittest.main()
