# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from scripts.experiments.qewc_v3.finalize_formal_results import build_result, load_oracle_ler


class TestQewcV3FormalResults(unittest.TestCase):

    def setUp(self):
        self.rows = {f"C{i}": [0.01 * (i + j) for j in range(5)] for i in range(5)}
        self.evaluation = {
            "track": "stress",
            "task_columns": ["T0", "T1", "T2", "T3", "T4"],
            "ler_rows": self.rows,
        }

    def test_continual_run_builds_matrix_summary_and_forgetting_curve(self):
        result = build_result(
            "S1",
            {"method": "seq", "order": ["T1", "T3", "T2", "T4"]},
            self.evaluation,
            {"T1": 0.01, "T2": 0.02, "T3": 0.03, "T4": 0.04},
        )
        self.assertEqual(result["kind"], "continual_matrix")
        self.assertEqual(result["row_context"], ["T0", "T1", "T3", "T2", "T4"])
        self.assertEqual(len(result["ler"]), 5)
        self.assertIn("final_average_ler", result["summary"])
        self.assertIsNone(result["forgetting_curve"]["T4"][3])
        self.assertEqual(result["forgetting_curve"]["T4"][4], 0.0)

    def test_mix_is_not_mislabeled_as_continual_forgetting(self):
        evaluation = dict(self.evaluation)
        evaluation["ler_rows"] = {f"M{i}": row for i, row in enumerate(self.rows.values())}
        result = build_result("S4", {"method": "mix_total"}, evaluation)
        self.assertEqual(result["kind"], "joint_optimization_trajectory")
        self.assertNotIn("summary", result)
        self.assertNotIn("forgetting_curve", result)
        result = build_result("S7", {"method": "mix_exposure"}, evaluation)
        self.assertEqual(result["kind"], "joint_optimization_trajectory")

    def test_oracle_uses_formal_run_id_despite_seq_training_method(self):
        evaluation = dict(self.evaluation)
        evaluation["ler_rows"] = {"C0": self.rows["C0"], "O3": self.rows["C1"]}
        result = build_result("O3", {"method": "seq", "order": ["T3"]}, evaluation)
        self.assertEqual(result["kind"], "single_task_oracle")
        self.assertEqual(result["row_labels"], ["C0", "O3"])
        self.assertEqual(result["row_context"], ["T0", "single_T3"])
        self.assertNotIn("summary", result)


    def test_realistic_oracles_use_o5_through_o8(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for offset in range(4):
                run_id = f"O{5 + offset}"
                path = root / run_id
                path.mkdir()
                rows = [0.0] * 5
                rows[offset + 1] = 0.1 + offset
                (path / "evaluation_manifest.json").write_text(
                    json.dumps({"ler_rows": {run_id: rows}}), encoding="utf-8"
                )
            self.assertEqual(
                load_oracle_ler(root, "realistic"),
                {"T1": 0.1, "T2": 1.1, "T3": 2.1, "T4": 3.1},
            )


    def test_rejects_missing_checkpoint_row(self):
        evaluation = dict(self.evaluation)
        evaluation["ler_rows"] = {"C0": self.rows["C0"]}
        with self.assertRaisesRegex(ValueError, "missing rows"):
            build_result(
                "R1",
                {"method": "seq", "order": ["T1", "T2", "T3", "T4"]},
                evaluation,
            )


if __name__ == "__main__":
    unittest.main()
