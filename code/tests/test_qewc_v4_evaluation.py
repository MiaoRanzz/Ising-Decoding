# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest
from pathlib import Path

from scripts.experiments.qewc_v4.aggregate_id_results import (
    build_run_result,
    lifelong_decision,
)
from scripts.experiments.qewc_v4.evaluate_formal_id import architecture_for_run
from scripts.experiments.qewc_v4.evaluation_common import (
    ARCHITECTURES,
    TASKS,
    checkpoint_items,
    config_path,
    final_checkpoint,
    run_ids,
)
from scripts.experiments.qewc_v4.run_oracles import htnet_source


class TestQewcV4Evaluation(unittest.TestCase):
    def test_architecture_model_ids_and_all_task_configs(self):
        self.assertEqual(ARCHITECTURES["htnet_a101"].model_id, 111)
        self.assertEqual(ARCHITECTURES["ising_fast"].model_id, 1)
        self.assertTrue(str(config_path("ising_fast", "T0")).endswith("ising/t0_base.yaml"))
        for architecture in ARCHITECTURES:
            for task in TASKS:
                self.assertTrue(config_path(architecture, task).is_file())

    def test_all_sixteen_run_ids_map_to_architecture(self):
        ids = (*run_ids("htnet_a101"), *run_ids("ising_fast"))
        self.assertEqual(len(ids), 16)
        self.assertTrue(all(architecture_for_run(item) == "htnet_a101" for item in ids[:8]))
        self.assertTrue(all(architecture_for_run(item) == "ising_fast" for item in ids[8:]))

    def test_final_checkpoint_supports_continual_and_mix(self):
        continual = {"method": "seq", "endpoints": {"C0": {"path": "/c0"}, "C4": {"path": "/c4"}}}
        mix = {"method": "mix_exposure", "checkpoints": {"M0": {"path": "/m0"}, "M5": {"path": "/m5"}}}
        self.assertEqual(checkpoint_items(continual)[-1], ("C4", Path("/c4")))
        self.assertEqual(final_checkpoint(mix), Path("/m5"))

    def test_htnet_oracle_reuses_exact_selected_budget_run(self):
        self.assertTrue(str(htnet_source("T3", 20)).endswith("htnet/T3_e20/run_manifest.json"))

    def test_id_aggregation_distinguishes_continual_mix_and_t0_control(self):
        order = ["T1", "T3", "T2", "T4"]
        seq_matrix = [
            [0.10, 0.50, 0.50, 0.50, 0.50],
            [0.12, 0.10, 0.50, 0.50, 0.50],
            [0.14, 0.15, 0.50, 0.10, 0.50],
            [0.16, 0.20, 0.10, 0.15, 0.50],
            [0.20, 0.25, 0.15, 0.20, 0.10],
        ]
        ewc_matrix = [
            [0.10, 0.50, 0.50, 0.50, 0.50],
            [0.10, 0.10, 0.50, 0.50, 0.50],
            [0.11, 0.11, 0.50, 0.10, 0.50],
            [0.12, 0.12, 0.10, 0.11, 0.50],
            [0.13, 0.13, 0.11, 0.12, 0.10],
        ]
        oracles = {task: 0.09 for task in TASKS[1:]}

        def continual(run_id, method, matrix):
            training = {
                "method": method,
                "order": order,
                "endpoints": {f"C{i}": {"path": f"/c{i}"} for i in range(5)},
            }
            evaluation = {
                "task_columns": list(TASKS),
                "ler_rows": {f"C{i}": row for i, row in enumerate(matrix)},
            }
            return build_run_result(run_id, training, evaluation, oracles)

        seq = continual("H_SEQ", "seq", seq_matrix)
        ewc = continual("H_EWC", "ewc", ewc_matrix)
        self.assertEqual(seq["kind"], "continual")
        self.assertTrue(lifelong_decision(seq, ewc)["passed"])

        mix_training = {
            "method": "mix_exposure",
            "checkpoints": {f"M{i}": {"path": f"/m{i}"} for i in range(6)},
        }
        mix_eval = {
            "task_columns": list(TASKS),
            "ler_rows": {f"M{i}": [0.1] * 5 for i in range(6)},
        }
        self.assertEqual(
            build_run_result("H_MIX_EXPOSURE", mix_training, mix_eval, oracles)["kind"],
            "joint_training_reference",
        )

        t0_training = {
            "method": "t0_only",
            "order": ["T0_stage1", "T0_stage2", "T0_stage3", "T0_stage4"],
            "endpoints": {f"C{i}": {"path": f"/c{i}"} for i in range(5)},
        }
        t0_eval = {
            "task_columns": list(TASKS),
            "ler_rows": {f"C{i}": [0.1] * 5 for i in range(5)},
        }
        self.assertEqual(
            build_run_result("H_T0", t0_training, t0_eval, oracles)["kind"],
            "t0_only_control",
        )


if __name__ == "__main__":
    unittest.main()
