# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import unittest

from omegaconf import OmegaConf

from scripts.experiments.qewc_v4.run_formal import (
    FORBIDDEN_LAMBDA,
    MANIFEST_PATH,
    RUNS,
    assert_frozen,
    build_command,
)


class TestQewcV4FormalLauncher(unittest.TestCase):
    def frozen_manifest(self):
        manifest = OmegaConf.load(MANIFEST_PATH)
        for field in (
            "design_confirmed", "stage0_endpoints", "legacy_epoch_semantics",
            "stress_tasks_and_orders", "budget_B", "ewc_parameters",
            "replay_parameters", "evaluation_protocols", "success_criteria",
        ):
            manifest.freeze[field] = True
        manifest.budget_pilot.selected_B = 5
        manifest.ewc_calibration.architectures = {
            "htnet_a101": {
                "selected_lambda": 10.0,
                "fisher_samples": 16384,
                "initial_fisher_path": "outputs/qewc_v4/calibration/htnet_a101/fisher/T0_n16384.pt",
                "positive_weight": 1.0,
            },
            "ising_fast": {
                "selected_lambda": 100.0,
                "fisher_samples": 65536,
                "initial_fisher_path": "outputs/qewc_v4/calibration/ising_fast/fisher/T0_n65536.pt",
                "positive_weight": 1.0,
            },
        }
        manifest.replay_calibration.architectures = {
            "htnet_a101": {
                "selected_ratio": 0.25,
                "capacity_samples": 1234,
                "target_bytes": 5200000,
            },
            "ising_fast": {
                "selected_ratio": 0.5,
                "capacity_samples": 5678,
                "target_bytes": 7300000,
            },
        }
        return manifest

    def test_current_manifest_blocks_formal_training(self):
        with self.assertRaisesRegex(RuntimeError, "not frozen"):
            assert_frozen(OmegaConf.load(MANIFEST_PATH))

    def test_all_sixteen_commands_build_after_freeze(self):
        manifest = self.frozen_manifest()
        assert_frozen(manifest)
        commands = {
            run_id: build_command(spec, manifest, gpu="1", resume=True)
            for run_id, spec in RUNS.items()
        }
        self.assertEqual(len(commands), 16)
        for command in commands.values():
            self.assertIn("--gpu", command)
            self.assertEqual(command[command.index("--gpu") + 1], "1")
            self.assertEqual(command[-1], "--resume")

    def test_t0_only_is_four_explicit_t0_stages(self):
        command = build_command(RUNS["H_T0"], self.frozen_manifest(), gpu="0", resume=False)
        self.assertEqual(command[command.index("--method") + 1], "t0_only")
        self.assertEqual(command[command.index("--order") + 1], "T0_stage1,T0_stage2,T0_stage3,T0_stage4")
        task_specs = [command[index + 1] for index, item in enumerate(command) if item == "--task"]
        self.assertEqual(len(task_specs), 4)
        self.assertTrue(all(spec.endswith(":experiments/qewc_v4/htnet/T0") for spec in task_specs))

    def test_ising_uses_existing_lowercase_t0_base_config(self):
        command = build_command(RUNS["I_T0"], self.frozen_manifest(), gpu="0", resume=False)
        self.assertIn("T0:experiments/qewc_v4/ising/t0_base", command)
        task_specs = [command[index + 1] for index, item in enumerate(command) if item == "--task"]
        self.assertEqual(len(task_specs), 4)
        self.assertTrue(
            all(spec.endswith(":experiments/qewc_v4/ising/t0_base") for spec in task_specs)
        )

    def test_mix_budgets_are_four_b_and_five_b(self):
        manifest = self.frozen_manifest()
        total = build_command(RUNS["H_MIX_TOTAL"], manifest, gpu="0", resume=False)
        exposure = build_command(RUNS["I_MIX_EXPOSURE"], manifest, gpu="0", resume=False)
        self.assertEqual(total[total.index("--total-legacy-epochs") + 1], "20")
        self.assertEqual(total[total.index("--checkpoint-count") + 1], "4")
        self.assertEqual(exposure[exposure.index("--total-legacy-epochs") + 1], "25")
        self.assertEqual(exposure[exposure.index("--checkpoint-count") + 1], "5")

    def test_architecture_specific_calibration_is_used(self):
        manifest = self.frozen_manifest()
        h_ewc = build_command(RUNS["H_EWC"], manifest, gpu="0", resume=False)
        i_ewc = build_command(RUNS["I_EWC"], manifest, gpu="0", resume=False)
        h_replay = build_command(RUNS["H_REPLAY"], manifest, gpu="0", resume=False)
        i_replay = build_command(RUNS["I_REPLAY"], manifest, gpu="0", resume=False)
        self.assertEqual(h_ewc[h_ewc.index("--ewc-lambda") + 1], "10.0")
        self.assertEqual(i_ewc[i_ewc.index("--ewc-lambda") + 1], "100.0")
        self.assertIn("--initial-fisher", h_ewc)
        self.assertIn("T0_n16384.pt", h_ewc[h_ewc.index("--initial-fisher") + 1])
        self.assertEqual(h_replay[h_replay.index("--replay-capacity") + 1], "1234")
        self.assertEqual(i_replay[i_replay.index("--replay-capacity") + 1], "5678")
        self.assertEqual(h_replay[h_replay.index("--replay-target-bytes") + 1], "5200000")
        self.assertEqual(i_replay[i_replay.index("--replay-target-bytes") + 1], "7300000")

    def test_forbidden_v3_lambda_is_rejected(self):
        manifest = self.frozen_manifest()
        manifest.ewc_calibration.architectures.htnet_a101.selected_lambda = FORBIDDEN_LAMBDA
        with self.assertRaisesRegex(RuntimeError, "forbidden"):
            assert_frozen(manifest)


if __name__ == "__main__":
    unittest.main()
