# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

from omegaconf import OmegaConf

from scripts.experiments.qewc_v3.run_formal import RUNS, assert_frozen, build_command


def frozen_manifest():
    return OmegaConf.create(
        {
            "stage0": {"checkpoint": "outputs/stage0.pt"},
            "seeds": {"continual_train": 17},
            "pilot": {
                "selected_optimizer_steps_per_task": 1000,
                "batch_size_per_gpu": 512,
                "validation_samples": 8192,
            },
            "ewc": {
                "lambda": 4.0,
                "gamma": 1.0,
                "positive_weight": 1.0,
                "fisher_samples": 65536,
                "fisher_batch_size": 2048,
                "per_sample_chunk_size": 16,
                "fisher_seed": 19,
            },
            "replay": {"ratio": 0.25, "capacity_samples": 594, "seed": 23},
            "freeze": {
                "realistic_tasks": True,
                "stress_tasks": True,
                "stress_orders": True,
                "continual_seed": True,
                "optimizer_steps_per_task": True,
                "ewc_lambda": True,
                "replay_ratio": True,
                "evaluation_streams": True,
                "success_criteria": True,
            },
        }
    )


class TestQewcV3FormalLauncher(unittest.TestCase):

    def test_blocks_unfrozen_manifest(self):
        manifest = frozen_manifest()
        manifest.freeze.ewc_lambda = False
        with self.assertRaisesRegex(RuntimeError, "ewc_lambda"):
            assert_frozen(manifest)

    def test_stress_ewc_uses_frozen_high_conflict_order(self):
        command = build_command(RUNS["S2"], frozen_manifest(), gpu="1", resume=False)
        self.assertIn("T1,T3,T2,T4", command)
        self.assertIn("4.0", command)
        self.assertIn("--fisher-per-sample-chunk-size", command)
        self.assertEqual(command[command.index("--gpu") + 1], "1")

    def test_replay_and_mix_are_bound_to_separate_entry_points(self):
        replay = build_command(RUNS["S3"], frozen_manifest(), gpu="0", resume=True)
        mix = build_command(RUNS["R3"], frozen_manifest(), gpu="1", resume=False)
        self.assertIn("--replay-capacity", replay)
        self.assertIn("--resume", replay)
        self.assertIn("run_mix_total.py", mix[2])
        self.assertIn("experiments/qewc_v3/realistic/mix_t0_t4", mix)

    def test_stretch_runs_use_frozen_protocols(self):
        stress_mix = build_command(RUNS["S7"], frozen_manifest(), gpu="1", resume=False)
        self.assertEqual(stress_mix[stress_mix.index("--total-steps") + 1], "5000")
        self.assertEqual(stress_mix[stress_mix.index("--method") + 1], "mix_exposure")
        realistic_replay = build_command(RUNS["R4"], frozen_manifest(), gpu="0", resume=False)
        self.assertIn("--replay-capacity", realistic_replay)
        realistic_oracle = build_command(RUNS["O5"], frozen_manifest(), gpu="0", resume=False)
        self.assertEqual(realistic_oracle[realistic_oracle.index("--order") + 1], "T1")


    def test_oracle_runs_one_frozen_stress_task(self):
        command = build_command(RUNS["O3"], frozen_manifest(), gpu="0", resume=False)
        self.assertEqual(command[command.index("--order") + 1], "T3")
        self.assertEqual(command[command.index("--expected-stages") + 1], "1")


if __name__ == "__main__":
    unittest.main()
