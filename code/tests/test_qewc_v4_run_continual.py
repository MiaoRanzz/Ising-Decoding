# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

from scripts.experiments.qewc_v4.run_continual import parse_args, stage_environment


class TestQewcV4RunContinual(unittest.TestCase):

    def test_seq_stage_uses_weights_only_and_disables_regularizers(self):
        args = parse_args(
            [
                "--method", "seq", "--run-name", "qewc_v4/test",
                "--base-task", "T0:experiments/qewc_v4/realistic/t0_base",
                "--task", "T1:experiments/qewc_v4/realistic/t1_meas_1p5",
                "--task", "T2:experiments/qewc_v4/realistic/t2_cnot_1p5",
                "--task", "T3:experiments/qewc_v4/realistic/t3_idle_1p5",
                "--task", "T4:experiments/qewc_v4/realistic/t4_z_bias_1p5",
                "--order", "T1,T2,T3,T4", "--stage0", "/tmp/c0.pt",
                "--epochs-per-task", "5", "--requested-samples-per-epoch", "4194304",
                "--batch-size", "2048", "--accumulate-steps", "2", "--seed", "7",
            ]
        )
        env, endpoint = stage_environment(args, args.tasks["T1"], 1, args.stage0)
        self.assertEqual(env["FRESH_START"], "1")
        self.assertEqual(env["PREDECODER_EWC_ENABLED"], "0")
        self.assertEqual(env["PREDECODER_REPLAY_ENABLED"], "0")
        self.assertEqual(env["PREDECODER_TRAIN_SAMPLES"], "4194304")
        self.assertEqual(env["PREDECODER_TRAIN_EPOCHS"], "5")
        self.assertEqual(env["PREDECODER_ACCUMULATE_STEPS"], "2")
        self.assertEqual(args.steps_per_task, 2560)
        self.assertIn("init_model_checkpoint_file=/tmp/c0.pt", env["EXTRA_PARAMS"])
        self.assertTrue(str(endpoint).endswith("stage_1_T1/models/HTnet.0.5.pt"))

    def test_ewc_requires_lambda(self):
        with self.assertRaises(SystemExit):
            parse_args(
                [
                    "--method", "ewc", "--run-name", "x",
                    "--base-task", "T0:a", "--task", "T1:b", "--task", "T2:c",
                    "--task", "T3:d", "--task", "T4:e", "--order", "T1,T2,T3,T4",
                    "--stage0", "/tmp/c0", "--requested-samples-per-epoch", "8", "--seed", "1",
                ]
            )


if __name__ == "__main__":
    unittest.main()
