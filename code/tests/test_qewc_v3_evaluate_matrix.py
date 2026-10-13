# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import unittest
from pathlib import Path

from scripts.experiments.qewc_v3.evaluate_realistic_matrix import (
    Checkpoint,
    TASKS,
    build_job,
    parse_checkpoint,
)


class TestQewcV3EvaluateMatrix(unittest.TestCase):

    def test_checkpoint_paths_are_root_relative(self):
        checkpoint = parse_checkpoint("C0:outputs/model.pt")
        self.assertEqual(checkpoint.name, "C0")
        self.assertTrue(checkpoint.path.is_absolute())

    def test_job_pairs_later_checkpoints_against_first(self):
        args = argparse.Namespace(
            python=Path("/env/python"),
            output_dir=Path("/tmp/eval"),
            num_samples=128,
            latency_num_samples=8,
            batch_size=32,
            model_id=1,
            seed=7,
            checkpoint=[
                Checkpoint("C0", Path("/tmp/c0.pt")),
                Checkpoint("C1", Path("/tmp/c1.pt")),
            ],
        )
        job = build_job(args, "T0", TASKS["T0"])
        self.assertIn("C1:C0", job.command)
        self.assertTrue(any(value.startswith("C0:1:/tmp/c0.pt:") for value in job.command))
        self.assertTrue(any(value.startswith("C1:1:/tmp/c1.pt:") for value in job.command))
        self.assertIn("both", job.command)
        self.assertEqual(job.output_path, Path("/tmp/eval/T0.json"))


if __name__ == "__main__":
    unittest.main()
