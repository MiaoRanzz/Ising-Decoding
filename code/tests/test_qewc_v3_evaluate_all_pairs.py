# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
from pathlib import Path
import unittest

from scripts.experiments.qewc_v3.evaluate_realistic_matrix import (
    Checkpoint,
    TASKS,
    build_job,
)


class TestQewcV3EvaluateAllPairs(unittest.TestCase):

    def test_build_job_requests_every_checkpoint_pair(self):
        args = argparse.Namespace(
            model_id=111,
            python=Path("/env/python"),
            output_dir=Path("/tmp/eval"),
            num_samples=128,
            latency_num_samples=8,
            batch_size=32,
            seed=7,
            checkpoint=[
                Checkpoint("C0", Path("/tmp/c0.pt")),
                Checkpoint("C1", Path("/tmp/c1.pt")),
                Checkpoint("C2", Path("/tmp/c2.pt")),
            ],
        )

        command = build_job(args, "T0", TASKS["T0"]).command

        self.assertIn("C1:C0", command)
        self.assertIn("C2:C0", command)
        self.assertIn("C2:C1", command)
        self.assertEqual(command.count("--paired-comparison"), 3)


if __name__ == "__main__":
    unittest.main()
