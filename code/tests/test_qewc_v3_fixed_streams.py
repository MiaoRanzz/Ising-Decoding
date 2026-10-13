# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
from pathlib import Path
import tempfile
import unittest

from scripts.experiments.qewc_v3.evaluate_realistic_matrix import (
    Checkpoint,
    TASKS,
    build_job,
)
from scripts.experiments.qewc_v3.generate_fixed_evaluation_streams import sha256


class TestQewcV3FixedStreams(unittest.TestCase):

    def test_evaluator_routes_each_task_to_its_frozen_directory(self):
        args = argparse.Namespace(
            model_id=111,
            python=Path("/env/python"),
            output_dir=Path("/tmp/eval"),
            sample_root=Path("/tmp/fixed"),
            num_samples=128,
            latency_num_samples=8,
            batch_size=32,
            seed=7,
            checkpoint=[Checkpoint("C0", Path("/tmp/c0.pt"))],
        )
        command = build_job(args, "T3", TASKS["T3"]).command
        index = command.index("--stim-samples-dir")
        self.assertEqual(command[index + 1], "/tmp/fixed/T3")

    def test_artifact_sha256_is_streamed_and_exact(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "samples.dets"
            path.write_bytes(b"fixed-samples")
            self.assertEqual(
                sha256(path),
                "b85307c2985710185a60860ee95c109920d6c2aaa9434a2562baa2fcde737205",
            )


if __name__ == "__main__":
    unittest.main()
