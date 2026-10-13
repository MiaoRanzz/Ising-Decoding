# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.experiments.qewc_v3.run_continual import Task, rebuild_replay


class TestQewcV3ReplayLauncher(unittest.TestCase):

    def test_rebuild_records_defined_provenance_fields(self):
        args = SimpleNamespace(
            python=Path("/env/bin/python"),
            replay_capacity=594,
            replay_dir=Path("/tmp/qewc-replay"),
            replay_seed=20260913,
            gpu="0",
            dry_run=False,
            replay_rebuild_timings=[],
        )
        tasks = [Task("T0", "base"), Task("T1", "shift")]

        with patch(
            "scripts.experiments.qewc_v3.run_continual.run_command",
            return_value=1.25,
        ) as run:
            rebuild_replay(args, tasks)

        self.assertEqual(
            args.replay_rebuild_timings,
            [
                {
                    "tasks": ["T0", "T1"],
                    "output": "/tmp/qewc-replay",
                    "wall_seconds": 1.25,
                }
            ],
        )
        command = run.call_args.args[0]
        self.assertIn("T0:base", command)
        self.assertIn("T1:shift", command)


if __name__ == "__main__":
    unittest.main()
