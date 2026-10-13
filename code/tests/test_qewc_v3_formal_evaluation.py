# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import unittest

from scripts.experiments.qewc_v3.evaluate_formal_run import checkpoint_items, task_items


class TestQewcV3FormalEvaluation(unittest.TestCase):

    def test_continual_and_mix_checkpoint_rows_are_preserved(self):
        continual = {
            "method": "ewc",
            "endpoints": {
                "C0": {"path": "/tmp/c0.pt"},
                "C1": {"path": "/tmp/c1.pt"},
            },
        }
        mix = {
            "method": "mix_total",
            "checkpoints": {
                "M0": {"path": "/tmp/m0.pt"},
                "M4": {"path": "/tmp/m4.pt"},
            },
        }
        self.assertEqual(
            checkpoint_items("S2", continual),
            [("C0", Path("/tmp/c0.pt")), ("C1", Path("/tmp/c1.pt"))],
        )
        self.assertEqual(checkpoint_items("S4", mix)[-1][0], "M4")
        mix["method"] = "mix_exposure"
        self.assertEqual(checkpoint_items("S7", mix)[-1][0], "M4")

    def test_oracle_endpoint_is_named_by_oracle_id(self):
        manifest = {
            "method": "seq",
            "endpoints": {
                "C0": {"path": "/tmp/c0.pt"},
                "C1": {"path": "/tmp/oracle.pt"},
            },
        }
        self.assertEqual(checkpoint_items("O3", manifest)[-1][0], "O3")

    def test_tracks_expose_exactly_t0_through_t4(self):
        for track in ("realistic", "stress"):
            self.assertEqual([name for name, _ in task_items(track)], ["T0", "T1", "T2", "T3", "T4"])


if __name__ == "__main__":
    unittest.main()
