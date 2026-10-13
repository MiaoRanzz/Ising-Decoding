# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import unittest

from scripts.experiments.qewc_v3.evaluate_formal_comparison import final_checkpoint


class TestQewcV3FormalComparison(unittest.TestCase):

    def test_selects_final_continual_endpoint(self):
        training = {
            "method": "ewc",
            "endpoints": {
                "C0": {"path": "/tmp/c0.pt"},
                "C4": {"path": "/tmp/c4.pt"},
                "C2": {"path": "/tmp/c2.pt"},
            },
        }
        self.assertEqual(final_checkpoint(training), Path("/tmp/c4.pt"))

    def test_selects_final_mix_checkpoint(self):
        training = {
            "method": "mix_total",
            "checkpoints": {"M0": {"path": "/tmp/m0.pt"}, "M4": {"path": "/tmp/m4.pt"}},
        }
        self.assertEqual(final_checkpoint(training), Path("/tmp/m4.pt"))
        training["method"] = "mix_exposure"
        self.assertEqual(final_checkpoint(training), Path("/tmp/m4.pt"))


if __name__ == "__main__":
    unittest.main()
