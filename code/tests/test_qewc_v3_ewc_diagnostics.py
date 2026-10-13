# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
import unittest

import torch

from scripts.experiments.qewc_v3.analyze_ewc_stage import summarize_stage
from training.ewc import EWCState


class TestQewcV3EwcDiagnostics(unittest.TestCase):

    def test_reports_layer_and_total_fisher_weighted_drift(self):
        reference = EWCState(
            task_name="T0",
            mean={
                "block.weight": torch.zeros(2),
                "block.bias": torch.zeros(1),
            },
            fisher={
                "block.weight": torch.full((2,), 2.0),
                "block.bias": torch.full((1,), 2.0),
            },
        )
        current_fisher = EWCState(
            task_name="T1",
            mean=reference.mean,
            fisher={
                "block.weight": torch.full((2,), 4.0),
                "block.bias": torch.full((1,), 4.0),
            },
        )
        checkpoint = {
            "block.weight": torch.tensor([1.0, 2.0]),
            "block.bias": torch.tensor([3.0]),
        }

        summary = summarize_stage(checkpoint, reference, current_fisher)

        layer = summary["layers"]["block"]
        self.assertEqual(layer["parameter_count"], 3)
        self.assertAlmostEqual(layer["drift_l2"], math.sqrt(14.0))
        self.assertAlmostEqual(layer["fisher_weighted_drift"], 14.0)
        self.assertAlmostEqual(layer["reference_fisher_mean"], 2.0)
        self.assertAlmostEqual(layer["current_fisher_mean"], 4.0)
        self.assertAlmostEqual(summary["totals"]["fisher_weighted_drift"], 14.0)

    def test_rejects_checkpoint_missing_a_reference_parameter(self):
        reference = EWCState(
            task_name="T0",
            mean={"weight": torch.zeros(1)},
            fisher={"weight": torch.ones(1)},
        )
        with self.assertRaisesRegex(ValueError, "missing"):
            summarize_stage(
                {},
                reference,
                EWCState(task_name="T1", mean={}, fisher={}),
            )


if __name__ == "__main__":
    unittest.main()
