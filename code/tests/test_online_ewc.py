# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

import torch

from training.ewc import EWCState
from training.online_ewc import update_online_ewc_state


class TestOnlineEWC(unittest.TestCase):

    def test_accumulates_fisher_and_anchors_at_current_parameters(self):
        previous = EWCState(
            task_name="T0",
            mean={"weight": torch.tensor([1.0, 2.0])},
            fisher={"weight": torch.tensor([3.0, 4.0])},
        )
        current = EWCState(
            task_name="T1",
            mean={"weight": torch.tensor([5.0, 6.0])},
            fisher={"weight": torch.tensor([7.0, 8.0])},
        )
        updated = update_online_ewc_state(previous, current, gamma=0.5)
        self.assertEqual(updated.task_name, "T1")
        torch.testing.assert_close(updated.mean["weight"], current.mean["weight"])
        torch.testing.assert_close(
            updated.fisher["weight"], torch.tensor([8.5, 10.0])
        )

    def test_rejects_invalid_gamma(self):
        state = EWCState(
            task_name="T0",
            mean={"weight": torch.ones(1)},
            fisher={"weight": torch.ones(1)},
        )
        with self.assertRaisesRegex(ValueError, "gamma"):
            update_online_ewc_state(state, state, gamma=1.1)


if __name__ == "__main__":
    unittest.main()
