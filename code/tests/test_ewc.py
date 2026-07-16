# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import inspect
import sys
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.ewc import (
    EWCState,
    add_ewc_penalty_to_loss,
    capture_parameter_snapshot,
    diagonal_ewc_penalty,
    estimate_diagonal_fisher,
    load_ewc_state,
    load_ewc_states,
    save_ewc_state,
)


class _FakeGenerator:

    def __init__(self):
        self.calls = 0

    def generate_batch(self, step, batch_size):
        self.calls += 1
        x = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, -0.5]], dtype=torch.float32
        )[:batch_size]
        y = torch.ones(batch_size, 1, dtype=torch.float32)
        return x, y


class _CompiledLikeWrapper(nn.Module):

    def __init__(self):
        super().__init__()
        self._orig_mod = nn.Linear(2, 1, bias=False)

    def forward(self, x):
        return self._orig_mod(x)


class TestEWC(unittest.TestCase):

    def test_train_epoch_exposes_ewc_options(self):
        from training.train import train_epoch

        parameters = inspect.signature(train_epoch).parameters
        self.assertIn("ewc_states", parameters)
        self.assertIn("ewc_lambda", parameters)

    def test_penalty_is_zero_for_snapshot_and_positive_after_parameter_change(self):
        model = nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            model.weight.copy_(torch.tensor([[1.0, 2.0]]))

        mean = capture_parameter_snapshot(model)
        fisher = {name: torch.ones_like(value) for name, value in mean.items()}
        state = EWCState(task_name="T0_base", mean=mean, fisher=fisher)

        self.assertEqual(float(diagonal_ewc_penalty(model, [state])), 0.0)

        with torch.no_grad():
            model.weight.add_(1.0)

        # 0.5 * sum([1^2, 1^2]) = 1.0
        self.assertAlmostEqual(float(diagonal_ewc_penalty(model, [state])), 1.0, places=6)

    def test_penalty_matches_parameters_with_compile_prefixes(self):
        model = _CompiledLikeWrapper()
        with torch.no_grad():
            model._orig_mod.weight.copy_(torch.tensor([[1.0, 2.0]]))

        state = EWCState(
            task_name="T0_base",
            mean={"weight": torch.tensor([[1.0, 2.0]])},
            fisher={"weight": torch.ones(1, 2)},
        )
        self.assertEqual(float(diagonal_ewc_penalty(model, [state])), 0.0)

        with torch.no_grad():
            model._orig_mod.weight.add_(2.0)

        # 0.5 * sum([2^2, 2^2]) = 4.0
        self.assertAlmostEqual(float(diagonal_ewc_penalty(model, [state])), 4.0, places=6)

    def test_estimate_diagonal_fisher_returns_nonnegative_tensors(self):
        model = nn.Linear(2, 1)
        generator = _FakeGenerator()

        state = estimate_diagonal_fisher(
            model,
            generator,
            task_name="T0_base",
            num_samples=4,
            batch_size=2,
            device=torch.device("cpu"),
        )

        self.assertEqual(state.task_name, "T0_base")
        self.assertEqual(generator.calls, 2)
        self.assertEqual(set(state.mean), set(state.fisher))
        self.assertTrue(all(torch.all(v >= 0) for v in state.fisher.values()))
        self.assertTrue(any(torch.any(v > 0) for v in state.fisher.values()))

    def test_save_and_load_ewc_state_directory(self):
        state0 = EWCState(
            task_name="T0_base",
            mean={"weight": torch.ones(1, 2)},
            fisher={"weight": torch.full((1, 2), 2.0)},
        )
        state1 = EWCState(
            task_name="T1_meas_1p5",
            mean={"weight": torch.full((1, 2), 3.0)},
            fisher={"weight": torch.full((1, 2), 4.0)},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            save_ewc_state(state1, Path(tmpdir) / "task_001_T1.pt")
            save_ewc_state(state0, Path(tmpdir) / "task_000_T0.pt")

            loaded_one = load_ewc_state(Path(tmpdir) / "task_000_T0.pt")
            self.assertEqual(loaded_one.task_name, "T0_base")

            loaded = load_ewc_states(tmpdir)
            self.assertEqual([s.task_name for s in loaded], ["T0_base", "T1_meas_1p5"])

    def test_add_ewc_penalty_scales_by_batch_size(self):
        base_loss = torch.tensor(10.0)
        penalty = torch.tensor(2.0)

        total_loss = add_ewc_penalty_to_loss(
            base_loss,
            penalty,
            ewc_lambda=3.0,
            batch_size=4,
        )

        self.assertEqual(float(total_loss), 34.0)


if __name__ == "__main__":
    unittest.main()
