# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

from training.ewc import estimate_diagonal_fisher


class _FixedGenerator:

    def __init__(self, inputs: torch.Tensor, targets: torch.Tensor):
        self.inputs = inputs
        self.targets = targets
        self.offset = 0

    def generate_batch(self, *, step: int, batch_size: int):
        del step
        start = self.offset
        self.offset += batch_size
        return self.inputs[start:self.offset], self.targets[start:self.offset]


class TestQewcV3EmpiricalFisher(unittest.TestCase):

    def test_matches_mean_of_squared_per_sample_gradients(self):
        model = nn.Linear(2, 1)
        with torch.no_grad():
            model.weight.copy_(torch.tensor([[0.25, -0.5]]))
            model.bias.copy_(torch.tensor([0.1]))
        inputs = torch.tensor([[1.0, 0.0], [0.0, 2.0], [1.0, -1.0]])
        targets = torch.tensor([[1.0], [0.0], [1.0]])

        expected = {
            name: torch.zeros_like(param)
            for name, param in model.named_parameters()
        }
        for sample_x, sample_y in zip(inputs, targets):
            loss = F.binary_cross_entropy_with_logits(
                model(sample_x.unsqueeze(0)),
                sample_y.unsqueeze(0),
                reduction="sum",
                pos_weight=torch.tensor(2.0),
            )
            gradients = torch.autograd.grad(loss, tuple(model.parameters()))
            for (name, _), gradient in zip(model.named_parameters(), gradients):
                expected[name] += gradient.detach().pow(2)
        expected = {name: value / len(inputs) for name, value in expected.items()}

        state = estimate_diagonal_fisher(
            model,
            _FixedGenerator(inputs, targets),
            task_name="T0",
            num_samples=len(inputs),
            batch_size=len(inputs),
            per_sample_chunk_size=2,
            positive_weight=2.0,
            device="cpu",
        )

        for name, value in expected.items():
            torch.testing.assert_close(state.fisher[name], value)

    def test_rejects_invalid_sample_chunk_size(self):
        with self.assertRaisesRegex(ValueError, "per_sample_chunk_size"):
            estimate_diagonal_fisher(
                nn.Linear(1, 1),
                _FixedGenerator(torch.ones(1, 1), torch.ones(1, 1)),
                task_name="T0",
                num_samples=1,
                batch_size=1,
                per_sample_chunk_size=0,
                device="cpu",
            )


if __name__ == "__main__":
    unittest.main()
