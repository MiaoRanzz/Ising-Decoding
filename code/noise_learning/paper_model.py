# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Noise-learning network from Chamberland et al., Eqs. (58)-(61)."""

from __future__ import annotations

import math

import torch
from torch import nn


class NoiseLearningNetwork(nn.Module):
    """Paper-faithful CNN/GAP/MLP model producing 25 noise probabilities."""

    def __init__(self, *, p_min: float = 1e-3, p_max: float = 1e-2):
        super().__init__()
        if not (0 < p_min < p_max):
            raise ValueError("expected 0 < p_min < p_max")
        self.output_min = float(p_min) / 100.0
        self.output_max = 3.0 * float(p_max)
        filters = (128, 256, 256, 128)
        layers: list[nn.Module] = []
        in_channels = 8
        for index, out_channels in enumerate(filters):
            layers.extend(
                [
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.GroupNorm(32, out_channels),
                    nn.GELU(approximate="tanh"),
                ]
            )
            if index == len(filters) - 1:
                layers.append(nn.Dropout2d(0.1))
            in_channels = out_channels
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Linear(128, 256),
            nn.GELU(approximate="tanh"),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.GELU(approximate="tanh"),
            nn.Dropout(0.2),
            nn.Linear(128, 25),
        )

    def per_sample_logits(self, syndrome_pairs: torch.Tensor) -> torch.Tensor:
        """Return Eq. (59) logits before post-MLP batch aggregation."""

        if syndrome_pairs.ndim == 5:
            if tuple(syndrome_pairs.shape[1:3]) != (4, 2):
                raise ValueError(
                    "5-D input must have shape (B, 4, 2, D, D)"
                )
            values = syndrome_pairs.flatten(1, 2)
        elif syndrome_pairs.ndim == 4 and syndrome_pairs.shape[1] == 8:
            values = syndrome_pairs
        else:
            raise ValueError(
                "input must have shape (B, 4, 2, D, D) or (B, 8, D, D)"
            )
        hidden = self.features(values.to(dtype=torch.float32))
        pooled = hidden.mean(dim=(-2, -1))
        return self.head(pooled)

    def bounded_log_space(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply the bounded log-space parameterization in Eq. (61)."""

        log_min = math.log(self.output_min)
        log_span = math.log(self.output_max) - log_min
        return torch.exp(log_min + log_span * torch.sigmoid(logits))

    def forward(self, syndrome_pairs: torch.Tensor) -> torch.Tensor:
        """Return one 25-vector after Eq. (60) batch-logit averaging."""

        averaged_logits = self.per_sample_logits(syndrome_pairs).mean(dim=0)
        return self.bounded_log_space(averaged_logits)
