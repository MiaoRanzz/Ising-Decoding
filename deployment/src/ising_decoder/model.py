# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standalone HTnet model used by the d=9/T=9 surface decoder.

This file is adapted from ``code/model/predecoder.py`` and intentionally keeps
the module names used by the released model-111 checkpoint.
"""

from __future__ import annotations

import torch
from torch import nn


def _activation(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU(approximate="tanh")
    if name == "leakyrelu":
        return nn.LeakyReLU()
    raise ValueError(f"Unsupported activation: {name}")


class AdaptiveBranchFusion3D(nn.Module):
    """Fuse spatial, temporal, and joint HTnet branches."""

    def __init__(self, channels: int, reduction: int, activation: nn.Module):
        super().__init__()
        self.num_branches = 3
        hidden_channels = max(1, channels // (reduction + 2))
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.weight_net = nn.Sequential(
            nn.Conv3d(channels * self.num_branches, hidden_channels, kernel_size=1),
            activation,
            nn.Conv3d(hidden_channels, channels * self.num_branches, kernel_size=1),
        )
        nn.init.zeros_(self.weight_net[-1].weight)
        nn.init.zeros_(self.weight_net[-1].bias)

    def forward(
        self,
        spatial: torch.Tensor,
        temporal: torch.Tensor,
        joint: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, channels = spatial.shape[:2]
        pooled = torch.cat(
            [self.pool(spatial), self.pool(temporal), self.pool(joint)],
            dim=1,
        )
        weights = self.weight_net(pooled).view(
            batch_size, self.num_branches, channels, 1, 1, 1
        )
        weights = torch.softmax(weights, dim=1)
        fused = (
            weights[:, 0] * spatial
            + weights[:, 1] * temporal
            + weights[:, 2] * joint
        )
        return fused * self.num_branches


class AxisChannelGate3D(nn.Module):
    """Apply channel, temporal, and spatial gates."""

    def __init__(self, channels: int, reduction: int, activation: nn.Module):
        super().__init__()
        hidden_channels = max(1, channels // reduction)
        self.channel_net = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels, hidden_channels, kernel_size=1),
            activation,
            nn.Conv3d(hidden_channels, channels, kernel_size=1),
        )
        self.temporal_conv = nn.Conv3d(
            1,
            1,
            kernel_size=(3, 1, 1),
            padding=(1, 0, 0),
        )
        self.spatial_conv = nn.Conv3d(
            1,
            1,
            kernel_size=(1, 3, 3),
            padding=(0, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channel_logits = self.channel_net(x)
        temporal_logits = self.temporal_conv(x.mean(dim=(1, 3, 4), keepdim=True))
        spatial_logits = self.spatial_conv(x.mean(dim=(1, 2), keepdim=True))
        return x * torch.sigmoid(channel_logits + temporal_logits + spatial_logits)


class STFusionBlockV2(nn.Module):
    """One full-path HTnet spatiotemporal fusion block."""

    def __init__(
        self,
        channels: int,
        expand_channels: int,
        joint_groups: int,
        norm_groups: int,
        se_reduction: int,
        dropout_p: float,
        activation_name: str,
    ):
        super().__init__()
        if expand_channels % joint_groups != 0:
            raise ValueError(
                f"expand_channels ({expand_channels}) must be divisible by "
                f"joint_groups ({joint_groups})"
            )
        if channels % norm_groups != 0 or expand_channels % norm_groups != 0:
            raise ValueError(
                "channels and expand_channels must be divisible by norm_groups "
                f"(got channels={channels}, expand_channels={expand_channels}, "
                f"norm_groups={norm_groups})"
            )

        self.pre = nn.Sequential(
            nn.GroupNorm(num_groups=norm_groups, num_channels=channels),
            nn.Conv3d(channels, expand_channels, kernel_size=1),
            _activation(activation_name),
        )
        self.spatial = nn.Conv3d(
            expand_channels,
            expand_channels,
            kernel_size=(1, 3, 3),
            padding=(0, 1, 1),
            groups=expand_channels,
        )
        self.temporal = nn.Conv3d(
            expand_channels,
            expand_channels,
            kernel_size=(3, 1, 1),
            padding=(1, 0, 0),
            groups=expand_channels,
        )
        self.joint = nn.Sequential(
            nn.GroupNorm(num_groups=norm_groups, num_channels=expand_channels),
            nn.Conv3d(
                expand_channels,
                expand_channels,
                kernel_size=3,
                padding=1,
                groups=joint_groups,
            ),
        )
        self.branch_fusion = AdaptiveBranchFusion3D(
            channels=expand_channels,
            reduction=se_reduction,
            activation=_activation(activation_name),
        )
        self.branch_mixer = nn.Sequential(
            nn.Conv3d(
                expand_channels,
                expand_channels,
                kernel_size=1,
                groups=joint_groups,
            ),
            _activation(activation_name),
        )
        self.project = nn.Sequential(
            nn.Conv3d(expand_channels, channels, kernel_size=1),
            _activation(activation_name),
        )
        self.gate = AxisChannelGate3D(
            channels=channels,
            reduction=se_reduction,
            activation=_activation(activation_name),
        )
        self.dropout = nn.Dropout3d(p=dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = self.pre(x)
        y = self.branch_fusion(self.spatial(y), self.temporal(y), self.joint(y))
        y = self.branch_mixer(y)
        y = self.project(y)
        y = self.gate(y)
        return residual + self.dropout(y)


class HTnet(nn.Module):
    """Fixed full-path HTnet architecture for public model ID 111."""

    def __init__(self):
        super().__init__()
        self.distance = 9
        self.n_rounds = 9
        self.dropout_p = 0.05
        input_channels = 4
        out_channels = 4
        channels = 112
        expand_channels = 168
        num_blocks = 3
        joint_groups = 6
        norm_groups = 8
        se_reduction = 4
        activation_name = "gelu"
        self.use_input_skip_head = True

        self.stem = nn.Sequential(
            nn.Conv3d(input_channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=norm_groups, num_channels=channels),
            _activation(activation_name),
        )
        self.blocks = nn.Sequential(
            *[
                STFusionBlockV2(
                    channels=channels,
                    expand_channels=expand_channels,
                    joint_groups=joint_groups,
                    norm_groups=norm_groups,
                    se_reduction=se_reduction,
                    dropout_p=self.dropout_p,
                    activation_name=activation_name,
                )
                for _ in range(num_blocks)
            ]
        )
        self.head_norm = nn.GroupNorm(
            num_groups=norm_groups,
            num_channels=channels,
        )
        self.head_hidden = nn.Conv3d(
            channels + input_channels,
            channels,
            kernel_size=1,
        )
        self.head_activation = _activation(activation_name)
        self.head_out = nn.Conv3d(channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.blocks(self.stem(x))
        y = self.head_norm(y)
        y = torch.cat([y, x], dim=1)
        y = self.head_activation(self.head_hidden(y))
        return self.head_out(y)
