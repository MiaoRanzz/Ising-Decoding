# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

## Model architecture with CNN networks for pre-decoders

import torch
import torch.nn as nn
from types import SimpleNamespace


class ResidualBlock3D(nn.Module):

    def __init__(self, channels, kernel_sizes, activation):
        """
        channels: List of 4 ints = [in1, out1, out2, out3]
        kernel_sizes: List of 3 ints (or tuples) = k1, k2, k3
        """
        super(ResidualBlock3D, self).__init__()
        self.activation = activation()  # instantiate once

        self.conv1 = nn.Sequential(
            nn.Conv3d(
                channels[0], channels[1], kernel_size=kernel_sizes[0], padding=kernel_sizes[0] // 2
            ),
            nn.BatchNorm3d(channels[1]),
            self.activation  # instance
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(
                channels[1], channels[2], kernel_size=kernel_sizes[1], padding=kernel_sizes[1] // 2
            ),
            nn.BatchNorm3d(channels[2]),
            self.activation  # instance
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(
                channels[2], channels[3], kernel_size=kernel_sizes[2], padding=kernel_sizes[2] // 2
            ), nn.BatchNorm3d(channels[3])
        )

        self.skip = nn.Identity()
        if channels[0] != channels[3]:
            self.skip = nn.Conv3d(channels[0], channels[3], kernel_size=1)

    def forward(self, x):
        identity = self.skip(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        return self.activation(out + identity)


class PreDecoderModelMemory_v1(nn.Module):

    def __init__(self, cfg):
        super(PreDecoderModelMemory_v1, self).__init__()

        self.distance = cfg.distance
        self.n_rounds = cfg.n_rounds
        self.dropout_p = cfg.model.dropout_p
        self.activation_fn = self._get_activation(cfg.model.activation)

        filters = cfg.model.num_filters
        kernel_sizes = cfg.model.kernel_size

        assert len(filters) == len(kernel_sizes), \
            "Mismatch: num_filters and kernel_size must be the same length."

        # === Configurable input and output channels ===
        input_channels = cfg.model.input_channels
        out_channels = cfg.model.out_channels
        assert filters[-1] == out_channels, \
            f"The last element of num_filters must match the configured out_channels ({out_channels}), but got {filters[-1]}"

        layers = []
        in_channels = input_channels  # 4 input channels from trainX

        for i in range(len(filters)):
            layers.append(
                nn.Conv3d(
                    in_channels=in_channels,
                    out_channels=filters[i],
                    kernel_size=kernel_sizes[i],
                    padding=kernel_sizes[i] // 2  # keeps same shape (optional)
                )
            )
            if i < len(filters) - 1:  # last layer should not have dropout or activation
                layers.append(nn.Dropout3d(p=self.dropout_p))
                layers.append(self.activation_fn)
            in_channels = filters[i]

        self.net = nn.Sequential(*layers)

    def _get_activation(self, name):
        if name == "relu":
            return nn.ReLU()
        elif name == "gelu":
            return nn.GELU(approximate='tanh')
        elif name == "leakyrelu":
            return nn.LeakyReLU()
        else:
            raise ValueError(f"Unsupported activation: {name}")

    def forward(self, x):
        return self.net(x)  # x: (B, 4, T, D, D)


class PreDecoderModelMemoryFactorized_v1(nn.Module):

    def __init__(self, cfg):
        super(PreDecoderModelMemoryFactorized_v1, self).__init__()

        self.distance = cfg.distance
        self.n_rounds = cfg.n_rounds
        self.dropout_p = cfg.model.dropout_p

        filters = cfg.model.num_filters
        kernel_sizes = cfg.model.kernel_size

        assert len(filters) == len(kernel_sizes), \
            "Mismatch: num_filters and kernel_size must be the same length."

        input_channels = cfg.model.input_channels
        out_channels = cfg.model.out_channels
        assert filters[-1] == out_channels, \
            f"The last element of num_filters must match the configured out_channels ({out_channels}), but got {filters[-1]}"

        layers = []
        in_channels = input_channels

        for i in range(len(filters)):
            k = kernel_sizes[i]
            if k == 1:
                layers.append(
                    nn.Conv3d(
                        in_channels=in_channels,
                        out_channels=filters[i],
                        kernel_size=1,
                        padding=0
                    )
                )
            else:
                layers.append(
                    nn.Conv3d(
                        in_channels=in_channels,
                        out_channels=filters[i],
                        kernel_size=(k, 1, 1),
                        padding=(k // 2, 0, 0)
                    )
                )
                layers.append(
                    nn.Conv3d(
                        in_channels=filters[i],
                        out_channels=filters[i],
                        kernel_size=(1, k, k),
                        padding=(0, k // 2, k // 2)
                    )
                )
            if i < len(filters) - 1:
                layers.append(nn.Dropout3d(p=self.dropout_p))
                layers.append(self._get_activation(cfg.model.activation))
            in_channels = filters[i]

        self.net = nn.Sequential(*layers)

    def _get_activation(self, name):
        if name == "relu":
            return nn.ReLU()
        elif name == "gelu":
            return nn.GELU(approximate='tanh')
        elif name == "leakyrelu":
            return nn.LeakyReLU()
        else:
            raise ValueError(f"Unsupported activation: {name}")

    def forward(self, x):
        return self.net(x)  # x: (B, 4, T, D, D)


class ChannelGate3D(nn.Module):

    def __init__(self, channels, reduction, activation):
        super(ChannelGate3D, self).__init__()
        hidden_channels = max(1, channels // reduction)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels, hidden_channels, kernel_size=1),
            activation,
            nn.Conv3d(hidden_channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.net(x)


class AdaptiveBranchFusion3D(nn.Module):

    def __init__(self, channels, reduction, activation):
        super(AdaptiveBranchFusion3D, self).__init__()
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

    def forward(self, spatial, temporal, joint):
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

    def __init__(self, channels, reduction, activation):
        super(AxisChannelGate3D, self).__init__()
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

    def forward(self, x):
        channel_logits = self.channel_net(x)
        temporal_logits = self.temporal_conv(x.mean(dim=(1, 3, 4), keepdim=True))
        spatial_logits = self.spatial_conv(x.mean(dim=(1, 2), keepdim=True))
        return x * torch.sigmoid(channel_logits + temporal_logits + spatial_logits)


class STFusionBlock(nn.Module):

    def __init__(
        self,
        channels,
        expand_channels,
        joint_groups,
        norm_groups,
        se_reduction,
        dropout_p,
        activation_name,
    ):
        super(STFusionBlock, self).__init__()
        if expand_channels % joint_groups != 0:
            raise ValueError(
                f"expand_channels ({expand_channels}) must be divisible by joint_groups ({joint_groups})"
            )
        if channels % norm_groups != 0 or expand_channels % norm_groups != 0:
            raise ValueError(
                "channels and expand_channels must be divisible by norm_groups "
                f"(got channels={channels}, expand_channels={expand_channels}, norm_groups={norm_groups})"
            )

        activation = self._get_activation(activation_name)
        self.pre = nn.Sequential(
            nn.GroupNorm(num_groups=norm_groups, num_channels=channels),
            nn.Conv3d(channels, expand_channels, kernel_size=1),
            activation,
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
        self.project = nn.Sequential(
            nn.Conv3d(expand_channels, channels, kernel_size=1),
            self._get_activation(activation_name),
        )
        self.gate = ChannelGate3D(
            channels=channels,
            reduction=se_reduction,
            activation=self._get_activation(activation_name),
        )
        self.dropout = nn.Dropout3d(p=dropout_p)

    def _get_activation(self, name):
        if name == "relu":
            return nn.ReLU()
        elif name == "gelu":
            return nn.GELU(approximate='tanh')
        elif name == "leakyrelu":
            return nn.LeakyReLU()
        else:
            raise ValueError(f"Unsupported activation: {name}")

    def forward(self, x):
        residual = x
        y = self.pre(x)
        y = self.spatial(y) + self.temporal(y) + self.joint(y)
        y = self.project(y)
        y = self.gate(y)
        return residual + self.dropout(y)


class STFusionBlockV2(nn.Module):

    def __init__(
        self,
        channels,
        expand_channels,
        joint_groups,
        norm_groups,
        se_reduction,
        dropout_p,
        activation_name,
    ):
        super(STFusionBlockV2, self).__init__()
        if expand_channels % joint_groups != 0:
            raise ValueError(
                f"expand_channels ({expand_channels}) must be divisible by joint_groups ({joint_groups})"
            )
        if channels % norm_groups != 0 or expand_channels % norm_groups != 0:
            raise ValueError(
                "channels and expand_channels must be divisible by norm_groups "
                f"(got channels={channels}, expand_channels={expand_channels}, norm_groups={norm_groups})"
            )

        activation = self._get_activation(activation_name)
        self.pre = nn.Sequential(
            nn.GroupNorm(num_groups=norm_groups, num_channels=channels),
            nn.Conv3d(channels, expand_channels, kernel_size=1),
            activation,
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
            activation=self._get_activation(activation_name),
        )
        self.branch_mixer = nn.Sequential(
            nn.Conv3d(
                expand_channels,
                expand_channels,
                kernel_size=1,
                groups=joint_groups,
            ),
            self._get_activation(activation_name),
        )
        self.project = nn.Sequential(
            nn.Conv3d(expand_channels, channels, kernel_size=1),
            self._get_activation(activation_name),
        )
        self.gate = AxisChannelGate3D(
            channels=channels,
            reduction=se_reduction,
            activation=self._get_activation(activation_name),
        )
        self.dropout = nn.Dropout3d(p=dropout_p)

    def _get_activation(self, name):
        if name == "relu":
            return nn.ReLU()
        elif name == "gelu":
            return nn.GELU(approximate='tanh')
        elif name == "leakyrelu":
            return nn.LeakyReLU()
        else:
            raise ValueError(f"Unsupported activation: {name}")

    def forward(self, x):
        residual = x
        y = self.pre(x)
        y = self.branch_fusion(self.spatial(y), self.temporal(y), self.joint(y))
        y = self.branch_mixer(y)
        y = self.project(y)
        y = self.gate(y)
        return residual + self.dropout(y)


class PreDecoderSTFusion_v1(nn.Module):

    def __init__(self, cfg):
        super(PreDecoderSTFusion_v1, self).__init__()

        self.distance = cfg.distance
        self.n_rounds = cfg.n_rounds
        self.dropout_p = cfg.model.dropout_p

        input_channels = cfg.model.input_channels
        out_channels = cfg.model.out_channels
        channels = int(cfg.model.channels)
        expand_channels = int(cfg.model.expand_channels)
        num_blocks = int(cfg.model.num_blocks)
        joint_groups = int(cfg.model.joint_groups)
        norm_groups = int(cfg.model.norm_groups)
        se_reduction = int(cfg.model.se_reduction)
        activation_name = cfg.model.activation

        self.stem = nn.Sequential(
            nn.Conv3d(input_channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=norm_groups, num_channels=channels),
            self._get_activation(activation_name),
        )
        self.blocks = nn.Sequential(*[
            STFusionBlock(
                channels=channels,
                expand_channels=expand_channels,
                joint_groups=joint_groups,
                norm_groups=norm_groups,
                se_reduction=se_reduction,
                dropout_p=self.dropout_p,
                activation_name=activation_name,
            )
            for _ in range(num_blocks)
        ])
        self.head = nn.Sequential(
            nn.GroupNorm(num_groups=norm_groups, num_channels=channels),
            nn.Conv3d(channels, channels, kernel_size=1),
            self._get_activation(activation_name),
            nn.Conv3d(channels, out_channels, kernel_size=1),
        )

    def _get_activation(self, name):
        if name == "relu":
            return nn.ReLU()
        elif name == "gelu":
            return nn.GELU(approximate='tanh')
        elif name == "leakyrelu":
            return nn.LeakyReLU()
        else:
            raise ValueError(f"Unsupported activation: {name}")

    def forward(self, x):
        return self.head(self.blocks(self.stem(x)))  # x: (B, 4, T, D, D)


class PreDecoderSTFusion_v2(nn.Module):

    def __init__(self, cfg):
        super(PreDecoderSTFusion_v2, self).__init__()

        self.distance = cfg.distance
        self.n_rounds = cfg.n_rounds
        self.dropout_p = cfg.model.dropout_p

        input_channels = cfg.model.input_channels
        out_channels = cfg.model.out_channels
        channels = int(cfg.model.channels)
        expand_channels = int(cfg.model.expand_channels)
        num_blocks = int(cfg.model.num_blocks)
        joint_groups = int(cfg.model.joint_groups)
        norm_groups = int(cfg.model.norm_groups)
        se_reduction = int(cfg.model.se_reduction)
        activation_name = cfg.model.activation

        self.stem = nn.Sequential(
            nn.Conv3d(input_channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=norm_groups, num_channels=channels),
            self._get_activation(activation_name),
        )
        self.blocks = nn.Sequential(*[
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
        ])
        self.head_norm = nn.GroupNorm(num_groups=norm_groups, num_channels=channels)
        self.head_hidden = nn.Conv3d(
            channels + input_channels,
            channels,
            kernel_size=1,
        )
        self.head_activation = self._get_activation(activation_name)
        self.head_out = nn.Conv3d(channels, out_channels, kernel_size=1)

    def _get_activation(self, name):
        if name == "relu":
            return nn.ReLU()
        elif name == "gelu":
            return nn.GELU(approximate='tanh')
        elif name == "leakyrelu":
            return nn.LeakyReLU()
        else:
            raise ValueError(f"Unsupported activation: {name}")

    def forward(self, x):
        y = self.blocks(self.stem(x))
        y = self.head_norm(y)
        y = torch.cat([y, x], dim=1)
        y = self.head_activation(self.head_hidden(y))
        return self.head_out(y)  # x: (B, 4, T, D, D)


class PreDecoderFastHyperRF13_v1(nn.Module):

    def __init__(self, cfg):
        super(PreDecoderFastHyperRF13_v1, self).__init__()

        self.distance = cfg.distance
        self.n_rounds = cfg.n_rounds
        self.dropout_p = cfg.model.dropout_p

        input_channels = cfg.model.input_channels
        out_channels = cfg.model.out_channels
        channels = int(cfg.model.channels)
        expand_channels = int(cfg.model.expand_channels)
        num_blocks = int(cfg.model.num_blocks)
        joint_groups = int(cfg.model.joint_groups)
        norm_groups = int(cfg.model.norm_groups)
        se_reduction = int(cfg.model.se_reduction)
        activation_name = cfg.model.activation

        self.stem = nn.Sequential(
            nn.Conv3d(input_channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=norm_groups, num_channels=channels),
            self._get_activation(activation_name),
        )
        self.blocks = nn.Sequential(*[
            STFusionBlock(
                channels=channels,
                expand_channels=expand_channels,
                joint_groups=joint_groups,
                norm_groups=norm_groups,
                se_reduction=se_reduction,
                dropout_p=self.dropout_p,
                activation_name=activation_name,
            )
            for _ in range(num_blocks)
        ])
        self.head = nn.Sequential(
            nn.GroupNorm(num_groups=norm_groups, num_channels=channels),
            nn.Conv3d(channels, channels, kernel_size=1),
            self._get_activation(activation_name),
            nn.Conv3d(channels, out_channels, kernel_size=1),
        )

    def _get_activation(self, name):
        if name == "relu":
            return nn.ReLU()
        elif name == "gelu":
            return nn.GELU(approximate='tanh')
        elif name == "leakyrelu":
            return nn.LeakyReLU()
        else:
            raise ValueError(f"Unsupported activation: {name}")

    def forward(self, x):
        return self.head(self.blocks(self.stem(x)))  # x: (B, 4, T, D, D)


# === Define a mock config using SimpleNamespace ===
def get_mock_config():
    cfg = SimpleNamespace()
    cfg.model = SimpleNamespace()
    cfg.distance = 11
    cfg.n_rounds = 3
    cfg.model.dropout_p = 0.1
    cfg.model.activation = 'relu'
    cfg.model.input_channels = 4
    cfg.model.out_channels = 2
    cfg.model.num_filters = [8, 4, 2]
    cfg.model.kernel_size = [3, 3, 3]
    return cfg


# === Run the test ===
def test_model():
    cfg = get_mock_config()
    model = PreDecoderModelMemory_v1(cfg)

    B, C_in, T, D = 2, cfg.model.input_channels, cfg.n_rounds, cfg.distance
    input_tensor = torch.randn(B, C_in, T, D, D)

    output = model(input_tensor)

    expected_shape = (B, cfg.model.out_channels, T, D, D)
    assert output.shape == expected_shape, \
        f"Output shape mismatch: expected {expected_shape}, got {output.shape}"

    print("✅ Model test passed. Output shape:", output.shape)


if __name__ == "__main__":
    test_model()
