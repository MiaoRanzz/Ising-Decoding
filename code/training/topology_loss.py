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
"""Differentiable topology and logical-frame losses for four-channel training."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional

from evaluation.surface_topology_adapter import SurfaceActionAdapter


@dataclass(frozen=True)
class TopologyLossWeights:
    topology: float = 0.0
    logical_frame: float = 0.0
    calibration: float = 0.0

    def __post_init__(self) -> None:
        if min(self.topology, self.logical_frame, self.calibration) < 0:
            raise ValueError("topology loss weights must be non-negative")


def soft_odd_probability(
    probabilities: torch.Tensor, incidence: torch.Tensor
) -> torch.Tensor:
    """Exact odd-parity probability for independent Bernoulli actions.

    Unlike the earlier log-domain proxy, this form preserves the sign of
    ``1 - 2p`` and therefore has valid gradients when an action probability is
    greater than 0.5.  Surface-code detector rows are low degree, so the direct
    product is stable for this experiment.
    """

    if probabilities.ndim != 2 or incidence.ndim != 2:
        raise ValueError("probabilities and incidence must both be matrices")
    if probabilities.shape[1] != incidence.shape[1]:
        raise ValueError("action dimensions do not match")
    probability = probabilities.clamp(1e-6, 1 - 1e-6)
    factors = torch.where(
        incidence.to(device=probability.device, dtype=torch.bool).unsqueeze(0),
        1 - 2 * probability.unsqueeze(1),
        torch.ones((), dtype=probability.dtype, device=probability.device),
    )
    return 0.5 * (1 - factors.prod(dim=2))


def topology_joint_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    train_x: torch.Tensor,
    adapter: SurfaceActionAdapter,
    *,
    weights: TopologyLossWeights,
    positive_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return local BCE plus optional topology/frame/calibration components."""

    if logits.shape != targets.shape:
        raise ValueError("logits and targets must have identical shapes")
    if float(positive_weight) <= 0:
        raise ValueError("positive_weight must be positive")
    pos_weight = torch.as_tensor(
        float(positive_weight), dtype=logits.dtype, device=logits.device
    )
    local = functional.binary_cross_entropy_with_logits(
        logits, targets.to(dtype=logits.dtype), pos_weight=pos_weight
    )
    zero = local.new_zeros(())
    components = {
        "local": local,
        "topology": zero,
        "logical_frame": zero,
        "calibration": zero,
    }
    if weights == TopologyLossWeights():
        components["total"] = local
        return local, components

    probabilities = adapter.flatten_spatial(torch.sigmoid(logits))
    target_actions = adapter.flatten_spatial(targets.to(dtype=logits.dtype))
    valid = adapter.valid_actions.to(device=logits.device, dtype=logits.dtype)
    probabilities = probabilities * valid
    target_actions = target_actions * valid
    h = adapter.extended_h.to(device=logits.device, dtype=logits.dtype)
    logical = adapter.extended_l.to(device=logits.device, dtype=logits.dtype)

    predicted_flip = soft_odd_probability(probabilities, h)
    detector_state = adapter.detector_state_from_train_x(train_x).to(logits.dtype)
    residual_probability = (
        detector_state * (1 - predicted_flip)
        + (1 - detector_state) * predicted_flip
    )
    detector_mask = adapter.detector_training_mask.to(
        device=logits.device, dtype=logits.dtype
    )
    topology = (residual_probability * detector_mask).sum() / (
        detector_mask.sum().clamp_min(1) * logits.shape[0]
    )

    predicted_frame = soft_odd_probability(probabilities, logical).squeeze(1)
    target_frame = torch.remainder(target_actions @ logical.t(), 2).squeeze(1)
    frame = functional.binary_cross_entropy(
        predicted_frame.clamp(1e-6, 1 - 1e-6), target_frame
    )
    calibration = (
        ((probabilities - target_actions).square() * valid).sum()
        / (valid.sum().clamp_min(1) * logits.shape[0])
    )
    total = (
        local
        + float(weights.topology) * topology
        + float(weights.logical_frame) * frame
        + float(weights.calibration) * calibration
    )
    components.update(
        topology=topology,
        logical_frame=frame,
        calibration=calibration,
        total=total,
    )
    return total, components


__all__ = ["TopologyLossWeights", "soft_odd_probability", "topology_joint_loss"]
