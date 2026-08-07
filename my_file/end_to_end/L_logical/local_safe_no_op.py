"""Shared local safe-no-op model and tensor conventions.

The frozen Ising-fast proposal has four correction channels.  A safe-no-op
decision must not independently remove the X and Z halves of a possible Y
data correction, so this module exposes three *packet* gates:

    0: both data channels at one (round, row, column) site;
    1: X-syndrome correction at one site;
    2: Z-syndrome correction at one site.

The gate is deliberately a separate model for the first experiment.  This
makes it possible to compare it with the unchanged proposal checkpoint and to
train it from counterfactual labels without destabilising the proposal model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn


PACKET_NAMES = ("data_packet", "x_syndrome_packet", "z_syndrome_packet")
NO_PACKET_LABEL = -2


def split_rows(
    total: int,
    train_fraction: float,
    validation_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create one deterministic, disjoint train/validation/test split."""
    if total < 3:
        raise ValueError("at least three shots are required for train/validation/test splitting")
    if train_fraction <= 0.0 or validation_fraction <= 0.0 or train_fraction + validation_fraction >= 1.0:
        raise ValueError("train_fraction and validation_fraction must be positive and sum to less than one")
    order = np.random.default_rng(seed).permutation(total)
    train_end = int(total * train_fraction)
    validation_end = train_end + int(total * validation_fraction)
    if train_end == 0 or validation_end == train_end or validation_end == total:
        raise ValueError("split fractions produced an empty train, validation, or test split")
    return np.sort(order[:train_end]), np.sort(order[train_end:validation_end]), np.sort(order[validation_end:])


@dataclass(frozen=True)
class GateArchitecture:
    input_channels: int = 12
    hidden_channels: int = 64
    num_hidden_layers: int = 3
    kernel_size: int = 3
    dropout: float = 0.05

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


class LocalSafeNoOpGate(nn.Module):
    """Small Conv3D gate yielding one logit for each correction packet."""

    def __init__(self, architecture: GateArchitecture = GateArchitecture()):
        super().__init__()
        if architecture.input_channels != 12:
            raise ValueError("the current gate feature layout has exactly 12 channels")
        if architecture.num_hidden_layers < 1:
            raise ValueError("num_hidden_layers must be positive")
        self.architecture = architecture
        layers: list[nn.Module] = []
        in_channels = architecture.input_channels
        for _ in range(architecture.num_hidden_layers):
            layers.extend((
                nn.Conv3d(in_channels, architecture.hidden_channels, architecture.kernel_size,
                          padding=architecture.kernel_size // 2),
                nn.GELU(approximate="tanh"),
                nn.Dropout3d(architecture.dropout),
            ))
            in_channels = architecture.hidden_channels
        layers.append(nn.Conv3d(in_channels, len(PACKET_NAMES), kernel_size=1))
        self.net = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


def proposal_actions_from_logits(proposal_logits: torch.Tensor) -> torch.Tensor:
    """Convert frozen proposal logits to its existing threshold-zero actions."""
    if proposal_logits.ndim != 5 or proposal_logits.shape[1] != 4:
        raise ValueError("proposal logits must have shape (batch, 4, rounds, distance, distance)")
    return proposal_logits >= 0.0


def packet_activity(proposal_actions: torch.Tensor) -> torch.Tensor:
    """Map four proposal action channels to the three coupled packet channels."""
    if proposal_actions.ndim != 5 or proposal_actions.shape[1] != 4:
        raise ValueError("proposal actions must have four channels")
    return torch.stack((
        proposal_actions[:, 0] | proposal_actions[:, 1],
        proposal_actions[:, 2],
        proposal_actions[:, 3],
    ), dim=1)


def gate_features(train_x: torch.Tensor, proposal_logits: torch.Tensor,
                  proposal_actions: torch.Tensor | None = None) -> torch.Tensor:
    """Build gate input: syndrome context, proposal strength, proposal action."""
    if train_x.shape != proposal_logits.shape:
        raise ValueError("train_x and proposal_logits must have identical shapes")
    # Clipping keeps a rare extreme proposal logit from dominating gate training.
    logits = proposal_logits.to(torch.float32).clamp(-12.0, 12.0) / 6.0
    if proposal_actions is None:
        proposal_actions = proposal_actions_from_logits(proposal_logits)
    if proposal_actions.shape != proposal_logits.shape:
        raise ValueError("proposal_actions and proposal_logits must have identical shapes")
    actions = proposal_actions.to(torch.float32)
    return torch.cat((train_x.to(torch.float32), logits, actions), dim=1)


def apply_packet_gate(proposal_actions: torch.Tensor, gate_logits: torch.Tensor,
                      threshold: float = 0.0) -> torch.Tensor:
    """Return four correction channels after packet-level abstention.

    A non-positive gate logit means no-op for that packet.  Inactive proposal
    bits always remain inactive, irrespective of the gate output.
    """
    if gate_logits.ndim != 5 or gate_logits.shape[1] != len(PACKET_NAMES):
        raise ValueError("gate logits must have three packet channels")
    if proposal_actions.shape[0] != gate_logits.shape[0] or proposal_actions.shape[2:] != gate_logits.shape[2:]:
        raise ValueError("proposal and gate tensors do not have matching batch/spatial shape")
    accepted = gate_logits >= threshold
    output = proposal_actions.clone()
    output[:, 0] &= accepted[:, 0]
    output[:, 1] &= accepted[:, 0]
    output[:, 2] &= accepted[:, 1]
    output[:, 3] &= accepted[:, 2]
    return output
