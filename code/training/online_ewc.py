# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Online EWC state consolidation."""

from __future__ import annotations

from training.ewc import EWCState


def update_online_ewc_state(
    previous: EWCState,
    current: EWCState,
    *,
    gamma: float = 1.0,
) -> EWCState:
    """Apply ``F* <- gamma * F* + F`` and anchor at current parameters."""
    if not 0.0 <= float(gamma) <= 1.0:
        raise ValueError("Online EWC gamma must be in [0, 1]")
    if set(current.mean) != set(current.fisher):
        raise ValueError("Current EWC mean and Fisher parameter names must match")
    if set(previous.mean) != set(previous.fisher):
        raise ValueError("Previous EWC mean and Fisher parameter names must match")

    fisher = {}
    for name, current_fisher in current.fisher.items():
        value = current_fisher.detach().cpu().float()
        if name in previous.fisher:
            old = previous.fisher[name].detach().cpu().float()
            if old.shape != value.shape:
                raise ValueError(f"Online EWC Fisher shape mismatch for {name!r}")
            value = value + float(gamma) * old
        fisher[name] = value
    return EWCState(
        task_name=current.task_name,
        mean={name: value.detach().cpu().float() for name, value in current.mean.items()},
        fisher=fisher,
    )


__all__ = ["update_online_ewc_state"]
