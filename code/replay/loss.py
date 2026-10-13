# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Loss helpers for current-data plus replay-data training."""

from __future__ import annotations

import torch


def combine_current_and_replay_loss(
    current_loss_sum: torch.Tensor,
    replay_loss_sum: torch.Tensor | None,
    *,
    current_batch_size: int,
    replay_batch_size: int,
    replay_lambda: float,
) -> torch.Tensor:
    """Return a sum-scaled, sample-weighted current-plus-Replay loss.

    The training loop divides accumulated gradients by the physical total
    batch size. With replay_lambda=1, this therefore reduces exactly to the
    ordinary mean over the fixed-size mixed batch, so the configured Replay
    sample fraction controls its influence. Other replay_lambda values weight
    each Replay sample relative to each current sample while preserving the
    overall loss scale.
    """
    current_batch_size = int(current_batch_size)
    replay_batch_size = int(replay_batch_size)
    replay_lambda = float(replay_lambda)
    if current_batch_size <= 0:
        raise ValueError("current_batch_size must be positive")
    if replay_lambda < 0:
        raise ValueError("replay_lambda must be non-negative")
    if replay_batch_size <= 0 or replay_loss_sum is None or replay_lambda == 0.0:
        return current_loss_sum

    total_batch_size = current_batch_size + replay_batch_size
    weighted_sample_count = current_batch_size + replay_lambda * replay_batch_size
    weighted_mean = (
        current_loss_sum + replay_lambda * replay_loss_sum
    ) / weighted_sample_count
    return total_batch_size * weighted_mean


__all__ = ["combine_current_and_replay_loss"]
