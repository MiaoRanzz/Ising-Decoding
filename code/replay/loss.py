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
    """Return a sum-scaled loss compatible with ``training.train``.

    ``train_epoch`` accumulates sum-reduced BCE gradients and divides them by
    the accumulated number of samples immediately before the optimizer step.
    This helper therefore multiplies the desired normalized objective by the
    combined batch size. After the existing gradient division, the effective
    objective is::

        (mean_current + replay_lambda * mean_replay) / (1 + replay_lambda)

    where each mean is per sample (and still sums over output elements). With
    no replay samples the returned tensor is exactly ``current_loss_sum``.
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
    current_mean = current_loss_sum / current_batch_size
    replay_mean = replay_loss_sum / replay_batch_size
    return (
        total_batch_size
        * (current_mean + replay_lambda * replay_mean)
        / (1.0 + replay_lambda)
    )


__all__ = ["combine_current_and_replay_loss"]
