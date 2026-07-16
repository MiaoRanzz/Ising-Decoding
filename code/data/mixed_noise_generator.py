# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Batch-level mixed-noise training generator wrapper."""

from __future__ import annotations


class MixedNoiseBatchGenerator:
    """Round-robin wrapper over per-noise QCDataGeneratorTorch instances.

    Each generated batch comes from exactly one noise task. This keeps the
    underlying generator assumptions intact while producing an equal-weight
    mixed-noise training stream across batches.
    """

    def __init__(self, generators, task_names):
        if not generators:
            raise ValueError("MixedNoiseBatchGenerator requires at least one generator")
        if len(generators) != len(task_names):
            raise ValueError("generators and task_names must have the same length")

        self.generators = list(generators)
        self.task_names = list(task_names)
        first = self.generators[0]
        self.distance = getattr(first, "distance", None)
        self.n_rounds = getattr(first, "n_rounds", None)
        self.mode = getattr(first, "mode", None)
        self.noise_model = None
        self.noise_model_mixture = True

    @property
    def num_tasks(self) -> int:
        return len(self.generators)

    def get_current_task(self, step):
        task_idx = int(step) % self.num_tasks
        return task_idx, self.task_names[task_idx]

    def generate_batch(self, step, batch_size, **kwargs):
        task_idx, _ = self.get_current_task(step)
        return self.generators[task_idx].generate_batch(
            step=step,
            batch_size=batch_size,
            **kwargs,
        )


__all__ = ["MixedNoiseBatchGenerator"]
