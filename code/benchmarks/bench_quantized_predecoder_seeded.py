#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the quantized benchmark with an explicitly seeded Stim sample stream."""

from __future__ import annotations

import hashlib

import numpy as np
import torch

from benchmarks import bench_quantized_predecoder as benchmark


_ORIGINAL_MAKE_DATASET = benchmark.make_dataset
_ORIGINAL_WRITE_JSON = benchmark.write_json
_SAMPLE_METADATA: dict[str, object] = {}


def seed_dataset(dataset, num_samples: int, seed: int):
    """Replace the eager, unseeded Stim samples with a reproducible stream."""
    circuit = dataset.circ.stim_circuit
    measurements = circuit.compile_sampler(seed=int(seed)).sample(shots=int(num_samples))
    converted = circuit.compile_m2d_converter().convert(
        measurements=measurements,
        append_observables=True,
    )
    dataset.dets_and_obs = torch.from_numpy(converted).to(torch.uint8)
    sample_bytes = np.ascontiguousarray(converted, dtype=np.uint8).tobytes()
    _SAMPLE_METADATA.clear()
    _SAMPLE_METADATA.update(
        {
            "stim_sampler_seeded": True,
            "stim_seed": int(seed),
            "dataset_sample_sha256": hashlib.sha256(sample_bytes).hexdigest(),
            "sampling_protocol": "stim.compile_sampler(seed)->sample->m2d(append_observables=True)",
        }
    )
    return dataset


def deterministic_make_dataset(cfg, seed: int):
    dataset = _ORIGINAL_MAKE_DATASET(cfg, seed)
    return seed_dataset(dataset, int(cfg.test.num_samples), seed)


def write_json_with_sample_metadata(path, payload) -> None:
    if isinstance(payload, dict) and "noninferiority" in payload:
        payload = {**payload, **_SAMPLE_METADATA}
    _ORIGINAL_WRITE_JSON(path, payload)


def main() -> None:
    benchmark.make_dataset = deterministic_make_dataset
    benchmark.write_json = write_json_with_sample_metadata
    benchmark.main()


if __name__ == "__main__":
    main()
