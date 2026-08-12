# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from benchmarks.patent_validation.topology_gating_v3_latency import (
    DEFAULT_CONFIG,
    _candidates,
    measure_batch_throughput,
    resolved_config,
    verify_latency_isolation,
)


class FakeMatcher:
    def __init__(self) -> None:
        self.decode_calls = 0
        self.decode_batch_calls = 0

    def decode(self, residual: np.ndarray) -> int:
        self.decode_calls += 1
        return int(np.asarray(residual).sum() % 2)

    def decode_batch(self, residuals: np.ndarray) -> np.ndarray:
        self.decode_batch_calls += 1
        return np.asarray(residuals).sum(axis=1) % 2


class TopologyGatingV3LatencyTest(unittest.TestCase):
    def test_registered_matrix_contains_96_configs(self):
        config = resolved_config(Path(DEFAULT_CONFIG), "full")
        candidates = _candidates(config)
        self.assertEqual(len(candidates), 96)
        self.assertEqual(len({name for name, _ in candidates}), 96)

    def test_throughput_harness_uses_decode_batch_and_balanced_order(self):
        matcher = FakeMatcher()
        residuals = {
            "pointwise": np.zeros((4, 3), dtype=np.uint8),
            "v3": np.ones((4, 3), dtype=np.uint8),
        }
        rows, sentinel = measure_batch_throughput(
            matcher,
            residuals,
            np.array([11, 11, 12, 12]),
            warmup=2,
            repeats=2,
            block_size=2,
            random_seed=7,
        )
        self.assertEqual(matcher.decode_calls, 0)
        self.assertEqual(matcher.decode_batch_calls, 12)
        self.assertEqual(len(rows), 8)
        self.assertTrue(all(row["measurement_mode"] == "decode_batch" for row in rows))
        for method in residuals:
            positions = {row["order"] for row in rows if row["method"] == method}
            self.assertEqual(positions, {0, 1})
        self.assertGreaterEqual(sentinel["raw_sentinel_drift"], 0.0)

    @patch(
        "benchmarks.patent_validation.topology_gating_v3_latency._cpu_utilisation",
        return_value=0.05,
    )
    @patch(
        "benchmarks.patent_validation.topology_gating_v3_latency.os.sched_getaffinity",
        return_value={27},
    )
    def test_latency_isolation_accepts_exact_affinity(self, _affinity, _utilisation):
        result = verify_latency_isolation(27, strict=True)
        self.assertEqual(result["affinity"], [27])

    @patch(
        "benchmarks.patent_validation.topology_gating_v3_latency._cpu_utilisation",
        return_value=0.05,
    )
    @patch(
        "benchmarks.patent_validation.topology_gating_v3_latency.os.sched_getaffinity",
        return_value={26, 27},
    )
    def test_latency_isolation_rejects_nonexclusive_affinity(self, _affinity, _utilisation):
        with self.assertRaises(RuntimeError):
            verify_latency_isolation(27, strict=True)


if __name__ == "__main__":
    unittest.main()
