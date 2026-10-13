# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.experiments.qewc_v3.summarize_resources import summarize_run


class TestQewcV3Resources(unittest.TestCase):

    def test_ewc_includes_training_fisher_and_online_state(self):
        with TemporaryDirectory() as raw_root:
            path = Path(raw_root) / "resource.json"
            path.write_text(json.dumps({"epochs": [{
                "gpu_hours": 1.0,
                "peak_allocated_bytes_max_rank": 10,
                "peak_reserved_bytes_max_rank": 20,
            }]}))
            training = {
                "method": "ewc",
                "total_optimizer_steps": 1000,
                "samples_processed": 512000,
                "resource_metrics": {"C1": str(path)},
                "method_parameters": {
                    "fisher_timings": [{"wall_seconds": 360.0}],
                    "online_state_bytes": 123,
                },
            }
            result = summarize_run("S2", training)
            self.assertAlmostEqual(result["total_gpu_hours_including_auxiliary"], 1.1)
            self.assertEqual(result["extra_method_storage_bytes"], 123)

    def test_replay_uses_serialized_buffer_size(self):
        with TemporaryDirectory() as raw_root:
            path = Path(raw_root) / "resource.json"
            path.write_text(json.dumps({"epochs": [{
                "gpu_hours": 0.5,
                "peak_allocated_bytes_max_rank": 10,
                "peak_reserved_bytes_max_rank": 20,
            }]}))
            training = {
                "method": "replay",
                "total_optimizer_steps": 1000,
                "samples_processed": 512000,
                "resource_metrics": {"C1": str(path)},
                "method_parameters": {
                    "rebuild_timings": [{"wall_seconds": 36.0}],
                    "final_buffer_manifest": {"serialized_bytes": 456},
                },
            }
            result = summarize_run("S3", training)
            self.assertAlmostEqual(result["total_gpu_hours_including_auxiliary"], 0.51)
            self.assertEqual(result["extra_method_storage_bytes"], 456)


if __name__ == "__main__":
    unittest.main()
