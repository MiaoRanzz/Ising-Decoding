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

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest

from benchmarks.patent_validation.common import sha256_file
from benchmarks.patent_validation.topology_gating_v2_small import (
    DEFAULT_CONFIG,
    _identity,
    _noise_mapping,
    aggregate,
    _resume_complete,
    resolved_config,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

class TopologyGatingV2SmallConfigTest(unittest.TestCase):
    def test_full_matrix_has_three_seeds_and_fixed_budget(self):
        config = resolved_config(DEFAULT_CONFIG, "full")
        self.assertEqual(config["seeds"], [20260807, 20260808, 20260809])
        self.assertEqual(config["training"]["epochs"], 10)
        self.assertEqual(config["training"]["samples_per_epoch"], 131072)
        self.assertEqual(config["gate_validation_shots_per_basis"], 8192)
        self.assertEqual(config["model"]["out_channels"], 4)
        self.assertEqual(config["training"]["checkpoint_epochs"], [2, 4, 6, 8, 10])
        self.assertEqual(config["training"]["batch_size"], 512)
        self.assertEqual(config["training"]["accumulate_steps"], 2)
        total_test_shots = (
            len(config["seeds"])
            * 2
            * sum(config["test_shots_per_basis"].values())
        )
        self.assertEqual(total_test_shots, 450000)

    def test_smoke_override_is_deep_merged(self):
        config = resolved_config(DEFAULT_CONFIG, "smoke")
        self.assertEqual(config["seeds"], [20260807])
        self.assertEqual(config["training"]["epochs"], 1)
        self.assertEqual(config["training"]["optimizer"]["name"], "Lion")
        self.assertEqual(config["gate"]["defaults"]["max_candidates"], 8)
        self.assertEqual(config["gate"]["defaults"]["max_combination_actions"], 4)
        self.assertEqual(config["test_shots_per_basis"]["t0"], 256)
        self.assertEqual(config["training"]["checkpoint_validation_shots"], 64)

    def test_measurement_drift_changes_only_measurement_rates(self):
        config = resolved_config(DEFAULT_CONFIG, "full")
        base = _noise_mapping(config, "t0")
        drift = _noise_mapping(config, "measurement_drift_1p5")
        changed = {key for key in base if base[key] != drift[key]}
        self.assertEqual(changed, {"p_meas_X", "p_meas_Z"})
        self.assertEqual(drift["p_meas_X"], 0.015)

    def test_resume_rejects_identity_mismatch(self):
        config = resolved_config(DEFAULT_CONFIG, "smoke")
        identity = _identity(config, "smoke", unit="test")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "artifact.txt"
            artifact.write_text("ok", encoding="utf-8")
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps({**identity, "status": "completed", "artifact_sha256": {str(artifact): sha256_file(artifact)}}), encoding="utf-8"
            )
            self.assertTrue(_resume_complete(manifest, identity, [artifact], True))
            artifact.write_text("tampered", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                _resume_complete(manifest, identity, [artifact], True)
            artifact.write_text("ok", encoding="utf-8")
            wrong = dict(identity)
            wrong["git_commit"] = "different"
            with self.assertRaises(RuntimeError):
                _resume_complete(manifest, wrong, [artifact], True)

    def test_incomplete_aggregate_writes_fixed_outputs(self):
        config = resolved_config(DEFAULT_CONFIG, "smoke")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            aggregate(config, SimpleNamespace(mode="smoke"), output)
            aggregate_dir = output / "smoke" / "aggregate"
            decision = json.loads(
                (aggregate_dir / "decisions.json").read_text(encoding="utf-8")
            )
            self.assertEqual(decision["status"], "INCOMPLETE")
            for name in ("summary.csv", "paired_deltas.csv", "ablation.csv", "results.md"):
                self.assertTrue((aggregate_dir / name).exists())

    def test_launch_script_dry_run_lists_all_smoke_stages(self):
        script = REPO_ROOT / "code/scripts/patent/run_topology_gating_v2_small.sh"
        result = subprocess.run(
            ["bash", str(script)],
            cwd=REPO_ROOT,
            env={
                "PATH": "/usr/bin:/bin",
                "PATENT_DRY_RUN": "1",
                "PATENT_MODE": "smoke",
                "PATENT_GPUS": "0,1,2,3",
            },
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("train seed=20260807 loss=bce", result.stdout)
        self.assertIn("train seed=20260807 loss=topology", result.stdout)
        self.assertIn("evaluate seed=20260807", result.stdout)
        self.assertIn("aggregate", result.stdout)
        self.assertIn("gpu=0", result.stdout)
        script_text = script.read_text(encoding="utf-8")
        self.assertIn("timeout --signal=TERM --kill-after=60", script_text)
        self.assertIn("aggregate_partial", script_text)

if __name__ == "__main__":
    unittest.main()
