# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from omegaconf import OmegaConf

from scripts.experiments.qewc_v3.audit_formal_runs import audit_run


class TestQewcV3FormalAudit(unittest.TestCase):

    def test_valid_oracle_manifest_passes_and_hash_mismatch_fails(self):
        with TemporaryDirectory() as raw_root:
            checkpoint = Path(raw_root) / "model.pt"
            checkpoint.write_bytes(b"checkpoint")
            digest = hashlib.sha256(b"checkpoint").hexdigest()
            training = {
                "method": "seq",
                "order": ["T1"],
                "total_optimizer_steps": 1000,
                "samples_processed": 512000,
                "endpoints": {
                    "C0": {"path": str(checkpoint), "sha256": digest},
                    "C1": {"path": str(checkpoint), "sha256": digest},
                },
            }
            experiment = OmegaConf.create({
                "pilot": {"selected_optimizer_steps_per_task": 1000, "batch_size_per_gpu": 512},
                "ewc": {"lambda": 4.0},
                "replay": {"ratio": 0.1, "capacity_samples": 594},
            })
            self.assertEqual(audit_run("O1", training, experiment)["status"], "passed")
            training["endpoints"]["C1"]["sha256"] = "wrong"
            report = audit_run("O1", training, experiment)
            self.assertEqual(report["status"], "failed")
            self.assertIn("checkpoint hash mismatch: C1", report["errors"])


    def test_mix_exposure_uses_five_task_exposure_budget(self):
        with TemporaryDirectory() as raw_root:
            checkpoint = Path(raw_root) / "model.pt"
            checkpoint.write_bytes(b"checkpoint")
            digest = hashlib.sha256(b"checkpoint").hexdigest()
            entry = {"path": str(checkpoint), "sha256": digest}
            training = {
                "method": "mix_exposure",
                "total_optimizer_steps": 5000,
                "samples_processed": 2560000,
                "checkpoints": {f"M{i}": dict(entry) for i in range(5)},
            }
            experiment = OmegaConf.create({
                "pilot": {"selected_optimizer_steps_per_task": 1000, "batch_size_per_gpu": 512},
                "ewc": {"lambda": 4.0},
                "replay": {"ratio": 0.1, "capacity_samples": 594},
            })
            self.assertEqual(audit_run("S7", training, experiment)["status"], "passed")


if __name__ == "__main__":
    unittest.main()
