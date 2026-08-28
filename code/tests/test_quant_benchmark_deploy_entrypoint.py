# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.experiments.quant_t0_compare.run import Job
from scripts.experiments.quant_t0_compare.run_deploy import ensure_reference_checkpoint


class TestDeployEntrypoint(unittest.TestCase):

    def test_creates_reference_parent_before_conversion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job = Job(
                model="ising_fast",
                model_id=1,
                epoch=87,
                checkpoint=root / "model.pt",
                reference_checkpoint=None,
                basis="X",
                precision="fp32",
                calibration_samples=None,
                calibration_seed=12345,
            )
            with patch(
                "scripts.experiments.quant_t0_compare.run_deploy._ensure_reference_checkpoint",
                return_value=root / "reference.safetensors",
            ):
                ensure_reference_checkpoint(job, root / "outputs", dry_run=False)
            self.assertTrue(
                (root / "outputs/artifacts/ising_fast/e87").is_dir()
            )


if __name__ == "__main__":
    unittest.main()
