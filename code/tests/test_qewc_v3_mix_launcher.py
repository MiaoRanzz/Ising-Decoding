# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from scripts.experiments.qewc_v3 import run_mix_total


class TestQewcV3MixLauncher(unittest.TestCase):

    def test_four_equal_epochs_emit_all_fraction_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stage0 = root / "stage0.pt"
            config = root / "mix.yaml"
            stage0.write_bytes(b"stage0")
            config.write_text("config\n", encoding="utf-8")
            args = SimpleNamespace(
                stage0=stage0,
                config_path=config,
                run_root=root / "run",
                steps_per_task=7,
                num_continual_tasks=4,
                total_steps=None,
                method="mix_total",
                batch_size=16,
                validation_samples=32,
                seed=20260907,
                gpu="0",
                python=Path("/env/bin/python"),
                config_name="mix",
                run_name="qewc_v3/test_mix",
                resume=False,
                dry_run=False,
            )

            def fake_run(_command, *, cwd, env, check):
                self.assertEqual(env["PREDECODER_TRAIN_EPOCHS"], "4")
                self.assertEqual(env["PREDECODER_TRAIN_SAMPLES"], str(7 * 16))
                self.assertTrue(check)
                models = args.run_root / "models"
                models.mkdir(parents=True)
                for index in range(1, 5):
                    (models / f"HTnet.0.{index}.pt").write_bytes(str(index).encode())

            with patch.object(run_mix_total, "parse_args", return_value=args), patch.object(
                run_mix_total.subprocess, "run", side_effect=fake_run
            ):
                run_mix_total.main()

            manifest = json.loads(
                (args.run_root / "run_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["total_optimizer_steps"], 28)
            self.assertEqual(
                [manifest["checkpoints"][f"M{i}"]["fraction"] for i in range(5)],
                [0.0, 0.25, 0.5, 0.75, 1.0],
            )
            self.assertTrue(manifest["endpoint"]["path"].endswith("HTnet.0.4.pt"))


if __name__ == "__main__":
    unittest.main()
