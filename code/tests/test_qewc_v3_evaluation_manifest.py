# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
from pathlib import Path
import tempfile
import unittest

from scripts.experiments.qewc_v3.evaluate_realistic_matrix import (
    Checkpoint,
    write_summary,
)


class TestQewcV3EvaluationManifest(unittest.TestCase):

    def test_carries_exact_test_stream_hashes_into_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "model.pt"
            config = root / "task.yaml"
            checkpoint.write_bytes(b"model")
            config.write_text("task\n", encoding="utf-8")
            streams = {
                "X": {"seed": 7, "sha256": "a" * 64},
                "Z": {"seed": 8, "sha256": "b" * 64},
            }
            (root / "T0.json").write_text(
                json.dumps(
                    {
                        "summary": [{"method": "C0", "ler_avg": 0.125}],
                        "paired_comparisons": [],
                        "sample_streams": streams,
                    }
                ),
                encoding="utf-8",
            )
            args = argparse.Namespace(
                model_id=111,
                checkpoint=[Checkpoint("C0", checkpoint)],
                tasks={"T0": config},
                output_dir=root,
                track="test",
                seed=7,
                num_samples=16,
                batch_size=8,
            )

            write_summary(args)

            manifest = json.loads(
                (root / "evaluation_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["test_streams"]["T0"], streams)


if __name__ == "__main__":
    unittest.main()
