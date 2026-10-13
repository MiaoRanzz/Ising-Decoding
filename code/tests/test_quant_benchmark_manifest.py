# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import tempfile
import unittest
from pathlib import Path

from benchmarks.quant_t0_common import sha256_file, write_json
from scripts.build_tensorrt_engine import (
    _matching_existing_report,
    build_command,
    layer_precision_counts,
)


class TestQuantBenchmarkManifest(unittest.TestCase):

    def test_sha_and_atomic_json(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.bin"
            source.write_bytes(b"source")
            report = root / "report.json"
            write_json(report, {"sha": sha256_file(source)})
            self.assertTrue(report.is_file())
            self.assertFalse(report.with_suffix(".json.tmp").exists())

    def test_precision_count_parsing(self):
        payload = {
            "Layers": [
                {"LayerPrecision": "Int8", "Outputs": [{"Format/Datatype": "Half"}]},
                {"LayerPrecision": "Float"},
            ]
        }
        self.assertEqual(layer_precision_counts(payload), {"int8": 1, "fp16": 1, "fp32": 1})

    def test_build_command_has_distinct_precision_flags(self):
        base = dict(
            trtexec=Path("/opt/tensorrt/bin/trtexec"),
            onnx=Path("model.onnx"),
            input_width=720,
            min_batch=1,
            opt_batch=32,
            max_batch=64,
        )
        fp32 = build_command(
            argparse.Namespace(**base, precision="fp32"), Path("a.engine"), Path("a.json")
        )
        fp16 = build_command(
            argparse.Namespace(**base, precision="fp16"), Path("b.engine"), Path("b.json")
        )
        int8 = build_command(
            argparse.Namespace(**base, precision="int8"), Path("c.engine"), Path("c.json")
        )
        self.assertIn("--noTF32", fp32)
        self.assertIn("--fp16", fp16)
        self.assertIn("--stronglyTyped", int8)

    def test_existing_hash_mismatch_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "model.onnx"
            engine = root / "model.engine"
            report_path = root / "report.json"
            source.write_bytes(b"onnx")
            engine.write_bytes(b"engine")
            args = argparse.Namespace(
                engine=engine,
                report=report_path,
                precision="fp16",
                input_width=720,
                min_batch=1,
                opt_batch=32,
                max_batch=64,
            )
            write_json(
                report_path,
                {
                    "source_onnx_sha256": "wrong",
                    "engine_sha256": sha256_file(engine),
                    "precision": "fp16",
                    "input_width": 720,
                    "min_batch": 1,
                    "opt_batch": 32,
                    "max_batch": 64,
                },
            )
            with self.assertRaises(FileExistsError):
                _matching_existing_report(args, sha256_file(source))


if __name__ == "__main__":
    unittest.main()
