# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for external-sample routing in paired_inference_compare.py."""

import argparse
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

_repo_code = Path(__file__).resolve().parent.parent
if str(_repo_code) not in sys.path:
    sys.path.insert(0, str(_repo_code))

from scripts.paired_inference_compare import (
    ModelSpec,
    build_cfg,
    paired_error_comparison,
    parse_args,
    resolve_stim_samples_dir,
)


class TestPairedInferenceExternalSamples(unittest.TestCase):

    def test_parser_accepts_external_sample_directory(self):
        argv = [
            "paired_inference_compare.py",
            "--model",
            "r9x:111:/tmp/r9x.pt",
            "--stim-samples-dir",
            "/tmp/qpu-dets",
        ]
        with patch.object(sys, "argv", argv):
            args = parse_args()
        self.assertEqual(args.stim_samples_dir, "/tmp/qpu-dets")

    def test_explicit_external_directory_takes_priority_over_environment(self):
        args = argparse.Namespace(stim_samples_dir="/tmp/qpu-dets")
        with patch.dict(
            "os.environ",
            {"PREDECODER_STIM_SAMPLES_DIR": "/tmp/stale-qpu-dets"},
        ):
            resolved = resolve_stim_samples_dir(args)
        self.assertEqual(resolved, Path("/tmp/qpu-dets"))

    def test_build_cfg_routes_relative_external_directory_to_datapipe(self):
        args = argparse.Namespace(
            config_name="experiments/external_qpu/config_domestic_fast_opt_stfusion_r9_x",
            distance=3,
            n_rounds=9,
            num_samples=128,
            latency_num_samples=16,
            batch_size=32,
            num_workers=0,
            stim_samples_dir="qpu_samples/run_001",
        )
        model = ModelSpec(
            name="r9x",
            model_id=111,
            checkpoint=Path("/tmp/r9x.pt"),
        )

        cfg = build_cfg(args, model, basis="X")

        expected = _repo_code.parent / "qpu_samples" / "run_001"
        self.assertEqual(Path(cfg.test.stim_samples_dir), expected)
        self.assertEqual(cfg.distance, 3)
        self.assertEqual(cfg.n_rounds, 9)
        self.assertEqual(cfg.test.meas_basis_test, "X")

    def test_paired_error_comparison_preserves_shot_pairing(self):
        comparison = paired_error_comparison(
            "ising_fast",
            [False, True, True, False, True],
            "r9_x",
            [False, False, True, True, False],
            basis="X",
        )
        self.assertEqual(comparison["both_error"], 1)
        self.assertEqual(comparison["a_only_error"], 2)
        self.assertEqual(comparison["b_only_error"], 1)
        self.assertEqual(comparison["neither_error"], 1)
        self.assertAlmostEqual(comparison["ler_delta_a_minus_b"], 0.2)


if __name__ == "__main__":
    unittest.main()
