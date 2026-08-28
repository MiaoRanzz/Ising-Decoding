#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""INT8-quantize one pre-decoder ONNX graph with fixed T0 calibration data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import modelopt.onnx.quantization as mq
import onnx

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from scripts.validate_deployment_parity import (  # noqa: E402
    build_cfg,
    make_detectors,
    sha256,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx-input", type=Path, required=True)
    parser.add_argument("--onnx-output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--model-id", type=int, default=111)
    parser.add_argument(
        "--config-name", default="examples/qadapt/config_qadapt_t0_base"
    )
    parser.add_argument("--distance", type=int, default=9)
    parser.add_argument("--n-rounds", type=int, default=9)
    parser.add_argument("--basis", choices=("X", "Z"), required=True)
    parser.add_argument("--num-samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=12345)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = build_cfg(args)
    calibration_data = make_detectors(cfg, args.seed)
    args.onnx_output.parent.mkdir(parents=True, exist_ok=True)
    mq.quantize(
        onnx_path=str(args.onnx_input),
        quantize_mode="int8",
        calibration_data={"dets": calibration_data},
        output_path=str(args.onnx_output),
    )
    if not args.onnx_output.is_file():
        raise RuntimeError("ModelOpt returned without writing the INT8 ONNX file")
    onnx.checker.check_model(onnx.load(str(args.onnx_output)))
    report = {
        "status": "PASS",
        "basis": args.basis,
        "calibration_samples": len(calibration_data),
        "source_onnx_sha256": sha256(args.onnx_input),
        "int8_onnx_sha256": sha256(args.onnx_output),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
