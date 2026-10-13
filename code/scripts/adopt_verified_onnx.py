#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adopt an already verified ONNX artifact into a hash-linked experiment directory."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from benchmarks.quant_t0_common import sha256_file, write_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-onnx", type=Path, required=True)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference-checkpoint", type=Path, required=True)
    parser.add_argument("--onnx-output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--basis", choices=("X", "Z"), required=True)
    args = parser.parse_args()

    source_report = json.loads(args.source_report.read_text(encoding="utf-8"))
    if source_report.get("status") != "PASS":
        raise AssertionError("source ONNX report did not pass")
    if source_report.get("basis") != args.basis:
        raise AssertionError("source ONNX report basis mismatch")
    if source_report.get("pt_vs_safetensors_mismatched_elements") != 0:
        raise AssertionError("source PyTorch/SafeTensors parity was not exact")
    if source_report.get("pt_vs_onnx_mismatched_elements") != 0:
        raise AssertionError("source PyTorch/ONNX parity was not exact")
    source_hash = sha256_file(args.source_onnx)
    if source_report.get("onnx_sha256") != source_hash:
        raise AssertionError("source ONNX hash does not match its report")

    args.onnx_output.parent.mkdir(parents=True, exist_ok=True)
    if args.onnx_output.exists() and sha256_file(args.onnx_output) != source_hash:
        raise FileExistsError("destination ONNX exists with a different hash")
    if not args.onnx_output.exists():
        shutil.copy2(args.source_onnx, args.onnx_output)
    shutil.copy2(args.source_report, args.report_output)
    write_json(
        args.manifest_output,
        {
            "status": "PASS",
            "model": args.model,
            "epoch": args.epoch,
            "basis": args.basis,
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "reference_checkpoint_sha256": sha256_file(args.reference_checkpoint),
            "onnx_sha256": source_hash,
            "adopted_from_onnx": str(args.source_onnx.resolve()),
            "adopted_from_report": str(args.source_report.resolve()),
        },
    )


if __name__ == "__main__":
    main()
