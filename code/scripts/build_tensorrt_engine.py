#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build one TensorRT engine with a fail-fast, hash-linked report."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from benchmarks.quant_t0_common import sha256_file, write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--layer-info", type=Path, required=True)
    parser.add_argument("--build-log", type=Path, required=True)
    parser.add_argument("--precision", choices=("fp32", "fp16", "int8"), required=True)
    parser.add_argument("--input-width", type=int, default=720)
    parser.add_argument("--min-batch", type=int, default=1)
    parser.add_argument("--opt-batch", type=int, default=2048)
    parser.add_argument("--max-batch", type=int, default=8192)
    parser.add_argument("--trtexec", type=Path, default=Path("/opt/tensorrt/bin/trtexec"))
    return parser.parse_args()


def _precision_strings(value: Any, *, parent_key: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            found.extend(_precision_strings(child, parent_key=str(key)))
    elif isinstance(value, list):
        for child in value:
            found.extend(_precision_strings(child, parent_key=parent_key))
    elif isinstance(value, str) and any(
        token in parent_key.lower() for token in ("precision", "datatype", "format")
    ):
        found.extend(
            match.lower()
            for match in re.findall(r"(?i)\b(?:int8|uint8|fp16|fp32|float|half)\b", value)
        )
    return found


def layer_precision_counts(payload: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    aliases = {"float": "fp32", "half": "fp16"}
    for value in _precision_strings(payload):
        normalized = aliases.get(value, value)
        counts[normalized] = counts.get(normalized, 0) + 1
    return counts


def build_command(args: argparse.Namespace, temporary_engine: Path, temporary_layers: Path) -> list[str]:
    shape = lambda batch: f"dets:{batch}x{args.input_width}"
    command = [
        str(args.trtexec),
        f"--onnx={args.onnx}",
        f"--saveEngine={temporary_engine}",
        f"--minShapes={shape(args.min_batch)}",
        f"--optShapes={shape(args.opt_batch)}",
        f"--maxShapes={shape(args.max_batch)}",
        "--profilingVerbosity=detailed",
        f"--exportLayerInfo={temporary_layers}",
        "--skipInference",
    ]
    if args.precision == "fp32":
        command.append("--noTF32")
    elif args.precision == "fp16":
        command.append("--fp16")
    else:
        command.append("--stronglyTyped")
    return command


def _validate_args(args: argparse.Namespace) -> None:
    if not args.onnx.is_file():
        raise FileNotFoundError(args.onnx)
    if not args.trtexec.is_file():
        raise FileNotFoundError(args.trtexec)
    if args.input_width <= 0:
        raise ValueError("input-width must be positive")
    if not 1 <= args.min_batch <= args.opt_batch <= args.max_batch:
        raise ValueError("batch profile must satisfy 1 <= min <= opt <= max")


def _matching_existing_report(args: argparse.Namespace, source_hash: str) -> dict[str, Any] | None:
    if not args.engine.exists() and not args.report.exists():
        return None
    if not args.engine.is_file() or not args.report.is_file():
        raise FileExistsError("partial TensorRT artifact set exists; refusing to overwrite")
    report = json.loads(args.report.read_text(encoding="utf-8"))
    expected = {
        "source_onnx_sha256": source_hash,
        "precision": args.precision,
        "input_width": args.input_width,
        "min_batch": args.min_batch,
        "opt_batch": args.opt_batch,
        "max_batch": args.max_batch,
        "engine_sha256": sha256_file(args.engine),
    }
    mismatches = {
        key: (report.get(key), value)
        for key, value in expected.items()
        if report.get(key) != value
    }
    if mismatches:
        raise FileExistsError(f"existing TensorRT artifacts do not match: {mismatches}")
    return report


def main() -> None:
    args = parse_args()
    _validate_args(args)
    source_hash = sha256_file(args.onnx)
    existing = _matching_existing_report(args, source_hash)
    if existing is not None:
        print(json.dumps(existing, indent=2, sort_keys=True))
        return

    for path in (args.engine, args.report, args.layer_info, args.build_log):
        path.parent.mkdir(parents=True, exist_ok=True)
    temporary_engine = args.engine.with_suffix(args.engine.suffix + ".tmp")
    temporary_layers = args.layer_info.with_suffix(args.layer_info.suffix + ".tmp")
    command = build_command(args, temporary_engine, temporary_layers)
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    build_output = completed.stdout + completed.stderr
    args.build_log.write_text(build_output, encoding="utf-8")
    if completed.returncode != 0 or not temporary_engine.is_file():
        write_json(
            args.report,
            {
                "status": "INVALID",
                "reason": "trtexec build failed",
                "returncode": completed.returncode,
                "source_onnx_sha256": source_hash,
                "precision": args.precision,
                "command": command,
            },
        )
        raise RuntimeError(f"trtexec build failed; see {args.build_log}")
    if not temporary_layers.is_file():
        raise RuntimeError("trtexec did not write layer information")

    layer_payload = json.loads(temporary_layers.read_text(encoding="utf-8"))
    precision_counts = layer_precision_counts(layer_payload)
    if args.precision == "int8" and precision_counts.get("int8", 0) == 0:
        raise RuntimeError("INT8 engine report contains no INT8 layer/tensor evidence")

    temporary_engine.replace(args.engine)
    temporary_layers.replace(args.layer_info)
    report = {
        "status": "PASS",
        "source_onnx": str(args.onnx.resolve()),
        "source_onnx_sha256": source_hash,
        "engine": str(args.engine.resolve()),
        "engine_sha256": sha256_file(args.engine),
        "layer_info": str(args.layer_info.resolve()),
        "layer_info_sha256": sha256_file(args.layer_info),
        "precision": args.precision,
        "layer_precision_counts": precision_counts,
        "input_width": args.input_width,
        "min_batch": args.min_batch,
        "opt_batch": args.opt_batch,
        "max_batch": args.max_batch,
        "command": command,
    }
    write_json(args.report, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
