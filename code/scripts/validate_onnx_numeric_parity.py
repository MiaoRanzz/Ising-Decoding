#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Record non-zero ONNX binary-output parity without relabeling it as a pass."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from benchmarks.quant_t0_common import sha256_file, write_json  # noqa: E402
from evaluation.logical_error_rate import PreDecoderMemoryEvalModule, _build_stab_maps  # noqa: E402
from scripts.validate_deployment_parity import (  # noqa: E402
    build_cfg,
    compare_torch_pipelines,
    export_onnx,
    load_models,
    make_detectors,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference-checkpoint", type=Path, required=True)
    parser.add_argument("--onnx-output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--model-id", type=int, required=True)
    parser.add_argument(
        "--config-name", default="examples/qadapt/config_qadapt_t0_base"
    )
    parser.add_argument("--distance", type=int, default=9)
    parser.add_argument("--n-rounds", type=int, default=9)
    parser.add_argument("--basis", choices=("X", "Z"), required=True)
    parser.add_argument("--num-samples", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device)
    cfg = build_cfg(args)
    load_args = argparse.Namespace(
        checkpoint=args.checkpoint,
        safetensors=args.reference_checkpoint,
        model_id=args.model_id,
    )
    pt_model, reference_model = load_models(load_args, cfg, device)
    maps = _build_stab_maps(args.distance, getattr(cfg, "rotation", "XV"))
    pt_pipeline = PreDecoderMemoryEvalModule(pt_model, cfg, maps, device).eval()
    reference_pipeline = PreDecoderMemoryEvalModule(reference_model, cfg, maps, device).eval()
    detectors = make_detectors(cfg, args.seed)
    expected, reference_mismatches = compare_torch_pipelines(
        pt_pipeline, reference_pipeline, detectors, args.batch_size, device
    )
    if reference_mismatches:
        raise AssertionError(
            f"reference checkpoint differs from .pt at {reference_mismatches} elements"
        )
    export_onnx(
        pt_pipeline,
        detectors[: min(args.batch_size, len(detectors))],
        args.onnx_output,
        device,
    )

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(args.onnx_output),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    mismatched_elements = 0
    mismatched_shots = 0
    for start in range(0, len(detectors), args.batch_size):
        actual = session.run(
            ["L_and_residual_dets"],
            {"dets": detectors[start : start + args.batch_size]},
        )[0]
        reference = expected[start : start + len(actual)]
        mismatch = actual != reference
        mismatched_elements += int(np.count_nonzero(mismatch))
        mismatched_shots += int(np.count_nonzero(np.any(mismatch, axis=1)))
    total_elements = int(expected.size)
    status = "PASS" if mismatched_elements == 0 else "NUMERIC_PARITY_WARN"
    report = {
        "status": status,
        "basis": args.basis,
        "distance": args.distance,
        "n_rounds": args.n_rounds,
        "samples": len(detectors),
        "pt_vs_reference_mismatched_elements": reference_mismatches,
        "pt_vs_onnx_mismatched_elements": mismatched_elements,
        "pt_vs_onnx_mismatched_element_rate": mismatched_elements / total_elements,
        "pt_vs_onnx_mismatched_shots": mismatched_shots,
        "pt_vs_onnx_mismatched_shot_rate": mismatched_shots / len(detectors),
        "onnx_provider": "CPUExecutionProvider",
        "onnx_sha256": sha256_file(args.onnx_output),
    }
    write_json(args.report, report)
    write_json(
        args.manifest,
        {
            "status": status,
            "model": args.model,
            "epoch": args.epoch,
            "basis": args.basis,
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "reference_checkpoint_sha256": sha256_file(args.reference_checkpoint),
            "onnx_sha256": sha256_file(args.onnx_output),
            "parity_report_sha256": sha256_file(args.report),
        },
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
