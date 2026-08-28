#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Export one pre-decoder pipeline to ONNX and fail on backend disagreement."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnx
import onnxruntime as ort
import torch
from omegaconf import OmegaConf

CODE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_ROOT.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from data.factory import DatapipeFactory  # noqa: E402
from evaluation.logical_error_rate import (  # noqa: E402
    PreDecoderMemoryEvalModule,
    _build_stab_maps,
)
from model.checkpoint_loader import load_model_checkpoint  # noqa: E402
from scripts.config_paths import config_path  # noqa: E402
from workflows.config_validator import (  # noqa: E402
    apply_public_defaults_and_model,
    validate_public_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--safetensors", type=Path, required=True)
    parser.add_argument("--onnx-output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--model-id", type=int, default=111)
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
    parser.add_argument(
        "--onnx-provider",
        choices=("CPUExecutionProvider", "CUDAExecutionProvider"),
        default="CPUExecutionProvider",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_cfg(args: argparse.Namespace):
    cfg = OmegaConf.load(config_path(args.config_name))
    cfg.model_id = args.model_id
    cfg.distance = args.distance
    cfg.n_rounds = args.n_rounds
    cfg.workflow.task = "inference"
    spec = validate_public_config(cfg)
    cfg = apply_public_defaults_and_model(cfg, spec)
    cfg.test.distance = args.distance
    cfg.test.n_rounds = args.n_rounds
    cfg.test.meas_basis_test = args.basis
    cfg.test.num_samples = args.num_samples
    cfg.test.batch_size = args.batch_size
    cfg.test.dataloader_num_workers = 0
    return cfg


def load_models(args: argparse.Namespace, cfg, device: torch.device):
    distributed = SimpleNamespace(rank=0, world_size=1, device=device)
    pt_model = load_model_checkpoint(
        cfg,
        checkpoint=args.checkpoint.resolve(),
        model_id=args.model_id,
        distributed=distributed,
    )
    safe_model = load_model_checkpoint(
        cfg,
        checkpoint=args.safetensors.resolve(),
        model_id=args.model_id,
        distributed=distributed,
    )
    for model in (pt_model, safe_model):
        model.eval()
        if device.type == "cuda":
            model.to(memory_format=torch.channels_last_3d)
    return pt_model, safe_model


def make_detectors(cfg, seed: int) -> np.ndarray:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    dataset = DatapipeFactory.create_datapipe_inference(cfg)
    num_observables = int(dataset.circ.stim_circuit.num_observables)
    values = np.asarray(dataset.dets_and_obs, dtype=np.uint8)
    if num_observables != 1:
        raise AssertionError(f"expected one observable, got {num_observables}")
    return np.ascontiguousarray(values[:, :-num_observables])


def compare_torch_pipelines(
    pt_pipeline: torch.nn.Module,
    safe_pipeline: torch.nn.Module,
    detectors: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, int]:
    outputs = []
    mismatched = 0
    with torch.no_grad():
        for start in range(0, len(detectors), batch_size):
            batch = torch.as_tensor(
                detectors[start : start + batch_size], dtype=torch.uint8, device=device
            )
            pt_output = pt_pipeline(batch)
            safe_output = safe_pipeline(batch)
            mismatched += int(torch.count_nonzero(pt_output != safe_output).item())
            outputs.append(pt_output.cpu().numpy())
    return np.concatenate(outputs, axis=0), mismatched


def export_onnx(
    pipeline: torch.nn.Module,
    example: np.ndarray,
    output_path: Path,
    device: torch.device,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    example_tensor = torch.as_tensor(example, dtype=torch.uint8, device=device)
    torch.onnx.export(
        pipeline,
        example_tensor,
        str(output_path),
        opset_version=18,
        external_data=False,
        input_names=["dets"],
        output_names=["L_and_residual_dets"],
        dynamic_axes={
            "dets": {0: "batch"},
            "L_and_residual_dets": {0: "batch"},
        },
        do_constant_folding=True,
        dynamo=False,
    )
    onnx.checker.check_model(onnx.load(str(output_path)))


def compare_onnx(
    onnx_path: Path,
    detectors: np.ndarray,
    expected: np.ndarray,
    batch_size: int,
    provider: str,
) -> tuple[int, list[str]]:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(onnx_path), sess_options=options, providers=[provider]
    )
    providers = session.get_providers()
    if not providers or providers[0] != provider:
        raise RuntimeError(f"requested provider was not activated: {providers}")
    mismatched = 0
    for start in range(0, len(detectors), batch_size):
        actual = session.run(
            ["L_and_residual_dets"],
            {"dets": detectors[start : start + batch_size]},
        )[0]
        reference = expected[start : start + batch_size]
        mismatched += int(np.count_nonzero(actual != reference))
    return mismatched, providers


def main() -> None:
    args = parse_args()
    if args.num_samples <= 0 or args.batch_size <= 0:
        raise ValueError("num-samples and batch-size must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA device is unavailable")

    cfg = build_cfg(args)
    pt_model, safe_model = load_models(args, cfg, device)
    maps = _build_stab_maps(args.distance, getattr(cfg, "rotation", "XV"))
    pt_pipeline = PreDecoderMemoryEvalModule(pt_model, cfg, maps, device).eval()
    safe_pipeline = PreDecoderMemoryEvalModule(safe_model, cfg, maps, device).eval()
    detectors = make_detectors(cfg, args.seed)

    expected, safe_mismatches = compare_torch_pipelines(
        pt_pipeline, safe_pipeline, detectors, args.batch_size, device
    )
    if safe_mismatches:
        raise AssertionError(
            f"SafeTensors pipeline differs from .pt at {safe_mismatches} output elements"
        )

    export_onnx(
        pt_pipeline,
        detectors[: min(args.batch_size, len(detectors))],
        args.onnx_output,
        device,
    )
    onnx_mismatches, providers = compare_onnx(
        args.onnx_output,
        detectors,
        expected,
        args.batch_size,
        args.onnx_provider,
    )
    if onnx_mismatches:
        raise AssertionError(
            f"ONNX pipeline differs from .pt at {onnx_mismatches} output elements"
        )

    report = {
        "status": "PASS",
        "basis": args.basis,
        "distance": args.distance,
        "n_rounds": args.n_rounds,
        "samples": len(detectors),
        "detector_width": int(detectors.shape[1]),
        "output_width": int(expected.shape[1]),
        "pt_vs_safetensors_mismatched_elements": safe_mismatches,
        "pt_vs_onnx_mismatched_elements": onnx_mismatches,
        "onnx_providers": providers,
        "onnx_sha256": sha256(args.onnx_output),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
