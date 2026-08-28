#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run a serialized TensorRT pre-decoder engine on fixed T0 inputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import tensorrt as trt
import torch

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from evaluation.logical_error_rate import (  # noqa: E402
    PreDecoderMemoryEvalModule,
    _build_stab_maps,
)
from model.checkpoint_loader import load_model_checkpoint  # noqa: E402
from scripts.paired_inference_compare import (  # noqa: E402
    build_matcher,
    make_dataset,
    paired_error_comparison,
)
from scripts.validate_deployment_parity import (  # noqa: E402
    build_cfg,
    sha256,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
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
    return parser.parse_args()


def torch_dtype(dtype: trt.DataType) -> torch.dtype:
    mapping = {
        trt.uint8: torch.uint8,
        trt.int8: torch.int8,
        trt.int32: torch.int32,
        trt.float16: torch.float16,
        trt.float32: torch.float32,
    }
    if dtype not in mapping:
        raise TypeError(f"unsupported TensorRT dtype: {dtype}")
    return mapping[dtype]


def load_engine(path: Path):
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(path.read_bytes())
    if engine is None:
        raise RuntimeError(f"TensorRT failed to deserialize {path}")
    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError("TensorRT failed to create an execution context")
    return runtime, engine, context


def run_engine(context, engine, batch: torch.Tensor) -> torch.Tensor:
    input_names = [
        engine.get_tensor_name(index)
        for index in range(engine.num_io_tensors)
        if engine.get_tensor_mode(engine.get_tensor_name(index)) == trt.TensorIOMode.INPUT
    ]
    output_names = [
        engine.get_tensor_name(index)
        for index in range(engine.num_io_tensors)
        if engine.get_tensor_mode(engine.get_tensor_name(index)) == trt.TensorIOMode.OUTPUT
    ]
    if input_names != ["dets"] or output_names != ["L_and_residual_dets"]:
        raise AssertionError(
            f"unexpected TensorRT IO tensors: inputs={input_names}, outputs={output_names}"
        )
    input_name = input_names[0]
    output_name = output_names[0]
    expected_input_dtype = torch_dtype(engine.get_tensor_dtype(input_name))
    if batch.dtype != expected_input_dtype:
        raise TypeError(
            f"engine input dtype is {expected_input_dtype}, batch dtype is {batch.dtype}"
        )
    if not context.set_input_shape(input_name, tuple(batch.shape)):
        raise RuntimeError(f"TensorRT rejected input shape {tuple(batch.shape)}")
    output_shape = tuple(context.get_tensor_shape(output_name))
    output = torch.empty(
        output_shape,
        dtype=torch_dtype(engine.get_tensor_dtype(output_name)),
        device=batch.device,
    )
    context.set_tensor_address(input_name, int(batch.data_ptr()))
    context.set_tensor_address(output_name, int(output.data_ptr()))
    stream = torch.cuda.current_stream(batch.device)
    if not context.execute_async_v3(stream_handle=stream.cuda_stream):
        raise RuntimeError("TensorRT execute_async_v3 returned false")
    stream.synchronize()
    return output


def logical_error_mask(output: torch.Tensor, observables: np.ndarray, matcher) -> np.ndarray:
    values = output.cpu().numpy()
    predecoder_logical = values[:, 0].astype(np.int32)
    residual = np.ascontiguousarray(values[:, 1:], dtype=np.uint8)
    global_logical = np.asarray(matcher.decode_batch(residual), dtype=np.int32).reshape(-1)
    predicted = (predecoder_logical + global_logical) % 2
    return predicted != observables.reshape(-1)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("TensorRT validation requires CUDA")
    cfg = build_cfg(args)
    distributed = SimpleNamespace(rank=0, world_size=1, device=device)
    model = load_model_checkpoint(
        cfg,
        checkpoint=args.checkpoint.resolve(),
        model_id=args.model_id,
        distributed=distributed,
    ).eval()
    maps = _build_stab_maps(args.distance, getattr(cfg, "rotation", "XV"))
    pipeline = PreDecoderMemoryEvalModule(model, cfg, maps, device).eval()
    dataset = make_dataset(cfg, args.seed)
    matcher, num_observables = build_matcher(dataset)
    values = np.asarray(dataset.dets_and_obs, dtype=np.uint8)
    detectors = np.ascontiguousarray(values[:, :-num_observables])
    observables = np.ascontiguousarray(values[:, -num_observables:])
    runtime, engine, context = load_engine(args.engine)

    mismatched_elements = 0
    mismatched_shots = 0
    total_elements = 0
    pytorch_error_chunks = []
    tensorrt_error_chunks = []
    with torch.no_grad():
        for start in range(0, len(detectors), args.batch_size):
            batch = torch.as_tensor(
                detectors[start : start + args.batch_size],
                dtype=torch.uint8,
                device=device,
            ).contiguous()
            expected = pipeline(batch)
            actual = run_engine(context, engine, batch)
            mismatch = actual != expected
            mismatched_elements += int(torch.count_nonzero(mismatch).item())
            mismatched_shots += int(torch.count_nonzero(mismatch.any(dim=1)).item())
            total_elements += int(mismatch.numel())
            batch_observables = observables[start : start + len(batch)]
            pytorch_error_chunks.append(
                logical_error_mask(expected, batch_observables, matcher)
            )
            tensorrt_error_chunks.append(
                logical_error_mask(actual, batch_observables, matcher)
            )

    pytorch_errors = np.concatenate(pytorch_error_chunks)
    tensorrt_errors = np.concatenate(tensorrt_error_chunks)
    paired_ler = paired_error_comparison(
        "tensorrt",
        tensorrt_errors,
        "pytorch",
        pytorch_errors,
        basis=args.basis,
    )
    ci_low, ci_high = paired_ler["ler_delta_ci95_normal"]
    ler_gate_passed = bool(ci_low <= 0.0 <= ci_high)

    report = {
        "status": "PASS" if ler_gate_passed else "FAIL_LER",
        "basis": args.basis,
        "distance": args.distance,
        "n_rounds": args.n_rounds,
        "samples": len(detectors),
        "mismatched_elements": mismatched_elements,
        "total_elements": total_elements,
        "mismatched_element_rate": mismatched_elements / total_elements,
        "mismatched_shots": mismatched_shots,
        "mismatched_shot_rate": mismatched_shots / len(detectors),
        "pytorch_logical_errors": int(np.count_nonzero(pytorch_errors)),
        "pytorch_ler": float(np.mean(pytorch_errors)),
        "tensorrt_logical_errors": int(np.count_nonzero(tensorrt_errors)),
        "tensorrt_ler": float(np.mean(tensorrt_errors)),
        "paired_ler": paired_ler,
        "ler_gate_passed": ler_gate_passed,
        "engine_sha256": sha256(args.engine),
        "tensorrt_version": trt.__version__,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    del context, engine, runtime
    if not ler_gate_passed:
        raise AssertionError(
            f"TensorRT LER delta confidence interval excludes zero: {ci_low}, {ci_high}"
        )


if __name__ == "__main__":
    main()
