#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Benchmark one TensorRT pre-decoder engine against its PyTorch checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from benchmarks.quant_t0_common import (  # noqa: E402
    latency_summary,
    paired_noninferiority,
    sha256_file,
    write_json,
)
from evaluation.logical_error_rate import (  # noqa: E402
    PreDecoderMemoryEvalModule,
    _build_stab_maps,
)
from model.checkpoint_loader import load_model_checkpoint  # noqa: E402
from scripts.build_tensorrt_engine import layer_precision_counts  # noqa: E402
from scripts.paired_inference_compare import build_matcher, make_dataset  # noqa: E402
from scripts.validate_deployment_parity import build_cfg  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--errors-output", type=Path, required=True)
    parser.add_argument("--layers-output", type=Path, required=True)
    parser.add_argument("--model-id", type=int, required=True)
    parser.add_argument(
        "--config-name", default="examples/qadapt/config_qadapt_t0_base"
    )
    parser.add_argument("--distance", type=int, default=9)
    parser.add_argument("--n-rounds", type=int, default=9)
    parser.add_argument("--basis", choices=("X", "Z"), required=True)
    parser.add_argument("--expected-precision", choices=("fp32", "fp16", "int8"), required=True)
    parser.add_argument("--num-samples", type=int, default=65536)
    parser.add_argument("--seed", type=int, default=45678)
    parser.add_argument("--eval-batch-size", type=int, default=2048)
    parser.add_argument("--batch-sizes", default="1,8,32,128,512,2048")
    parser.add_argument("--warmup-iterations", type=int, default=200)
    parser.add_argument("--latency-iterations", type=int, default=1000)
    parser.add_argument("--absolute-ler-margin", type=float, default=0.001)
    parser.add_argument("--relative-ler-margin", type=float, default=0.03)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def parse_batch_sizes(value: str) -> list[int]:
    try:
        values = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("batch sizes must be comma-separated integers") from exc
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("batch sizes must be positive")
    return list(dict.fromkeys(values))


class TensorRTRunner:

    def __init__(self, path: Path, device: torch.device) -> None:
        import tensorrt as trt

        self.trt = trt
        self.device = device
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"TensorRT failed to deserialize {path}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("TensorRT failed to create an execution context")
        self.input_name = "dets"
        self.output_name = "L_and_residual_dets"
        names = {
            self.engine.get_tensor_name(index)
            for index in range(self.engine.num_io_tensors)
        }
        if names != {self.input_name, self.output_name}:
            raise AssertionError(f"unexpected TensorRT I/O tensors: {sorted(names)}")

    def inspect(self) -> tuple[object, dict[str, int]]:
        inspector = self.engine.create_engine_inspector()
        if inspector is None:
            raise RuntimeError("TensorRT did not provide an engine inspector")
        raw = inspector.get_engine_information(self.trt.LayerInformationFormat.JSON)
        payload = json.loads(raw)
        return payload, layer_precision_counts(payload)

    def allocate(self, batch: torch.Tensor) -> torch.Tensor:
        if batch.dtype != torch.uint8 or not batch.is_cuda:
            raise TypeError("TensorRT input must be a CUDA uint8 tensor")
        if not self.context.set_input_shape(self.input_name, tuple(batch.shape)):
            raise RuntimeError(f"TensorRT rejected input shape {tuple(batch.shape)}")
        shape = tuple(self.context.get_tensor_shape(self.output_name))
        return torch.empty(shape, dtype=torch.uint8, device=self.device)

    def execute(self, batch: torch.Tensor, output: torch.Tensor) -> None:
        self.context.set_tensor_address(self.input_name, int(batch.data_ptr()))
        self.context.set_tensor_address(self.output_name, int(output.data_ptr()))
        stream = torch.cuda.current_stream(self.device)
        if not self.context.execute_async_v3(stream_handle=stream.cuda_stream):
            raise RuntimeError("TensorRT execute_async_v3 returned false")


def logical_error_mask(output: np.ndarray, observables: np.ndarray, matcher) -> np.ndarray:
    predecoder_logical = output[:, 0].astype(np.int32)
    residual = np.ascontiguousarray(output[:, 1:], dtype=np.uint8)
    global_logical = np.asarray(matcher.decode_batch(residual), dtype=np.int32).reshape(-1)
    predicted = (predecoder_logical + global_logical) % 2
    return predicted != observables.reshape(-1)


def evaluate_pytorch(pipeline, detectors, observables, matcher, batch_size, device) -> np.ndarray:
    chunks = []
    with torch.no_grad():
        for start in range(0, len(detectors), batch_size):
            batch = torch.as_tensor(
                detectors[start : start + batch_size], dtype=torch.uint8, device=device
            ).contiguous()
            actual = pipeline(batch).cpu().numpy()
            chunks.append(
                logical_error_mask(actual, observables[start : start + len(batch)], matcher)
            )
    return np.concatenate(chunks)


def evaluate_tensorrt(runner, detectors, observables, matcher, batch_size, device) -> tuple[np.ndarray, float]:
    chunks = []
    started = time.perf_counter()
    with torch.no_grad():
        for start in range(0, len(detectors), batch_size):
            batch = torch.as_tensor(
                detectors[start : start + batch_size], dtype=torch.uint8, device=device
            ).contiguous()
            output = runner.allocate(batch)
            runner.execute(batch, output)
            actual = output.cpu().numpy()
            chunks.append(
                logical_error_mask(actual, observables[start : start + len(batch)], matcher)
            )
    torch.cuda.synchronize(device)
    return np.concatenate(chunks), time.perf_counter() - started


def benchmark_backend(runner, detectors, batch_size, warmup, iterations, device) -> dict:
    batch = torch.as_tensor(detectors[:batch_size], dtype=torch.uint8, device=device).contiguous()
    output = runner.allocate(batch)
    for _ in range(warmup):
        runner.execute(batch, output)
    torch.cuda.synchronize(device)

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    for start, end in zip(starts, ends):
        start.record()
        runner.execute(batch, output)
        end.record()
    torch.cuda.synchronize(device)
    values = [start.elapsed_time(end) for start, end in zip(starts, ends)]
    summary = latency_summary(values)
    total_s = float(sum(values)) / 1000.0
    return {
        "batch_size": batch_size,
        "request_latency": summary,
        "us_per_shot_p50": summary["p50_ms"] * 1000.0 / batch_size,
        "shots_per_second": batch_size * iterations / total_s,
        "iterations": iterations,
    }


def benchmark_decoder(runner, detectors, matcher, batch_size, iterations, device) -> dict:
    host = np.ascontiguousarray(detectors[:batch_size], dtype=np.uint8)
    values_ms = []
    for _ in range(iterations):
        started = time.perf_counter_ns()
        batch = torch.as_tensor(host, dtype=torch.uint8, device=device).contiguous()
        output = runner.allocate(batch)
        runner.execute(batch, output)
        actual = output.cpu().numpy()
        matcher.decode_batch(np.ascontiguousarray(actual[:, 1:], dtype=np.uint8))
        values_ms.append((time.perf_counter_ns() - started) / 1e6)
    summary = latency_summary(values_ms)
    return {
        "batch_size": batch_size,
        "request_latency": summary,
        "us_per_shot_p50": summary["p50_ms"] * 1000.0 / batch_size,
        "shots_per_second": batch_size / (summary["mean_ms"] / 1000.0),
        "iterations": iterations,
    }


def main() -> None:
    args = parse_args()
    if args.num_samples <= 0 or args.eval_batch_size <= 0:
        raise ValueError("sample and evaluation batch sizes must be positive")
    if args.warmup_iterations < 0 or args.latency_iterations <= 0:
        raise ValueError("warmup must be non-negative and latency iterations positive")
    batch_sizes = parse_batch_sizes(args.batch_sizes)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("quantized benchmark requires CUDA")
    if not args.checkpoint.is_file() or not args.engine.is_file():
        raise FileNotFoundError("checkpoint and engine must exist")

    cfg_args = SimpleNamespace(
        config_name=args.config_name,
        model_id=args.model_id,
        distance=args.distance,
        n_rounds=args.n_rounds,
        basis=args.basis,
        num_samples=args.num_samples,
        batch_size=args.eval_batch_size,
    )
    cfg = build_cfg(cfg_args)
    dataset_started = time.perf_counter()
    dataset = make_dataset(cfg, args.seed)
    dataset_generation_s = time.perf_counter() - dataset_started
    matcher_started = time.perf_counter()
    matcher, num_observables = build_matcher(dataset)
    matcher_build_s = time.perf_counter() - matcher_started
    values = np.asarray(dataset.dets_and_obs, dtype=np.uint8)
    detectors = np.ascontiguousarray(values[:, :-num_observables])
    observables = np.ascontiguousarray(values[:, -num_observables:])

    model_started = time.perf_counter()
    distributed = SimpleNamespace(rank=0, world_size=1, device=device)
    model = load_model_checkpoint(
        cfg,
        checkpoint=args.checkpoint.resolve(),
        model_id=args.model_id,
        distributed=distributed,
    ).eval()
    maps = _build_stab_maps(args.distance, getattr(cfg, "rotation", "XV"))
    pipeline = PreDecoderMemoryEvalModule(model, cfg, maps, device).eval()
    model_load_s = time.perf_counter() - model_started

    engine_started = time.perf_counter()
    runner = TensorRTRunner(args.engine, device)
    engine_load_s = time.perf_counter() - engine_started
    layers, precision_counts = runner.inspect()
    if args.expected_precision == "int8" and precision_counts.get("int8", 0) == 0:
        raise AssertionError("expected INT8 engine has no INT8 inspector evidence")
    write_json(args.layers_output, layers)

    pytorch_errors = evaluate_pytorch(
        pipeline, detectors, observables, matcher, args.eval_batch_size, device
    )
    tensorrt_errors, trt_decode_s = evaluate_tensorrt(
        runner, detectors, observables, matcher, args.eval_batch_size, device
    )
    noninferiority = paired_noninferiority(
        tensorrt_errors,
        pytorch_errors,
        absolute_margin=args.absolute_ler_margin,
        relative_margin=args.relative_ler_margin,
        alpha=args.alpha,
    )

    args.errors_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.errors_output,
        samples=np.asarray([len(pytorch_errors)], dtype=np.int64),
        pytorch=np.packbits(pytorch_errors),
        tensorrt=np.packbits(tensorrt_errors),
    )

    backend_rows = []
    decoder_rows = []
    previous_throughput = None
    stagnant = 0
    resource_limit = None
    for batch_size in batch_sizes:
        if batch_size > len(detectors):
            break
        try:
            backend = benchmark_backend(
                runner,
                detectors,
                batch_size,
                args.warmup_iterations,
                args.latency_iterations,
                device,
            )
            decoder = benchmark_decoder(
                runner,
                detectors,
                matcher,
                batch_size,
                args.latency_iterations,
                device,
            )
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            resource_limit = {"batch_size": batch_size, "reason": "CUDA OOM"}
            break
        backend_rows.append(backend)
        decoder_rows.append(decoder)
        throughput = float(backend["shots_per_second"])
        if previous_throughput is not None:
            gain = throughput / previous_throughput - 1.0
            stagnant = stagnant + 1 if gain < 0.05 else 0
        previous_throughput = throughput
        if stagnant >= 2:
            break

    torch.cuda.synchronize(device)
    properties = torch.cuda.get_device_properties(device)
    report = {
        "status": (
            "VALID_PASS"
            if noninferiority["noninferiority_passed"]
            else "ACCURACY_FAIL"
        ),
        "basis": args.basis,
        "distance": args.distance,
        "n_rounds": args.n_rounds,
        "model_id": args.model_id,
        "samples": len(detectors),
        "seed": args.seed,
        "expected_precision": args.expected_precision,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "engine": str(args.engine.resolve()),
        "engine_sha256": sha256_file(args.engine),
        "engine_layer_precision_counts": precision_counts,
        "layers_output": str(args.layers_output.resolve()),
        "layers_output_sha256": sha256_file(args.layers_output),
        "errors_output": str(args.errors_output.resolve()),
        "errors_output_sha256": sha256_file(args.errors_output),
        "resolved_config_sha256": hashlib.sha256(str(cfg).encode("utf-8")).hexdigest(),
        "noninferiority": noninferiority,
        "one_time_s": {
            "dataset_generation": dataset_generation_s,
            "matcher_build": matcher_build_s,
            "model_load": model_load_s,
            "engine_load": engine_load_s,
        },
        "simulation": {
            "tensorrt_decode_s": trt_decode_s,
            "steady_state_shots_per_s": len(detectors) / trt_decode_s,
            "generation_plus_decode_shots_per_s": len(detectors)
            / (dataset_generation_s + matcher_build_s + trt_decode_s),
        },
        "backend": backend_rows,
        "decoder": decoder_rows,
        "resource_limit": resource_limit,
        "software": {
            "python": sys.version,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "tensorrt": runner.trt.__version__,
        },
        "gpu": {
            "name": properties.name,
            "total_memory": properties.total_memory,
            "compute_capability": [properties.major, properties.minor],
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        },
    }
    write_json(args.report, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
