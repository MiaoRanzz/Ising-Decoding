# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hydra-backed replay and latency commands for the standalone decoder."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter
from typing import Sequence

from hydra import compose, initialize_config_dir
import numpy as np
from omegaconf import DictConfig
import stim

from .config import DecoderConfig, deployment_root
from .decoder import SurfaceDecoderCore


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ising-decode")
    parser.add_argument("command", choices=("replay", "benchmark"))
    parser.add_argument("input", type=Path, help="Stim .dets file")
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Hydra overrides, for example basis=X device=cuda",
    )
    return parser


def _compose_config(overrides: Sequence[str]) -> DictConfig:
    config_dir = deployment_root() / "conf"
    with initialize_config_dir(
        version_base="1.3",
        config_dir=str(config_dir),
    ):
        return compose(config_name="config", overrides=list(overrides))


def _read_samples(
    path: Path,
    *,
    num_detectors: int,
    num_observables: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Stim sample file not found: {path}")
    detectors, observables = stim.read_shot_data_file(
        path=path,
        format="dets",
        num_detectors=num_detectors,
        num_observables=num_observables,
        separate_observables=True,
    )
    return (
        np.ascontiguousarray(detectors, dtype=np.uint8),
        np.ascontiguousarray(observables, dtype=np.uint8),
    )


def _percentiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
    }


def _run_replay(
    decoder: SurfaceDecoderCore,
    detectors: np.ndarray,
    observables: np.ndarray,
    limit: int,
) -> dict[str, object]:
    count = min(len(detectors), limit)
    predictions: list[int] = []
    predecoder: list[int] = []
    pymatching_predictions: list[int] = []
    for row in detectors[:count]:
        result = decoder.decode_detectors(row)
        predictions.append(result.logical_flip)
        predecoder.append(result.predecoder_flip)
        pymatching_predictions.append(result.pymatching_flip)
    expected = observables[:count, 0].astype(np.uint8)
    predicted = np.asarray(predictions, dtype=np.uint8)
    errors = int(np.count_nonzero(predicted != expected))
    return {
        "samples": count,
        "logical_errors": errors,
        "logical_error_rate": float(errors / count) if count else 0.0,
        "predictions": predictions,
        "predecoder_predictions": predecoder,
        "pymatching_predictions": pymatching_predictions,
    }


def _run_benchmark(
    decoder: SurfaceDecoderCore,
    detectors: np.ndarray,
    *,
    warmup: int,
    samples: int,
) -> dict[str, object]:
    if len(detectors) == 0:
        raise ValueError("benchmark input contains no shots")
    for index in range(warmup):
        decoder.decode_detectors(detectors[index % len(detectors)])

    stage_names = tuple(asdict(decoder.decode_detectors(detectors[0]).stage_latencies))
    timings: dict[str, list[float]] = {name: [] for name in stage_names}
    for index in range(samples):
        result = decoder.decode_detectors(detectors[index % len(detectors)])
        for name, value in asdict(result.stage_latencies).items():
            timings[name].append(float(value))
    return {
        "warmup_samples": warmup,
        "measured_samples": samples,
        "latency_us": {
            name.removesuffix("_us"): _percentiles(values)
            for name, values in timings.items()
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    cfg = _compose_config(args.overrides)
    decoder_cfg = DecoderConfig.from_omegaconf(cfg)

    file_start = perf_counter()
    detectors, observables = _read_samples(
        args.input.expanduser().resolve(),
        num_detectors=decoder_cfg.num_detectors,
        num_observables=int(cfg.input_num_observables),
    )
    file_load_ms = (perf_counter() - file_start) * 1_000.0

    init_start = perf_counter()
    decoder = SurfaceDecoderCore(decoder_cfg)
    decoder_init_ms = (perf_counter() - init_start) * 1_000.0

    if args.command == "replay":
        payload = _run_replay(
            decoder,
            detectors,
            observables,
            int(cfg.replay_samples),
        )
    else:
        payload = _run_benchmark(
            decoder,
            detectors,
            warmup=int(cfg.benchmark.warmup),
            samples=int(cfg.benchmark.samples),
        )
    report = {
        "command": args.command,
        "basis": decoder_cfg.basis,
        "device": str(decoder.device),
        "distance": decoder_cfg.distance,
        "n_rounds": decoder_cfg.n_rounds,
        "input": str(args.input.expanduser().resolve()),
        "file_load_ms": file_load_ms,
        "decoder_init_ms": decoder_init_ms,
        **payload,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if cfg.output_path:
        output_path = Path(str(cfg.output_path)).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
