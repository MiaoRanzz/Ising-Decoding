#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Evaluate released pre-decoders on downloaded Google Willow QEC data."""

from __future__ import annotations

import argparse
import os
import shlex
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from scripts.qadapt_example_utils import (  # noqa: E402
    add_common_inference_args,
    checkpoint_specs,
    parse_gpus,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        default=Path(
            "benchmarks/google_qec/google_105Q_surface_code_d3_d5_d7"
        ),
    )
    parser.add_argument("--distances", nargs="+", type=int, default=[3, 5, 7])
    parser.add_argument("--rounds", nargs="+", type=int, default=[13])
    add_common_inference_args(
        parser,
        default_output_dir=Path("outputs/examples/released_models/willow"),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_path = args.output_dir / "results.json"
    if args.resume and output_path.is_file():
        print(f"[resume] output exists: {output_path}")
        return 0
    selected_gpus = parse_gpus(args.gpus)
    bases = ["X", "Z"] if args.basis == "both" else [args.basis]
    specs = checkpoint_specs(args)
    command_preview = [
        str(args.python),
        "-m",
        "scripts.providers.google_qec_decoder_benchmark",
        "--benchmark-root",
        str(args.benchmark_root),
        "--distances",
        *(str(value) for value in args.distances),
        "--rounds",
        *(str(value) for value in args.rounds),
        "--bases",
        *bases,
        "--models",
        *(spec.name for spec in specs),
        "--max-shots",
        str(args.num_samples),
        "--batch-size",
        str(args.batch_size),
        "--latency-shots",
        str(args.latency_num_samples),
        "--output",
        str(output_path),
    ]
    if args.dry_run:
        print(
            f"[dry-run] gpu={selected_gpus[0]} seed={args.seed} "
            + shlex.join(command_preview)
        )
        for spec in specs:
            print(
                f"[dry-run] model {spec.name}: "
                f"model_id={spec.model_id} checkpoint={spec.checkpoint}"
            )
        return 0

    os.environ["CUDA_VISIBLE_DEVICES"] = selected_gpus[0]
    from scripts.providers import google_qec_decoder_benchmark as benchmark

    benchmark.DEFAULT_MODELS = {
        spec.name: benchmark.BenchmarkModel(
            spec.name,
            spec.model_id,
            spec.checkpoint,
        )
        for spec in specs
    }
    benchmark.DEFAULT_BENCHMARK_ROOT = args.benchmark_root
    benchmark_args = command_preview[3:]
    return benchmark.main(benchmark_args)


if __name__ == "__main__":
    raise SystemExit(main())
