#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run QAdapt seq+EWC on the five T0-T4 simulated noise tasks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from scripts.qadapt_example_utils import (  # noqa: E402
    InferenceJob,
    TASK_CONFIGS,
    add_common_inference_args,
    build_paired_command,
    parse_gpus,
    run_jobs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distance", type=int, default=9)
    parser.add_argument("--n-rounds", type=int, default=9)
    add_common_inference_args(
        parser,
        default_output_dir=Path("outputs/examples/qadapt/t0_t4"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    jobs = []
    for task_key, config_name in TASK_CONFIGS:
        output_path = args.output_dir / f"{task_key}.json"
        jobs.append(
            InferenceJob(
                label=task_key,
                command=build_paired_command(
                    args,
                    config_name=config_name,
                    output_path=output_path,
                    distance=args.distance,
                    n_rounds=args.n_rounds,
                ),
                output_path=output_path,
            )
        )
    run_jobs(
        jobs,
        gpus=parse_gpus(args.gpus),
        parallelism=args.parallelism,
        resume=args.resume,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
