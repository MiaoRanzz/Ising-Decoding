#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run released pre-decoders on the five T0-T4 simulated noise tasks."""

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


TASK_BY_ID = {
    f"T{index}": (task_key, config_name)
    for index, (task_key, config_name) in enumerate(TASK_CONFIGS)
}


def parse_distances(value: str) -> list[int]:
    result = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not result or result != sorted(set(result)):
        raise argparse.ArgumentTypeError(
            "distances must be a non-empty, increasing comma-separated list"
        )
    return result


def parse_tasks(value: str) -> list[str]:
    result = [item.strip().upper() for item in value.split(",") if item.strip()]
    if not result or len(result) != len(set(result)):
        raise argparse.ArgumentTypeError(
            "tasks must be a non-empty comma-separated subset of T0,T1,T2,T3,T4"
        )
    unknown = [item for item in result if item not in TASK_BY_ID]
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown task(s): {','.join(unknown)}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--distances",
        type=parse_distances,
        default=[9],
        help=(
            "Comma-separated distances. Use 7,9 with --tasks T0 for the "
            "paper's mapped-noise geometry; the default is release coverage at d=9."
        ),
    )
    parser.add_argument(
        "--tasks",
        type=parse_tasks,
        default=list(TASK_BY_ID),
        help="Comma-separated task subset; defaults to T0,T1,T2,T3,T4.",
    )
    parser.add_argument("--n-rounds", type=int, default=9)
    add_common_inference_args(
        parser,
        default_output_dir=Path("outputs/examples/released_models/t0_t4"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    jobs = []
    for distance in args.distances:
        for task_id in args.tasks:
            task_key, config_name = TASK_BY_ID[task_id]
            label = f"d{distance}_{task_key}"
            output_path = args.output_dir / f"d{distance}" / f"{task_key}.json"
            jobs.append(
                InferenceJob(
                    label=label,
                    command=build_paired_command(
                        args,
                        config_name=config_name,
                        output_path=output_path,
                        distance=distance,
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
