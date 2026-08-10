#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run released pre-decoders on the fixed training-axis OOD grid."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from scripts.experiments.unknown_noise.generate_unknown_axismix_grid_u1p2_5p0_configs import (  # noqa: E402
    write_axismix_grid_configs,
)
from scripts.qadapt_example_utils import (  # noqa: E402
    InferenceJob,
    add_common_inference_args,
    build_paired_command,
    parse_gpus,
    run_jobs,
)


PAPER_DISTANCES = (7, 9)
PAPER_MULTIPLIERS = (1.2, 1.5, 2.0, 2.5, 3.0)


def parse_distances(value: str) -> list[int]:
    result = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not result or result != sorted(set(result)):
        raise argparse.ArgumentTypeError(
            "distances must be a non-empty, increasing comma-separated list"
        )
    return result


def parse_multipliers(value: str) -> list[float]:
    result = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not result or result != sorted(set(result)) or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError(
            "multipliers must be a non-empty, increasing comma-separated list "
            "of positive numbers"
        )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--distances",
        type=parse_distances,
        default=list(PAPER_DISTANCES),
        help="Comma-separated distances; defaults to the paper's d=7,9 grid.",
    )
    parser.add_argument("--n-rounds", type=int, default=9)
    parser.add_argument(
        "--multipliers",
        type=parse_multipliers,
        default=list(PAPER_MULTIPLIERS),
        help="Comma-separated OOD multipliers; defaults to the paper's 1.2--3.0 grid.",
    )
    parser.add_argument(
        "--generated-config-dir",
        type=Path,
        default=Path("outputs/generated_configs/ood"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("outputs/generated_configs/ood/manifest.json"),
    )
    add_common_inference_args(
        parser,
        default_output_dir=Path("outputs/examples/released_models/ood"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _, manifest = write_axismix_grid_configs(
        base_config="conf/examples/qadapt/config_qadapt_t0_base.yaml",
        output_dir=args.generated_config_dir,
        manifest=args.manifest,
        grid_multipliers=args.multipliers,
    )
    jobs = []
    for distance in args.distances:
        for environment in manifest["environments"]:
            config_file = args.generated_config_dir / environment["config_filename"]
            label = (
                f"d{distance}_{environment['env_key']}_"
                f"{environment['multiplier_key']}"
            )
            output_path = args.output_dir / f"d{distance}" / f"{label}.json"
            jobs.append(
                InferenceJob(
                    label=label,
                    command=build_paired_command(
                        args,
                        config_file=config_file,
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
