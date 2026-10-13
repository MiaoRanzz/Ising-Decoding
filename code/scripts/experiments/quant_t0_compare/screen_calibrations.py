#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build and screen the remaining 1024/4096-sample INT8 calibrations."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[3]
REPO_ROOT = CODE_ROOT.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from scripts.experiments.quant_t0_compare import run as workflow  # noqa: E402
from scripts.experiments.quant_t0_compare.screen_existing import (  # noqa: E402
    run_job,
    write_summary,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan", type=Path, default=Path(__file__).with_name("experiment.yaml")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=REPO_ROOT / "outputs/quant_t0_compare_v1"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--only-model", choices=("htnet", "ising_fast"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    plan = workflow.load_plan(args.plan)
    jobs = [
        job
        for job in workflow.expand_jobs(
            plan, smoke=False, only_model=args.only_model
        )
        if job.precision == "int8" and int(job.calibration_samples) > 256
    ]
    random.Random(int(plan["schedule_seed"]) + 1).shuffle(jobs)
    for job in jobs:
        run_job(job, plan, args.output_dir, args.device, args.dry_run)
    if not args.dry_run:
        write_summary(args.output_dir)


if __name__ == "__main__":
    main()
