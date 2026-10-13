#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Extract formal EWC penalty telemetry from TensorBoard event files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tensorboard.backend.event_processing import event_accumulator


REPO_ROOT = Path(__file__).resolve().parents[4]
RATIO_TAG = "EWC/lambda_penalty_to_data_ratio"
PENALTY_TAG = "Loss/ewc_penalty"


def summarize_values(values: list[float]) -> dict:
    if not values:
        raise ValueError("scalar series is empty")
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "first": float(array[0]),
        "last": float(array[-1]),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def read_stage(stage_dir: Path) -> dict:
    events = sorted((stage_dir / "tensorboard").glob("events.out.tfevents.*"))
    if len(events) != 1:
        raise ValueError(f"expected one root TensorBoard event file in {stage_dir}, found {len(events)}")
    accumulator = event_accumulator.EventAccumulator(str(events[0]), size_guidance={"scalars": 0})
    accumulator.Reload()
    tags = set(accumulator.Tags()["scalars"])
    missing = {RATIO_TAG, PENALTY_TAG} - tags
    if missing:
        raise ValueError(f"missing EWC scalar tags in {events[0]}: {sorted(missing)}")
    return {
        "event_file": str(events[0]),
        "penalty_to_data_ratio": summarize_values([item.value for item in accumulator.Scalars(RATIO_TAG)]),
        "unscaled_ewc_penalty": summarize_values([item.value for item in accumulator.Scalars(PENALTY_TAG)]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for run_id in args.run_id:
        run_root = REPO_ROOT / "outputs" / "qewc_v3" / "formal" / run_id
        training = json.loads((run_root / "run_manifest.json").read_text(encoding="utf-8"))
        if training["method"] != "ewc":
            raise ValueError(f"{run_id} is not an EWC run")
        for stage, task in enumerate(training["order"], 1):
            rows.append({"run_id": run_id, "checkpoint": f"C{stage}", "task": task, **read_stage(run_root / f"stage_{stage}_{task}")})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(f"[QEWC v3] EWC telemetry: {args.output}")


if __name__ == "__main__":
    main()
