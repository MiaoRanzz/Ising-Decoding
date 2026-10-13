#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Apply the frozen QEWC-v3 optimizer-step pilot selection rule."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path


TASK_INDEX = {f"T{i}": i for i in range(5)}


@dataclass(frozen=True)
class Candidate:
    name: str
    steps: int
    train_loss: float
    validation_loss: float
    wall_seconds: float


def parse_candidate(value: str) -> Candidate:
    fields = value.split(":")
    if len(fields) != 5:
        raise argparse.ArgumentTypeError(
            "--candidate must be NAME:STEPS:TRAIN_LOSS:VALIDATION_LOSS:WALL_SECONDS"
        )
    name, steps, train_loss, validation_loss, wall_seconds = fields
    return Candidate(name, int(steps), float(train_loss), float(validation_loss), float(wall_seconds))


def paired_row(payload: dict, task: str, candidate: str, baseline: str) -> dict:
    for row in payload["paired_comparisons"][task]:
        if row["basis"] != "both":
            continue
        if row["candidate"] == candidate and row["baseline"] == baseline:
            return row
        if row["candidate"] == baseline and row["baseline"] == candidate:
            reversed_row = dict(row)
            reversed_row["candidate"] = candidate
            reversed_row["baseline"] = baseline
            reversed_row["delta_ler"] = -float(row["delta_ler"])
            reversed_row["ci95_low"] = -float(row["ci95_high"])
            reversed_row["ci95_high"] = -float(row["ci95_low"])
            return reversed_row
    raise ValueError(f"missing paired comparison for {task}: {candidate}, {baseline}")


def select(
    payload: dict,
    candidates: list[Candidate],
    plateaued: set[str],
    max_projected_gpu_days: float,
) -> dict:
    candidates = sorted(candidates, key=lambda item: item.steps)
    if len({item.name for item in candidates}) != len(candidates):
        raise ValueError("candidate names must be unique")
    rows = payload["ler_rows"]
    records = []
    for item in candidates:
        if item.name not in rows:
            raise ValueError(f"missing evaluation row: {item.name}")
        projected_days = item.wall_seconds * 4.0 / 86400.0
        records.append(
            {
                "name": item.name,
                "steps": item.steps,
                "train_loss": item.train_loss,
                "validation_loss": item.validation_loss,
                "wall_seconds": item.wall_seconds,
                "projected_four_stage_gpu_days": projected_days,
                "plateaued": item.name in plateaued,
                "within_gpu_budget": projected_days < max_projected_gpu_days,
                "ler": dict(zip(TASK_INDEX, map(float, rows[item.name]))),
            }
        )
    eligible = [item for item in records if item["plateaued"] and item["within_gpu_budget"]]
    if not eligible:
        raise ValueError("no plateaued candidate is within the GPU budget")

    lowest = eligible[0]
    lowest["comparison_to_lowest"] = None
    improvements = []
    for item in eligible[1:]:
        t1 = paired_row(payload, "T1", item["name"], lowest["name"])
        t0 = paired_row(payload, "T0", item["name"], lowest["name"])
        item["comparison_to_lowest"] = {
            "T1": t1,
            "T0": t0,
            "significant_current_task_gain": float(t1["ci95_high"]) < 0.0,
            "measurable_historical_task_change": not (
                float(t0["ci95_low"]) <= 0.0 <= float(t0["ci95_high"])
            ),
        }
        if item["comparison_to_lowest"]["significant_current_task_gain"]:
            improvements.append(item)
    selected = min(
        improvements,
        key=lambda item: (item["ler"]["T1"], item["steps"]),
        default=lowest,
    )
    return {
        "kind": "optimizer_steps_per_task",
        "selection_rule": (
            "lowest plateaued in-budget candidate unless a larger candidate has a paired "
            "95-percent T1 LER interval strictly below zero"
        ),
        "max_projected_four_stage_gpu_days": max_projected_gpu_days,
        "selected": {"name": selected["name"], "steps": selected["steps"]},
        "larger_candidate_has_significant_t1_gain": bool(improvements),
        "candidates": records,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluation_manifest", type=Path)
    parser.add_argument("--candidate", action="append", type=parse_candidate, required=True)
    parser.add_argument("--plateaued", action="append", required=True)
    parser.add_argument("--max-projected-gpu-days", type=float, default=2.4)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.evaluation_manifest.read_text(encoding="utf-8"))
    result = select(payload, args.candidate, set(args.plateaued), args.max_projected_gpu_days)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"[QEWC v3] selected {result['selected']}; wrote {args.output}")


if __name__ == "__main__":
    main()
