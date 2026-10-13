#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Apply the frozen QEWC-v3 first-switch hyperparameter selection rules."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path


TASKS = ("T0", "T1", "T2", "T3", "T4")


@dataclass(frozen=True)
class Candidate:
    name: str
    value: float


def parse_candidate(value: str) -> Candidate:
    name, separator, raw_value = value.partition(":")
    if not separator or not name or not raw_value:
        raise argparse.ArgumentTypeError("--candidate must be NAME:VALUE")
    return Candidate(name=name, value=float(raw_value))


def paired_half_width(payload: dict, task: str, first: str, second: str) -> float:
    for row in payload["paired_comparisons"][task]:
        if row.get("basis") != "both":
            continue
        if {row["candidate"], row["baseline"]} == {first, second}:
            return (float(row["ci95_high"]) - float(row["ci95_low"])) / 2.0
    raise ValueError(f"missing paired both-basis comparison for {task}: {first}, {second}")


def select(payload: dict, kind: str, baseline: str, candidates: list[Candidate]) -> dict:
    rows = payload["ler_rows"]
    if baseline not in rows:
        raise ValueError(f"missing baseline row: {baseline}")
    if len(rows[baseline]) != len(TASKS):
        raise ValueError("baseline LER row must have five tasks")
    primary_tasks = ("T0", "T1") if kind == "ewc" else TASKS
    task_index = {task: index for index, task in enumerate(TASKS)}
    baseline_t1 = float(rows[baseline][task_index["T1"]])
    records = []
    for candidate in candidates:
        if candidate.name not in rows or len(rows[candidate.name]) != len(TASKS):
            raise ValueError(f"missing five-task candidate row: {candidate.name}")
        values = [float(value) for value in rows[candidate.name]]
        records.append(
            {
                "name": candidate.name,
                "value": candidate.value,
                "ler": dict(zip(TASKS, values)),
                "primary_score": sum(values[task_index[task]] for task in primary_tasks)
                / len(primary_tasks),
                "five_task_macro_ler": sum(values) / len(values),
                "eligible": values[task_index["T1"]] <= 1.05 * baseline_t1,
                "dominated_on_t0_t1": False,
            }
        )
    eligible = [record for record in records if record["eligible"]]
    if not eligible:
        raise ValueError("no candidate satisfies the T1 5-percent plasticity constraint")

    if kind == "ewc":
        for record in eligible:
            for other in eligible:
                if other is record:
                    continue
                no_worse = all(other["ler"][task] <= record["ler"][task] for task in ("T0", "T1"))
                strictly_better = any(other["ler"][task] < record["ler"][task] for task in ("T0", "T1"))
                if no_worse and strictly_better:
                    record["dominated_on_t0_t1"] = True
                    break
        eligible = [record for record in eligible if not record["dominated_on_t0_t1"]]

    best = min(eligible, key=lambda record: (record["primary_score"], record["value"]))
    tied = []
    for record in eligible:
        uncertainty = 0.0 if record is best else max(
            paired_half_width(payload, task, record["name"], best["name"])
            for task in primary_tasks
        )
        record["paired_tie_half_width"] = uncertainty
        if record["primary_score"] - best["primary_score"] <= uncertainty:
            tied.append(record)
    selected = min(tied, key=lambda record: record["value"])
    return {
        "kind": kind,
        "baseline": baseline,
        "primary_tasks": list(primary_tasks),
        "plasticity_limit_relative": 0.05,
        "selected": {"name": selected["name"], "value": selected["value"]},
        "best_raw_primary": {"name": best["name"], "value": best["value"]},
        "candidates": records,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluation_manifest", type=Path)
    parser.add_argument("--kind", choices=("ewc", "replay"), required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", action="append", type=parse_candidate, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.evaluation_manifest.read_text(encoding="utf-8"))
    result = select(payload, args.kind, args.baseline, args.candidate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"[QEWC v3] selected {result['selected']}; wrote {args.output}")


if __name__ == "__main__":
    main()
