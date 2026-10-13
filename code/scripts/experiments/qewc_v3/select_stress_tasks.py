#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Freeze Stress tasks and a greedy high-conflict order from pilot metrics."""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path


FAMILIES = ("measurement", "cnot", "idle", "z_bias")
TASK_BY_FAMILY = dict(zip(FAMILIES, ("T1", "T2", "T3", "T4")))


def mean_pairwise(matrix: dict, names: tuple[str, ...]) -> float:
    values = [matrix[left][right] for left, right in itertools.combinations(names, 2)]
    return sum(values) / len(values)


def cumulative_cosine(next_name: str, history: list[str], cosine: dict, norms: dict) -> float:
    numerator = sum(
        norms[next_name] * norms[old] * cosine[next_name][old] for old in history
    )
    history_norm_sq = sum(norms[old] ** 2 for old in history)
    history_norm_sq += 2.0 * sum(
        norms[left] * norms[right] * cosine[left][right]
        for left, right in itertools.combinations(history, 2)
    )
    denominator = norms[next_name] * math.sqrt(max(0.0, history_norm_sq))
    return numerator / denominator if denominator else float("inf")


def greedy_order(selected: tuple[str, ...], cosine: dict, norms: dict) -> list[str]:
    first = min(
        selected,
        key=lambda name: (
            sum(cosine[name][other] for other in selected if other != name) / (len(selected) - 1),
            name,
        ),
    )
    order = [first]
    remaining = set(selected) - {first}
    while remaining:
        next_name = min(
            remaining,
            key=lambda name: (cumulative_cosine(name, order, cosine, norms), name),
        )
        order.append(next_name)
        remaining.remove(next_name)
    return order


def select(ler_payload: dict, screen_payload: dict, difficulty_slack: float) -> dict:
    ler = {
        name: float(values["stage0_ler"])
        for name, values in ler_payload["candidates"].items()
    }
    candidates = {
        family: sorted(name for name in ler if name.startswith(family + "_"))
        for family in FAMILIES
    }
    if any(len(names) != 4 for names in candidates.values()):
        raise ValueError("Expected four candidates for every Stress family")
    combinations = []
    for names in itertools.product(*(candidates[family] for family in FAMILIES)):
        ler_range = max(ler[name] for name in names) - min(ler[name] for name in names)
        combinations.append(
            {
                "names": names,
                "ler_range": ler_range,
                "mean_gradient_cosine": mean_pairwise(screen_payload["gradient_cosine"], names),
                "mean_representation_distance": mean_pairwise(
                    screen_payload["representation_distance"], names
                ),
            }
        )
    best_range = min(row["ler_range"] for row in combinations)
    eligible = [
        row for row in combinations
        if row["ler_range"] <= best_range + float(difficulty_slack)
    ]
    chosen = min(
        eligible,
        key=lambda row: (
            row["mean_gradient_cosine"],
            -row["mean_representation_distance"],
            row["names"],
        ),
    )
    selected = tuple(chosen["names"])
    norms = {
        name: float(screen_payload["metrics"][name]["gradient_norm"])
        for name in selected
    }
    order = greedy_order(selected, screen_payload["gradient_cosine"], norms)
    candidate_to_task = {
        candidate: TASK_BY_FAMILY[family]
        for family, candidate in zip(FAMILIES, selected)
    }
    task_order = [candidate_to_task[name] for name in order]
    return {
        "selection_rule": {
            "difficulty": "minimum Stage-0 LER range plus absolute slack",
            "difficulty_slack": difficulty_slack,
            "tie_break_1": "minimum mean pairwise gradient cosine",
            "tie_break_2": "maximum mean pairwise representation distance",
            "order": "lowest mean conflict first, then greedy conflict with cumulative history gradient",
        },
        "minimum_ler_range": best_range,
        "eligible_combinations": len(eligible),
        "selected_by_task": {
            TASK_BY_FAMILY[family]: {
                "family": family,
                "candidate": candidate,
                "stage0_ler": ler[candidate],
            }
            for family, candidate in zip(FAMILIES, selected)
        },
        "selection_metrics": chosen,
        "high_conflict_order": task_order,
        "reverse_order": list(reversed(task_order)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ler", type=Path, required=True)
    parser.add_argument("--screen", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--difficulty-slack", type=float, default=0.005)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = select(
        json.loads(args.ler.read_text(encoding="utf-8")),
        json.loads(args.screen.read_text(encoding="utf-8")),
        args.difficulty_slack,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"[QEWC v3] selected Stress tasks: {args.output}")


if __name__ == "__main__":
    main()
