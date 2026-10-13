#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Summarize a QEWC v3 stage-by-task LER matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


TASKS = ("T0", "T1", "T2", "T3", "T4")


def summarize(
    ler: list[list[float]],
    task_order: list[str],
    oracle_ler: dict[str, float] | None = None,
) -> dict:
    matrix = np.asarray(ler, dtype=np.float64)
    if matrix.shape != (5, 5):
        raise ValueError(f"LER matrix must be 5x5, got {matrix.shape}")
    if not np.all(np.isfinite(matrix)) or np.any((matrix < 0) | (matrix > 1)):
        raise ValueError("LER values must be finite and in [0, 1]")
    if len(task_order) != 4 or set(task_order) != set(TASKS[1:]):
        raise ValueError("task_order must be a permutation of T1,T2,T3,T4")

    score = 1.0 - matrix
    learned_stage = {"T0": 0}
    learned_stage.update({task: stage for stage, task in enumerate(task_order, 1)})

    incremental_scores = []
    for stage in range(5):
        learned = ["T0", *task_order[:stage]]
        incremental_scores.append(
            float(np.mean([score[stage, TASKS.index(task)] for task in learned]))
        )

    historical = [task for task in TASKS if learned_stage[task] < 4]
    forgetting = {}
    bwt_terms = []
    for task in historical:
        column = TASKS.index(task)
        acquisition = learned_stage[task]
        forgetting[task] = float(
            np.max(score[acquisition:4, column]) - score[4, column]
        )
        bwt_terms.append(float(score[4, column] - score[acquisition, column]))
    forgetting[task_order[-1]] = None

    fwt_terms = []
    fwt_by_task = {task_order[0]: 0.0}
    for stage in range(2, 5):
        task = task_order[stage - 1]
        column = TASKS.index(task)
        value = float(score[stage - 1, column] - score[0, column])
        fwt_by_task[task] = value
        fwt_terms.append(value)

    acquisition_gap = None
    if oracle_ler is not None:
        missing = set(TASKS[1:]) - set(oracle_ler)
        if missing:
            raise ValueError(f"oracle_ler is missing tasks: {sorted(missing)}")
        acquisition_gap = {
            task: float(
                matrix[learned_stage[task], TASKS.index(task)] - oracle_ler[task]
            )
            for task in TASKS[1:]
        }

    return {
        "task_columns": list(TASKS),
        "task_order": list(task_order),
        "ler": matrix.tolist(),
        "score": score.tolist(),
        "final_average_ler": float(np.mean(matrix[4])),
        "average_incremental_performance": float(np.mean(incremental_scores)),
        "incremental_scores": incremental_scores,
        "forgetting": forgetting,
        "average_forgetting": float(np.mean([forgetting[t] for t in historical])),
        "backward_transfer": float(np.mean(bwt_terms)),
        "forward_transfer": float(np.mean(fwt_terms)),
        "forward_transfer_by_task": fwt_by_task,
        "acquisition_gap": acquisition_gap,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    result = summarize(
        payload["ler"],
        payload["task_order"],
        payload.get("oracle_ler"),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"[QEWC v3] wrote matrix summary: {args.output}")


if __name__ == "__main__":
    main()
