#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Aggregate all QEWC-v4 formal ID matrices and evaluate lifelong criteria."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[4]
CODE_ROOT = REPO_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from scripts.experiments.qewc_v3.summarize_matrix import summarize  # noqa: E402
from scripts.experiments.qewc_v4.evaluation_common import (  # noqa: E402
    ARCHITECTURES,
    TASKS,
    load_training,
    run_ids,
)


def relative_change(candidate: float, baseline: float) -> float:
    if baseline == 0:
        return 0.0 if candidate == 0 else math.inf
    return (candidate - baseline) / baseline


def oracle_ler(architecture: str, root: Path) -> dict[str, float]:
    evaluation = json.loads(
        (root / "id_oracles" / architecture / "evaluation_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    if evaluation["task_columns"] != list(TASKS):
        raise ValueError("Oracle task columns do not match T0-T4")
    return {
        task: float(evaluation["ler_rows"][f"O_{task}"][TASKS.index(task)])
        for task in TASKS[1:]
    }


def build_run_result(run_id: str, training: dict, evaluation: dict, oracles: dict) -> dict:
    if evaluation["task_columns"] != list(TASKS):
        raise ValueError(f"{run_id}: task columns do not match T0-T4")
    field = "checkpoints" if training["method"] in {"mix_total", "mix_exposure"} else "endpoints"
    labels = list(training[field])
    missing = [label for label in labels if label not in evaluation["ler_rows"]]
    if missing:
        raise ValueError(f"{run_id}: missing evaluation rows {missing}")
    matrix = np.asarray([evaluation["ler_rows"][label] for label in labels], dtype=np.float64)
    if matrix.shape != (len(labels), len(TASKS)):
        raise ValueError(f"{run_id}: invalid LER matrix shape {matrix.shape}")
    if not np.all(np.isfinite(matrix)) or np.any((matrix < 0) | (matrix > 1)):
        raise ValueError(f"{run_id}: invalid LER value")
    order = list(training.get("order", []))
    if training["method"] in {"seq", "ewc", "replay"} and set(order) == set(TASKS[1:]):
        kind = "continual"
        metrics = summarize(matrix.tolist(), order, oracles)
    elif training["method"] in {"mix_total", "mix_exposure"}:
        kind = "joint_training_reference"
        metrics = None
    elif training["method"] == "t0_only":
        kind = "t0_only_control"
        metrics = None
    else:
        raise ValueError(f"{run_id}: unsupported formal training shape")
    return {
        "run_id": run_id,
        "method": training["method"],
        "kind": kind,
        "checkpoint_labels": labels,
        "stage_context": order if order else [training["method"]] * (len(labels) - 1),
        "task_columns": list(TASKS),
        "ler": matrix.tolist(),
        "final_ler_by_task": dict(zip(TASKS, matrix[-1].tolist(), strict=True)),
        "final_average_ler": float(matrix[-1].mean()),
        "metrics": metrics,
    }


def lifelong_decision(seq: dict, ewc: dict) -> dict:
    seq_metrics = seq["metrics"]
    ewc_metrics = ewc["metrics"]
    order = ewc_metrics["task_order"]
    current_task_relative_changes = {}
    for stage, task in enumerate(order, 1):
        column = TASKS.index(task)
        current_task_relative_changes[task] = relative_change(
            float(ewc["ler"][stage][column]), float(seq["ler"][stage][column])
        )
    last = order[-1]
    last_relative = current_task_relative_changes[last]
    faler_relative = relative_change(
        float(ewc_metrics["final_average_ler"]),
        float(seq_metrics["final_average_ler"]),
    )
    criteria = {
        "average_forgetting_lower": (
            float(ewc_metrics["average_forgetting"])
            < float(seq_metrics["average_forgetting"])
        ),
        "backward_transfer_closer_to_zero": (
            abs(float(ewc_metrics["backward_transfer"]))
            < abs(float(seq_metrics["backward_transfer"]))
        ),
        "final_average_ler_relative_change_at_most_1pct": faler_relative <= 0.01,
        "last_task_relative_change_at_most_5pct": last_relative <= 0.05,
        "all_acquisition_stages_relative_change_at_most_5pct": all(
            value <= 0.05 for value in current_task_relative_changes.values()
        ),
    }
    return {
        "criteria": criteria,
        "passed": all(criteria.values()),
        "faler_relative_change": faler_relative,
        "last_task": last,
        "last_task_relative_change": last_relative,
        "current_task_relative_changes": current_task_relative_changes,
        "ewc_acquisition_gap": ewc_metrics["acquisition_gap"],
        "seq_acquisition_gap": seq_metrics["acquisition_gap"],
        "ag_interpretation_rule": (
            "No separate numeric AG threshold was frozen; the explicit no-adaptation-failure "
            "check is EWC current-task LER no more than 5% above SEQ at every acquisition stage."
        ),
    }


def flatten_row(architecture: str, result: dict) -> dict:
    metrics = result["metrics"] or {}
    return {
        "architecture": architecture,
        "run_id": result["run_id"],
        "method": result["method"],
        "kind": result["kind"],
        "final_average_ler": result["final_average_ler"],
        "average_forgetting": metrics.get("average_forgetting"),
        "backward_transfer": metrics.get("backward_transfer"),
        "average_incremental_performance": metrics.get("average_incremental_performance"),
        "forward_transfer": metrics.get("forward_transfer"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evaluation-root",
        type=Path,
        default=REPO_ROOT / "outputs" / "qewc_v4" / "evaluations",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "qewc_v4" / "summary" / "id",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = {"architectures": {}}
    csv_rows = []
    for architecture in ARCHITECTURES:
        oracles = oracle_ler(architecture, args.evaluation_root)
        results = {}
        for run_id in run_ids(architecture):
            evaluation = json.loads(
                (args.evaluation_root / "id" / run_id / "evaluation_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            result = build_run_result(run_id, load_training(run_id), evaluation, oracles)
            results[run_id] = result
            csv_rows.append(flatten_row(architecture, result))
        prefix = ARCHITECTURES[architecture].short
        payload["architectures"][architecture] = {
            "oracle_ler": oracles,
            "runs": results,
            "main_order_ewc_vs_seq": lifelong_decision(
                results[f"{prefix}_SEQ"], results[f"{prefix}_EWC"]
            ),
            "reverse_order_ewc_vs_seq": lifelong_decision(
                results[f"{prefix}_REV_SEQ"], results[f"{prefix}_REV_EWC"]
            ),
        }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "id_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "id_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"[QEWC v4] ID summary: {args.output_dir / 'id_summary.json'}")


if __name__ == "__main__":
    main()
