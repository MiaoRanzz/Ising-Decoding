#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Audit HTNet budget pilots and select the frozen QEWC-v4 budget B."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
TASKS = ("T1", "T2", "T3", "T4")
EPOCHS = (1, 5, 20, 50)
EXTENDED_EPOCHS = (*EPOCHS, 100)
LOSS = re.compile(r"LOSS train ([0-9.eE+-]+) valid ([0-9.eE+-]+)")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_gain(before: float, after: float) -> float:
    return (before - after) / before if before > 0 else float("nan")


def sustained_decline(values: list[float], threshold: float = 0.01) -> bool | None:
    """Whether the last fifth improves by at least ``threshold`` vs the prior fifth."""

    if len(values) < 5:
        return None
    width = max(1, len(values) // 5)
    previous = sum(values[-2 * width : -width]) / width
    recent = sum(values[-width:]) / width
    return relative_gain(previous, recent) >= threshold


def paired_row(payload: dict, task: str, candidate: str, baseline: str) -> dict:
    for row in payload["paired_comparisons"][task]:
        if row["basis"] != "both":
            continue
        if row["candidate"] == candidate and row["baseline"] == baseline:
            return row
        if row["candidate"] == baseline and row["baseline"] == candidate:
            result = dict(row)
            result["candidate"] = candidate
            result["baseline"] = baseline
            result["delta_ler"] = -float(row["delta_ler"])
            result["ci95_low"] = -float(row["ci95_high"])
            result["ci95_high"] = -float(row["ci95_low"])
            return result
    raise ValueError(f"missing paired row for {task}: {candidate} vs {baseline}")


def select_task(
    records: list[dict],
    *,
    ler_threshold: float = 0.001,
    bce_threshold: float = 0.01,
) -> dict:
    ordered = sorted(records, key=lambda row: int(row["legacy_epochs"]))
    epoch_grid = [int(row["legacy_epochs"]) for row in ordered]
    if epoch_grid not in (list(EPOCHS), list(EXTENDED_EPOCHS)):
        raise ValueError(
            "budget records must contain 1, 5, 20, 50 epochs, with optional 100"
        )
    comparisons = []
    selected = None
    for current, following in zip(ordered, ordered[1:]):
        ler_gain = float(current["current_task_ler"]) - float(
            following["current_task_ler"]
        )
        bce_gain = relative_gain(
            float(current["validation_bce"]), float(following["validation_bce"])
        )
        enough = (
            int(current["legacy_epochs"]) >= 5
            and bool(current["stable"])
            and current["train_loss_sustained_decline"] is False
            and current["validation_loss_sustained_decline"] is False
            and ler_gain < ler_threshold
            and bce_gain < bce_threshold
        )
        comparisons.append(
            {
                "from_legacy_epochs": int(current["legacy_epochs"]),
                "to_legacy_epochs": int(following["legacy_epochs"]),
                "ler_improvement_absolute": ler_gain,
                "validation_bce_improvement_relative": bce_gain,
                "current_budget_is_sufficient": enough,
            }
        )
        if enough and selected is None:
            selected = int(current["legacy_epochs"])
    last = ordered[-1]
    needs_extension = False
    selected_at_cap_still_improving = False
    if selected is None:
        last_plateaued = (
            bool(last["stable"])
            and last["train_loss_sustained_decline"] is False
            and last["validation_loss_sustained_decline"] is False
        )
        if last_plateaued:
            selected = int(last["legacy_epochs"])
        elif int(last["legacy_epochs"]) == 100 and bool(last["stable"]):
            # 100 epochs is the preregistered ceiling.  Keep the limitation
            # explicit instead of requesting an unbounded follow-up grid.
            selected = 100
            selected_at_cap_still_improving = True
        else:
            needs_extension = True
    return {
        "selected_legacy_epochs": selected,
        "needs_extension_to_100": needs_extension,
        "selected_at_cap_still_improving": selected_at_cap_still_improving,
        "comparisons": comparisons,
        "records": ordered,
    }


def _latest_log(log_root: Path, task: str, epochs: int) -> Path:
    pattern = f"{task}_e{epochs}/stage_1_{task}_*/train.log"
    paths = sorted(log_root.glob(pattern))
    if not paths:
        raise FileNotFoundError(log_root / pattern)
    return paths[-1]


def _record(
    root: Path,
    log_root: Path,
    evaluation: dict,
    task: str,
    epochs: int,
) -> dict:
    run_root = root / "htnet" / f"{task}_e{epochs}"
    training_path = run_root / "run_manifest.json"
    training = json.loads(training_path.read_text(encoding="utf-8"))
    endpoint = training["endpoints"]["C1"]
    endpoint_path = Path(endpoint["path"])
    resource_path = run_root / f"stage_1_{task}" / "resource_metrics.json"
    resource = json.loads(resource_path.read_text(encoding="utf-8"))
    log_path = _latest_log(log_root, task, epochs)
    losses = [
        (float(train), float(validation))
        for train, validation in LOSS.findall(log_path.read_text(encoding="utf-8"))
    ]
    errors = []
    expected_updates = 512 * epochs
    expected_samples = 2097152 * epochs
    if int(training["total_optimizer_steps"]) != expected_updates:
        errors.append("optimizer update count mismatch")
    if int(training["samples_processed"]) != expected_samples:
        errors.append("actual sample count mismatch")
    if len(resource.get("epochs", [])) != epochs:
        errors.append("resource epoch count mismatch")
    if len(losses) != epochs:
        errors.append("training/validation BCE row count mismatch")
    if not endpoint_path.is_file() or sha256(endpoint_path) != endpoint["sha256"]:
        errors.append("endpoint missing or SHA256 mismatch")
    task_index = list(evaluation["task_columns"]).index(task)
    current_ler = float(evaluation["ler_rows"][f"E{epochs}"][task_index])
    train_losses = [row[0] for row in losses]
    validation_losses = [row[1] for row in losses]
    stable = not errors and all(
        math.isfinite(value)
        for value in train_losses + validation_losses + [current_ler]
    )
    return {
        "legacy_epochs": epochs,
        "optimizer_updates": int(training["total_optimizer_steps"]),
        "actual_samples": int(training["samples_processed"]),
        "current_task_ler": current_ler,
        "train_bce": train_losses[-1],
        "validation_bce": validation_losses[-1],
        "train_bce_trajectory": train_losses,
        "validation_bce_trajectory": validation_losses,
        "train_loss_sustained_decline": sustained_decline(train_losses),
        "validation_loss_sustained_decline": sustained_decline(validation_losses),
        "stable": stable,
        "audit_errors": errors,
        "wall_seconds": sum(float(row["wall_seconds"]) for row in resource["epochs"]),
        "gpu_hours": sum(float(row["gpu_hours"]) for row in resource["epochs"]),
        "peak_reserved_bytes": max(
            int(row["peak_reserved_bytes_max_rank"]) for row in resource["epochs"]
        ),
        "endpoint": endpoint,
        "training_manifest": str(training_path),
        "resource_metrics": str(resource_path),
        "log": str(log_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path("outputs/qewc_v4/budget_pilot")
    )
    parser.add_argument(
        "--log-root", type=Path, default=Path("logs/qewc_v4/budget_pilot/htnet")
    )
    parser.add_argument("--ler-threshold", type=float, default=0.001)
    parser.add_argument("--bce-threshold", type=float, default=0.01)
    parser.add_argument(
        "--include-100",
        action="store_true",
        help="Require and include the preregistered 100-epoch extension.",
    )
    parser.add_argument(
        "--include-100-task",
        action="append",
        choices=TASKS,
        default=[],
        help="Include E100 only for this task; may be repeated.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/qewc_v4/budget_pilot/selection.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root if args.root.is_absolute() else REPO_ROOT / args.root
    log_root = (
        args.log_root if args.log_root.is_absolute() else REPO_ROOT / args.log_root
    )
    task_results = {}
    extended_tasks = set(TASKS if args.include_100 else args.include_100_task)
    for task in TASKS:
        evaluation_path = root / "evaluations" / task / "evaluation_manifest.json"
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        epochs_grid = EXTENDED_EPOCHS if task in extended_tasks else EPOCHS
        records = [
            _record(root, log_root, evaluation, task, epochs)
            for epochs in epochs_grid
        ]
        result = select_task(
            records,
            ler_threshold=args.ler_threshold,
            bce_threshold=args.bce_threshold,
        )
        for comparison in result["comparisons"]:
            current = f"E{comparison['from_legacy_epochs']}"
            following = f"E{comparison['to_legacy_epochs']}"
            comparison["paired_ler"] = paired_row(evaluation, task, following, current)
        task_results[task] = result
    needs_extension = any(
        row["needs_extension_to_100"] for row in task_results.values()
    )
    selected = [row["selected_legacy_epochs"] for row in task_results.values()]
    payload = {
        "status": "needs_100_epoch_extension" if needs_extension else "selected",
        "authority_architecture": "htnet_a101_only",
        "selection_rule": {
            "ler_improvement_absolute_below": args.ler_threshold,
            "validation_bce_improvement_relative_below": args.bce_threshold,
            "minimum_epochs_with_loss_trend": 5,
            "no_sustained_train_or_validation_loss_decline": True,
            "B": "maximum selected sufficient budget across T1-T4",
        },
        "selected_B": None if needs_extension else max(int(value) for value in selected),
        "selected_at_preregistered_cap_still_improving": any(
            row["selected_at_cap_still_improving"] for row in task_results.values()
        ),
        "tasks": task_results,
    }
    output = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"[QEWC v4] budget selection status={payload['status']} output={output}")


if __name__ == "__main__":
    main()
