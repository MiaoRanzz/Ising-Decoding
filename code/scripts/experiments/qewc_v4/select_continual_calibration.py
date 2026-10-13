#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Select V4 EWC lambda or Replay ratio using full ID continual matrices."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
CODE_ROOT = REPO_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from scripts.experiments.qewc_v3.summarize_matrix import TASKS, summarize  # noqa: E402


ORDER = ("T1", "T3", "T2", "T4")


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    value: float
    evaluation: Path
    training: Path


def parse_candidate(value: str) -> CandidateSpec:
    parts = value.split(":", 3)
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            "--candidate must be NAME:VALUE:EVALUATION_JSON:TRAINING_JSON"
        )
    name, raw_value, evaluation, training = parts
    return CandidateSpec(name, float(raw_value), Path(evaluation), Path(training))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def matrix_from_evaluation(evaluation: dict) -> list[list[float]]:
    if evaluation.get("task_columns") != list(TASKS):
        raise ValueError("calibration task columns must be T0,T1,T2,T3,T4")
    labels = [f"C{index}" for index in range(5)]
    if any(label not in evaluation.get("ler_rows", {}) for label in labels):
        raise ValueError("calibration evaluation must contain C0-C4")
    return [[float(value) for value in evaluation["ler_rows"][label]] for label in labels]


def training_resources(training: dict, expected_value: float, kind: str) -> tuple[bool, float, list[str]]:
    errors = []
    if training.get("method") != kind:
        errors.append(f"training method is {training.get('method')}, expected {kind}")
    if training.get("order") != list(ORDER):
        errors.append("training order does not match frozen high-conflict order")
    parameters = training.get("method_parameters", {})
    key = "lambda" if kind == "ewc" else "ratio"
    if float(parameters.get(key, -1)) != float(expected_value):
        errors.append(f"training {key} does not match candidate")
    gpu_hours = 0.0
    for checkpoint, raw_path in training.get("resource_metrics", {}).items():
        path = Path(raw_path)
        if not path.is_file():
            errors.append(f"missing resource metrics for {checkpoint}")
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for epoch in payload.get("epochs", []):
            value = float(epoch.get("gpu_hours", float("nan")))
            if not math.isfinite(value):
                errors.append(f"non-finite GPU-hours for {checkpoint}")
            else:
                gpu_hours += value
    for checkpoint, item in training.get("endpoints", {}).items():
        path = Path(item["path"])
        if not path.is_file() or sha256(path) != item.get("sha256"):
            errors.append(f"missing or corrupt endpoint {checkpoint}")
    return not errors, gpu_hours, errors


def make_record(
    name: str,
    value: float,
    matrix: list[list[float]],
    baseline_matrix: list[list[float]],
    *,
    kind: str,
    stable: bool,
    gpu_hours: float,
    audit_errors: list[str] | None = None,
    current_task_relative_limit: float = 0.05,
    final_relative_limit: float = 0.02,
) -> dict:
    metrics = summarize(matrix, list(ORDER))
    baseline = summarize(baseline_matrix, list(ORDER))
    degradation = {}
    for stage, task in enumerate(ORDER, 1):
        column = list(TASKS).index(task)
        base = float(baseline_matrix[stage][column])
        candidate = float(matrix[stage][column])
        degradation[task] = (candidate - base) / base if base > 0 else float("inf")
    faler_relative = (
        (metrics["final_average_ler"] - baseline["final_average_ler"])
        / baseline["final_average_ler"]
    )
    eligible = stable
    if kind == "ewc":
        eligible = (
            eligible
            and max(degradation.values()) <= current_task_relative_limit
            and faler_relative <= final_relative_limit
        )
    return {
        "name": name,
        "value": float(value),
        "stable": bool(stable),
        "audit_errors": audit_errors or [],
        "current_task_relative_degradation_vs_seq": degradation,
        "maximum_current_task_relative_degradation_vs_seq": max(degradation.values()),
        "faler_relative_degradation_vs_seq": faler_relative,
        "average_forgetting": metrics["average_forgetting"],
        "backward_transfer": metrics["backward_transfer"],
        "final_average_ler": metrics["final_average_ler"],
        "gpu_hours": gpu_hours,
        "eligible": eligible,
    }


def choose(records: list[dict], kind: str) -> dict:
    eligible = [row for row in records if row["eligible"]]
    if not eligible:
        raise ValueError(f"no stable eligible {kind} candidate")
    if kind == "ewc":
        key = lambda row: (
            float(row["average_forgetting"]),
            abs(float(row["backward_transfer"])),
            float(row["final_average_ler"]),
            float(row["value"]),
        )
    else:
        key = lambda row: (
            float(row["average_forgetting"]),
            float(row["final_average_ler"]),
            float(row["gpu_hours"]),
            float(row["value"]),
        )
    return min(eligible, key=key)


def next_ewc_step(records: list[dict], selected: dict, phase: str, boundary_extended: bool) -> dict:
    if phase == "refined":
        return {"action": "complete", "candidate_values": []}
    values = sorted(float(row["value"]) for row in records)
    value = float(selected["value"])
    index = values.index(value)
    if not boundary_extended and index in (0, len(values) - 1):
        extension = value / 10.0 if index == 0 else value * 10.0
        return {"action": "extend_one_decade", "candidate_values": [extension]}
    neighbors = []
    if index > 0:
        neighbors.append(math.sqrt(values[index - 1] * value))
    if index + 1 < len(values):
        neighbors.append(math.sqrt(value * values[index + 1]))
    return {"action": "refine_geometric_midpoints", "candidate_values": neighbors}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("ewc", "replay"), required=True)
    parser.add_argument("--baseline-evaluation", type=Path, required=True)
    parser.add_argument("--candidate", action="append", type=parse_candidate, required=True)
    parser.add_argument("--phase", choices=("coarse", "refined"), default="coarse")
    parser.add_argument("--boundary-already-extended", action="store_true")
    parser.add_argument("--current-task-relative-limit", type=float, default=0.05)
    parser.add_argument("--final-relative-limit", type=float, default=0.02)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline_payload = json.loads(args.baseline_evaluation.read_text(encoding="utf-8"))
    baseline_matrix = matrix_from_evaluation(baseline_payload)
    records = []
    for spec in args.candidate:
        evaluation = json.loads(spec.evaluation.read_text(encoding="utf-8"))
        training = json.loads(spec.training.read_text(encoding="utf-8"))
        stable, gpu_hours, errors = training_resources(training, spec.value, args.kind)
        records.append(make_record(
            spec.name, spec.value, matrix_from_evaluation(evaluation), baseline_matrix,
            kind=args.kind, stable=stable, gpu_hours=gpu_hours, audit_errors=errors,
            current_task_relative_limit=args.current_task_relative_limit,
            final_relative_limit=args.final_relative_limit,
        ))
    selected = choose(records, args.kind)
    payload = {
        "kind": args.kind,
        "selection_order": (
            ["minimum_AF", "minimum_abs_BWT", "minimum_FALER", "smaller_lambda"]
            if args.kind == "ewc"
            else ["minimum_AF", "minimum_FALER", "minimum_GPU_hours", "smaller_ratio"]
        ),
        "current_task_relative_limit": args.current_task_relative_limit,
        "final_relative_limit": args.final_relative_limit,
        "selected": {"name": selected["name"], "value": selected["value"]},
        "records": records,
    }
    if args.kind == "ewc":
        payload["phase"] = args.phase
        payload["next_step"] = next_ewc_step(
            records, selected, args.phase, args.boundary_already_extended
        )
    output = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"[QEWC v4] selected {args.kind} {payload['selected']}: {output}")


if __name__ == "__main__":
    main()
