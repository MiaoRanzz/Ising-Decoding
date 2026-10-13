#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert one formal QEWC-v3 evaluation into report-ready artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
CODE_ROOT = REPO_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from scripts.experiments.qewc_v3.summarize_matrix import TASKS, summarize




def build_result(
    run_id: str,
    training: dict,
    evaluation: dict,
    oracle_ler: dict[str, float] | None = None,
) -> dict:
    task_columns = evaluation["task_columns"]
    if task_columns != list(TASKS):
        raise ValueError(f"task columns must be {list(TASKS)}, got {task_columns}")

    method = training["method"]
    if method in {"mix_total", "mix_exposure"}:
        row_labels = [f"M{i}" for i in range(5)]
        row_context = ["0%", "25%", "50%", "75%", "100%"]
        kind = "joint_optimization_trajectory"
    elif run_id.startswith("O"):
        row_labels = ["C0", run_id]
        row_context = ["T0", f"single_{training['order'][0]}"]
        kind = "single_task_oracle"
    else:
        order = list(training["order"])
        if len(order) != 4 or set(order) != set(TASKS[1:]):
            raise ValueError("continual task order must be a permutation of T1,T2,T3,T4")
        row_labels = [f"C{i}" for i in range(5)]
        row_context = ["T0", *order]
        kind = "continual_matrix"

    missing = [name for name in row_labels if name not in evaluation["ler_rows"]]
    if missing:
        raise ValueError(f"evaluation is missing rows: {missing}")
    ler = [evaluation["ler_rows"][name] for name in row_labels]
    matrix = np.asarray(ler, dtype=np.float64)
    if matrix.shape != (len(row_labels), 5):
        raise ValueError(f"expected {len(row_labels)}x5 LER data, got {matrix.shape}")

    result = {
        "run_id": run_id,
        "method": method,
        "track": evaluation["track"],
        "kind": kind,
        "row_labels": row_labels,
        "row_context": row_context,
        "task_columns": task_columns,
        "ler": matrix.tolist(),
    }
    if kind == "continual_matrix":
        result["summary"] = summarize(matrix.tolist(), row_context[1:], oracle_ler)
        result["forgetting_curve"] = forgetting_curve(matrix, row_context[1:])
    return result


def forgetting_curve(matrix: np.ndarray, task_order: list[str]) -> dict[str, list[float | None]]:
    score = 1.0 - matrix
    acquired = {"T0": 0, **{task: i for i, task in enumerate(task_order, 1)}}
    curves: dict[str, list[float | None]] = {}
    for task in TASKS:
        column = TASKS.index(task)
        start = acquired[task]
        values: list[float | None] = []
        for stage in range(5):
            if stage < start:
                values.append(None)
            else:
                values.append(float(np.max(score[start : stage + 1, column]) - score[stage, column]))
        curves[task] = values
    return curves


def load_oracle_ler(evaluation_root: Path, track: str) -> dict[str, float] | None:
    first_oracle = 1 if track == "stress" else 5
    result = {}
    for offset, task in enumerate(TASKS[1:]):
        run_id = f"O{first_oracle + offset}"
        path = evaluation_root / run_id / "evaluation_manifest.json"
        if not path.is_file():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        result[task] = float(payload["ler_rows"][run_id][TASKS.index(task)])
    return result


def write_csv(path: Path, result: dict) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["checkpoint", "stage_or_progress", *TASKS])
        for label, context, row in zip(
            result["row_labels"], result["row_context"], result["ler"], strict=True
        ):
            writer.writerow([label, context, *row])


def write_forgetting_csv(path: Path, result: dict) -> None:
    curves = result["forgetting_curve"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["checkpoint", "stage_task", *TASKS])
        for stage, (label, context) in enumerate(
            zip(result["row_labels"], result["row_context"], strict=True)
        ):
            writer.writerow([label, context, *[curves[task][stage] for task in TASKS]])


def plot_heatmap(path: Path, result: dict) -> None:
    import matplotlib.pyplot as plt

    matrix = np.asarray(result["ler"])
    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    image = axis.imshow(matrix, cmap="viridis", aspect="auto")
    axis.set_xticks(range(5), TASKS)
    axis.set_yticks(range(len(result["row_labels"])), result["row_labels"])
    axis.set_xlabel("Evaluation task")
    axis.set_ylabel("Checkpoint")
    axis.set_title(f"{result['run_id']} logical error rate")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(column, row, f"{matrix[row, column]:.4f}", ha="center", va="center", color="white")
    figure.colorbar(image, ax=axis, label="LER")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_forgetting(path: Path, result: dict) -> None:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    stages = np.arange(5)
    for task in TASKS:
        values = np.asarray(
            [np.nan if item is None else item for item in result["forgetting_curve"][task]],
            dtype=np.float64,
        )
        axis.plot(stages, values, marker="o", label=task)
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xticks(stages, result["row_labels"])
    axis.set_xlabel("Checkpoint")
    axis.set_ylabel("Forgetting in score space")
    axis.set_title(f"{result['run_id']} task-wise forgetting")
    axis.legend(ncol=3)
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--training-manifest", type=Path)
    parser.add_argument("--evaluation-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--evaluation-root", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    training_path = args.training_manifest or REPO_ROOT / "outputs" / "qewc_v3" / "formal" / args.run_id / "run_manifest.json"
    evaluation_root = args.evaluation_root or REPO_ROOT / "outputs" / "qewc_v3" / "evaluations" / "formal"
    evaluation_path = args.evaluation_manifest or evaluation_root / args.run_id / "evaluation_manifest.json"
    output_dir = args.output_dir or evaluation_path.parent / "artifacts"
    training = json.loads(training_path.read_text(encoding="utf-8"))
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    oracle_ler = load_oracle_ler(evaluation_root, evaluation["track"])
    result = build_result(args.run_id, training, evaluation, oracle_ler)

    output_dir.mkdir(parents=True, exist_ok=True)
    matrix_json = output_dir / "ler_matrix.json"
    matrix_csv = output_dir / "ler_matrix.csv"
    matrix_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    write_csv(matrix_csv, result)
    generated = [matrix_json, matrix_csv]
    if "summary" in result:
        summary = output_dir / "metrics_summary.json"
        forgetting_csv = output_dir / "forgetting_curve.csv"
        forgetting_plot = output_dir / "forgetting_curve.png"
        summary.write_text(json.dumps(result["summary"], indent=2) + "\n", encoding="utf-8")
        write_forgetting_csv(forgetting_csv, result)
        plot_forgetting(forgetting_plot, result)
        generated.extend((summary, forgetting_csv, forgetting_plot))
    heatmap = output_dir / "ler_heatmap.png"
    plot_heatmap(heatmap, result)
    generated.append(heatmap)
    manifest = {
        "run_id": args.run_id,
        "training_manifest": str(training_path),
        "evaluation_manifest": str(evaluation_path),
        "oracle_ler_available": oracle_ler is not None,
        "artifacts": {item.name: {"path": str(item), "sha256": sha256(item)} for item in generated},
    }
    manifest_path = output_dir / "artifact_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[QEWC v3] formal artifacts: {manifest_path}")


if __name__ == "__main__":
    main()
