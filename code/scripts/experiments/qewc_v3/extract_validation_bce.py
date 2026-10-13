#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Extract acquisition/joint validation BCE from QEWC-v3 formal logs."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
LOSS = re.compile(r"LOSS train ([0-9.eE+-]+) valid ([0-9.eE+-]+)")
MIX_VALIDATION = re.compile(r"\[Mix Validation\] (T[0-4]) loss=([0-9.eE+-]+)")


def continual_row(text: str, checkpoint: str, task: str) -> dict:
    matches = LOSS.findall(text)
    if len(matches) != 1:
        raise ValueError(f"expected one LOSS row for {checkpoint}, found {len(matches)}")
    train, validation = matches[0]
    return {
        "checkpoint": checkpoint,
        "stage_or_progress": task,
        "train_bce": float(train),
        "macro_validation_bce": float(validation),
        **{f"validation_bce_{name}": float(validation) if name == task else None for name in ("T0", "T1", "T2", "T3", "T4")},
    }


def mix_rows(text: str) -> list[dict]:
    losses = LOSS.findall(text)
    validation = MIX_VALIDATION.findall(text)
    if len(losses) != 4 or len(validation) != 20:
        raise ValueError(f"expected 4 LOSS and 20 Mix Validation rows, found {len(losses)} and {len(validation)}")
    rows = []
    for index, (train, macro) in enumerate(losses, 1):
        task_values = validation[(index - 1) * 5 : index * 5]
        if [task for task, _ in task_values] != [f"T{i}" for i in range(5)]:
            raise ValueError(f"invalid Mix Validation task order at M{index}")
        rows.append({
            "checkpoint": f"M{index}",
            "stage_or_progress": f"{index * 25}%",
            "train_bce": float(train),
            "macro_validation_bce": float(macro),
            **{f"validation_bce_{task}": float(value) for task, value in task_values},
        })
    return rows


def latest(pattern: str) -> Path:
    paths = sorted(REPO_ROOT.glob(pattern))
    if not paths:
        raise FileNotFoundError(pattern)
    return paths[-1]


def run_rows(run_id: str, training: dict) -> list[dict]:
    if training["method"] in {"mix_total", "mix_exposure"}:
        log = latest(f"logs/qewc_v3/formal/{run_id}_*/train.log")
        return mix_rows(log.read_text(encoding="utf-8"))
    rows = []
    for stage, task in enumerate(training["order"], 1):
        log = latest(f"logs/qewc_v3/formal/{run_id}/stage_{stage}_{task}_*/train.log")
        rows.append(continual_row(log.read_text(encoding="utf-8"), f"C{stage}", task))
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for run_id in args.run_id:
        manifest = REPO_ROOT / "outputs" / "qewc_v3" / "formal" / run_id / "run_manifest.json"
        training = json.loads(manifest.read_text(encoding="utf-8"))
        rows.extend({"run_id": run_id, "method": training["method"], **row} for row in run_rows(run_id, training))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "validation_bce.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    with (args.output_dir / "validation_bce.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[QEWC v3] validation BCE summaries: {args.output_dir}")


if __name__ == "__main__":
    main()
