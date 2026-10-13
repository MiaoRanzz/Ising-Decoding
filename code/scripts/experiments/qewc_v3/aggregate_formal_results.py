#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Aggregate per-run QEWC-v3 artifacts into the formal metric table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[4]


def summary_row(result: dict) -> dict:
    summary = result.get("summary")
    if summary is None:
        final_average_ler = float(np.mean(np.asarray(result["ler"], dtype=np.float64)[-1]))
        return {
            "run_id": result["run_id"],
            "track": result["track"],
            "method": result["method"],
            "kind": result["kind"],
            "final_average_ler": final_average_ler,
            "average_incremental_performance": None,
            "average_forgetting": None,
            "backward_transfer": None,
            "forward_transfer": None,
            "mean_acquisition_gap": None,
            "ag_T1": None,
            "ag_T2": None,
            "ag_T3": None,
            "ag_T4": None,
        }
    gaps = summary.get("acquisition_gap")
    return {
        "run_id": result["run_id"],
        "track": result["track"],
        "method": result["method"],
        "kind": result["kind"],
        "final_average_ler": summary["final_average_ler"],
        "average_incremental_performance": summary["average_incremental_performance"],
        "average_forgetting": summary["average_forgetting"],
        "backward_transfer": summary["backward_transfer"],
        "forward_transfer": summary["forward_transfer"],
        "mean_acquisition_gap": float(np.mean(list(gaps.values()))) if gaps else None,
        "ag_T1": gaps.get("T1") if gaps else None,
        "ag_T2": gaps.get("T2") if gaps else None,
        "ag_T3": gaps.get("T3") if gaps else None,
        "ag_T4": gaps.get("T4") if gaps else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", action="append")
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.artifact_root or REPO_ROOT / "outputs" / "qewc_v3" / "evaluations" / "formal"
    run_ids = args.run_id or sorted(path.parent.parent.name for path in root.glob("*/artifacts/ler_matrix.json"))
    rows = [
        summary_row(json.loads((root / run_id / "artifacts" / "ler_matrix.json").read_text(encoding="utf-8")))
        for run_id in run_ids
    ]
    if not rows:
        raise ValueError("no finalized formal runs found")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    with (args.output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[QEWC v3] aggregate formal metrics: {args.output_dir}")


if __name__ == "__main__":
    main()
