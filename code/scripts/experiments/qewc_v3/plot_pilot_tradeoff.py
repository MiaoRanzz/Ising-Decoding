#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plot the deterministic QEWC-v3 stability-plasticity pilot comparison."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


def points(
    ewc_evaluation: dict,
    replay_evaluation: dict,
    ewc_selection: dict,
    replay_selection: dict,
) -> list[dict]:
    if ewc_evaluation["test_streams"] != replay_evaluation["test_streams"]:
        raise ValueError("EWC and Replay evaluations do not use identical test streams")
    result = []
    definitions = [
        ("baseline", ewc_evaluation, ["C0", "SEQ"], None),
        ("ewc", ewc_evaluation, [name for name in ewc_evaluation["ler_rows"] if name.startswith("EWC_")], ewc_selection),
        ("replay", replay_evaluation, [name for name in replay_evaluation["ler_rows"] if name.startswith("R")], replay_selection),
    ]
    for method, evaluation, names, selection in definitions:
        selected_name = selection["selected"]["name"] if selection else None
        for name in names:
            row = list(map(float, evaluation["ler_rows"][name]))
            result.append(
                {
                    "method": method,
                    "candidate": name,
                    "selected": name == selected_name,
                    "t0_ler": row[0],
                    "t1_ler": row[1],
                    "five_task_macro_ler": sum(row) / len(row),
                }
            )
    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(path: Path, rows: list[dict]) -> None:
    colors = {"baseline": "black", "ewc": "tab:blue", "replay": "tab:orange"}
    figure, axis = plt.subplots(figsize=(7.4, 5.4))
    for row in rows:
        marker = "*" if row["selected"] else "o"
        size = 170 if row["selected"] else 55
        axis.scatter(row["t0_ler"], row["t1_ler"], color=colors[row["method"]], marker=marker, s=size)
        axis.annotate(row["candidate"], (row["t0_ler"], row["t1_ler"]), xytext=(5, 4), textcoords="offset points", fontsize=8)
    axis.set_xlabel("T0 LER (stability; lower is better)")
    axis.set_ylabel("T1 LER (plasticity; lower is better)")
    axis.set_title("QEWC-v3 deterministic first-switch trade-off")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ewc-evaluation", type=Path, required=True)
    parser.add_argument("--replay-evaluation", type=Path, required=True)
    parser.add_argument("--ewc-selection", type=Path, required=True)
    parser.add_argument("--replay-selection", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in (
        args.ewc_evaluation,
        args.replay_evaluation,
        args.ewc_selection,
        args.replay_selection,
    )]
    rows = points(*payloads)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "stability_plasticity.csv", rows)
    plot(args.output_dir / "stability_plasticity.png", rows)
    print(f"[QEWC v3] pilot trade-off artifacts: {args.output_dir}")


if __name__ == "__main__":
    main()
