#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Aggregate the QEWC-v3 per-configuration paired OOD evaluations."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def paired_summary(items: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(items)
    samples = sum(int(row["samples"]) for row in rows)
    a_only = sum(int(row["a_only_error"]) for row in rows)
    b_only = sum(int(row["b_only_error"]) for row in rows)
    both = sum(int(row["both_error"]) for row in rows)
    neither = sum(int(row["neither_error"]) for row in rows)
    delta = (a_only - b_only) / samples
    if samples > 1:
        variance = ((a_only + b_only) - samples * delta**2) / (samples - 1)
        standard_error = math.sqrt(max(variance, 0.0) / samples)
    else:
        standard_error = 0.0
    margin = 1.96 * standard_error
    first = rows[0]
    return {
        "method_a": first["method_a"],
        "method_b": first["method_b"],
        "basis": first["basis"],
        "configurations": len({row["_configuration"] for row in rows}),
        "samples": samples,
        "both_error": both,
        "a_only_error": a_only,
        "b_only_error": b_only,
        "neither_error": neither,
        "ler_delta_a_minus_b": delta,
        "paired_standard_error": standard_error,
        "ler_delta_ci95_normal": [max(-1.0, delta - margin), min(1.0, delta + margin)],
    }


def method_summary(rows: list[dict[str, Any]], configurations: int) -> dict[str, Any]:
    samples = sum(int(row["samples"]) for row in rows)
    logical_errors = sum(int(row["logical_errors"]) for row in rows)
    result = {
        "method": rows[0]["method"],
        "configurations": configurations,
        "basis_rows": len(rows),
        "samples": samples,
        "logical_errors": logical_errors,
        "ler": logical_errors / samples,
        "latency_us_per_round_mean": sum(float(row["latency_us_per_round"]) for row in rows) / len(rows),
        "speedup_vs_pymatching_mean": sum(float(row["speedup_vs_pymatching"]) for row in rows) / len(rows),
        "input_syndrome_density": (
            sum(int(row["input_syndrome_ones"]) for row in rows)
            / sum(int(row["input_syndrome_elements"]) for row in rows)
        ),
    }
    residual_elements = sum(int(row.get("residual_syndrome_elements", 0)) for row in rows)
    if residual_elements:
        result["residual_syndrome_density"] = (
            sum(int(row["residual_syndrome_ones"]) for row in rows) / residual_elements
        )
        result["density_reduction_fraction"] = (
            1.0 - result["residual_syndrome_density"] / result["input_syndrome_density"]
        )
    return result


def load_results(root: Path) -> list[tuple[Path, dict[str, Any]]]:
    results = []
    for path in sorted(root.glob("d*/d*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "rows" not in payload or "paired_comparisons" not in payload:
            raise ValueError(f"invalid paired OOD result: {path}")
        results.append((path, payload))
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-configurations", type=int, default=110)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = load_results(args.input_root)
    if len(results) != args.expected_configurations:
        raise ValueError(
            f"expected {args.expected_configurations} configurations, found {len(results)}"
        )

    config_rows = []
    method_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    pair_groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    distance_counts: dict[str, int] = defaultdict(int)
    for path, payload in results:
        distance = str(int(payload["distance"]))
        distance_counts[distance] += 1
        for row in payload["rows"]:
            config_rows.append(
                {
                    "configuration": path.stem,
                    "distance": int(distance),
                    "basis": row["basis"],
                    "method": row["method"],
                    "samples": row["samples"],
                    "logical_errors": row["logical_errors"],
                    "ler": row["ler"],
                    "latency_us_per_round": row["latency_us_per_round"],
                    "speedup_vs_pymatching": row["speedup_vs_pymatching"],
                    "input_syndrome_density": row["input_syndrome_density"],
                    "residual_syndrome_density": row.get("residual_syndrome_density"),
                    "density_reduction_fraction": row.get("density_reduction_fraction"),
                }
            )
            method_groups[(distance, row["method"])].append(row)
            method_groups[("all", row["method"])].append(row)
        for row in payload["paired_comparisons"]:
            item = {**row, "_configuration": path.stem}
            key = (distance, row["basis"], row["method_a"], row["method_b"])
            pair_groups[key].append(item)
            pair_groups[(distance, "both", row["method_a"], row["method_b"])].append(
                {**item, "basis": "both"}
            )
            pair_groups[("all", row["basis"], row["method_a"], row["method_b"])].append(item)
            pair_groups[("all", "both", row["method_a"], row["method_b"])].append(
                {**item, "basis": "both"}
            )

    method_aggregate = []
    for (distance, method), rows in sorted(method_groups.items()):
        configurations = len(results) if distance == "all" else distance_counts[distance]
        method_aggregate.append({"distance": distance, **method_summary(rows, configurations)})
    paired_aggregate = []
    for (distance, _basis, _method_a, _method_b), rows in sorted(pair_groups.items()):
        paired_aggregate.append({"distance": distance, **paired_summary(rows)})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "input_root": str(args.input_root),
        "configurations": len(results),
        "distances": {key: distance_counts[key] for key in sorted(distance_counts)},
        "method_aggregate": method_aggregate,
        "paired_aggregate": paired_aggregate,
    }
    (args.output_dir / "ood_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "ood_per_configuration.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(config_rows[0]))
        writer.writeheader()
        writer.writerows(config_rows)
    print(f"[QEWC v3] OOD summary: {args.output_dir / 'ood_summary.json'}")


if __name__ == "__main__":
    main()
