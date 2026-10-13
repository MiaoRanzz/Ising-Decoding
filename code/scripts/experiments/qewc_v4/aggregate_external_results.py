#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Aggregate V4 OOD and Willow results and apply the frozen generalization rule."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[4]
CODE_ROOT = REPO_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from scripts.experiments.qewc_v4.evaluation_common import ARCHITECTURES  # noqa: E402


LATENCY_FIELDS = (
    "model_latency_us_per_shot",
    "residual_pymatching_batch_us_per_shot",
    "end_to_end_batch_us_per_shot",
    "pymatching_batch_us_per_shot",
    "pymatching_latency_us_per_round",
    "latency_us_per_round",
)


def aggregate_rows(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    items = list(rows)
    if not items:
        raise ValueError("cannot aggregate an empty row collection")
    sample_field = "samples" if "samples" in items[0] else "shots"
    samples = sum(int(row[sample_field]) for row in items)
    errors = sum(int(row["logical_errors"]) for row in items)
    result = {
        "rows": len(items),
        "samples": samples,
        "logical_errors": errors,
        "micro_ler": errors / samples,
        "macro_ler": statistics.fmean(float(row["ler"]) for row in items),
    }
    for field in LATENCY_FIELDS:
        values = [float(row[field]) for row in items if row.get(field) is not None]
        if values:
            result[f"{field}_mean"] = statistics.fmean(values)
    return result


def paired_from_counts(
    candidate: str,
    baseline: str,
    samples: int,
    candidate_only: int,
    baseline_only: int,
    both: int,
    neither: int,
) -> dict[str, Any]:
    delta = (candidate_only - baseline_only) / samples
    variance = (
        ((candidate_only + baseline_only) - samples * delta**2) / (samples - 1)
        if samples > 1
        else 0.0
    )
    standard_error = math.sqrt(max(variance, 0.0) / max(samples, 1))
    margin = 1.96 * standard_error
    return {
        "candidate": candidate,
        "baseline": baseline,
        "samples": samples,
        "both_error": both,
        "candidate_only_error": candidate_only,
        "baseline_only_error": baseline_only,
        "neither_error": neither,
        "ler_delta_candidate_minus_baseline": delta,
        "paired_standard_error": standard_error,
        "ler_delta_ci95_normal": [delta - margin, delta + margin],
    }


def normalize_ood_pair(row: dict, candidate: str, baseline: str) -> tuple[int, int, int, int, int]:
    if {row["method_a"], row["method_b"]} != {candidate, baseline}:
        raise ValueError("OOD pair does not match requested methods")
    candidate_only = int(row["a_only_error"] if row["method_a"] == candidate else row["b_only_error"])
    baseline_only = int(row["b_only_error"] if row["method_a"] == candidate else row["a_only_error"])
    return (
        int(row["samples"]),
        candidate_only,
        baseline_only,
        int(row["both_error"]),
        int(row["neither_error"]),
    )


def aggregate_ood_pair(rows: Iterable[dict], candidate: str, baseline: str) -> dict:
    counts = [normalize_ood_pair(row, candidate, baseline) for row in rows]
    if not counts:
        raise ValueError(f"missing OOD pair {candidate} vs {baseline}")
    totals = [sum(values) for values in zip(*counts, strict=True)]
    return paired_from_counts(candidate, baseline, *totals)


def ood_metadata(manifest: dict) -> dict[tuple[str, str], dict]:
    return {
        (str(row["env_key"]), str(row["multiplier_key"])): row
        for row in manifest["environments"]
    }


def load_ood(architecture: str, root: Path) -> dict:
    input_root = root / "ood" / architecture
    generated = REPO_ROOT / "outputs" / "qewc_v4" / "generated_configs" / "ood" / architecture
    manifest = json.loads((generated / "manifest.json").read_text(encoding="utf-8"))
    metadata = ood_metadata(manifest)
    results = []
    for path in sorted(input_root.glob("d*/d*.json")):
        parts = path.stem.split("_")
        if len(parts) != 3 or (parts[1], parts[2]) not in metadata:
            raise ValueError(f"cannot map OOD result to frozen environment: {path}")
        results.append((path, json.loads(path.read_text(encoding="utf-8")), metadata[(parts[1], parts[2])]))
    if len(results) != 110:
        raise ValueError(f"{architecture}: expected 110 OOD results, found {len(results)}")
    by_method: dict[str, list[dict]] = defaultdict(list)
    by_distance: dict[tuple[str, int], list[dict]] = defaultdict(list)
    by_category: dict[tuple[str, str], list[dict]] = defaultdict(list)
    by_multiplier: dict[tuple[str, float], list[dict]] = defaultdict(list)
    pair_rows = []
    stream_hashes = {}
    for path, payload, meta in results:
        distance = int(payload["distance"])
        stream_hashes[path.stem] = {
            basis: payload["sample_streams"][basis]["sha256"] for basis in ("X", "Z")
        }
        for row in payload["rows"]:
            method = str(row["method"])
            by_method[method].append(row)
            by_distance[(method, distance)].append(row)
            by_category[(method, str(meta["axis_signature"]))].append(row)
            by_multiplier[(method, float(meta["multiplier"]))].append(row)
        pair_rows.extend(payload["paired_comparisons"])
    methods = {method: aggregate_rows(rows) for method, rows in sorted(by_method.items())}
    distance = {
        method: {
            str(value): aggregate_rows(by_distance[(method, value)]) for value in (7, 9)
        }
        for method in methods
    }
    categories = {
        method: {
            category: aggregate_rows(rows)
            for (name, category), rows in sorted(by_category.items())
            if name == method
        }
        for method in methods
    }
    multipliers = {
        method: {
            str(value): aggregate_rows(rows)
            for (name, value), rows in sorted(by_multiplier.items())
            if name == method
        }
        for method in methods
    }
    prefix = ARCHITECTURES[architecture].short
    candidate = f"{prefix}_EWC"
    baseline = f"{prefix}_SEQ"
    selected_pairs = [
        row for row in pair_rows if {row["method_a"], row["method_b"]} == {candidate, baseline}
    ]
    paired = aggregate_ood_pair(selected_pairs, candidate, baseline)
    paired["relative_ler_improvement"] = (
        methods[baseline]["micro_ler"] - methods[candidate]["micro_ler"]
    ) / methods[baseline]["micro_ler"]
    worst_categories = {
        method: max(values.items(), key=lambda item: item[1]["macro_ler"])[0]
        for method, values in categories.items()
    }
    return {
        "architecture": architecture,
        "configurations": len(results),
        "basis_rows_per_method": len(next(iter(by_method.values()))),
        "methods": methods,
        "by_distance": distance,
        "by_noise_category": categories,
        "by_multiplier": multipliers,
        "worst_noise_category": worst_categories,
        "ewc_vs_seq": paired,
        "stream_hashes": stream_hashes,
        "noise_hashes": {
            f"{row['env_key']}_{row['multiplier_key']}": row["noise_model_sha256"]
            for row in manifest["environments"]
        },
    }


def normalize_willow_pair(row: dict, candidate: str, baseline: str) -> dict:
    if {row["candidate"], row["baseline"]} != {candidate, baseline}:
        raise ValueError("Willow pair does not match requested methods")
    if row["candidate"] == candidate:
        candidate_only = int(row["candidate_only_errors"])
        baseline_only = int(row["baseline_only_errors"])
    else:
        candidate_only = int(row["baseline_only_errors"])
        baseline_only = int(row["candidate_only_errors"])
    return paired_from_counts(
        candidate,
        baseline,
        int(row["samples"]),
        candidate_only,
        baseline_only,
        int(row["both_errors"]),
        int(row["neither_errors"]),
    )


def load_willow(architecture: str, root: Path) -> dict:
    path = root / "willow" / architecture / "results.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload["filters"]["max_shots"]) != 0 or payload["filters"]["distances"] != [5, 7]:
        raise ValueError(f"{architecture}: Willow protocol does not match the frozen all-shot d5/d7 design")
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in payload["rows"]:
        grouped[str(row["method"])].append(row)
    methods = {method: aggregate_rows(rows) for method, rows in sorted(grouped.items())}
    prefix = ARCHITECTURES[architecture].short
    candidate = f"{prefix}_EWC"
    baseline = f"{prefix}_SEQ"
    matches = [
        row for row in payload["paired_aggregate"]
        if {row["candidate"], row["baseline"]} == {candidate, baseline}
    ]
    if len(matches) != 1:
        raise ValueError(f"{architecture}: expected exactly one Willow EWC/SEQ pair")
    paired = normalize_willow_pair(matches[0], candidate, baseline)
    paired["relative_ler_improvement"] = (
        methods[baseline]["micro_ler"] - methods[candidate]["micro_ler"]
    ) / methods[baseline]["micro_ler"]
    return {
        "architecture": architecture,
        "cases": len(payload["rows"]) // len(grouped),
        "shots_per_method": methods[candidate]["samples"],
        "methods": methods,
        "ewc_vs_seq": paired,
        "benchmark_root": payload["benchmark_root"],
        "filters": payload["filters"],
    }


def generalization_decision(ood: dict, willow: dict) -> dict:
    decisions = {}
    for name, result in (("synthetic_ood", ood), ("willow", willow)):
        pair = result["ewc_vs_seq"]
        relative = float(pair["relative_ler_improvement"])
        ci = pair["ler_delta_ci95_normal"]
        decisions[name] = {
            "relative_ler_improvement": relative,
            "paired_ci95": ci,
            "relative_improvement_at_least_1pct": relative >= 0.01,
            "paired_ci_excludes_zero_in_ewc_favor": float(ci[1]) < 0.0,
        }
        decisions[name]["passed"] = (
            decisions[name]["relative_improvement_at_least_1pct"]
            and decisions[name]["paired_ci_excludes_zero_in_ewc_favor"]
        )
    return {"endpoints": decisions, "passed_both": all(row["passed"] for row in decisions.values())}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evaluation-root",
        type=Path,
        default=REPO_ROOT / "outputs" / "qewc_v4" / "evaluations",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "outputs" / "qewc_v4" / "summary" / "external" / "external_summary.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ood = {architecture: load_ood(architecture, args.evaluation_root) for architecture in ARCHITECTURES}
    willow = {architecture: load_willow(architecture, args.evaluation_root) for architecture in ARCHITECTURES}
    if ood["htnet_a101"]["stream_hashes"] != ood["ising_fast"]["stream_hashes"]:
        raise RuntimeError("HTNet and Ising OOD runs did not consume identical generated sample streams")
    if ood["htnet_a101"]["noise_hashes"] != ood["ising_fast"]["noise_hashes"]:
        raise RuntimeError("HTNet and Ising OOD generated noise models differ")
    decisions = {
        architecture: generalization_decision(ood[architecture], willow[architecture])
        for architecture in ARCHITECTURES
    }
    payload = {
        "synthetic_ood": ood,
        "willow": willow,
        "generalization_decisions": decisions,
        "claim_limit": (
            "One train seed supports conclusions only for this fixed training repetition; "
            "it does not establish cross-seed stability."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"[QEWC v4] external summary: {args.output}")


if __name__ == "__main__":
    main()
