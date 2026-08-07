#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Aggregate paired QAdapt unseen-noise A/B inference into auditable tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from scripts.experiments.unknown_noise.run_qadapt_unseen_noise_ab_compare import (
    DEFAULT_MANIFEST,
    DEFAULT_OUTPUT_DIR,
    rel,
    result_path,
)


DEFAULT_DETAILS = "outputs/analysis/qadapt_unseen_noise_ab_details.csv"
DEFAULT_SUMMARY = "outputs/analysis/qadapt_unseen_noise_ab_summary.csv"
DEFAULT_REPORT = "outputs/analysis/qadapt_unseen_noise_ab_report.md"
EWC = "r9x_seq_ewc_e100"
NOEWC = "r9x_seq_noewc_e100"


def _method_rows(payload: dict[str, Any], method: str) -> dict[str, dict[str, Any]]:
    return {
        str(row["basis"]): row
        for row in payload["rows"]
        if row["method"] == method
    }


def _summary_row(payload: dict[str, Any], method: str) -> dict[str, Any]:
    return next(row for row in payload["summary"] if row["method"] == method)


def _comparison_rows(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(row["basis"]): row
        for row in payload["paired_comparisons"]
        if row.get("candidate") == EWC and row.get("baseline") == NOEWC
    }


def _detail_rows(
    manifest: dict[str, Any],
    output_dir: Path,
    distances: Iterable[int],
    *,
    allow_partial: bool,
) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for environment in manifest["environments"]:
        for distance in distances:
            path = result_path(output_dir, environment, int(distance))
            if not path.exists():
                missing.append(str(path))
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            ewc_rows = _method_rows(payload, EWC)
            noewc_rows = _method_rows(payload, NOEWC)
            comparisons = _comparison_rows(payload)
            ewc_summary = _summary_row(payload, EWC)
            noewc_summary = _summary_row(payload, NOEWC)
            for basis in ("X", "Z", "both"):
                comparison = comparisons[basis]
                if basis == "both":
                    ewc_ler = float(ewc_summary["ler_avg"])
                    noewc_ler = float(noewc_summary["ler_avg"])
                    ewc_density = float(ewc_summary["residual_syndrome_density"])
                    noewc_density = float(noewc_summary["residual_syndrome_density"])
                    ewc_latency = float(ewc_summary["latency_us_per_round_avg"])
                    noewc_latency = float(noewc_summary["latency_us_per_round_avg"])
                else:
                    ewc_ler = float(ewc_rows[basis]["ler"])
                    noewc_ler = float(noewc_rows[basis]["ler"])
                    ewc_density = float(ewc_rows[basis]["residual_syndrome_density"])
                    noewc_density = float(noewc_rows[basis]["residual_syndrome_density"])
                    ewc_latency = float(ewc_rows[basis]["latency_us_per_round"])
                    noewc_latency = float(noewc_rows[basis]["latency_us_per_round"])
                axis_multipliers = environment.get("axis_multipliers", {})
                sampled_totals = environment.get("sampled_totals", {})
                rows.append(
                    {
                        "design_sha256": manifest["design_sha256"],
                        "family": environment["family"],
                        "family_key": environment["family_key"],
                        "env_key": environment["env_key"],
                        "replicate_key": environment["replicate_key"],
                        "config_name": environment["config_name"],
                        "noise_model_sha256": environment["noise_model_sha256"],
                        "axis_signature": environment["axis_signature"],
                        "axis_multiplier_meas": axis_multipliers.get("meas_all", ""),
                        "axis_multiplier_cnot": axis_multipliers.get("cnot_all", ""),
                        "axis_multiplier_idle": axis_multipliers.get("idle_all", ""),
                        "axis_multiplier_z_bias": axis_multipliers.get("z_bias", ""),
                        "multiplier_min_observed": environment.get(
                            "multiplier_min_observed", ""
                        ),
                        "multiplier_geometric_mean": environment.get(
                            "multiplier_geometric_mean", ""
                        ),
                        "multiplier_max_observed": environment.get(
                            "multiplier_max_observed", ""
                        ),
                        "sampled_idle_cnot_total": sampled_totals.get(
                            "idle_cnot_total", ""
                        ),
                        "sampled_idle_spam_total": sampled_totals.get(
                            "idle_spam_total", ""
                        ),
                        "sampled_cnot_total": sampled_totals.get(
                            "cnot_total", ""
                        ),
                        "outside_training_envelope_count": environment.get(
                            "outside_training_envelope_count", ""
                        ),
                        "min_log10_rms_distance_to_training": environment.get(
                            "min_log10_rms_distance_to_training", ""
                        ),
                        "distance": int(distance),
                        "basis": basis,
                        "samples": int(comparison["samples"]),
                        "ewc_ler": ewc_ler,
                        "noewc_ler": noewc_ler,
                        "delta_ler": float(comparison["delta_ler"]),
                        "standard_error": float(comparison["standard_error"]),
                        "ci95_low": float(comparison["ci95_low"]),
                        "ci95_high": float(comparison["ci95_high"]),
                        "ewc_only_errors": int(comparison["candidate_only_errors"]),
                        "noewc_only_errors": int(comparison["baseline_only_errors"]),
                        "both_errors": int(comparison["both_errors"]),
                        "neither_errors": int(comparison["neither_errors"]),
                        "ewc_win": int(float(comparison["delta_ler"]) < 0),
                        "ewc_residual_density": ewc_density,
                        "noewc_residual_density": noewc_density,
                        "residual_density_delta": ewc_density - noewc_density,
                        "ewc_backend_latency_us_per_round": ewc_latency,
                        "noewc_backend_latency_us_per_round": noewc_latency,
                        "backend_latency_delta_us_per_round": ewc_latency
                        - noewc_latency,
                        "result_json": str(path),
                    }
                )
    if missing and not allow_partial:
        preview = "\n".join(missing[:10])
        raise FileNotFoundError(
            f"missing {len(missing)} result files; first paths:\n{preview}"
        )
    return rows, missing


def _bootstrap_mean_ci(values: np.ndarray, seed: int, repeats: int = 10000) -> tuple[float, float]:
    if values.size == 0:
        return float("nan"), float("nan")
    if values.size == 1:
        value = float(values[0])
        return value, value
    rng = np.random.default_rng(seed)
    sampled = rng.choice(values, size=(repeats, values.size), replace=True).mean(axis=1)
    low, high = np.quantile(sampled, [0.025, 0.975])
    return float(low), float(high)


def _wilson_interval(wins: int, total: int) -> tuple[float, float]:
    if total == 0:
        return float("nan"), float("nan")
    z = 1.96
    p = wins / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def _summarize_group(rows: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    deltas = np.asarray([float(row["delta_ler"]) for row in rows], dtype=np.float64)
    wins = sum(int(row["ewc_win"]) for row in rows)
    samples = sum(int(row["samples"]) for row in rows)
    candidate_only = sum(int(row["ewc_only_errors"]) for row in rows)
    baseline_only = sum(int(row["noewc_only_errors"]) for row in rows)
    pooled_delta = (candidate_only - baseline_only) / samples
    discordant_fraction = (candidate_only + baseline_only) / samples
    pooled_se = math.sqrt(
        max(0.0, discordant_fraction - pooled_delta * pooled_delta) / samples
    )
    bootstrap_low, bootstrap_high = _bootstrap_mean_ci(deltas, seed)
    win_low, win_high = _wilson_interval(wins, len(rows))
    return {
        "config_count": len(rows),
        "ewc_ler_macro_mean": float(np.mean([row["ewc_ler"] for row in rows])),
        "noewc_ler_macro_mean": float(np.mean([row["noewc_ler"] for row in rows])),
        "delta_macro_mean": float(deltas.mean()),
        "delta_macro_std": float(deltas.std(ddof=1)) if len(rows) > 1 else 0.0,
        "delta_macro_median": float(np.median(deltas)),
        "delta_macro_q10": float(np.quantile(deltas, 0.10)),
        "delta_macro_q90": float(np.quantile(deltas, 0.90)),
        "delta_macro_min": float(deltas.min()),
        "delta_macro_max": float(deltas.max()),
        "config_bootstrap_ci95_low": bootstrap_low,
        "config_bootstrap_ci95_high": bootstrap_high,
        "ewc_win_count": wins,
        "ewc_win_rate": wins / len(rows),
        "ewc_win_rate_ci95_low": win_low,
        "ewc_win_rate_ci95_high": win_high,
        "pooled_samples": samples,
        "pooled_ewc_only_errors": candidate_only,
        "pooled_noewc_only_errors": baseline_only,
        "pooled_delta_ler": pooled_delta,
        "pooled_standard_error": pooled_se,
        "pooled_ci95_low": pooled_delta - 1.96 * pooled_se,
        "pooled_ci95_high": pooled_delta + 1.96 * pooled_se,
        "residual_density_delta_macro_mean": float(
            np.mean([row["residual_density_delta"] for row in rows])
        ),
        "backend_latency_delta_macro_mean": float(
            np.mean([row["backend_latency_delta_us_per_round"] for row in rows])
        ),
    }


def build_summary(rows: list[dict[str, Any]], seed: int = 20260806) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    both_rows = [row for row in rows if row["basis"] == "both"]
    family_keys = list(dict.fromkeys(str(row["family_key"]) for row in both_rows))
    for family_key in family_keys:
        family_rows = [row for row in both_rows if row["family_key"] == family_key]
        for distance_label, selected in [
            ("7", [row for row in family_rows if row["distance"] == 7]),
            ("9", [row for row in family_rows if row["distance"] == 9]),
            ("all", family_rows),
        ]:
            if not selected:
                continue
            summary.append(
                {
                    "family_key": family_key,
                    "family": selected[0]["family"],
                    "distance": distance_label,
                    "basis": "both",
                    **_summarize_group(selected, seed + len(summary)),
                }
            )
    return summary


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(
    path: Path,
    manifest: dict[str, Any],
    summary: list[dict[str, Any]],
    missing: list[str],
) -> None:
    lines = [
        f"# {manifest.get('design', 'QAdapt unseen-noise paired evaluation')}",
        "",
        f"- Frozen design SHA-256: `{manifest['design_sha256']}`",
        f"- Configurations: {manifest['num_configs']}",
        "- Primary contrast: final EWC minus final no-EWC LER; negative favors EWC.",
        f"- Missing result files at aggregation time: {len(missing)}",
    ]
    sampling = manifest.get("sampling")
    if sampling:
        outside_counts = [
            int(environment["outside_training_envelope_count"])
            for environment in manifest["environments"]
        ]
        distances = [
            float(environment["min_log10_rms_distance_to_training"])
            for environment in manifest["environments"]
        ]
        lines.extend(
            [
                f"- T0-anchored noise sampling: {manifest.get('t0_anchored_noise_sampling')}",
                f"- Scalar/total sampling: {sampling['scalar_method']}; "
                f"compositions: {sampling['composition_method']}.",
                f"- Parameters outside the T0--T4 envelope per config: "
                f"{min(outside_counts)}--{max(outside_counts)} of 25.",
                f"- Minimum log10-RMS distance to a training task: "
                f"{min(distances):.6f}--{max(distances):.6f}.",
                f"- Scalar absolute ranges: `{json.dumps(sampling['scalar_absolute_ranges'], sort_keys=True)}`",
                f"- Channel-total absolute ranges: `{json.dumps(sampling['channel_total_absolute_ranges'], sort_keys=True)}`",
            ]
        )
    lines.extend(
        [
            "",
            "| Family | Distance | Configs | EWC LER | no-EWC LER | Mean delta | Config-bootstrap 95% CI | Median delta | EWC wins |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary:
        lines.append(
            "| {family_key} | {distance} | {config_count} | {ewc_ler_macro_mean:.6f} | "
            "{noewc_ler_macro_mean:.6f} | {delta_macro_mean:+.6f} | "
            "[{config_bootstrap_ci95_low:+.6f}, {config_bootstrap_ci95_high:+.6f}] | "
            "{delta_macro_median:+.6f} | {ewc_win_count}/{config_count} |".format(**row)
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--input-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--distances", default="7,9")
    parser.add_argument("--details", default=DEFAULT_DETAILS)
    parser.add_argument("--summary", default=DEFAULT_SUMMARY)
    parser.add_argument("--report", default=DEFAULT_REPORT)
    parser.add_argument("--allow-partial", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads(rel(args.manifest).read_text(encoding="utf-8"))
    distances = [int(item.strip()) for item in args.distances.split(",") if item.strip()]
    rows, missing = _detail_rows(
        manifest,
        rel(args.input_dir),
        distances,
        allow_partial=args.allow_partial,
    )
    if not rows:
        raise RuntimeError("no completed A/B inference results found")
    summary = build_summary(rows)
    _write_csv(rel(args.details), rows)
    _write_csv(rel(args.summary), summary)
    _write_markdown(rel(args.report), manifest, summary, missing)
    print(f"[write] {rel(args.details)}")
    print(f"[write] {rel(args.summary)}")
    print(f"[write] {rel(args.report)}")
    if missing:
        print(f"[partial] missing={len(missing)}")


if __name__ == "__main__":
    main()
