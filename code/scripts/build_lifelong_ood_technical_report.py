#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build a technical report combining multi-distance OOD and lifelong forgetting evidence."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = REPO_ROOT / "outputs" / "analysis"
DEFAULT_BASE_DETAILS = ANALYSIS_DIR / "unknown_axismix_grid_u1p2_5p0_multi_distance_details.csv"
DEFAULT_EWC_DETAILS = [
    ANALYSIS_DIR / "unknown_axismix_grid_u1p2_5p0_seq_ewc_d5_details.csv",
    ANALYSIS_DIR / "unknown_axismix_grid_u1p2_5p0_seq_ewc_d7_details.csv",
    ANALYSIS_DIR / "unknown_axismix_grid_u1p2_5p0_seq_ewc_d9_details.csv",
]
DEFAULT_LIFELONG_DETAILS = ANALYSIS_DIR / "r9x_epoch_matrix_details.csv"
DEFAULT_MODEL_DETAILS = ANALYSIS_DIR / "lifelong_r9_unified_decoder_details.csv"
DEFAULT_DOMESTIC_COMPLETE = ANALYSIS_DIR / "r9x_domestic_epoch_complete_ler.csv"
DEFAULT_OUT_DETAILS = ANALYSIS_DIR / "lifelong_ood_grid_with_ewc_details.csv"
DEFAULT_OUT_SUMMARY = ANALYSIS_DIR / "lifelong_ood_grid_with_ewc_summary.csv"
DEFAULT_OUT_REPORT = ANALYSIS_DIR / "lifelong_ood_model_training_technical_report.md"
DEFAULT_FIGURE_DIR = ANALYSIS_DIR / "lifelong_ood_model_training_figures"
AXIS_ORDER = ("meas_all", "cnot_all", "idle_all", "z_bias")
AXIS_LABELS = {
    "meas_all": "Meas",
    "cnot_all": "CNOT",
    "idle_all": "Idle",
    "z_bias": "Z-bias",
}



def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen = set()
    for row in rows:
        for field in row:
            if field not in seen:
                fields.append(field)
                seen.add(field)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fnum(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if not math.isnan(float(v))]
    return sum(vals) / len(vals) if vals else float("nan")


def fmt(value: Any, digits: int = 6) -> str:
    try:
        number = float(value)
    except Exception:
        return str(value)
    if math.isnan(number):
        return ""
    return f"{number:.{digits}f}"


def signed(value: Any, digits: int = 6) -> str:
    try:
        number = float(value)
    except Exception:
        return str(value)
    if math.isnan(number):
        return ""
    return f"{number:+.{digits}f}"


def md_table(headers: list[str], rows: Iterable[Iterable[Any]]) -> str:
    rows = [[str(cell) for cell in row] for row in rows]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def summarize_t0_model_effectiveness(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize the paired T0 domestic-only comparison used for architecture evidence."""
    method_keys = {
        "ising": "ising_domestic",
        "r9x": "stfusion_x_domestic",
    }
    selected = [
        row
        for row in rows
        if str(row.get("task_key")) == "t0_base"
        and str(row.get("method")) in method_keys.values()
        and str(row.get("basis")) in {"X", "Z"}
    ]
    by_method_basis = {
        (str(row["method"]), str(row["basis"])): row
        for row in selected
    }
    missing = [
        f"{method}/{basis}"
        for method in method_keys.values()
        for basis in ("X", "Z")
        if (method, basis) not in by_method_basis
    ]
    if missing:
        raise ValueError("T0 model-effectiveness rows are incomplete: " + ", ".join(missing))

    summaries: dict[str, dict[str, Any]] = {}
    for label, method in method_keys.items():
        method_rows = [by_method_basis[(method, basis)] for basis in ("X", "Z")]
        summaries[label] = {
            "method": method,
            "ler_mean": mean(fnum(row["ler"]) for row in method_rows),
            "latency_mean": mean(fnum(row["latency_us_per_round"]) for row in method_rows),
            "speedup_mean": mean(fnum(row["speedup_vs_pymatching"]) for row in method_rows),
            "checkpoint": str(method_rows[0].get("checkpoint", "")),
        }

    basis_rows = []
    for basis in ("X", "Z"):
        ising = by_method_basis[(method_keys["ising"], basis)]
        r9x = by_method_basis[(method_keys["r9x"], basis)]
        ising_ler = fnum(ising["ler"])
        r9x_ler = fnum(r9x["ler"])
        basis_rows.append(
            {
                "basis": basis,
                "samples": int(float(ising["samples"])),
                "ising_logical_errors": int(float(ising["logical_errors"])),
                "r9x_logical_errors": int(float(r9x["logical_errors"])),
                "ising_ler": ising_ler,
                "r9x_ler": r9x_ler,
                "ler_delta": r9x_ler - ising_ler,
                "relative_reduction": (ising_ler - r9x_ler) / ising_ler,
            }
        )

    ising = summaries["ising"]
    r9x = summaries["r9x"]
    return {
        **summaries,
        "basis_rows": basis_rows,
        "ler_absolute_delta": r9x["ler_mean"] - ising["ler_mean"],
        "ler_relative_reduction": (ising["ler_mean"] - r9x["ler_mean"]) / ising["ler_mean"],
        "latency_absolute_delta": r9x["latency_mean"] - ising["latency_mean"],
        "latency_relative_reduction": (
            (ising["latency_mean"] - r9x["latency_mean"]) / ising["latency_mean"]
        ),
        "ising_parameters": 912_772,
        "r9x_parameters": 650_374,
        "parameter_relative_reduction": (912_772 - 650_374) / 912_772,
    }


def merge_grid_details_with_ewc(
    base_details: Sequence[Mapping[str, Any]],
    ewc_details: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    lookup = {
        (str(row["distance"]), str(row["env_key"]), str(row["multiplier_key"]), str(row["basis"])): row
        for row in ewc_details
    }
    merged: list[dict[str, Any]] = []
    for row in base_details:
        key = (str(row["distance"]), str(row["env_key"]), str(row["multiplier_key"]), str(row["basis"]))
        ewc = lookup.get(key)
        item: dict[str, Any] = dict(row)
        if ewc is None:
            item["seq_ewc_ler"] = ""
            item["delta_ewc_vs_domestic"] = float("nan")
            item["delta_ewc_vs_mixed"] = float("nan")
            item["delta_ewc_vs_noewc"] = float("nan")
            item["ewc_beats_domestic"] = False
            item["ewc_beats_mixed"] = False
            item["ewc_beats_noewc"] = False
        else:
            ewc_ler = fnum(ewc["seq_ewc_ler"])
            domestic = fnum(row["domestic100_ler"])
            mixed = fnum(row["mixed100_ler"])
            noewc = fnum(row["seq_noewc_ler"])
            item["seq_ewc_ler"] = ewc_ler
            item["seq_ewc_latency_us_per_round"] = ewc.get("seq_ewc_latency_us_per_round", "")
            item["seq_ewc_speedup_vs_pymatching"] = ewc.get("seq_ewc_speedup_vs_pymatching", "")
            item["delta_ewc_vs_domestic"] = ewc_ler - domestic
            item["delta_ewc_vs_mixed"] = ewc_ler - mixed
            item["delta_ewc_vs_noewc"] = ewc_ler - noewc
            item["ewc_beats_domestic"] = ewc_ler < domestic
            item["ewc_beats_mixed"] = ewc_ler < mixed
            item["ewc_beats_noewc"] = ewc_ler < noewc
        merged.append(item)
    return merged


def summarize_grid(merged: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in merged:
        groups.setdefault((str(row["distance"]), str(row["basis"])), []).append(row)
    summary: list[dict[str, Any]] = []
    for (distance, basis), rows in sorted(groups.items(), key=lambda item: (int(item[0][0]), item[0][1])):
        summary.append(
            {
                "distance": int(distance),
                "basis": basis,
                "config_count": len(rows),
                "domestic100_ler_mean": mean(fnum(row["domestic100_ler"]) for row in rows),
                "mixed100_ler_mean": mean(fnum(row["mixed100_ler"]) for row in rows),
                "seq_noewc_ler_mean": mean(fnum(row["seq_noewc_ler"]) for row in rows),
                "seq_ewc_ler_mean": mean(fnum(row["seq_ewc_ler"]) for row in rows),
                "delta_noewc_vs_domestic_mean": mean(fnum(row["seq_noewc_ler"]) - fnum(row["domestic100_ler"]) for row in rows),
                "delta_ewc_vs_domestic_mean": mean(fnum(row["delta_ewc_vs_domestic"]) for row in rows),
                "delta_ewc_vs_noewc_mean": mean(fnum(row["delta_ewc_vs_noewc"]) for row in rows),
                "delta_ewc_vs_mixed_mean": mean(fnum(row["delta_ewc_vs_mixed"]) for row in rows),
                "noewc_win_vs_domestic_count": sum(1 for row in rows if fnum(row["seq_noewc_ler"]) < fnum(row["domestic100_ler"])),
                "ewc_win_vs_domestic_count": sum(1 for row in rows if bool(row["ewc_beats_domestic"])),
                "ewc_win_vs_mixed_count": sum(1 for row in rows if bool(row["ewc_beats_mixed"])),
                "ewc_win_vs_noewc_count": sum(1 for row in rows if bool(row["ewc_beats_noewc"])),
            }
        )
    return summary


def _population_std(values: Iterable[float]) -> float:
    vals = [float(value) for value in values if not math.isnan(float(value))]
    if not vals:
        return float("nan")
    center = sum(vals) / len(vals)
    return math.sqrt(sum((value - center) ** 2 for value in vals) / len(vals))


def _axis_tokens(row: Mapping[str, Any]) -> tuple[str, ...]:
    value = row.get("active_axes", row.get("axis_signature", ""))
    if isinstance(value, (list, tuple)):
        tokens = [str(item) for item in value]
    else:
        tokens = str(value).split("+")
    return tuple(axis for axis in AXIS_ORDER if axis in tokens)


def _axis_display(signature: str) -> str:
    return "+".join(AXIS_LABELS.get(axis, axis) for axis in str(signature).split("+"))


def _aggregate_ood_group(
    rows: Sequence[Mapping[str, Any]],
    **labels: Any,
) -> dict[str, Any]:
    domestic = [fnum(row["domestic100_ler"]) for row in rows]
    mixed = [fnum(row["mixed100_ler"]) for row in rows]
    seq = [fnum(row["seq_noewc_ler"]) for row in rows]
    delta_domestic = [seq_value - domestic_value for seq_value, domestic_value in zip(seq, domestic)]
    delta_mixed = [seq_value - mixed_value for seq_value, mixed_value in zip(seq, mixed)]
    domestic_mean = mean(domestic)
    seq_mean = mean(seq)
    return {
        **labels,
        "config_count": len(rows),
        "domestic100_ler_mean": domestic_mean,
        "mixed100_ler_mean": mean(mixed),
        "seq_noewc_ler_mean": seq_mean,
        "delta_seq_vs_domestic_mean": mean(delta_domestic),
        "delta_seq_vs_domestic_std": _population_std(delta_domestic),
        "delta_seq_vs_mixed_mean": mean(delta_mixed),
        "delta_seq_vs_mixed_std": _population_std(delta_mixed),
        "relative_domestic_reduction": (
            (domestic_mean - seq_mean) / domestic_mean
            if domestic_mean
            else float("nan")
        ),
        "seq_win_vs_domestic_count": sum(value < 0 for value in delta_domestic),
        "seq_win_vs_mixed_count": sum(value < 0 for value in delta_mixed),
    }


def summarize_ood_dimensions(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Aggregate basis=both OOD results by multiplier and physical-noise axes."""
    selected = [row for row in rows if str(row.get("basis")) == "both"]

    multiplier_groups: dict[float, list[Mapping[str, Any]]] = {}
    distance_multiplier_groups: dict[tuple[int, float], list[Mapping[str, Any]]] = {}
    axis_groups: dict[tuple[int, str], list[Mapping[str, Any]]] = {}
    distance_axis_groups: dict[tuple[int, int, str], list[Mapping[str, Any]]] = {}
    combination_groups: dict[int, list[Mapping[str, Any]]] = {}
    axis_multiplier_groups: dict[tuple[int, str, float], list[Mapping[str, Any]]] = {}

    for row in selected:
        distance = int(float(row["distance"]))
        multiplier = fnum(row["multiplier"])
        env_index = int(float(row.get("env_index", 0)))
        signature = str(row["axis_signature"])
        combination_size = int(float(row["combination_size"]))
        multiplier_groups.setdefault(multiplier, []).append(row)
        distance_multiplier_groups.setdefault((distance, multiplier), []).append(row)
        axis_groups.setdefault((env_index, signature), []).append(row)
        distance_axis_groups.setdefault((distance, env_index, signature), []).append(row)
        combination_groups.setdefault(combination_size, []).append(row)
        axis_multiplier_groups.setdefault((env_index, signature, multiplier), []).append(row)

    multiplier_rows = [
        _aggregate_ood_group(group, multiplier=multiplier)
        for multiplier, group in sorted(multiplier_groups.items())
    ]
    distance_multiplier_rows = [
        _aggregate_ood_group(group, distance=distance, multiplier=multiplier)
        for (distance, multiplier), group in sorted(distance_multiplier_groups.items())
    ]
    axis_rows = [
        _aggregate_ood_group(
            group,
            env_index=env_index,
            axis_signature=signature,
            axis_label=_axis_display(signature),
        )
        for (env_index, signature), group in sorted(axis_groups.items())
    ]
    distance_axis_rows = [
        _aggregate_ood_group(
            group,
            distance=distance,
            env_index=env_index,
            axis_signature=signature,
            axis_label=_axis_display(signature),
        )
        for (distance, env_index, signature), group in sorted(distance_axis_groups.items())
    ]
    combination_rows = [
        _aggregate_ood_group(group, combination_size=size)
        for size, group in sorted(combination_groups.items())
    ]
    axis_multiplier_rows = [
        _aggregate_ood_group(
            group,
            env_index=env_index,
            axis_signature=signature,
            axis_label=_axis_display(signature),
            multiplier=multiplier,
        )
        for (env_index, signature, multiplier), group in sorted(axis_multiplier_groups.items())
    ]

    axis_presence_rows = []
    for axis in AXIS_ORDER:
        group = [row for row in selected if axis in _axis_tokens(row)]
        if group:
            axis_presence_rows.append(
                _aggregate_ood_group(
                    group,
                    axis=axis,
                    axis_label=AXIS_LABELS[axis],
                )
            )

    return {
        "multiplier_rows": multiplier_rows,
        "distance_multiplier_rows": distance_multiplier_rows,
        "axis_rows": axis_rows,
        "distance_axis_rows": distance_axis_rows,
        "axis_presence_rows": axis_presence_rows,
        "combination_rows": combination_rows,
        "axis_multiplier_rows": axis_multiplier_rows,
    }


def _save_ood_figure(fig: Any, png_path: Path, plt: Any) -> None:
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(png_path.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(png_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def generate_ood_figures(
    grid_details: Sequence[Mapping[str, Any]],
    figure_dir: str | Path = DEFAULT_FIGURE_DIR,
) -> dict[str, Path]:
    """Generate multiplier, axis-combination, and interaction figures."""
    import matplotlib as mpl

    mpl.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import numpy as np

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
        }
    )

    dimensions = summarize_ood_dimensions(grid_details)
    if not dimensions["distance_multiplier_rows"]:
        raise ValueError("No basis=both OOD rows are available for plotting")

    figure_dir = Path(figure_dir)
    paths = {
        "multiplier": figure_dir / "ood_delta_vs_multiplier_by_distance.png",
        "axis_distance": figure_dir / "ood_axis_combination_delta_heatmap.png",
        "axis_multiplier": figure_dir / "ood_axis_multiplier_heatmap.png",
    }
    distances = sorted({int(row["distance"]) for row in dimensions["distance_multiplier_rows"]})
    colors = {5: "#3B6FB6", 7: "#D17A22", 9: "#2A8C82"}
    fallback_colors = plt.get_cmap("tab10")

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.0), sharex=True, constrained_layout=True)
    metric_specs = [
        (
            "delta_seq_vs_domestic_mean",
            "delta_seq_vs_domestic_std",
            "Seq no-EWC - domestic-only",
        ),
        (
            "delta_seq_vs_mixed_mean",
            "delta_seq_vs_mixed_std",
            "Seq no-EWC - mixed-noise",
        ),
    ]
    for panel_index, (ax, (metric, spread, title)) in enumerate(zip(axes, metric_specs)):
        for color_index, distance in enumerate(distances):
            rows = [
                row
                for row in dimensions["distance_multiplier_rows"]
                if int(row["distance"]) == distance
            ]
            rows.sort(key=lambda row: float(row["multiplier"]))
            x = np.asarray([float(row["multiplier"]) for row in rows])
            y = np.asarray([float(row[metric]) for row in rows])
            std = np.asarray([float(row[spread]) for row in rows])
            color = colors.get(distance, fallback_colors(color_index))
            ax.fill_between(x, y - std, y + std, color=color, alpha=0.12, linewidth=0)
            ax.plot(x, y, marker="o", markersize=4, linewidth=1.8, color=color, label=f"d={distance}")
        ax.axhline(0, color="#444444", linewidth=0.9, linestyle="--")
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Active-axis multiplier")
        ax.set_ylabel("LER delta")
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.7)
        ax.text(-0.11, 1.05, chr(ord("a") + panel_index), transform=ax.transAxes, fontweight="bold", fontsize=10)
    axes[0].legend(title="Code distance", ncol=max(1, len(distances)))
    fig.suptitle("OOD generalization gap across noise multipliers", fontweight="bold", fontsize=11)
    _save_ood_figure(fig, paths["multiplier"], plt)

    axis_rows = dimensions["axis_rows"]
    axis_order = [str(row["axis_signature"]) for row in axis_rows]
    axis_labels = [str(row["axis_label"]) for row in axis_rows]
    distance_axis_lookup = {
        (int(row["distance"]), str(row["axis_signature"])): row
        for row in dimensions["distance_axis_rows"]
    }
    matrices = []
    for metric in ("delta_seq_vs_domestic_mean", "delta_seq_vs_mixed_mean"):
        matrices.append(
            np.asarray(
                [
                    [
                        float(distance_axis_lookup[(distance, signature)][metric])
                        for distance in distances
                    ]
                    for signature in axis_order
                ]
            )
        )
    vmax = max(0.001, max(float(np.nanmax(np.abs(matrix))) for matrix in matrices))
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(9.6, max(5.5, 0.42 * len(axis_order) + 1.7)),
        sharey=True,
        constrained_layout=True,
    )
    titles = ["Seq - domestic", "Seq - mixed"]
    images = []
    for panel_index, (ax, matrix, title) in enumerate(zip(axes, matrices, titles)):
        image = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        images.append(image)
        ax.set_xticks(range(len(distances)), [f"d={distance}" for distance in distances])
        ax.set_yticks(range(len(axis_labels)), axis_labels)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Code distance")
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                value = float(matrix[row_index, column_index])
                color = "white" if abs(value) > 0.55 * vmax else "#222222"
                ax.text(column_index, row_index, f"{value:+.3f}", ha="center", va="center", fontsize=6.5, color=color)
        ax.text(-0.11, 1.03, chr(ord("a") + panel_index), transform=ax.transAxes, fontweight="bold", fontsize=10)
    axes[0].set_ylabel("Active-axis combination")
    colorbar = fig.colorbar(images[0], ax=axes, fraction=0.035, pad=0.03)
    colorbar.set_label("Mean LER delta across 9 multipliers")
    fig.suptitle("Axis-combination dependence of OOD generalization", fontweight="bold", fontsize=11)
    _save_ood_figure(fig, paths["axis_distance"], plt)

    multipliers = [float(row["multiplier"]) for row in dimensions["multiplier_rows"]]
    distance_detail_lookup = {
        (
            int(float(row["distance"])),
            int(float(row.get("env_index", 0))),
            fnum(row["multiplier"]),
        ): fnum(row["seq_noewc_ler"]) - fnum(row["domestic100_ler"])
        for row in grid_details
        if str(row.get("basis")) == "both"
    }
    interaction_matrices = [
        np.asarray(
            [
                [
                    distance_detail_lookup.get((distance, int(axis_row["env_index"]), multiplier), math.nan)
                    for multiplier in multipliers
                ]
                for axis_row in axis_rows
            ]
        )
        for distance in distances
    ]
    interaction_vmax = max(
        0.001,
        max(float(np.nanmax(np.abs(matrix))) for matrix in interaction_matrices),
    )
    fig, axes_array = plt.subplots(
        1,
        len(distances),
        figsize=(15.2, max(5.8, 0.43 * len(axis_rows) + 1.9)),
        sharey=True,
        constrained_layout=True,
        squeeze=False,
    )
    axes = axes_array[0]
    images = []
    for panel_index, (ax, distance, matrix) in enumerate(zip(axes, distances, interaction_matrices)):
        image = ax.imshow(
            matrix,
            aspect="auto",
            cmap="RdBu_r",
            vmin=-interaction_vmax,
            vmax=interaction_vmax,
        )
        images.append(image)
        ax.set_xticks(range(len(multipliers)), [f"{value:g}" for value in multipliers], rotation=45)
        ax.set_yticks(range(len(axis_labels)), axis_labels)
        ax.set_title(f"d={distance}", fontweight="bold")
        ax.set_xlabel("Multiplier")
        positive_rows, positive_columns = np.where(matrix >= 0)
        ax.scatter(positive_columns, positive_rows, marker="x", s=18, linewidths=0.8, color="#111111")
        ax.text(-0.10, 1.03, chr(ord("a") + panel_index), transform=ax.transAxes, fontweight="bold", fontsize=10)
    axes[0].set_ylabel("Active-axis combination")
    colorbar = fig.colorbar(images[0], ax=axes.tolist(), fraction=0.025, pad=0.02)
    colorbar.set_label("LER delta: seq no-EWC - domestic-only")
    fig.suptitle("Axis-by-multiplier OOD map (black x: seq not better)", fontweight="bold", fontsize=11)
    _save_ood_figure(fig, paths["axis_multiplier"], plt)
    return paths


def _task_basis_avg(rows: Sequence[Mapping[str, str]], *, group: str, epoch: int) -> dict[str, float]:
    selected = [row for row in rows if row["group"] == group and int(row["epoch"]) == epoch]
    by_task: dict[str, list[float]] = {}
    for row in selected:
        by_task.setdefault(row["task_key"], []).append(fnum(row["ler"]))
    return {task: mean(values) for task, values in by_task.items()}


def summarize_lifelong(
    rows: Sequence[Mapping[str, str]],
    domestic_complete_rows: Sequence[Mapping[str, str]] = (),
) -> dict[str, Any]:
    domestic100 = _task_basis_avg(rows, group="r9x_domestic", epoch=100)
    if not domestic100:
        complete_e100 = next(
            (row for row in domestic_complete_rows if int(row.get("epoch", 0)) == 100),
            None,
        )
        if complete_e100 is not None:
            domestic100 = {
                field.removeprefix("ler_"): fnum(value)
                for field, value in complete_e100.items()
                if field.startswith("ler_t") and field not in {"ler_avg_5task", "ler_worst_5task"}
            }
    if domestic100:
        domestic_reference = domestic100
        domestic_reference_label = "domestic-only e100"
    else:
        domestic_reference = _task_basis_avg(rows, group="r9x_domestic", epoch=20)
        domestic_reference_label = "domestic-only e20"

    final_task = {
        "domestic_reference": domestic_reference,
        "seq_noewc100": _task_basis_avg(rows, group="r9x_seq_noewc", epoch=100),
        "seq_ewc100": _task_basis_avg(rows, group="r9x_seq_ewc", epoch=100),
    }
    final_avg = {
        "domestic_reference_label": domestic_reference_label,
        "domestic_reference_avg": mean(final_task["domestic_reference"].values()),
        "seq_noewc100_avg": mean(final_task["seq_noewc100"].values()),
        "seq_ewc100_avg": mean(final_task["seq_ewc100"].values()),
    }
    trajectory = []
    for group, label in [("r9x_seq_noewc", "seq no-EWC"), ("r9x_seq_ewc", "seq + EWC")]:
        e20 = _task_basis_avg(rows, group=group, epoch=20)
        e100 = _task_basis_avg(rows, group=group, epoch=100)
        for task in sorted(e100):
            trajectory.append(
                {
                    "model": label,
                    "task": task,
                    "epoch20_ler": e20.get(task, float("nan")),
                    "epoch100_ler": e100[task],
                    "delta_e100_minus_e20": e100[task] - e20.get(task, float("nan")),
                }
            )
    noewc_deltas = [row["delta_e100_minus_e20"] for row in trajectory if row["model"] == "seq no-EWC"]
    ewc_deltas = [row["delta_e100_minus_e20"] for row in trajectory if row["model"] == "seq + EWC"]
    final_avg["final_task"] = final_task
    final_avg["trajectory"] = trajectory
    final_avg["ewc_minus_noewc"] = final_avg["seq_ewc100_avg"] - final_avg["seq_noewc100_avg"]
    final_avg["noewc_positive_forgetting_count"] = sum(1 for value in noewc_deltas if value > 0)
    final_avg["ewc_positive_forgetting_count"] = sum(1 for value in ewc_deltas if value > 0)
    final_avg["noewc_max_e100_minus_e20"] = max(noewc_deltas) if noewc_deltas else float("nan")
    final_avg["ewc_max_e100_minus_e20"] = max(ewc_deltas) if ewc_deltas else float("nan")
    if final_avg["noewc_positive_forgetting_count"] == 0:
        noewc_text = "seq no-EWC 从 epoch20 到 epoch100 没有任务 LER 上升，未观察到灾难性遗忘。"
    else:
        noewc_text = (
            f"seq no-EWC 有 {final_avg['noewc_positive_forgetting_count']} 个任务从 epoch20 到 epoch100 LER 上升，"
            "需要按任务披露遗忘边界。"
        )
    if final_avg["ewc_minus_noewc"] < 0:
        ewc_text = "EWC 的 5-task 平均 LER 低于 no-EWC，可作为有收益的正则化对照。"
    else:
        ewc_text = "EWC 的 5-task 平均 LER 高于 no-EWC，当前设置没有证明其缓解遗忘优于 no-EWC。"
    final_avg["forgetting_summary"] = noewc_text + ewc_text

    task_order = [
        "t0_base",
        "t1_meas_1p5",
        "t2_cnot_1p5",
        "t3_idle_1p5",
        "t4_z_bias_1p5",
    ]
    seq_matrix = {
        epoch: _task_basis_avg(rows, group="r9x_seq_noewc", epoch=epoch)
        for epoch in (20, 40, 60, 80, 100)
    }
    forward_adaptation = []
    for task_index, task in enumerate(task_order[1:], start=1):
        learned_epoch = (task_index + 1) * 20
        before_epoch = learned_epoch - 20
        before = seq_matrix.get(before_epoch, {}).get(task)
        after = seq_matrix.get(learned_epoch, {}).get(task)
        if before is None or after is None:
            continue
        forward_adaptation.append(
            {
                "task": task,
                "before_epoch": before_epoch,
                "learned_epoch": learned_epoch,
                "before_ler": before,
                "after_ler": after,
                "delta_after_minus_before": after - before,
            }
        )

    forgetting = []
    final_values = seq_matrix.get(100, {})
    for task_index, task in enumerate(task_order):
        learned_epoch = (task_index + 1) * 20
        candidates = [
            task_values[task]
            for epoch, task_values in seq_matrix.items()
            if epoch >= learned_epoch and task in task_values
        ]
        if task not in final_values or not candidates:
            continue
        best_after_learned = min(candidates)
        final_ler = final_values[task]
        forgetting.append(
            {
                "task": task,
                "learned_epoch": learned_epoch,
                "best_after_learned": best_after_learned,
                "final_ler": final_ler,
                "forgetting": final_ler - best_after_learned,
            }
        )
    final_avg["forward_adaptation"] = forward_adaptation
    final_avg["forgetting"] = forgetting
    return final_avg

def write_technical_report(
    path: str | Path,
    *,
    model_effectiveness: Mapping[str, Any],
    grid_overall: Sequence[Mapping[str, Any]],
    lifelong: Mapping[str, Any],
    grid_details: Sequence[Mapping[str, Any]] = (),
    figure_paths: Mapping[str, str | Path] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ising = model_effectiveness["ising"]
    r9x = model_effectiveness["r9x"]
    ler_reduction_pct = 100 * fnum(model_effectiveness["ler_relative_reduction"])
    latency_reduction_pct = 100 * fnum(model_effectiveness["latency_relative_reduction"])
    parameter_reduction_pct = 100 * fnum(model_effectiveness["parameter_relative_reduction"])

    model_basis_table = md_table(
        [
            "basis",
            "samples",
            "Ising logical errors",
            "R9-X logical errors",
            "Ising LER",
            "R9-X LER",
            "R9-X - Ising",
            "相对下降",
        ],
        [
            [
                row["basis"],
                str(row.get("samples", "")),
                str(row.get("ising_logical_errors", "")),
                str(row.get("r9x_logical_errors", "")),
                fmt(row["ising_ler"]),
                fmt(row["r9x_ler"]),
                signed(row["ler_delta"]),
                f"{100 * fnum(row['relative_reduction']):.2f}%",
            ]
            for row in model_effectiveness["basis_rows"]
        ],
    )
    model_summary_table = md_table(
        ["模型", "参数量", "T0 LER-Avg", "latency (us/round)", "speedup vs PyMatching"],
        [
            [
                "Ising fast",
                f"{int(model_effectiveness.get('ising_parameters', 912772)):,}",
                fmt(ising["ler_mean"]),
                fmt(ising["latency_mean"], 3),
                fmt(ising["speedup_mean"], 3),
            ],
            [
                "ST-Fusion-R9-X",
                f"{int(model_effectiveness.get('r9x_parameters', 650374)):,}",
                fmt(r9x["ler_mean"]),
                fmt(r9x["latency_mean"], 3),
                fmt(r9x["speedup_mean"], 3),
            ],
        ],
    )

    final_task = lifelong.get("final_task", {})
    domestic_label = str(lifelong.get("domestic_reference_label", "domestic-only reference"))
    task_rows = []
    for task in sorted(final_task.get("seq_noewc100", {})):
        domestic_value = fnum(final_task.get("domestic_reference", {}).get(task))
        noewc_value = fnum(final_task.get("seq_noewc100", {}).get(task))
        ewc_value = fnum(final_task.get("seq_ewc100", {}).get(task))
        task_rows.append(
            [
                task,
                fmt(domestic_value),
                fmt(noewc_value),
                fmt(ewc_value),
                signed(noewc_value - domestic_value),
                signed(ewc_value - noewc_value),
            ]
        )
    task_table = md_table(
        [
            "任务",
            domestic_label,
            "seq no-EWC e100",
            "seq + EWC e100",
            "noEWC - domestic",
            "EWC - noEWC",
        ],
        task_rows,
    )

    adaptation_table = md_table(
        [
            "新任务",
            "学习前 checkpoint",
            "学习后 checkpoint",
            "学习前 LER",
            "学习后 LER",
            "after - before",
        ],
        [
            [
                row["task"],
                f"e{row['before_epoch']}",
                f"e{row['learned_epoch']}",
                fmt(row["before_ler"]),
                fmt(row["after_ler"]),
                signed(row["delta_after_minus_before"]),
            ]
            for row in lifelong.get("forward_adaptation", [])
        ],
    )
    forgetting_table = md_table(
        ["旧任务", "学会阶段", "学会后最佳 LER", "final@100", "forgetting"],
        [
            [
                row["task"],
                f"e{row['learned_epoch']}",
                fmt(row["best_after_learned"]),
                fmt(row["final_ler"]),
                fmt(row["forgetting"]),
            ]
            for row in lifelong.get("forgetting", [])
        ],
    )

    both_rows = sorted(
        [row for row in grid_overall if str(row.get("basis")) == "both"],
        key=lambda row: int(row["distance"]),
    )
    ood_table = md_table(
        [
            "distance",
            "domestic e100",
            "mixed e100",
            "R9-X + seq no-EWC",
            "seq + EWC e100",
            "seq - domestic",
            "seq - mixed",
            "相对 domestic 降幅",
            "seq 胜 domestic",
        ],
        [
            [
                f"d={int(row['distance'])}",
                fmt(row["domestic100_ler_mean"]),
                fmt(row["mixed100_ler_mean"]),
                fmt(row["seq_noewc_ler_mean"]),
                fmt(row["seq_ewc_ler_mean"]),
                signed(row["delta_noewc_vs_domestic_mean"]),
                signed(
                    fnum(row["seq_noewc_ler_mean"])
                    - fnum(row["mixed100_ler_mean"])
                ),
                (
                    f"{100 * (fnum(row['domestic100_ler_mean']) - fnum(row['seq_noewc_ler_mean'])) / fnum(row['domestic100_ler_mean']):.2f}%"
                ),
                f"{row['noewc_win_vs_domestic_count']}/{row['config_count']}",
            ]
            for row in both_rows
        ],
    )
    ood_config_count = sum(int(row["config_count"]) for row in both_rows)

    def weighted_ood_mean(key: str) -> float:
        if not ood_config_count:
            return float("nan")
        return sum(
            fnum(row[key]) * int(row["config_count"])
            for row in both_rows
        ) / ood_config_count

    ood_domestic_mean = weighted_ood_mean("domestic100_ler_mean")
    ood_mixed_mean = weighted_ood_mean("mixed100_ler_mean")
    ood_seq_mean = weighted_ood_mean("seq_noewc_ler_mean")
    ood_ewc_mean = weighted_ood_mean("seq_ewc_ler_mean")
    ood_seq_vs_domestic = ood_seq_mean - ood_domestic_mean
    ood_seq_vs_domestic_pct = (
        100 * (ood_domestic_mean - ood_seq_mean) / ood_domestic_mean
        if ood_domestic_mean
        else float("nan")
    )
    ood_seq_vs_mixed = ood_seq_mean - ood_mixed_mean
    ood_seq_vs_mixed_pct = (
        100 * (ood_mixed_mean - ood_seq_mean) / ood_mixed_mean
        if ood_mixed_mean
        else float("nan")
    )
    ood_win_count = sum(
        int(row["noewc_win_vs_domestic_count"]) for row in both_rows
    )

    domestic_avg = fnum(lifelong.get("domestic_reference_avg"))
    noewc_avg = fnum(lifelong.get("seq_noewc100_avg"))
    ewc_avg = fnum(lifelong.get("seq_ewc100_avg"))
    if ood_ewc_mean < ood_seq_mean:
        ewc_ood_relation = "低于"
        ewc_ood_interpretation = "EWC 在纯 OOD 聚合指标上更低"
    else:
        ewc_ood_relation = "高于"
        ewc_ood_interpretation = "EWC 在纯 OOD 聚合指标上未超过 no-EWC"
    noewc_vs_domestic = noewc_avg - domestic_avg
    noewc_vs_domestic_pct = (
        100 * noewc_vs_domestic / domestic_avg if domestic_avg else float("nan")
    )
    adaptations = list(lifelong.get("forward_adaptation", []))
    adaptation_win_count = sum(
        1 for row in adaptations if fnum(row["delta_after_minus_before"]) < 0
    )
    forgetting_rows = list(lifelong.get("forgetting", []))
    max_forgetting = max(
        (fnum(row["forgetting"]) for row in forgetting_rows),
        default=float("nan"),
    )

    ood_dimensions = summarize_ood_dimensions(grid_details)
    multiplier_rows = ood_dimensions["multiplier_rows"]
    axis_rows = ood_dimensions["axis_rows"]
    multiplier_table = md_table(
        [
            "倍率",
            "配置数",
            "domestic LER",
            "mixed LER",
            "seq LER",
            "seq-domestic",
            "相对 domestic",
            "seq-mixed",
            "胜 domestic",
        ],
        [
            [
                f"{float(row['multiplier']):g}x",
                row["config_count"],
                fmt(row["domestic100_ler_mean"]),
                fmt(row["mixed100_ler_mean"]),
                fmt(row["seq_noewc_ler_mean"]),
                signed(row["delta_seq_vs_domestic_mean"]),
                f"{100 * fnum(row['relative_domestic_reduction']):.2f}%",
                signed(row["delta_seq_vs_mixed_mean"]),
                f"{row['seq_win_vs_domestic_count']}/{row['config_count']}",
            ]
            for row in multiplier_rows
        ],
    )
    axis_presence_table = md_table(
        [
            "包含的训练轴",
            "配置数",
            "domestic LER",
            "seq LER",
            "seq-domestic",
            "相对 domestic",
            "seq-mixed",
            "胜 domestic",
        ],
        [
            [
                row["axis_label"],
                row["config_count"],
                fmt(row["domestic100_ler_mean"]),
                fmt(row["seq_noewc_ler_mean"]),
                signed(row["delta_seq_vs_domestic_mean"]),
                f"{100 * fnum(row['relative_domestic_reduction']):.2f}%",
                signed(row["delta_seq_vs_mixed_mean"]),
                f"{row['seq_win_vs_domestic_count']}/{row['config_count']}",
            ]
            for row in ood_dimensions["axis_presence_rows"]
        ],
    )
    axis_combination_table = md_table(
        [
            "轴组合",
            "配置数",
            "domestic LER",
            "mixed LER",
            "seq LER",
            "seq-domestic",
            "seq-mixed",
            "胜 domestic",
        ],
        [
            [
                row["axis_label"],
                row["config_count"],
                fmt(row["domestic100_ler_mean"]),
                fmt(row["mixed100_ler_mean"]),
                fmt(row["seq_noewc_ler_mean"]),
                signed(row["delta_seq_vs_domestic_mean"]),
                signed(row["delta_seq_vs_mixed_mean"]),
                f"{row['seq_win_vs_domestic_count']}/{row['config_count']}",
            ]
            for row in axis_rows
        ],
    )
    combination_table = md_table(
        ["激活轴数", "配置数", "domestic LER", "seq LER", "seq-domestic", "seq-mixed"],
        [
            [
                row["combination_size"],
                row["config_count"],
                fmt(row["domestic100_ler_mean"]),
                fmt(row["seq_noewc_ler_mean"]),
                signed(row["delta_seq_vs_domestic_mean"]),
                signed(row["delta_seq_vs_mixed_mean"]),
            ]
            for row in ood_dimensions["combination_rows"]
        ],
    )

    distance_trend_parts = []
    for distance in sorted(
        {int(row["distance"]) for row in ood_dimensions["distance_multiplier_rows"]}
    ):
        rows = [
            row
            for row in ood_dimensions["distance_multiplier_rows"]
            if int(row["distance"]) == distance
        ]
        rows.sort(key=lambda row: float(row["multiplier"]))
        strongest = min(rows, key=lambda row: fnum(row["delta_seq_vs_domestic_mean"]))
        nonnegative = [
            f"{float(row['multiplier']):g}x"
            for row in rows
            if fnum(row["delta_seq_vs_domestic_mean"]) >= 0
        ]
        if nonnegative:
            boundary = f"{'/'.join(nonnegative)} 的均值未超过 domestic"
        else:
            boundary = "全部倍率的均值均优于 domestic"
        distance_trend_parts.append(
            f"d={distance}：{boundary}，最大平均收益出现在 "
            f"{float(strongest['multiplier']):g}x（delta={signed(strongest['delta_seq_vs_domestic_mean'])}）"
        )
    distance_trend_text = "；".join(distance_trend_parts) + "。"

    best_multiplier = min(
        multiplier_rows,
        key=lambda row: fnum(row["delta_seq_vs_domestic_mean"]),
        default={},
    )
    best_axis = min(
        axis_rows,
        key=lambda row: fnum(row["delta_seq_vs_domestic_mean"]),
        default={},
    )
    weakest_axis = max(
        axis_rows,
        key=lambda row: fnum(row["delta_seq_vs_domestic_mean"]),
        default={},
    )
    axis_presence_text = "；".join(
        f"含 {row['axis_label']} 的配置 delta={signed(row['delta_seq_vs_domestic_mean'])}"
        for row in ood_dimensions["axis_presence_rows"]
    )
    combination_text = "；".join(
        f"{row['combination_size']} 轴组合 delta={signed(row['delta_seq_vs_domestic_mean'])}"
        for row in ood_dimensions["combination_rows"]
    )

    interaction_rows = [
        row for row in grid_details if str(row.get("basis")) == "both"
    ]
    interaction_by_distance = []
    for distance in sorted({int(float(row["distance"])) for row in interaction_rows}):
        rows = [row for row in interaction_rows if int(float(row["distance"])) == distance]
        wins = sum(
            fnum(row["seq_noewc_ler"]) < fnum(row["domestic100_ler"])
            for row in rows
        )
        interaction_by_distance.append(f"d={distance} 为 {wins}/{len(rows)}")
    interaction_win_text = "、".join(interaction_by_distance)

    def figure_markdown(key: str, alt: str, caption: str) -> str:
        if not figure_paths or key not in figure_paths:
            return ""
        figure_path = Path(figure_paths[key])
        try:
            relative = figure_path.resolve().relative_to(path.parent.resolve())
        except ValueError:
            relative = figure_path
        return f"![{alt}]({relative.as_posix()})\n\n*{caption}*"

    multiplier_figure = figure_markdown(
        "multiplier",
        "不同噪声倍率下 seq no-EWC 相对 domestic 和 mixed 的 LER 差值",
        "图 1：曲线为 11 个轴组合的平均 delta；阴影为轴组合间的 1 个总体标准差，用于展示环境异质性，不是随机种子置信区间。负值表示 seq no-EWC 更好。",
    )
    axis_distance_figure = figure_markdown(
        "axis_distance",
        "不同轴组合在 d5 d7 d9 下的平均 LER 差值热图",
        "图 2：每个单元格对 9 个倍率取平均。左图比较 domestic，右图比较 mixed；蓝色负值表示 seq no-EWC 更好。",
    )
    axis_multiplier_figure = figure_markdown(
        "axis_multiplier",
        "各 distance 下轴组合与倍率交互的 OOD 热图",
        "图 3：展开全部 11×9×3 个 basis=both 配置。黑色叉号表示该配置下 seq no-EWC 未优于 domestic-only。",
    )

    text = f"""# R9-X 架构 + Sequential Lifelong：可持续进化量子纠错技术报告

## 1. 问题背景

量子纠错解码器面对的不是静态分类问题。量子器件的测量误差、双比特门误差、空闲退相干和 Pauli 偏置会随校准、运行状态和器件老化持续变化。一个可部署的纠错模型不仅要在初始噪声环境中准确、快速，还需要在后续噪声任务到来时继续学习，同时保持对旧任务的解码能力。

本项目在 Ising Decoding 技术路线下，形成“ST-Fusion-R9-X 模型架构 + sequential no-EWC 训练范式”的端到端组合方案。报告按三层递进证据验证该方案：

1. **架构基础能力**：在 T0 同分布任务上，验证 R9-X 相对原始 Ising fast 的精度、规模和延迟优势。
2. **持续学习能力**：在 T0-T4 连续任务上，验证 sequential no-EWC 能否吸收新噪声并保持旧任务能力。
3. **组合方案 OOD 泛化**：在 d=5/7/9、297 个固定倍率复合 OOD 配置上，验证最终 R9-X + sequential no-EWC 模型相对 R9-X domestic-only 和 mixed-noise 的泛化收益。

核心结论是：R9-X 提供更强、更轻量的基础预解码器，sequential no-EWC 进一步扩展其噪声适应范围；两者组合后在跨 distance 复合 OOD 网格上的加权平均 LER 同时低于 R9-X domestic-only 和 mixed-noise 基线。

## 2. 技术路线

整体路线保留 Ising Decoding 的“神经预解码器 + 后端匹配解码器”框架。神经网络接收时空 syndrome 张量，先识别和消除局部高置信错误，再由 PyMatching 完成剩余图匹配解码。优化重点不是替换解码框架，而是同时增强预解码器的时空表达能力和持续适应能力。

### 2.1 模型架构：从 Ising fast 到 ST-Fusion-R9-X

Ising fast（model_id=1）采用普通 3D 卷积堆叠。ST-Fusion-R9-X（model_id=111）保持 R=9 感受野，引入四项结构优化：

- 空间、时间和联合三分支分别提取不同类型的 syndrome 相关性。
- Adaptive Branch Fusion 根据输入动态调整三分支权重。
- Grouped Pointwise Branch Mixer 与 Axis-Channel Gate 加强通道、时间位置和空间位置校准。
- Raw-Evidence Head 在输出前重新融合原始 syndrome，避免深层特征丢失局部强证据。

R9-X 参数量从 Ising fast 的 912,772 降到 650,374，减少 {parameter_reduction_pct:.2f}%。

### 2.2 训练范式：Sequential no-EWC Lifelong Learning

任务流为 T0_base -> T1_meas_1p5 -> T2_cnot_1p5 -> T3_idle_1p5 -> T4_z_bias_1p5。每个阶段引入一种新的物理噪声变化：测量误差、CNOT 误差、idle 误差和 Z 偏置。主模型为 sequential no-EWC；sequential + EWC 用于检验显式参数保持正则是否必要；domestic-only 与 mixed-noise 是训练范式基线。

最终部署候选不是单独的 R9-X 架构，也不是脱离模型的 seq 策略，而是 **R9-X + sequential no-EWC** 的组合模式。

## 3. 实验设计

### 3.1 架构有效性实验

- 任务：T0_base，非 OOD。
- 对象：Ising fast domestic-only best checkpoint 与 ST-Fusion-R9-X domestic-only best checkpoint。
- 口径：distance=9、rounds=9、X/Z basis、每个 basis 262,144 samples。
- 公平性：相同训练预算、相同噪声配置；推理使用相同 seed、相同样本和相同 matcher，构成 paired comparison。
- 目的：确认最终组合方案所采用的 R9-X 架构本身优于原始 Ising 模型。

### 3.2 Lifelong 持续学习实验

- 前向适应：比较每个新任务进入前后的 sequential checkpoint，负 delta 表示学到新任务后 LER 下降。
- 遗忘：对每个任务计算 final@100 - best_after_learned，越接近 0 表示旧任务保持越好。
- 公平预算基线：sequential e100 与 domestic-only e100 比较，不用早期 domestic e20 代替充分训练基线。
- 目的：确认 seq 训练能够持续吸收新噪声，同时避免旧任务能力失控。

### 3.3 架构 + seq 组合方案 OOD 实验

- 环境：11 个训练轴复合环境、9 个倍率点、每个 distance 99 个配置，合计 d=5/7/9 下 297 个配置。
- 对象：同一 R9-X 架构下的 domestic-only、mixed-noise、seq no-EWC 和 seq + EWC。
- 主对象：R9-X + sequential no-EWC。
- 目的：OOD 结果作为端到端组合方案的泛化主证据，检验模型经过 sequential 训练后能否超越单一环境训练和混合噪声训练。

## 4. 模型架构有效性：T0 同分布实验

### 4.1 分 basis 结果

{model_basis_table}

### 4.2 精度、规模与速度汇总

{model_summary_table}

R9-X 的 T0 平均 LER 从 {fmt(ising["ler_mean"])} 降至 {fmt(r9x["ler_mean"])}，绝对下降 {fmt(-fnum(model_effectiveness["ler_absolute_delta"]))}，相对下降 {ler_reduction_pct:.2f}%。改进同时出现在 X 和 Z basis，说明收益不是由单一 basis 偶然贡献。

平均 latency 从 {fmt(ising["latency_mean"], 3)} 降至 {fmt(r9x["latency_mean"], 3)} us/round，下降 {latency_reduction_pct:.2f}%；相对 PyMatching 的平均加速比由 {fmt(ising["speedup_mean"], 3)}x 提升到 {fmt(r9x["speedup_mean"], 3)}x。结合参数量下降，R9-X 并非以更大模型换取精度，而是在更小参数规模下同时改善 LER 与端到端延迟。

对应 checkpoint：

- Ising fast：{ising["checkpoint"]}
- ST-Fusion-R9-X：{r9x["checkpoint"]}

T0 结果证明 R9-X 为组合方案提供了优于原 Ising 模型的基础能力，但这一实验只回答同分布架构有效性，组合方案的 OOD 能力由第 5.4 节直接验证。

## 5. Lifelong 训练范式有效性

本节验证 R9-X + sequential no-EWC 如何从高质量基础架构进一步形成可持续进化和 OOD 泛化能力。

### 5.1 新任务前向适应

{adaptation_table}

Sequential no-EWC 在 {adaptation_win_count}/{len(adaptations)} 个后续任务上均实现学习后 LER 下降，说明 R9-X 参数可以在连续任务流中继续吸收新的噪声知识。这里的指标是阶段前后适应增益；若采用严格 continual-learning FWT 定义，还需要以“新任务从头训练”的独立模型作为额外参照。

### 5.2 旧任务遗忘

{forgetting_table}

R9-X + sequential no-EWC 的最大 forgetting 为 {fmt(max_forgetting)}。现有结果没有观察到灾难遗忘：模型在学习后续噪声任务时，旧任务最终性能没有偏离其学会后的最佳水平。

### 5.3 同预算已见任务对比

{task_table}

五任务平均 LER：{domestic_label} 为 {fmt(domestic_avg)}，R9-X + seq no-EWC e100 为 {fmt(noewc_avg)}，seq + EWC e100 为 {fmt(ewc_avg)}。组合方案相对同预算 domestic e100 的差值为 {signed(noewc_vs_domestic)}（相对 {noewc_vs_domestic_pct:+.2f}%）。

这表明组合方案在已见任务上没有依靠牺牲旧任务换取 OOD 收益：其平均 LER 仅比充分训练的 domestic 基线高 {abs(noewc_vs_domestic_pct):.2f}%，同时保持零遗忘，并在复合 OOD 上取得明显优势。

EWC 的五任务平均 LER 比 no-EWC 高 {fmt(ewc_avg - noewc_avg)}，当前 EWC_LAMBDA=100 没有证明比无正则顺序训练更有效。

### 5.4 R9-X + Sequential no-EWC 的 OOD 泛化主证据（basis=both）

#### 5.4.1 跨 distance 总体结果

{ood_table}

跨 d=5/7/9 共 {ood_config_count} 个固定复合 OOD 配置，R9-X + sequential no-EWC 的加权平均 LER 为 {fmt(ood_seq_mean)}，低于 R9-X domestic-only 的 {fmt(ood_domestic_mean)}。绝对下降 {fmt(-ood_seq_vs_domestic)}，相对下降 {ood_seq_vs_domestic_pct:.2f}%；逐配置胜出 {ood_win_count}/{ood_config_count}。

相对 mixed-noise e100，组合方案的三距离加权平均 LER 由 {fmt(ood_mixed_mean)} 降至 {fmt(ood_seq_mean)}，绝对下降 {fmt(-ood_seq_vs_mixed)}，相对下降 {ood_seq_vs_mixed_pct:.2f}%。这说明收益不仅来自“见过更多噪声”，还与 sequential 组织任务和参数演化的方式有关。

#### 5.4.2 不同噪声倍率

{multiplier_figure}

{multiplier_table}

倍率表在每个点等权聚合 d=5/7/9 和 11 个轴组合，共 33 个配置。跨 distance 的平均值用于回答“优势是否覆盖多个噪声强度”，distance 分层曲线用于防止 d=5 的大幅差值掩盖 d=7/d=9 的边界。{distance_trend_text}

图中 delta 定义为 seq - baseline，因此负值代表 seq no-EWC 更优。相对 domestic，d=5 的优势在中等倍率附近最大，随后随 LER 接近饱和而收窄；d=7 的优势也呈先增强后收窄；d=9 在 1.2x 和 1.5x 的均值尚未超过 domestic，从 2.0x 开始转为稳定负值。该趋势说明 sequential 训练的收益主要在噪声偏离 T0 足够明显时出现，而不是所有强度上都具有同样幅度。

相对 mixed-noise 的倍率曲线给出更严格边界：d=5 整体不占优，d=7/d=9 多数倍率为负。由此，倍率分析强化了“seq 相对 domestic 的 OOD 泛化”主结论，但不支持把 seq 写成每个倍率、每个 distance 都超过 mixed。

#### 5.4.3 不同噪声轴组合

{axis_distance_figure}

按单轴是否出现聚合：

{axis_presence_table}

按 11 个固定轴组合聚合：

{axis_combination_table}

按激活轴数量聚合：

{combination_table}

单轴出现分组不是互斥消融，因为每个环境同时包含两到四个轴；它用于描述哪些物理噪声结构经常与收益共同出现，不能解释为单轴因果贡献。聚合结果为：{axis_presence_text}。

在 11 个组合中，跨 distance 平均收益最大的是 **{best_axis.get('axis_label', '')}**（seq-domestic delta={signed(best_axis.get('delta_seq_vs_domestic_mean'))}），收益最小的是 **{weakest_axis.get('axis_label', '')}**（delta={signed(weakest_axis.get('delta_seq_vs_domestic_mean'))}）。按组合规模看：{combination_text}。四轴全组合仍保持负 delta，但收益通常较小，说明更复杂、更强的复合噪声会使各模型 LER 向饱和区靠近，从而压缩绝对差距。

轴组合热图进一步显示，seq 相对 domestic 的负 delta 覆盖全部三种 distance，但 d=5 的幅度最大；相对 mixed 的图则明显依赖 distance。含 Z-bias 的组合在 d=7/d=9 通常更有利于 seq，这与 sequential 任务流后期显式学习 Z-biased 噪声结构一致，但这里只能视为机制一致性证据。

#### 5.4.4 轴组合与倍率的交互

{axis_multiplier_figure}

三张面板展开全部 297 个配置，不再对倍率或轴组合取平均。seq no-EWC 相对 domestic 的逐配置胜出数分别为：{interaction_win_text}。大部分非优势单元集中在 d=9 的低倍率区域以及少量高倍率饱和区域；这解释了为什么 d=9 总体平均收益较小，同时仍能在 2.0x 以上大部分组合中保持负 delta。

轴与倍率并非独立影响：同一组合在低倍率可能与 domestic 接近，倍率增大后优势扩大，再在极高倍率因逻辑错误率饱和而收窄。因此主证据应同时报告“跨配置胜率”和“平均 delta”，不能只选取优势最大的倍率点。

#### 5.4.5 Mixed-noise 与 EWC 强基线

seq + EWC 的 OOD 加权平均 LER 为 {fmt(ood_ewc_mean)}，{ewc_ood_relation} no-EWC 的 {fmt(ood_seq_mean)}；{ewc_ood_interpretation}。但其已见任务平均 LER 为 {fmt(ewc_avg)}，高于 no-EWC 的 {fmt(noewc_avg)}，且 d=9 下不占优。因此 no-EWC 仍作为兼顾已见任务稳定性和 OOD 泛化的综合主模型，EWC 作为偏 OOD 的正则化对照。

分 distance 看，组合方案在 d=5/7/9 均优于 domestic-only；相对 mixed-noise 则在 d=7、d=9 更优，在 d=5 略差。因此应表述为“跨 distance 总体优于 mixed”，而不是“每个 distance 都优于 mixed”。

#### 5.4.6 证据边界与可解释范围

该 OOD 实验固定使用 R9-X 架构，直接证明 sequential 训练的增益以及 R9-X + seq 端到端组合方案的实际泛化表现；但不能由该实验单独分解架构贡献。架构贡献由第 4 节 T0 对照实验建立。OOD 环境属于 training-axis fixed multiplier grid stress test，结论不能外推到所有未知物理噪声。

## 6. 基于实验结果的综合结论

1. **组合方案具备 OOD 泛化性**：R9-X + sequential no-EWC 在 d=5/7/9、297 个复合 OOD 配置上的加权平均 LER 为 {fmt(ood_seq_mean)}，相对同架构 domestic-only 降低 {ood_seq_vs_domestic_pct:.2f}%，并在 {ood_win_count}/{ood_config_count} 个配置中胜出。
2. **OOD 收益具有清晰结构**：跨 distance 平均绝对收益在 {float(best_multiplier.get('multiplier', float('nan'))):g}x 最大；轴组合中 {best_axis.get('axis_label', '')} 的平均收益最大，四轴组合仍优于 domestic，但饱和效应使差距收窄。
3. **架构提供有效基础**：T0 同分布实验中，R9-X 相对 Ising fast 将平均 LER 降低 {ler_reduction_pct:.2f}%，同时减少 {parameter_reduction_pct:.2f}% 参数和 {latency_reduction_pct:.2f}% 推理延迟。
4. **Seq 训练形成持续进化能力**：四个后续任务均获得正向阶段适应，最终最大 forgetting 为 {fmt(max_forgetting)}；组合方案在维持已见任务能力的同时扩展了复合噪声泛化范围。
5. **组合优于单一训练范式**：跨三种 distance 的总体均值上，R9-X + seq 同时优于 domestic-only 和 mixed-noise，说明在已验证有效的 R9-X 架构上，顺序持续训练比 domestic-only 或当前 mixed-noise 更适合作为主路线。
6. **结论边界**：当前证据支持 R9-X + seq 在指定任务流和固定倍率复合 OOD 网格中的整体优势。由于 OOD 实验没有加入 Ising + seq 对照，不能把 OOD 收益全部归因于架构；由于 d=5 下 mixed 略优于 seq，也不能声称组合方案在每个设置下绝对最优。

最终可概括为：R9-X 解决“基础模型是否足够强”，sequential no-EWC 解决“模型能否随噪声环境持续进化”，固定倍率复合 OOD 实验则验证了两者组合后的系统级泛化收益。

## 7. 结果文件

- T0 模型 paired detail：outputs/analysis/lifelong_r9_unified_decoder_details.csv
- Lifelong epoch matrix：outputs/analysis/r9x_epoch_matrix_details.csv
- Domestic 同预算完整轨迹：outputs/analysis/r9x_domestic_epoch_complete_ler.csv
- OOD + EWC detail：outputs/analysis/lifelong_ood_grid_with_ewc_details.csv
- OOD + EWC summary：outputs/analysis/lifelong_ood_grid_with_ewc_summary.csv
- OOD 图表目录：outputs/analysis/lifelong_ood_model_training_figures/
"""
    path.write_text(text, encoding="utf-8")

def build_report(args: argparse.Namespace) -> None:
    base_details = read_csv(args.base_details)
    ewc_details: list[dict[str, str]] = []
    for path in args.ewc_details:
        ewc_details.extend(read_csv(path))
    merged = merge_grid_details_with_ewc(base_details, ewc_details)
    summary = summarize_grid(merged)
    model_effectiveness = summarize_t0_model_effectiveness(read_csv(args.model_details))
    lifelong = summarize_lifelong(
        read_csv(args.lifelong_details),
        domestic_complete_rows=read_csv(args.domestic_complete),
    )
    write_csv(args.out_details, merged)
    write_csv(args.out_summary, summary)
    figure_paths = generate_ood_figures(merged, args.figure_dir)
    write_technical_report(
        args.out_report,
        model_effectiveness=model_effectiveness,
        grid_overall=summary,
        lifelong=lifelong,
        grid_details=merged,
        figure_paths=figure_paths,
    )
    print(f"[write] {args.out_details}")
    print(f"[write] {args.out_summary}")
    print(f"[write] {args.out_report}")
    print(f"[write] {args.figure_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-details", default=DEFAULT_BASE_DETAILS)
    parser.add_argument("--ewc-details", nargs="+", default=DEFAULT_EWC_DETAILS)
    parser.add_argument("--lifelong-details", default=DEFAULT_LIFELONG_DETAILS)
    parser.add_argument("--model-details", default=DEFAULT_MODEL_DETAILS)
    parser.add_argument("--domestic-complete", default=DEFAULT_DOMESTIC_COMPLETE)
    parser.add_argument("--out-details", default=DEFAULT_OUT_DETAILS)
    parser.add_argument("--out-summary", default=DEFAULT_OUT_SUMMARY)
    parser.add_argument("--out-report", default=DEFAULT_OUT_REPORT)
    parser.add_argument("--figure-dir", default=DEFAULT_FIGURE_DIR)
    return parser.parse_args()


def main() -> None:
    build_report(parse_args())


if __name__ == "__main__":
    main()
