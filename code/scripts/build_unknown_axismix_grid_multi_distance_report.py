#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build a fused d=5/7/9 report for the fixed multiplier-grid axis-mix OOD test."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = REPO_ROOT / "outputs" / "analysis"

DEFAULT_DISTANCE_DETAILS = [
    (5, ANALYSIS_DIR / "unknown_axismix_grid_u1p2_5p0_d5_details.csv"),
    (7, ANALYSIS_DIR / "unknown_axismix_grid_u1p2_5p0_d7_details.csv"),
    (9, ANALYSIS_DIR / "unknown_axismix_grid_u1p2_5p0_details.csv"),
]
DEFAULT_OUT_DETAILS = ANALYSIS_DIR / "unknown_axismix_grid_u1p2_5p0_multi_distance_details.csv"
DEFAULT_OUT_SUMMARY = ANALYSIS_DIR / "unknown_axismix_grid_u1p2_5p0_multi_distance_summary.csv"
DEFAULT_OUT_REPORT = ANALYSIS_DIR / "unknown_axismix_grid_u1p2_5p0_multi_distance_comparison.md"

BASIS_ORDER = {"both": 0, "X": 1, "Z": 2}
GROUP_ORDER = {
    "all": 0,
    "multiplier": 1,
    "env": 2,
    "combination_size": 3,
    "contains_z_bias": 4,
    "contains_cnot_z_bias": 5,
}


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen: set[str] = set()
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
        number = float(value)
    except Exception:
        return float("nan")
    return number


def mean(values: Iterable[float]) -> float:
    vals = [float(value) for value in values]
    vals = [value for value in vals if not math.isnan(value)]
    return sum(vals) / len(vals) if vals else float("nan")


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def fmt(value: Any, digits: int = 6) -> str:
    number = fnum(value)
    if math.isnan(number):
        return ""
    return f"{number:.{digits}f}"


def signed(value: Any, digits: int = 6) -> str:
    number = fnum(value)
    if math.isnan(number):
        return ""
    return f"{number:+.{digits}f}"


def pct(delta: Any, baseline: Any) -> str:
    delta_v = fnum(delta)
    base_v = fnum(baseline)
    if math.isnan(delta_v) or math.isnan(base_v) or base_v == 0:
        return ""
    return f"{100.0 * delta_v / base_v:+.2f}%"


def md_table(headers: Sequence[str], rows: Iterable[Iterable[Any]]) -> str:
    rendered = [[str(cell) for cell in row] for row in rows]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rendered)
    return "\n".join(lines)


def _group_sort_value(group_type: str, group_value: str) -> tuple[int, Any]:
    if group_type == "multiplier":
        return (0, fnum(group_value[1:].replace("p", ".")))
    if group_type == "env":
        return (0, int(group_value[1:]) if group_value.startswith("e") else group_value)
    if group_type == "combination_size":
        return (0, int(group_value))
    if group_value in {"no", "False", "false"}:
        return (0, group_value)
    if group_value in {"yes", "True", "true"}:
        return (1, group_value)
    return (0, group_value)


def merge_distance_details(distance_rows: Sequence[tuple[int, Sequence[Mapping[str, Any]]]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for distance, rows in distance_rows:
        for row in rows:
            domestic = fnum(row.get("domestic100_ler"))
            mixed = fnum(row.get("mixed100_ler"))
            seq = fnum(row.get("seq_noewc_ler"))
            item: dict[str, Any] = {
                "distance": int(distance),
                "distance_label": f"d{int(distance)}",
            }
            item.update(row)
            item["distance"] = int(distance)
            item["distance_label"] = f"d{int(distance)}"
            item["delta_seq_vs_domestic"] = seq - domestic
            item["delta_seq_vs_mixed"] = seq - mixed
            item["delta_mixed_vs_domestic"] = mixed - domestic
            item["seq_beats_domestic"] = seq < domestic
            item["seq_beats_mixed"] = seq < mixed
            item["mixed_beats_domestic"] = mixed < domestic
            merged.append(item)
    merged.sort(
        key=lambda row: (
            int(row["distance"]),
            int(row.get("env_index", 0)),
            fnum(row.get("multiplier", 0.0)),
            BASIS_ORDER.get(str(row.get("basis")), 99),
        )
    )
    return merged


def summarize_multi_distance(detail_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    group_defs = [
        ("all", lambda row: "all"),
        ("multiplier", lambda row: str(row["multiplier_key"])),
        ("env", lambda row: str(row["env_key"])),
        ("combination_size", lambda row: str(row["combination_size"])),
        ("contains_z_bias", lambda row: "yes" if as_bool(row["contains_z_bias"]) else "no"),
        ("contains_cnot_z_bias", lambda row: "yes" if as_bool(row["contains_cnot_z_bias"]) else "no"),
    ]
    grouped: dict[tuple[int, str, str, str], list[Mapping[str, Any]]] = {}
    for row in detail_rows:
        distance = int(row["distance"])
        basis = str(row["basis"])
        for group_type, group_fn in group_defs:
            grouped.setdefault((distance, group_type, group_fn(row), basis), []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for (distance, group_type, group_value, basis), rows in grouped.items():
        summary_rows.append(
            {
                "distance": int(distance),
                "distance_label": f"d{int(distance)}",
                "group_type": group_type,
                "group_value": group_value,
                "basis": basis,
                "config_count": len(rows),
                "env_count": len({str(row["env_key"]) for row in rows}),
                "domestic100_ler_mean": mean(fnum(row["domestic100_ler"]) for row in rows),
                "mixed100_ler_mean": mean(fnum(row["mixed100_ler"]) for row in rows),
                "seq_noewc_ler_mean": mean(fnum(row["seq_noewc_ler"]) for row in rows),
                "delta_seq_vs_domestic_mean": mean(fnum(row["delta_seq_vs_domestic"]) for row in rows),
                "delta_seq_vs_mixed_mean": mean(fnum(row["delta_seq_vs_mixed"]) for row in rows),
                "delta_mixed_vs_domestic_mean": mean(fnum(row["delta_mixed_vs_domestic"]) for row in rows),
                "seq_win_vs_domestic_count": sum(1 for row in rows if as_bool(row["seq_beats_domestic"])),
                "seq_win_vs_mixed_count": sum(1 for row in rows if as_bool(row["seq_beats_mixed"])),
                "mixed_win_vs_domestic_count": sum(1 for row in rows if as_bool(row["mixed_beats_domestic"])),
            }
        )
    summary_rows.sort(
        key=lambda row: (
            int(row["distance"]),
            GROUP_ORDER.get(str(row["group_type"]), 99),
            _group_sort_value(str(row["group_type"]), str(row["group_value"])),
            BASIS_ORDER.get(str(row["basis"]), 99),
        )
    )
    return summary_rows


def _summary_lookup(summary_rows: Sequence[Mapping[str, Any]]) -> dict[tuple[int, str, str, str], Mapping[str, Any]]:
    return {
        (int(row["distance"]), str(row["group_type"]), str(row["group_value"]), str(row["basis"])): row
        for row in summary_rows
    }


def _overall_rows(summary_rows: Sequence[Mapping[str, Any]], basis: str = "both") -> list[Mapping[str, Any]]:
    rows = [
        row
        for row in summary_rows
        if row["group_type"] == "all" and row["group_value"] == "all" and row["basis"] == basis
    ]
    return sorted(rows, key=lambda row: int(row["distance"]))


def _rows_for(summary_rows: Sequence[Mapping[str, Any]], *, group_type: str, basis: str) -> list[Mapping[str, Any]]:
    return [
        row
        for row in summary_rows
        if row["group_type"] == group_type and row["basis"] == basis
    ]


def _verdict(row: Mapping[str, Any]) -> str:
    seq_dom = fnum(row["delta_seq_vs_domestic_mean"])
    seq_mixed = fnum(row["delta_seq_vs_mixed_mean"])
    if seq_dom < 0 and seq_mixed < 0:
        return "seq 同时优于 domestic 和 mixed"
    if seq_dom < 0:
        return "seq 优于 domestic，但不优于 mixed"
    return "seq 未优于 domestic"


def _main_table(summary_rows: Sequence[Mapping[str, Any]]) -> str:
    rows = []
    for row in _overall_rows(summary_rows, "both"):
        rows.append(
            [
                f"d={int(row['distance'])}",
                fmt(row["domestic100_ler_mean"]),
                fmt(row["mixed100_ler_mean"]),
                fmt(row["seq_noewc_ler_mean"]),
                signed(row["delta_seq_vs_domestic_mean"]),
                pct(row["delta_seq_vs_domestic_mean"], row["domestic100_ler_mean"]),
                signed(row["delta_seq_vs_mixed_mean"]),
                pct(row["delta_seq_vs_mixed_mean"], row["mixed100_ler_mean"]),
                f"{row['seq_win_vs_domestic_count']}/{row['config_count']}",
                f"{row['seq_win_vs_mixed_count']}/{row['config_count']}",
                _verdict(row),
            ]
        )
    return md_table(
        [
            "distance",
            "domestic LER",
            "mixed LER",
            "seq LER",
            "seq-domestic",
            "相对 domestic",
            "seq-mixed",
            "相对 mixed",
            "seq 胜 domestic",
            "seq 胜 mixed",
            "判读",
        ],
        rows,
    )


def _basis_table(summary_rows: Sequence[Mapping[str, Any]]) -> str:
    rows = []
    for distance in sorted({int(row["distance"]) for row in summary_rows}):
        for basis in ("both", "X", "Z"):
            selected = [
                row
                for row in summary_rows
                if int(row["distance"]) == distance
                and row["group_type"] == "all"
                and row["group_value"] == "all"
                and row["basis"] == basis
            ]
            for row in selected:
                rows.append(
                    [
                        f"d={distance}",
                        basis,
                        fmt(row["domestic100_ler_mean"]),
                        fmt(row["mixed100_ler_mean"]),
                        fmt(row["seq_noewc_ler_mean"]),
                        signed(row["delta_seq_vs_domestic_mean"]),
                        signed(row["delta_seq_vs_mixed_mean"]),
                        f"{row['seq_win_vs_domestic_count']}/{row['config_count']}",
                        f"{row['seq_win_vs_mixed_count']}/{row['config_count']}",
                    ]
                )
    return md_table(
        [
            "distance",
            "basis",
            "domestic LER",
            "mixed LER",
            "seq LER",
            "seq-domestic",
            "seq-mixed",
            "seq 胜 domestic",
            "seq 胜 mixed",
        ],
        rows,
    )


def _multiplier_table(summary_rows: Sequence[Mapping[str, Any]]) -> str:
    rows = []
    selected = _rows_for(summary_rows, group_type="multiplier", basis="both")
    selected.sort(key=lambda row: (int(row["distance"]), _group_sort_value("multiplier", str(row["group_value"]))))
    for row in selected:
        rows.append(
            [
                f"d={int(row['distance'])}",
                row["group_value"],
                fmt(row["seq_noewc_ler_mean"]),
                signed(row["delta_seq_vs_domestic_mean"]),
                signed(row["delta_seq_vs_mixed_mean"]),
                f"{row['seq_win_vs_domestic_count']}/{row['config_count']}",
                f"{row['seq_win_vs_mixed_count']}/{row['config_count']}",
            ]
        )
    return md_table(
        ["distance", "倍率", "seq LER", "seq-domestic", "seq-mixed", "seq 胜 domestic", "seq 胜 mixed"],
        rows,
    )


def _group_table(summary_rows: Sequence[Mapping[str, Any]]) -> str:
    rows = []
    for group_type, label in [("combination_size", "组合轴数"), ("contains_z_bias", "是否含 z_bias")]:
        selected = _rows_for(summary_rows, group_type=group_type, basis="both")
        selected.sort(
            key=lambda row: (
                int(row["distance"]),
                _group_sort_value(group_type, str(row["group_value"])),
            )
        )
        for row in selected:
            rows.append(
                [
                    f"d={int(row['distance'])}",
                    label,
                    row["group_value"],
                    row["config_count"],
                    fmt(row["domestic100_ler_mean"]),
                    fmt(row["mixed100_ler_mean"]),
                    fmt(row["seq_noewc_ler_mean"]),
                    signed(row["delta_seq_vs_domestic_mean"]),
                    signed(row["delta_seq_vs_mixed_mean"]),
                ]
            )
    return md_table(
        [
            "distance",
            "分组维度",
            "分组值",
            "config 数",
            "domestic LER",
            "mixed LER",
            "seq LER",
            "seq-domestic",
            "seq-mixed",
        ],
        rows,
    )


def _env_table(summary_rows: Sequence[Mapping[str, Any]]) -> str:
    rows = []
    selected = _rows_for(summary_rows, group_type="env", basis="both")
    selected.sort(key=lambda row: (int(row["distance"]), _group_sort_value("env", str(row["group_value"]))))
    for row in selected:
        rows.append(
            [
                f"d={int(row['distance'])}",
                row["group_value"],
                fmt(row["domestic100_ler_mean"]),
                fmt(row["mixed100_ler_mean"]),
                fmt(row["seq_noewc_ler_mean"]),
                signed(row["delta_seq_vs_domestic_mean"]),
                signed(row["delta_seq_vs_mixed_mean"]),
                f"{row['seq_win_vs_domestic_count']}/{row['config_count']}",
                f"{row['seq_win_vs_mixed_count']}/{row['config_count']}",
            ]
        )
    return md_table(
        [
            "distance",
            "env",
            "domestic LER",
            "mixed LER",
            "seq LER",
            "seq-domestic",
            "seq-mixed",
            "seq 胜 domestic",
            "seq 胜 mixed",
        ],
        rows,
    )


def write_multi_distance_report(
    path: str | Path,
    *,
    summary_rows: Sequence[Mapping[str, Any]],
    details_csv: str | Path = DEFAULT_OUT_DETAILS,
    summary_csv: str | Path = DEFAULT_OUT_SUMMARY,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    overall = _overall_rows(summary_rows, "both")
    lookup = _summary_lookup(summary_rows)
    distances = [int(row["distance"]) for row in overall]
    seq_domestic = ", ".join(
        f"d={int(row['distance'])} `{signed(row['delta_seq_vs_domestic_mean'])}`" for row in overall
    )
    seq_mixed_negative = [
        row for row in overall if fnum(row["delta_seq_vs_mixed_mean"]) < 0
    ]
    seq_mixed_positive = [
        row for row in overall if fnum(row["delta_seq_vs_mixed_mean"]) >= 0
    ]
    mixed_positive_text = "、".join(f"d={int(row['distance'])}" for row in seq_mixed_positive) or "无"
    mixed_negative_text = "、".join(f"d={int(row['distance'])}" for row in seq_mixed_negative) or "无"

    xz_notes = []
    for distance in distances:
        for basis in ("X", "Z"):
            row = lookup.get((distance, "all", "all", basis))
            if row:
                xz_notes.append(
                    f"d={distance} {basis}: seq-domestic {signed(row['delta_seq_vs_domestic_mean'])}, "
                    f"seq-mixed {signed(row['delta_seq_vs_mixed_mean'])}"
                )

    lines = [
        "# 三距离固定倍率网格 OOD 融合分析报告",
        "",
        "## 结论摘要",
        "",
        "- 本报告融合 `d=5`、`d=7`、`d=9` 三个 code distance 下的 `training-axis fixed multiplier grid OOD composite test`。每个 distance 均覆盖 11 个训练轴混合 env × 9 个固定倍率，共 99 个 paired config；三模型为 domestic-only e100、mixed-noise e100、seq no-EWC e100。",
        f"- 相对 domestic-only：seq no-EWC 在三个 distance 的 `basis=both` 下均更低。delta 分别为 {seq_domestic}。这支持 sequential no-EWC 相对单一 domestic 训练的跨 distance OOD 收益。",
        f"- 相对 mixed-noise：seq no-EWC 优于 mixed 的 distance 为 {mixed_negative_text}；不优于 mixed 的 distance 为 {mixed_positive_text}。因此是否能声明“超过 mixed 基线”必须按 distance 说明。",
        "- X/Z 拆分显示：Z basis 通常更稳定；X basis 上 seq 的优势更容易被 mixed-noise 或低倍率区间抵消。",
        "- 推荐论文/组会主表述：在固定训练轴复合 OOD 网格中，seq no-EWC 在 d=5/7/9 上均优于 domestic-only；若讨论 mixed-noise，则只声明实际为负 delta 的 distance。",
        "- 不推荐表述：不要写成随机未知噪声，不要写成所有 code distance 和所有 OOD 分布上 seq 都优于 mixed，也不要隐藏 d=5 的 mixed 基线边界。",
        "",
        "## 实验口径",
        "",
        "- 噪声族：`unknown_axismix_grid_u1p2_5p0`，即训练轴固定倍率网格 OOD，不是随机噪声。",
        "- 训练轴：`meas_all`、`cnot_all`、`idle_all`、`z_bias`。",
        "- 环境：任意两轴、任意三轴、四轴全组合，共 11 个 env。",
        "- 倍率：`[1.2, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]`，激活参数统一乘当前倍率，重叠参数只乘一次。",
        "- 每个 distance：`num_samples=262144`，`basis=both`，并拆分 X/Z；每个 config 内三个模型共享同一批 samples 和 matcher。",
        "- distances：d=5 使用 `n_rounds=5`，d=7 使用 `n_rounds=7`，d=9 使用 `n_rounds=9`。",
        "",
        "## 主结果：basis=both",
        "",
        _main_table(summary_rows),
        "",
        "从 domestic-only 基线看，三个 distance 均支持 seq no-EWC 更优。这个结论的意义是训练范式相对单一 domestic 训练具备跨距离 OOD 收益，而不是只在 d=9 单点成立。",
        "",
        "从 mixed-noise 基线看，结论需要更谨慎：mixed-noise 是多噪声训练基线，是否被 seq 超过取决于 distance、basis 和倍率区间。因此报告应把 `seq vs domestic` 作为主结论，把 `seq vs mixed` 作为更强基线的补充结论。",
        "",
        "## X/Z basis 拆分",
        "",
        _basis_table(summary_rows),
        "",
        "X/Z 拆分用于解释平均值来源：" + "；".join(xz_notes) + "。整体上 Z basis 更有利于 seq 展示稳定优势，X basis 则更能暴露与 mixed-noise 基线的差异边界。",
        "",
        "## 倍率趋势：basis=both",
        "",
        _multiplier_table(summary_rows),
        "",
        "倍率趋势用于判断 seq 优势是否只来自少数噪声强度。若某个 distance 在低倍率不稳定但高倍率转负，结论应写成“压力增强后更支持 seq 泛化”；若某个 distance 全倍率相对 mixed 为正，则不能把它纳入 seq 超过 mixed 的证据。",
        "",
        "## 环境组合差异：basis=both",
        "",
        _group_table(summary_rows),
        "",
        "组合维度用于给结论加机制解释：二轴、三轴、四轴组合都属于训练轴的未见组合，含 `z_bias` 的组合通常更接近 sequential 后续任务暴露过的结构，因此更可能贡献 seq 的 OOD 收益。",
        "",
        "## env 级结果：basis=both",
        "",
        _env_table(summary_rows),
        "",
        "env 级表适合放在 appendix 或组会备份页。主文建议聚焦三点：seq 相对 domestic 的跨 d 一致性、相对 mixed 的 distance 边界、以及 X/Z 与倍率趋势如何解释该边界。",
        "",
        "## 推荐结论写法",
        "",
        "> 在 d=5/7/9 三个 code distance 的固定训练轴复合 OOD 网格评估中，seq no-EWC epoch 100 在所有 distance 上均低于 domestic-only epoch 100，说明顺序训练范式相对单一 domestic 训练具备跨距离的 OOD 泛化收益。相对 mixed-noise epoch 100 的结论需要按 distance 披露：只有实测 seq-mixed delta 为负的 distance 才能声明 seq 超过 mixed 基线。",
        "",
        "## 文件索引",
        "",
        f"- 融合 summary CSV: `{details_csv.parent / summary_csv.name if isinstance(summary_csv, Path) else summary_csv}`",
        f"- 融合 details CSV: `{details_csv}`",
        f"- 融合报告: `{path}`",
        "- d=5 原报告: `outputs/analysis/unknown_axismix_grid_u1p2_5p0_d5_comparison.md`",
        "- d=7 原报告: `outputs/analysis/unknown_axismix_grid_u1p2_5p0_d7_comparison.md`",
        "- d=9 原报告: `outputs/analysis/unknown_axismix_grid_u1p2_5p0_comparison.md`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_report(args: argparse.Namespace) -> None:
    distance_rows = [(distance, read_csv(path)) for distance, path in args.distance_details]
    detail_rows = merge_distance_details(distance_rows)
    summary_rows = summarize_multi_distance(detail_rows)
    write_csv(args.out_details, detail_rows)
    write_csv(args.out_summary, summary_rows)
    write_multi_distance_report(
        args.out_report,
        summary_rows=summary_rows,
        details_csv=args.out_details,
        summary_csv=args.out_summary,
    )
    print(f"[write] {args.out_details}")
    print(f"[write] {args.out_summary}")
    print(f"[write] {args.out_report}")


def _distance_detail_arg(value: str) -> tuple[int, Path]:
    if ":" not in value:
        raise argparse.ArgumentTypeError("expected DISTANCE:CSV_PATH")
    distance, path = value.split(":", 1)
    return int(distance), Path(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--distance-details",
        type=_distance_detail_arg,
        nargs="+",
        default=DEFAULT_DISTANCE_DETAILS,
        help="Distance details as DISTANCE:CSV_PATH. Defaults to d5/d7/d9 full outputs.",
    )
    parser.add_argument("--out-details", type=Path, default=DEFAULT_OUT_DETAILS)
    parser.add_argument("--out-summary", type=Path, default=DEFAULT_OUT_SUMMARY)
    parser.add_argument("--out-report", type=Path, default=DEFAULT_OUT_REPORT)
    return parser.parse_args()


def main() -> None:
    build_report(parse_args())


if __name__ == "__main__":
    main()
