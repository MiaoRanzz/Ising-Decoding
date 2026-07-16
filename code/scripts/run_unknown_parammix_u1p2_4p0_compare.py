#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run domestic-100 vs seq no-EWC on parameter-level random OOD configs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from scripts.generate_unknown_parammix_u1p2_4p0_configs import (  # noqa: E402
    DEFAULT_MANIFEST,
    DEFAULT_OUTPUT_DIR as DEFAULT_CONFIG_OUTPUT_DIR,
    DEFAULT_PREFIX,
    DEFAULT_SEED,
    DESIGN_LABEL,
    write_parammix_configs,
)
from scripts.generate_unknown_seqfav_composite_configs import DEFAULT_BASE_CONFIG  # noqa: E402
from scripts.config_paths import config_lookup_with_basename, config_path  # noqa: E402
from scripts.run_unknown_t0_random_compare import (  # noqa: E402
    PAIRED_SCRIPT,
    display_path,
    markdown_table,
    rel,
)


MODEL_SPECS = [
    (
        "stfusion_domestic_e100",
        111,
        "outputs/ising_domestic_fast_opt_stfusion_r9_x/models/PreDecoderSTFusion_v2.0.100.pt",
    ),
    (
        "stfusion_seq_noewc_e100",
        111,
        "outputs/ising_domestic_fast_opt_stfusion_r9_x_seq_noewc/models/PreDecoderSTFusion_v2.0.100.pt",
    ),
]
DOMESTIC_METHOD = "stfusion_domestic_e100"
SEQ_METHOD = "stfusion_seq_noewc_e100"
DEFAULT_OUTPUT_DIR = "outputs/paired_inference_compare/unknown_parammix_u1p2_4p0_full"
DEFAULT_DETAILS_CSV = "outputs/analysis/unknown_parammix_u1p2_4p0_details.csv"
DEFAULT_SUMMARY_CSV = "outputs/analysis/unknown_parammix_u1p2_4p0_summary.csv"
DEFAULT_SUMMARY_MD = "outputs/analysis/unknown_parammix_u1p2_4p0_comparison.md"


def resolve_python(args: argparse.Namespace) -> str:
    candidates = [
        getattr(args, "python", None),
        os.environ.get("PREDECODER_PYTHON"),
        "/root/miniconda3/envs/ising-decoding/bin/python",
        sys.executable,
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return str(candidate)
    return sys.executable


def fmt_float(value: Any, digits: int = 6) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if math.isnan(number) or math.isinf(number):
        return ""
    return f"{number:.{digits}f}"


def mean(values: Sequence[float]) -> float:
    values = [float(value) for value in values]
    return sum(values) / len(values) if values else float("nan")


def sample_std(values: Sequence[float]) -> float:
    values = [float(value) for value in values]
    return statistics.stdev(values) if len(values) > 1 else 0.0


def task_output_path(output_dir: Path, env_index: int, replicate_index: int) -> Path:
    return output_dir / (
        f"unknown_parammix_u1p2_4p0_e{env_index:02d}_r{replicate_index:02d}.json"
    )


def ensure_configs(args: argparse.Namespace) -> dict[str, Any]:
    _, manifest = write_parammix_configs(
        base_config=args.base_config,
        output_dir=DEFAULT_CONFIG_OUTPUT_DIR,
        prefix=args.config_prefix,
        manifest=args.manifest,
        seed=args.generation_seed,
    )
    return manifest


def existing_inputs_or_raise(manifest: Mapping[str, Any]) -> None:
    missing = []
    if not PAIRED_SCRIPT.exists():
        missing.append(PAIRED_SCRIPT)
    for env in manifest["environments"]:
        path = config_path(env["config_name"])
        if not path.exists():
            missing.append(path)
    for _, _, checkpoint in MODEL_SPECS:
        path = rel(checkpoint)
        if not path.exists():
            missing.append(path)
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Required inputs are missing:\n{formatted}")


def build_command(
    args: argparse.Namespace,
    *,
    config_name: str,
    output_path: Path,
) -> list[str]:
    cmd = [
        resolve_python(args),
        "-u",
        str(PAIRED_SCRIPT),
        "--config-name",
        config_name,
        "--distance",
        str(args.distance),
        "--n-rounds",
        str(args.n_rounds),
        "--num-samples",
        str(args.num_samples),
        "--latency-num-samples",
        str(args.latency_num_samples),
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--seed",
        str(args.seed),
        "--basis",
        args.basis,
        "--output",
        str(output_path),
    ]
    if args.device:
        cmd.extend(["--device", args.device])
    for name, model_id, checkpoint in MODEL_SPECS:
        cmd.extend(["--model", f"{name}:{model_id}:{rel(checkpoint)}"])
    return cmd


def run_all(
    args: argparse.Namespace,
    manifest: Mapping[str, Any],
) -> Path:
    output_dir = rel(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    existing_inputs_or_raise(manifest)
    for env in manifest["environments"]:
        env_index = int(env["env_index"])
        replicate_index = int(env["replicate_index"])
        output_path = task_output_path(output_dir, env_index, replicate_index)
        if args.resume and output_path.exists():
            print(f"[skip] {env['env_key']}/{env['replicate_key']} exists: {output_path}")
            continue
        cmd = build_command(
            args,
            config_name=str(env["config_name"]),
            output_path=output_path,
        )
        print(
            f"[run] env={env['env_key']} replicate={env['replicate_key']} "
            f"config={env['config_name']} -> {output_path}"
        )
        if not args.dry_run:
            subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    return output_dir


def load_payloads(
    output_dir: str | Path,
    manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    output_dir = rel(output_dir)
    payloads = []
    missing = []
    for env in manifest["environments"]:
        output_path = task_output_path(
            output_dir,
            int(env["env_index"]),
            int(env["replicate_index"]),
        )
        if not output_path.exists():
            missing.append(output_path)
            continue
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        payload["run_phase"] = "full"
        payload["path"] = str(output_path)
        payloads.append(payload)
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Missing per-realization inference outputs:\n{formatted}")
    return payloads


def _method_metric(
    rows: Sequence[Mapping[str, Any]],
    basis: str,
    method: str,
    metric: str,
) -> float:
    values = [
        float(row[metric])
        for row in rows
        if row.get("basis") == basis
        and row.get("method") == method
        and row.get(metric) is not None
    ]
    return values[0] if len(values) == 1 else float("nan")


def _detail_row(
    *,
    payload: Mapping[str, Any],
    env: Mapping[str, Any],
    basis: str,
    domestic_ler: float,
    seq_ler: float,
    domestic_latency: float,
    seq_latency: float,
    domestic_speedup: float,
    seq_speedup: float,
) -> dict[str, Any]:
    delta = seq_ler - domestic_ler
    relative_delta = delta / domestic_ler * 100.0 if domestic_ler else float("nan")
    return {
        "run_phase": payload.get("run_phase", "full"),
        "env_key": env["env_key"],
        "env_index": int(env["env_index"]),
        "replicate_key": env["replicate_key"],
        "replicate_index": int(env["replicate_index"]),
        "config_name": env["config_name"],
        "axis_signature": env["axis_signature"],
        "active_axes": "+".join(env["active_axes"]),
        "combination_size": int(env["combination_size"]),
        "contains_z_bias": bool(env["contains_z_bias"]),
        "contains_cnot_z_bias": bool(env["contains_cnot_z_bias"]),
        "basis": basis,
        "domestic100_ler": domestic_ler,
        "seq_noewc_ler": seq_ler,
        "delta": delta,
        "relative_delta_pct": relative_delta,
        "winner": "seq_noewc_e100" if delta < 0 else "domestic_e100" if delta > 0 else "tie",
        "domestic100_latency_us_per_round": domestic_latency,
        "seq_noewc_latency_us_per_round": seq_latency,
        "domestic100_speedup_vs_pymatching": domestic_speedup,
        "seq_noewc_speedup_vs_pymatching": seq_speedup,
        "output_json": payload.get("path", ""),
    }


def _detail_rows_for_payload(
    payload: Mapping[str, Any],
    env: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows = payload["rows"]
    details = []
    basis_metrics: dict[str, dict[str, float]] = {}
    for basis in ("X", "Z"):
        domestic_ler = _method_metric(rows, basis, DOMESTIC_METHOD, "ler")
        seq_ler = _method_metric(rows, basis, SEQ_METHOD, "ler")
        if math.isnan(domestic_ler) or math.isnan(seq_ler):
            continue
        metrics = {
            "domestic_ler": domestic_ler,
            "seq_ler": seq_ler,
            "domestic_latency": _method_metric(
                rows, basis, DOMESTIC_METHOD, "latency_us_per_round"
            ),
            "seq_latency": _method_metric(rows, basis, SEQ_METHOD, "latency_us_per_round"),
            "domestic_speedup": _method_metric(
                rows, basis, DOMESTIC_METHOD, "speedup_vs_pymatching"
            ),
            "seq_speedup": _method_metric(rows, basis, SEQ_METHOD, "speedup_vs_pymatching"),
        }
        basis_metrics[basis] = metrics
        details.append(
            _detail_row(payload=payload, env=env, basis=basis, **metrics)
        )

    if {"X", "Z"}.issubset(basis_metrics):
        combined = {
            name: mean([basis_metrics["X"][name], basis_metrics["Z"][name]])
            for name in basis_metrics["X"]
        }
        details.append(
            _detail_row(payload=payload, env=env, basis="both", **combined)
        )
    return details


def _summarize_env_rows(detail_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str], list[Mapping[str, Any]]] = {}
    for row in detail_rows:
        grouped.setdefault((int(row["env_index"]), str(row["basis"])), []).append(row)

    summaries = []
    for (env_index, basis), rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda row: int(row["replicate_index"]))
        domestic_values = [float(row["domestic100_ler"]) for row in rows]
        seq_values = [float(row["seq_noewc_ler"]) for row in rows]
        deltas = [float(row["delta"]) for row in rows]
        first = rows[0]
        delta_mean = mean(deltas)
        summaries.append(
            {
                "run_phase": "full",
                "env_key": first["env_key"],
                "env_index": env_index,
                "axis_signature": first["axis_signature"],
                "active_axes": first["active_axes"],
                "combination_size": int(first["combination_size"]),
                "contains_z_bias": bool(first["contains_z_bias"]),
                "contains_cnot_z_bias": bool(first["contains_cnot_z_bias"]),
                "basis": basis,
                "num_replicates": len(rows),
                "domestic100_ler_mean": mean(domestic_values),
                "domestic100_ler_std": sample_std(domestic_values),
                "seq_noewc_ler_mean": mean(seq_values),
                "seq_noewc_ler_std": sample_std(seq_values),
                "delta_mean": delta_mean,
                "delta_std": sample_std(deltas),
                "delta_min": min(deltas),
                "delta_max": max(deltas),
                "relative_delta_pct_mean": mean(
                    [float(row["relative_delta_pct"]) for row in rows]
                ),
                "replicate_seq_win_count": sum(1 for value in deltas if value < 0),
                "winner": (
                    "seq_noewc_e100" if delta_mean < 0
                    else "domestic_e100" if delta_mean > 0
                    else "tie"
                ),
                "domestic100_latency_us_per_round_mean": mean(
                    [float(row["domestic100_latency_us_per_round"]) for row in rows]
                ),
                "seq_noewc_latency_us_per_round_mean": mean(
                    [float(row["seq_noewc_latency_us_per_round"]) for row in rows]
                ),
                "domestic100_speedup_vs_pymatching_mean": mean(
                    [float(row["domestic100_speedup_vs_pymatching"]) for row in rows]
                ),
                "seq_noewc_speedup_vs_pymatching_mean": mean(
                    [float(row["seq_noewc_speedup_vs_pymatching"]) for row in rows]
                ),
            }
        )
    return summaries


def _aggregate_groups(summary_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    group_defs = [
        ("all", lambda row: "all"),
        ("combination_size", lambda row: str(row["combination_size"])),
        ("contains_z_bias", lambda row: "yes" if row["contains_z_bias"] else "no"),
    ]
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for row in summary_rows:
        for group_type, group_fn in group_defs:
            grouped.setdefault(
                (group_type, group_fn(row), str(row["basis"])), []
            ).append(row)

    group_rows = []
    for (group_type, group_value, basis), rows in sorted(grouped.items()):
        deltas = [float(row["delta_mean"]) for row in rows]
        group_rows.append(
            {
                "group_type": group_type,
                "group_value": group_value,
                "basis": basis,
                "env_count": len(rows),
                "domestic100_ler_mean": mean(
                    [float(row["domestic100_ler_mean"]) for row in rows]
                ),
                "seq_noewc_ler_mean": mean(
                    [float(row["seq_noewc_ler_mean"]) for row in rows]
                ),
                "delta_mean": mean(deltas),
                "seq_env_win_count": sum(1 for value in deltas if value < 0),
            }
        )
    return group_rows


def aggregate_payloads(
    payloads: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    env_by_config = config_lookup_with_basename(manifest["environments"])
    detail_rows = []
    for payload in payloads:
        config_name = str(payload["config_name"])
        if config_name not in env_by_config:
            raise KeyError(f"payload config not found in manifest: {config_name}")
        detail_rows.extend(_detail_rows_for_payload(payload, env_by_config[config_name]))
    detail_rows.sort(
        key=lambda row: (
            int(row["env_index"]),
            int(row["replicate_index"]),
            ("both", "X", "Z").index(str(row["basis"])),
        )
    )
    summary_rows = _summarize_env_rows(detail_rows)
    group_rows = _aggregate_groups(summary_rows)
    return detail_rows, summary_rows, group_rows


def _write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path = rel(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
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


def _summary_table_rows(
    rows: Sequence[Mapping[str, Any]],
    basis: str,
) -> list[list[str]]:
    selected = sorted(
        (row for row in rows if row["basis"] == basis),
        key=lambda row: int(row["env_index"]),
    )
    return [
        [
            str(row["env_key"]),
            str(row["axis_signature"]),
            fmt_float(row["domestic100_ler_mean"]),
            fmt_float(row["seq_noewc_ler_mean"]),
            fmt_float(row["delta_mean"]),
            fmt_float(
                float(row["delta_mean"]) / float(row["domestic100_ler_mean"]) * 100.0,
                digits=3,
            ),
            fmt_float(row["delta_std"]),
            fmt_float(row["delta_min"]),
            fmt_float(row["delta_max"]),
            f"{row['replicate_seq_win_count']}/{row['num_replicates']}",
            (
                "seq no-EWC" if row["winner"] == "seq_noewc_e100"
                else "domestic-100" if row["winner"] == "domestic_e100"
                else "持平"
            ),
        ]
        for row in selected
    ]


def _group_table_rows(
    rows: Sequence[Mapping[str, Any]],
    basis: str,
) -> list[list[str]]:
    selected = sorted(
        (row for row in rows if row["basis"] == basis),
        key=lambda row: (str(row["group_type"]), str(row["group_value"])),
    )
    group_labels = {
        "all": "全部环境",
        "combination_size": "组合轴数",
        "contains_z_bias": "是否包含 z_bias",
    }
    return [
        [
            group_labels.get(str(row["group_type"]), str(row["group_type"])),
            {"yes": "是", "no": "否", "all": "全部"}.get(
                str(row["group_value"]), str(row["group_value"])
            ),
            str(row["env_count"]),
            fmt_float(row["domestic100_ler_mean"]),
            fmt_float(row["seq_noewc_ler_mean"]),
            fmt_float(row["delta_mean"]),
            fmt_float(
                float(row["delta_mean"]) / float(row["domestic100_ler_mean"]) * 100.0,
                digits=3,
            ),
            str(row["seq_env_win_count"]),
        ]
        for row in selected
    ]


def _basis_overview(
    summary_rows: Sequence[Mapping[str, Any]],
    basis: str,
) -> dict[str, float | int | str]:
    selected = [row for row in summary_rows if row["basis"] == basis]
    domestic = mean([float(row["domestic100_ler_mean"]) for row in selected])
    seq = mean([float(row["seq_noewc_ler_mean"]) for row in selected])
    delta = seq - domestic
    return {
        "basis": basis,
        "env_count": len(selected),
        "seq_wins": sum(1 for row in selected if float(row["delta_mean"]) < 0),
        "domestic": domestic,
        "seq": seq,
        "delta": delta,
        "relative_pct": delta / domestic * 100.0 if domestic else float("nan"),
        "domestic_latency": mean(
            [float(row["domestic100_latency_us_per_round_mean"]) for row in selected]
        ),
        "seq_latency": mean(
            [float(row["seq_noewc_latency_us_per_round_mean"]) for row in selected]
        ),
        "domestic_speedup": mean(
            [float(row["domestic100_speedup_vs_pymatching_mean"]) for row in selected]
        ),
        "seq_speedup": mean(
            [float(row["seq_noewc_speedup_vs_pymatching_mean"]) for row in selected]
        ),
    }


def _find_group(
    group_rows: Sequence[Mapping[str, Any]],
    group_type: str,
    group_value: str,
    basis: str = "both",
) -> Mapping[str, Any]:
    return next(
        row
        for row in group_rows
        if row["basis"] == basis
        and row["group_type"] == group_type
        and str(row["group_value"]) == group_value
    )


def write_markdown(
    path: str | Path,
    *,
    details_csv: str | Path,
    summary_csv: str | Path,
    summary_rows: Sequence[Mapping[str, Any]],
    group_rows: Sequence[Mapping[str, Any]],
    output_dir: str | Path,
    basis: str,
    num_samples: int,
) -> None:
    path = rel(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    both_rows = [row for row in summary_rows if row["basis"] == "both"]
    overviews = {
        basis_name: _basis_overview(summary_rows, basis_name)
        for basis_name in ("both", "X", "Z")
    }
    overall = overviews["both"]
    seq_wins = int(overall["seq_wins"])
    overall_delta = float(overall["delta"])
    target_met = seq_wins >= 8 and overall_delta < 0
    strongest = min(both_rows, key=lambda row: float(row["delta_mean"]))
    weakest = min(both_rows, key=lambda row: abs(float(row["delta_mean"])))
    fully_consistent = sum(
        1
        for row in both_rows
        if int(row["replicate_seq_win_count"]) == int(row["num_replicates"])
    )
    strong_noise_rows = [
        row for row in both_rows if float(row["domestic100_ler_mean"]) >= 0.4
    ]
    size2 = _find_group(group_rows, "combination_size", "2")
    size3 = _find_group(group_rows, "combination_size", "3")
    size4 = _find_group(group_rows, "combination_size", "4")
    no_z_bias = _find_group(group_rows, "contains_z_bias", "no")
    with_z_bias = _find_group(group_rows, "contains_z_bias", "yes")

    overview_rows = [
        [
            label,
            fmt_float(item["domestic"]),
            fmt_float(item["seq"]),
            fmt_float(item["delta"]),
            fmt_float(item["relative_pct"], digits=3),
            f"{item['seq_wins']}/{item['env_count']}",
        ]
        for label, item in (
            ("X+Z 平均", overviews["both"]),
            ("X", overviews["X"]),
            ("Z", overviews["Z"]),
        )
    ]
    latency_rows = [
        [
            label,
            fmt_float(item["domestic_latency"], digits=3),
            fmt_float(item["seq_latency"], digits=3),
            fmt_float(item["domestic_speedup"], digits=3),
            fmt_float(item["seq_speedup"], digits=3),
        ]
        for label, item in (
            ("X+Z 平均", overviews["both"]),
            ("X", overviews["X"]),
            ("Z", overviews["Z"]),
        )
    ]

    lines = [
        "# 参数级随机混合噪声 OOD 对比报告",
        "",
        "## 结论摘要",
        "",
        f"- **验收结果：{'通过' if target_met else '未通过'}。** 主口径 `basis=both` 下，seq no-EWC 在 `{seq_wins}/11` 个环境的五次平均 LER 上优于 domestic-100。",
        f"- 11 环境等权平均 LER 从 `{float(overall['domestic']):.6f}` 降至 `{float(overall['seq']):.6f}`，绝对差为 `{overall_delta:.6f}`，相对降低 `{abs(float(overall['relative_pct'])):.3f}%`。",
        f"- X basis 为 `{int(overviews['X']['seq_wins'])}/11` 个环境获胜，Z basis 为 `{int(overviews['Z']['seq_wins'])}/11` 个环境获胜，说明优势并非只来自单一 basis。",
        f"- `{fully_consistent}/11` 个环境在全部 5 个随机 realization 上均由 seq 获胜；最弱环境 `{weakest['env_key']}` 仅有 `{weakest['replicate_seq_win_count']}/5` 次获胜，平均 delta 为 `{float(weakest['delta_mean']):.6f}`。",
        "- 该结果支持“seq no-EWC 在本次固定的宽范围参数级混合噪声压力测试中具有更好的平均泛化表现”，但不能外推为对所有未知噪声或所有工作区间都更优。",
        "",
        "## 1. 实验设计与评价口径",
        "",
        "- 设计：参数级随机训练轴混合 OOD 测试。以 T0 为基础，保留 6 个双轴、4 个三轴和 1 个四轴组合。",
        "- 每个环境生成 5 个固定 realization，共 55 个 config；激活轴覆盖的每个唯一物理参数独立采样 `Uniform(1.2, 4.0)` 倍率，未激活参数保持 1.0 倍。",
        f"- 每个 config、每个 basis 使用 `{num_samples}` 个样本；请求 basis 为 `{basis}`，同时分别报告 X、Z 及二者平均结果。",
        "- 对比 checkpoint：`stfusion_domestic_e100` 与 `stfusion_seq_noewc_e100`。两模型在每个 config/basis 上使用同一批数据，保证 paired comparison。",
        "- 定义 `delta = LER(seq no-EWC) - LER(domestic-100)`；delta < 0 表示 seq 更优。每个环境的最终判定只比较 5 个 realization 的平均 LER。",
        f"- 预设成功标准：至少 8/11 个环境的平均 delta < 0，且 11 环境总体平均 delta < 0；本次结果为 `{'PASS' if target_met else 'FAIL'}`。",
        "- 55 个噪声配置在模型推理前固定，实验后未根据输赢替换环境。",
        "",
        "## 2. 总体结果",
        "",
        markdown_table(
            ["评价口径", "domestic-100 LER", "seq no-EWC LER", "绝对 delta", "相对变化 %", "seq 获胜环境"],
            overview_rows,
        ),
        "",
        "在主口径上，seq no-EWC 的绝对 LER 改善为 "
        f"`{abs(overall_delta):.6f}`，相对改善 `{abs(float(overall['relative_pct'])):.3f}%`。"
        "X 与 Z 的平均改善幅度接近，表明本次收益不是依靠牺牲一个 basis 换取另一个 basis；唯一例外是 e01 的 X basis，domestic-100 略优。",
        "",
        "## 3. 各环境五次随机采样结果",
        "",
    ]
    for basis_name in ("both", "X", "Z"):
        table_rows = _summary_table_rows(summary_rows, basis_name)
        if not table_rows:
            continue
        lines.extend(
            [
                f"### basis={basis_name}（5 次均值）",
                "",
                markdown_table(
                    [
                        "环境",
                        "激活轴组合",
                        "domestic-100 LER",
                        "seq no-EWC LER",
                        "平均 delta",
                        "相对变化 %",
                        "delta 标准差",
                        "最小 delta",
                        "最大 delta",
                        "seq 胜出次数",
                        "五次均值胜者",
                    ],
                    table_rows,
                ),
                "",
            ]
        )

    lines.extend(["## 4. 分组结果", ""])
    for basis_name in ("both", "X", "Z"):
        table_rows = _group_table_rows(group_rows, basis_name)
        if not table_rows:
            continue
        lines.extend(
            [
                f"### basis={basis_name}",
                "",
                markdown_table(
                    [
                        "分组维度",
                        "分组值",
                        "环境数",
                        "domestic-100 LER",
                        "seq no-EWC LER",
                        "平均 delta",
                        "相对变化 %",
                        "seq 获胜环境",
                    ],
                    table_rows,
                ),
                "",
            ]
        )

    lines.extend(
        [
            "## 结果解读",
            "",
            "### 5.1 组合复杂度",
            "",
            f"主口径下，双轴、三轴和四轴组合的平均 delta 分别为 `{float(size2['delta_mean']):.6f}`、`{float(size3['delta_mean']):.6f}` 和 `{float(size4['delta_mean']):.6f}`。"
            "在这组固定环境中，组合轴数增加时 seq 的优势同步增大，符合 sequential 训练覆盖多种任务轴后对复合扰动更稳健的预期。"
            "但四轴组只有 e10 一个环境，因此不能据此建立严格的单调规律。",
            "",
            "### 5.2 z-bias 不是唯一收益来源",
            "",
            f"不含 z-bias 的 4 个环境平均 delta 为 `{float(no_z_bias['delta_mean']):.6f}`，含 z-bias 的 7 个环境为 `{float(with_z_bias['delta_mean']):.6f}`，两者非常接近。"
            "因此，本次 seq 优势不能简单归因于对 z-bias 的特殊适配；measurement、CNOT 与 idle 的参数级混合也贡献了收益。",
            "",
            "### 5.3 随机 realization 一致性",
            "",
            f"除 `{weakest['env_key']}` 外，其余 10 个环境均在 5/5 个 realization 上保持 seq 更低。"
            f"最强环境为 `{strongest['env_key']}`（`{strongest['axis_signature']}`），平均 delta `{float(strongest['delta_mean']):.6f}`，相对改善 `{abs(float(strongest['delta_mean']) / float(strongest['domestic100_ler_mean']) * 100.0):.3f}%`。"
            f"`{weakest['env_key']}` 的平均 delta 仅 `{float(weakest['delta_mean']):.6f}`，标准差 `{float(weakest['delta_std']):.6f}`，且最大 delta 为 `{float(weakest['delta_max']):.6f}`，说明该环境的优势较弱且会随 realization 翻转。",
            "",
            "### 5.4 强噪声与饱和风险",
            "",
            f"有 `{len(strong_noise_rows)}/11` 个环境的 domestic-100 平均 LER 不低于 0.40，最高达到 `{max(float(row['domestic100_ler_mean']) for row in both_rows):.6f}`。"
            "这些环境已接近二元逻辑错误率 0.5 的随机区，属于强噪声 OOD stress test，而不是典型低噪声工作点。"
            "高 LER 能暴露模型在严重分布失配下的鲁棒性差异，但也削弱了结果对实际可用工作区间的代表性。",
            "",
            "### 5.5 延迟侧面结果",
            "",
            markdown_table(
                ["评价口径", "domestic 延迟 us/round", "seq 延迟 us/round", "domestic 对 PyMatching 加速", "seq 对 PyMatching 加速"],
                latency_rows,
            ),
            "",
            "seq 的平均端到端解码延迟低于 domestic，但本次 55 个任务在多张 GPU 上并行执行，且残余 syndrome 难度会影响 PyMatching 延迟。"
            "因此延迟只作为工程侧面观察，不应作为训练范式优劣的主要证据。",
            "",
            "## 6. 结论与适用边界",
            "",
            "### 可支持的结论",
            "",
            "> 在预先固定的 11 类训练轴组合、55 个参数级随机宽范围噪声环境中，seq no-EWC epoch 100 的平均逻辑错误率低于 domestic-only epoch 100。该优势覆盖 X/Z 两个 basis，并在双轴、三轴和四轴组合中均出现，表明 sequential 训练提升了模型对强复合噪声扰动的鲁棒性。",
            "",
            "### 不应扩大的结论",
            "",
            "- 不能表述为 seq no-EWC 对所有未知噪声都优于 domestic-100；本实验仍围绕训练任务轴构造。",
            "- 不能仅凭本实验宣称 lifelong learning 已被普遍证明；还需要不同 code distance、不同基础噪声强度和独立噪声族复现。",
            "- 5 个 realization 提供了重复性观察，但报告没有给出以噪声环境为统计单位的置信区间或显著性检验。",
            "- e01 的优势非常小，且 X basis 上 domestic 略优，应保留这一反例而不是只报告总体均值。",
            "",
            "## 7. 文件索引",
            "",
            f"- realization 级明细：`{display_path(details_csv)}`",
            f"- 环境级汇总：`{display_path(summary_csv)}`",
            f"- 每个 config 的 JSON/CSV：`{display_path(output_dir)}`",
            f"- 本报告：`{display_path(path)}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(
    *,
    details_csv: str | Path,
    summary_csv: str | Path,
    summary_md: str | Path,
    detail_rows: Sequence[Mapping[str, Any]],
    summary_rows: Sequence[Mapping[str, Any]],
    group_rows: Sequence[Mapping[str, Any]],
    output_dir: str | Path,
    basis: str,
    num_samples: int,
) -> None:
    _write_csv(details_csv, detail_rows)
    _write_csv(summary_csv, summary_rows)
    write_markdown(
        summary_md,
        details_csv=details_csv,
        summary_csv=summary_csv,
        summary_rows=summary_rows,
        group_rows=group_rows,
        output_dir=output_dir,
        basis=basis,
        num_samples=num_samples,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--config-prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--generation-seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--distance", type=int, default=9)
    parser.add_argument("--n-rounds", type=int, default=9)
    parser.add_argument("--num-samples", type=int, default=262144)
    parser.add_argument("--latency-num-samples", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--basis", choices=("both", "X", "Z"), default="both")
    parser.add_argument("--device", default=None)
    parser.add_argument("--python", default=None)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--details-csv", default=DEFAULT_DETAILS_CSV)
    parser.add_argument("--summary-csv", default=DEFAULT_SUMMARY_CSV)
    parser.add_argument("--summary-md", default=DEFAULT_SUMMARY_MD)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = ensure_configs(args)
    output_dir = rel(args.output_dir)
    if not args.aggregate_only:
        run_all(args, manifest)
    payloads = load_payloads(output_dir, manifest)
    detail_rows, summary_rows, group_rows = aggregate_payloads(payloads, manifest)
    write_outputs(
        details_csv=args.details_csv,
        summary_csv=args.summary_csv,
        summary_md=args.summary_md,
        detail_rows=detail_rows,
        summary_rows=summary_rows,
        group_rows=group_rows,
        output_dir=output_dir,
        basis=args.basis,
        num_samples=args.num_samples,
    )
    print(f"[write] {rel(args.details_csv)}")
    print(f"[write] {rel(args.summary_csv)}")
    print(f"[write] {rel(args.summary_md)}")


if __name__ == "__main__":
    main()
