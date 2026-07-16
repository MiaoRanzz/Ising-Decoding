#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run and aggregate unknown T0-random generalization paired inference."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.config_paths import config_path  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[2]
PAIRED_SCRIPT = REPO_ROOT / "code" / "scripts" / "paired_inference_compare.py"

DEFAULT_UNKNOWN_SEED = 20260714
DEFAULT_NUM_ENVS = 5
DEFAULT_PREFIX = "config_unknown_t0_random"
DEFAULT_CONFIG_DIR = "experiments/unknown_t0_random"

MODEL_SPECS = [
    (
        "stfusion_domestic_e20",
        "ST-Fusion-R9-X domestic-only epoch 20",
        111,
        "outputs/ising_domestic_fast_opt_stfusion_r9_x/models/PreDecoderSTFusion_v2.0.20.pt",
    ),
    (
        "stfusion_domestic_e100",
        "ST-Fusion-R9-X domestic-only epoch 100",
        111,
        "outputs/ising_domestic_fast_opt_stfusion_r9_x/models/PreDecoderSTFusion_v2.0.100.pt",
    ),
    (
        "stfusion_mixed_e100",
        "ST-Fusion-R9-X mixed-noise epoch 100",
        111,
        "outputs/ising_domestic_fast_opt_stfusion_r9_x_mixed_noise/models/PreDecoderSTFusion_v2.0.100.pt",
    ),
    (
        "stfusion_seq_noewc_e100",
        "ST-Fusion-R9-X sequential no-EWC epoch 100",
        111,
        "outputs/ising_domestic_fast_opt_stfusion_r9_x_seq_noewc/models/PreDecoderSTFusion_v2.0.100.pt",
    ),
    (
        "stfusion_seq_ewc_e100",
        "ST-Fusion-R9-X sequential + EWC epoch 100",
        111,
        "outputs/ising_domestic_fast_opt_stfusion_r9_x_seq_ewc/models/PreDecoderSTFusion_v2.0.100.pt",
    ),
]

METHOD_ORDER = ["pymatching"] + [spec[0] for spec in MODEL_SPECS]
METHOD_LABELS = {
    "pymatching": "PyMatching / no-predecoder",
    **{spec[0]: spec[1] for spec in MODEL_SPECS},
}
SEQ_METHODS = {"stfusion_seq_noewc_e100", "stfusion_seq_ewc_e100"}
BASELINE_METHODS = [
    "stfusion_domestic_e20",
    "stfusion_domestic_e100",
    "stfusion_mixed_e100",
]


def rel(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def display_path(path: str | Path) -> str:
    path = Path(path)
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def default_config_names(*, unknown_seed: int = DEFAULT_UNKNOWN_SEED, num_envs: int = DEFAULT_NUM_ENVS,
                         prefix: str = DEFAULT_PREFIX) -> list[str]:
    return [f"{DEFAULT_CONFIG_DIR}/{prefix}_s{int(unknown_seed)}_e{i:02d}" for i in range(int(num_envs))]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distance", type=int, default=9)
    parser.add_argument("--n-rounds", type=int, default=9)
    parser.add_argument("--num-samples", type=int, default=262144)
    parser.add_argument("--latency-num-samples", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=12345, help="Inference data seed.")
    parser.add_argument("--basis", choices=("both", "X", "Z"), default="both")
    parser.add_argument("--device", default=None)
    parser.add_argument("--python", default=None)
    parser.add_argument("--unknown-seed", type=int, default=DEFAULT_UNKNOWN_SEED)
    parser.add_argument("--num-envs", type=int, default=DEFAULT_NUM_ENVS)
    parser.add_argument("--config-prefix", default=DEFAULT_PREFIX)
    parser.add_argument(
        "--config-name",
        action="append",
        default=None,
        help="Repeat to override the default unknown environment config names.",
    )
    parser.add_argument("--output-dir", default="outputs/paired_inference_compare/unknown_t0_random")
    parser.add_argument(
        "--summary-csv",
        default="outputs/analysis/unknown_t0_random_generalization_summary.csv",
    )
    parser.add_argument(
        "--detail-csv",
        default="outputs/analysis/unknown_t0_random_generalization_details.csv",
    )
    parser.add_argument(
        "--summary-md",
        default="outputs/analysis/unknown_t0_random_generalization_comparison.md",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    return parser.parse_args()


def resolve_python(args: argparse.Namespace) -> str:
    candidates = [
        args.python,
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


def fmt_latency(value: Any) -> str:
    return fmt_float(value, digits=3)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def method_sort_key(method: str) -> int:
    return METHOD_ORDER.index(method) if method in METHOD_ORDER else len(METHOD_ORDER)


def config_names_from_args(args: argparse.Namespace) -> list[str]:
    if args.config_name:
        return list(args.config_name)
    return default_config_names(
        unknown_seed=args.unknown_seed,
        num_envs=args.num_envs,
        prefix=args.config_prefix,
    )


def task_output_path(output_dir: Path, env_index: int) -> Path:
    return output_dir / f"unknown_t0_random_e{env_index:02d}.json"


def existing_inputs_or_raise(config_names: list[str]) -> None:
    missing = []
    if not PAIRED_SCRIPT.exists():
        missing.append(PAIRED_SCRIPT)
    for config_name in config_names:
        path = config_path(config_name)
        if not path.exists():
            missing.append(path)
    for _, _, _, checkpoint in MODEL_SPECS:
        path = rel(checkpoint)
        if not path.exists():
            missing.append(path)
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Required inputs are missing:\n{formatted}")


def build_command(args: argparse.Namespace, config_name: str, output_path: Path) -> list[str]:
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
    for name, _, model_id, checkpoint in MODEL_SPECS:
        cmd.extend(["--model", f"{name}:{model_id}:{rel(checkpoint)}"])
    return cmd


def run_inference(args: argparse.Namespace, config_names: list[str], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for env_index, config_name in enumerate(config_names):
        output_path = task_output_path(output_dir, env_index)
        cmd = build_command(args, config_name, output_path)
        print(f"[env] e{env_index:02d} config={config_name} -> {output_path}")
        print("[cmd] " + " ".join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def load_payloads(output_dir: str | Path, config_names: list[str]) -> list[dict[str, Any]]:
    output_dir = rel(output_dir)
    payloads = []
    missing = []
    for env_index, config_name in enumerate(config_names):
        output_path = task_output_path(output_dir, env_index)
        if not output_path.exists():
            missing.append(output_path)
            continue
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        payload["env_index"] = env_index
        payload["env_key"] = f"e{env_index:02d}"
        payload["config_name"] = config_name
        payload["path"] = str(output_path)
        payloads.append(payload)
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Missing per-environment inference outputs:\n{formatted}")
    return payloads


def _init_summary_entry(method: str) -> dict[str, Any]:
    return {
        "method": method,
        "label": METHOD_LABELS.get(method, method),
        "env_ler_avgs": [],
        "env_latency_avgs": [],
        "env_speedup_avgs": [],
        "x_lers": [],
        "z_lers": [],
        "env_values": {},
    }


def aggregate_rows(payloads: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    detail_rows = []
    summary_by_method: dict[str, dict[str, Any]] = {}

    for payload in payloads:
        env_key = payload["env_key"]
        env_index = payload["env_index"]
        config_name = payload["config_name"]
        for row in payload["rows"]:
            detail = {
                "env_key": env_key,
                "env_index": env_index,
                "config_name": config_name,
                "basis": row.get("basis"),
                "method": row.get("method"),
                "model_id": row.get("model_id", ""),
                "logical_errors": row.get("logical_errors"),
                "samples": row.get("samples"),
                "ler": row.get("ler"),
                "latency_us_per_round": row.get("latency_us_per_round"),
                "speedup_vs_pymatching": row.get("speedup_vs_pymatching"),
                "checkpoint": row.get("checkpoint", ""),
            }
            detail_rows.append(detail)

        for item in payload["summary"]:
            method = item["method"]
            entry = summary_by_method.setdefault(method, _init_summary_entry(method))
            ler_avg = float(item["ler_avg"])
            entry["env_ler_avgs"].append(ler_avg)
            entry["env_latency_avgs"].append(float(item["latency_us_per_round_avg"]))
            entry["env_speedup_avgs"].append(float(item["speedup_vs_pymatching_avg"]))
            entry["env_values"][env_key] = ler_avg

    for detail in detail_rows:
        method = detail["method"]
        if method not in summary_by_method:
            continue
        if detail["basis"] == "X":
            summary_by_method[method]["x_lers"].append(float(detail["ler"]))
        elif detail["basis"] == "Z":
            summary_by_method[method]["z_lers"].append(float(detail["ler"]))

    summary_rows = []
    env_keys = [payload["env_key"] for payload in payloads]
    for method, entry in sorted(summary_by_method.items(), key=lambda kv: method_sort_key(kv[0])):
        env_lers = entry["env_ler_avgs"]
        row = {
            "method": method,
            "label": entry["label"],
            "ler_avg_unknown_envs": mean(env_lers),
            "ler_worst_env": max(env_lers) if env_lers else float("nan"),
            "ler_x_avg": mean(entry["x_lers"]),
            "ler_z_avg": mean(entry["z_lers"]),
            "latency_us_per_round_avg": mean(entry["env_latency_avgs"]),
            "speedup_vs_pymatching_avg": mean(entry["env_speedup_avgs"]),
        }
        for env_key in env_keys:
            row[f"ler_{env_key}"] = entry["env_values"].get(env_key, float("nan"))
        summary_rows.append(row)

    by_method = {row["method"]: row for row in summary_rows}
    for row in summary_rows:
        for baseline_method in BASELINE_METHODS:
            delta_key = f"delta_vs_{baseline_method}"
            rel_key = f"rel_delta_pct_vs_{baseline_method}"
            row[delta_key] = ""
            row[rel_key] = ""
            if row["method"] not in SEQ_METHODS or baseline_method not in by_method:
                continue
            baseline = float(by_method[baseline_method]["ler_avg_unknown_envs"])
            value = float(row["ler_avg_unknown_envs"])
            row[delta_key] = value - baseline
            row[rel_key] = ((value - baseline) / baseline * 100.0) if baseline else float("nan")
    return detail_rows, summary_rows


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = rel(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _comparison_rows(summary_rows: list[dict[str, Any]]) -> list[list[str]]:
    by_method = {row["method"]: row for row in summary_rows}
    labels = {
        "stfusion_domestic_e20": "domestic 20",
        "stfusion_domestic_e100": "domestic 100",
        "stfusion_mixed_e100": "mixed 100",
    }
    seq_labels = {
        "stfusion_seq_noewc_e100": "seq no-EWC",
        "stfusion_seq_ewc_e100": "seq EWC",
    }
    rows = []
    for seq_method in ["stfusion_seq_noewc_e100", "stfusion_seq_ewc_e100"]:
        if seq_method not in by_method:
            continue
        for baseline in BASELINE_METHODS:
            if baseline not in by_method:
                continue
            delta = by_method[seq_method].get(f"delta_vs_{baseline}")
            rel = by_method[seq_method].get(f"rel_delta_pct_vs_{baseline}")
            rows.append(
                [
                    f"{seq_labels[seq_method]} vs {labels[baseline]}",
                    fmt_float(delta),
                    fmt_float(rel, digits=2),
                ]
            )
    return rows


def write_markdown(
    path: str | Path,
    summary_rows: list[dict[str, Any]],
    config_names: list[str],
    *,
    distance: int,
    n_rounds: int,
    num_samples: int,
    latency_num_samples: int,
    seed: int,
    basis: str,
    output_dir: str | Path,
    summary_csv: str | Path,
    detail_csv: str | Path,
) -> None:
    path = rel(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    env_keys = [f"e{i:02d}" for i in range(len(config_names))]

    summary_table = markdown_table(
        [
            "模型",
            "unknown-env LER-Avg",
            "worst-env LER",
            "X LER 均值",
            "Z LER 均值",
            "latency Avg (us/round)",
            "speedup vs PyMatching",
        ],
        [
            [
                row["label"],
                fmt_float(row["ler_avg_unknown_envs"]),
                fmt_float(row["ler_worst_env"]),
                fmt_float(row["ler_x_avg"]),
                fmt_float(row["ler_z_avg"]),
                fmt_latency(row["latency_us_per_round_avg"]),
                fmt_float(row["speedup_vs_pymatching_avg"], digits=3),
            ]
            for row in summary_rows
        ],
    )

    env_table = markdown_table(
        ["模型"] + env_keys,
        [
            [row["label"]] + [fmt_float(row.get(f"ler_{env_key}")) for env_key in env_keys]
            for row in summary_rows
        ],
    )
    comparison_table = markdown_table(
        ["对比", "LER-Avg delta", "relative delta (%)"],
        _comparison_rows(summary_rows),
    )
    config_lines = "\n".join(f"- `{name}`" for name in config_names)

    text = f"""# Unknown T0 Random-Noise Generalization Comparison

## 1. 推理口径

- distance: `{distance}`
- n_rounds: `{n_rounds}`
- num_samples: `{num_samples}`
- latency_num_samples: `{latency_num_samples}`
- seed: `{seed}`
- basis: `{basis}`
- unknown environments: `{len(config_names)}`
- 评估脚本: `code/scripts/paired_inference_compare.py`
- 聚合脚本: `code/scripts/run_unknown_t0_random_compare.py`

## 2. 未知环境 configs

{config_lines}

## 3. 汇总结果

{summary_table}

## 4. 分环境 LER-Avg

{env_table}

## 5. Sequential 泛化对比

负值表示 sequential 方法在 unknown-env LER-Avg 上更低。

{comparison_table}

## 6. 解释边界

- 若 sequential 只优于 domestic epoch 20，应表述为“连续训练后的模型优于早期 domestic checkpoint”。
- 若 sequential 同时优于 domestic epoch 100 和 mixed-noise epoch 100，才表述为“训练范式带来未知环境泛化收益”。
- EWC 不应被写成最优，除非本表中 sequential + EWC 的 LER 明确低于 sequential no-EWC。

## 7. 结果文件

- Per-env JSON/CSV: `{display_path(output_dir)}/unknown_t0_random_e<idx>.json|csv`
- Summary CSV: `{display_path(summary_csv)}`
- Detail CSV: `{display_path(detail_csv)}`
- Markdown: `{display_path(path)}`
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    config_names = config_names_from_args(args)
    output_dir = rel(args.output_dir)
    summary_csv = rel(args.summary_csv)
    detail_csv = rel(args.detail_csv)
    summary_md = rel(args.summary_md)

    existing_inputs_or_raise(config_names)
    if not args.aggregate_only:
        run_inference(args, config_names, output_dir)
    if args.dry_run:
        return

    payloads = load_payloads(output_dir, config_names)
    detail_rows, summary_rows = aggregate_rows(payloads)
    write_csv(detail_csv, detail_rows)
    write_csv(summary_csv, summary_rows)
    write_markdown(
        summary_md,
        summary_rows,
        config_names,
        distance=args.distance,
        n_rounds=args.n_rounds,
        num_samples=args.num_samples,
        latency_num_samples=args.latency_num_samples,
        seed=args.seed,
        basis=args.basis,
        output_dir=output_dir,
        summary_csv=summary_csv,
        detail_csv=detail_csv,
    )
    print(f"[write] {detail_csv}")
    print(f"[write] {summary_csv}")
    print(f"[write] {summary_md}")


if __name__ == "__main__":
    main()
