#!/usr/bin/env python3
"""Run and aggregate R9 lifelong paired inference comparisons.

The script keeps the existing paired inference implementation as the single
source of evaluation logic. It only orchestrates five noise-task runs and then
aggregates their JSON outputs into CSV/Markdown summaries.
"""

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


REPO_ROOT = Path(__file__).resolve().parents[2]
PAIRED_SCRIPT = REPO_ROOT / "code" / "scripts" / "paired_inference_compare.py"

TASKS = [
    ("t0_base", "T0_base", "config_domestic_fast_opt_stfusion_r9_x_seq_ewc_t0_base"),
    ("t1_meas_1p5", "T1_meas_1p5", "config_domestic_fast_opt_stfusion_r9_x_seq_ewc_t1_meas_1p5"),
    ("t2_cnot_1p5", "T2_cnot_1p5", "config_domestic_fast_opt_stfusion_r9_x_seq_ewc_t2_cnot_1p5"),
    ("t3_idle_1p5", "T3_idle_1p5", "config_domestic_fast_opt_stfusion_r9_x_seq_ewc_t3_idle_1p5"),
    ("t4_z_bias_1p5", "T4_z_bias_1p5", "config_domestic_fast_opt_stfusion_r9_x_seq_ewc_t4_z_bias_1p5"),
]

MODEL_SPECS = [
    (
        "ising_domestic",
        1,
        "outputs/ising_domestic_fast/models/best_model/PreDecoderModelMemory_v1.0.53.pt",
        "domestic-only best",
    ),
    (
        "ising_mixed",
        1,
        "outputs/ising_domestic_fast_mixed_noise_r9/models/PreDecoderModelMemory_v1.0.100.pt",
        "mixed-noise final",
    ),
    (
        "stfusion_x_domestic",
        111,
        "outputs/ising_domestic_fast_opt_stfusion_r9_x/models/best_model/PreDecoderSTFusion_v2.0.89.pt",
        "domestic-only best",
    ),
    (
        "stfusion_x_mixed",
        111,
        "outputs/ising_domestic_fast_opt_stfusion_r9_x_mixed_noise/models/PreDecoderSTFusion_v2.0.100.pt",
        "mixed-noise final",
    ),
    (
        "stfusion_x_seq_noewc",
        111,
        "outputs/ising_domestic_fast_opt_stfusion_r9_x_seq_noewc/models/PreDecoderSTFusion_v2.0.100.pt",
        "sequential no-EWC final",
    ),
    (
        "stfusion_x_seq_ewc",
        111,
        "outputs/ising_domestic_fast_opt_stfusion_r9_x_seq_ewc/models/PreDecoderSTFusion_v2.0.100.pt",
        "sequential + EWC final",
    ),
]

METHOD_ORDER = [
    "pymatching",
    "ising_domestic",
    "ising_mixed",
    "stfusion_x_domestic",
    "stfusion_x_mixed",
    "stfusion_x_seq_noewc",
    "stfusion_x_seq_ewc",
]

METHOD_LABELS = {
    "pymatching": "PyMatching / no-predecoder",
    "ising_domestic": "Ising fast domestic-only",
    "ising_mixed": "Ising fast mixed-noise",
    "stfusion_x_domestic": "ST-Fusion-R9-X domestic-only",
    "stfusion_x_mixed": "ST-Fusion-R9-X mixed-noise",
    "stfusion_x_seq_noewc": "ST-Fusion-R9-X sequential no-EWC",
    "stfusion_x_seq_ewc": "ST-Fusion-R9-X sequential + EWC",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run five-task R9 lifelong paired inference comparison and aggregate results."
    )
    parser.add_argument("--distance", type=int, default=9)
    parser.add_argument("--n-rounds", type=int, default=9)
    parser.add_argument("--num-samples", type=int, default=262144)
    parser.add_argument("--latency-num-samples", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--basis", choices=("both", "X", "Z"), default="both")
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--ising-domestic-checkpoint",
        default="outputs/ising_domestic_fast/models/best_model/PreDecoderModelMemory_v1.0.53.pt",
        help="Checkpoint for Ising fast domestic-only baseline.",
    )
    parser.add_argument(
        "--stfusion-domestic-checkpoint",
        default="outputs/ising_domestic_fast_opt_stfusion_r9_x/models/best_model/PreDecoderSTFusion_v2.0.89.pt",
        help="Checkpoint for ST-Fusion-R9-X domestic-only baseline.",
    )
    parser.add_argument(
        "--python",
        default=None,
        help=(
            "Python executable used to launch paired_inference_compare.py. "
            "Defaults to PREDECODER_PYTHON, then the ising-decoding conda env, then sys.executable."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/paired_inference_compare/lifelong_r9",
        help="Directory for per-task JSON/CSV outputs.",
    )
    parser.add_argument(
        "--summary-md",
        default="outputs/analysis/lifelong_r9_unified_decoder_comparison.md",
        help="Markdown summary path.",
    )
    parser.add_argument(
        "--summary-csv",
        default="outputs/analysis/lifelong_r9_unified_decoder_summary.csv",
        help="Aggregated method summary CSV path.",
    )
    parser.add_argument(
        "--detail-csv",
        default="outputs/analysis/lifelong_r9_unified_decoder_details.csv",
        help="Aggregated per-task/per-basis CSV path.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running inference.")
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Skip inference and aggregate existing per-task JSON outputs.",
    )
    return parser.parse_args()


def rel(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def fmt_float(value: Any, digits: int = 6) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if math.isnan(number) or math.isinf(number):
        return ""
    return f"{number:.{digits}f}"


def fmt_latency(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if math.isnan(number) or math.isinf(number):
        return ""
    return f"{number:.3f}"


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def model_specs(args: argparse.Namespace):
    specs = []
    for name, model_id, checkpoint, policy in MODEL_SPECS:
        if name == "ising_domestic":
            checkpoint = args.ising_domestic_checkpoint
        elif name == "stfusion_x_domestic":
            checkpoint = args.stfusion_domestic_checkpoint
        specs.append((name, model_id, checkpoint, policy))
    return specs


def existing_inputs_or_raise(args: argparse.Namespace) -> None:
    missing = []
    if not PAIRED_SCRIPT.exists():
        missing.append(PAIRED_SCRIPT)
    for _, _, config_name in TASKS:
        path = REPO_ROOT / "conf" / f"{config_name}.yaml"
        if not path.exists():
            missing.append(path)
    for _, _, checkpoint, _ in model_specs(args):
        path = rel(checkpoint)
        if not path.exists():
            missing.append(path)
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Required inputs are missing:\n{formatted}")


def task_output_path(output_dir: Path, task_key: str) -> Path:
    return output_dir / f"lifelong_r9_{task_key}.json"


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
    for name, model_id, checkpoint, _ in model_specs(args):
        cmd.extend(["--model", f"{name}:{model_id}:{rel(checkpoint)}"])
    return cmd


def run_inference(args: argparse.Namespace, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for task_key, task_name, config_name in TASKS:
        output_path = task_output_path(output_dir, task_key)
        cmd = build_command(args, config_name, output_path)
        print(f"[task] {task_name} -> {output_path}")
        print("[cmd] " + " ".join(cmd))
        if args.dry_run:
            continue
        subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def load_payloads(output_dir: Path) -> list[dict[str, Any]]:
    payloads = []
    missing = []
    for task_key, task_name, _ in TASKS:
        output_path = task_output_path(output_dir, task_key)
        if not output_path.exists():
            missing.append(output_path)
            continue
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        payload["task_key"] = task_key
        payload["task_name"] = task_name
        payload["path"] = str(output_path)
        payloads.append(payload)
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Missing per-task inference outputs:\n{formatted}")
    return payloads


def method_sort_key(method: str) -> int:
    return METHOD_ORDER.index(method) if method in METHOD_ORDER else len(METHOD_ORDER)


def aggregate_rows(payloads: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    detail_rows = []
    summary_by_method: dict[str, dict[str, Any]] = {}

    for payload in payloads:
        task_key = payload["task_key"]
        task_name = payload["task_name"]
        for row in payload["rows"]:
            detail = {
                "task_key": task_key,
                "task_name": task_name,
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
            entry = summary_by_method.setdefault(
                method,
                {
                    "method": method,
                    "task_ler_avgs": [],
                    "task_latency_avgs": [],
                    "task_speedup_avgs": [],
                    "x_lers": [],
                    "z_lers": [],
                    "task_values": {},
                },
            )
            ler_avg = float(item["ler_avg"])
            latency_avg = float(item["latency_us_per_round_avg"])
            speedup_avg = float(item["speedup_vs_pymatching_avg"])
            entry["task_ler_avgs"].append(ler_avg)
            entry["task_latency_avgs"].append(latency_avg)
            entry["task_speedup_avgs"].append(speedup_avg)
            entry["task_values"][task_key] = ler_avg

    for detail in detail_rows:
        method = detail["method"]
        if method not in summary_by_method:
            continue
        if detail["basis"] == "X":
            summary_by_method[method]["x_lers"].append(float(detail["ler"]))
        elif detail["basis"] == "Z":
            summary_by_method[method]["z_lers"].append(float(detail["ler"]))

    summary_rows = []
    for method, entry in sorted(summary_by_method.items(), key=lambda kv: method_sort_key(kv[0])):
        task_lers = entry["task_ler_avgs"]
        summary_rows.append(
            {
                "method": method,
                "label": METHOD_LABELS.get(method, method),
                "ler_avg_5task": mean(task_lers),
                "ler_worst_task": max(task_lers) if task_lers else float("nan"),
                "ler_x_avg": mean(entry["x_lers"]),
                "ler_z_avg": mean(entry["z_lers"]),
                "latency_us_per_round_avg": mean(entry["task_latency_avgs"]),
                "speedup_vs_pymatching_avg": mean(entry["task_speedup_avgs"]),
                **{f"ler_{task_key}": entry["task_values"].get(task_key, float("nan")) for task_key, _, _ in TASKS},
            }
        )
    return detail_rows, summary_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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


def write_markdown(path: Path, args: argparse.Namespace, summary_rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    by_method = {row["method"]: row for row in summary_rows}
    ewc = by_method.get("stfusion_x_seq_ewc", {})
    mixed = by_method.get("stfusion_x_mixed", {})
    ising_mixed = by_method.get("ising_mixed", {})
    domestic = by_method.get("stfusion_x_domestic", {})

    def delta(a: dict[str, Any], b: dict[str, Any]) -> str:
        if not a or not b:
            return ""
        return fmt_float(float(a["ler_avg_5task"]) - float(b["ler_avg_5task"]))

    summary_table = markdown_table(
        [
            "模型",
            "5环境 LER-Avg",
            "最差任务 LER-Avg",
            "X LER 均值",
            "Z LER 均值",
            "latency Avg (us/round)",
            "speedup vs PyMatching",
        ],
        [
            [
                row["label"],
                fmt_float(row["ler_avg_5task"]),
                fmt_float(row["ler_worst_task"]),
                fmt_float(row["ler_x_avg"]),
                fmt_float(row["ler_z_avg"]),
                fmt_latency(row["latency_us_per_round_avg"]),
                fmt_float(row["speedup_vs_pymatching_avg"], digits=3),
            ]
            for row in summary_rows
        ],
    )

    task_headers = ["模型"] + [task_name for _, task_name, _ in TASKS]
    task_rows = []
    for row in summary_rows:
        task_rows.append(
            [row["label"]]
            + [fmt_float(row.get(f"ler_{task_key}")) for task_key, _, _ in TASKS]
        )
    task_table = markdown_table(task_headers, task_rows)

    checkpoint_rows = []
    for name, model_id, checkpoint, policy in model_specs(args):
        checkpoint_rows.append([METHOD_LABELS.get(name, name), str(model_id), policy, f"`{checkpoint}`"])
    checkpoint_table = markdown_table(["模型", "model_id", "checkpoint 口径", "checkpoint"], checkpoint_rows)

    latency_limit = ""
    if ewc and domestic:
        latency_limit_value = float(domestic["latency_us_per_round_avg"]) * 1.05
        latency_limit = (
            f"- latency 约束：EWC={fmt_latency(ewc['latency_us_per_round_avg'])} us/round；"
            f"ST-Fusion-R9-X domestic-only ×1.05={fmt_latency(latency_limit_value)} us/round。\n"
        )

    text = f"""# R9 Lifelong Decoder 统一泛化推理对比

## 1. 推理口径

- distance: `{args.distance}`
- n_rounds: `{args.n_rounds}`
- num_samples: `{args.num_samples}`
- latency_num_samples: `{args.latency_num_samples}`
- batch_size: `{args.batch_size}`
- num_workers: `{args.num_workers}`
- seed: `{args.seed}`
- basis: `{args.basis}`
- 评估脚本: `code/scripts/paired_inference_compare.py`
- 聚合脚本: `code/scripts/run_lifelong_r9_paired_compare.py`

## 2. Checkpoint 口径

mixed-noise、sequential no-EWC、sequential + EWC 使用最终 checkpoint 作为主结果；domestic-only baseline 使用对应 best checkpoint。

{checkpoint_table}

## 3. 5 环境汇总结果

{summary_table}

## 4. 分任务 LER-Avg

{task_table}

## 5. 主结论检查

- EWC vs ST-Fusion-R9-X mixed-noise LER-Avg 差值：`{delta(ewc, mixed)}`，负值表示 EWC 更好。
- EWC vs Ising fast mixed-noise LER-Avg 差值：`{delta(ewc, ising_mixed)}`，负值表示 EWC 更好。
{latency_limit}- 单任务稳定性：以第 4 节分任务 LER-Avg 表为准，重点检查 EWC 是否在任一任务上相对 ST-Fusion-R9-X mixed-noise 明显失控。

## 6. 结果文件

- 聚合 Markdown: `{path.relative_to(REPO_ROOT)}`
- 聚合 summary CSV: `{rel(args.summary_csv).relative_to(REPO_ROOT)}`
- 聚合 detail CSV: `{rel(args.detail_csv).relative_to(REPO_ROOT)}`
- 单任务 JSON/CSV: `{rel(args.output_dir).relative_to(REPO_ROOT)}/lifelong_r9_<task>.json|csv`
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = rel(args.output_dir)
    summary_md = rel(args.summary_md)
    summary_csv = rel(args.summary_csv)
    detail_csv = rel(args.detail_csv)

    existing_inputs_or_raise(args)
    if not args.aggregate_only:
        run_inference(args, output_dir)
    if args.dry_run:
        return

    payloads = load_payloads(output_dir)
    detail_rows, summary_rows = aggregate_rows(payloads)
    write_csv(detail_csv, detail_rows)
    write_csv(summary_csv, summary_rows)
    write_markdown(summary_md, args, summary_rows)
    print(f"[write] {detail_csv}")
    print(f"[write] {summary_csv}")
    print(f"[write] {summary_md}")


if __name__ == "__main__":
    main()
