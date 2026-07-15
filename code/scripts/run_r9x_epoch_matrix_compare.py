#!/usr/bin/env python3
"""Run R9-X epoch-by-epoch paired inference matrices across five noise tasks."""

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

GROUPS = [
    (
        "r9x_domestic",
        "R9-X domestic-only",
        [20],
        "outputs/ising_domestic_fast_opt_stfusion_r9_x/models/PreDecoderSTFusion_v2.0.{epoch}.pt",
    ),
    (
        "r9x_seq_noewc",
        "R9-X sequential no-EWC",
        [20, 40, 60, 80, 100],
        "outputs/ising_domestic_fast_opt_stfusion_r9_x_seq_noewc/models/PreDecoderSTFusion_v2.0.{epoch}.pt",
    ),
    (
        "r9x_seq_ewc",
        "R9-X sequential + EWC",
        [20, 40, 60, 80, 100],
        "outputs/ising_domestic_fast_opt_stfusion_r9_x_seq_ewc/models/PreDecoderSTFusion_v2.0.{epoch}.pt",
    ),
    (
        "r9x_mixed",
        "R9-X mixed-noise",
        [20, 40, 60, 80, 100],
        "outputs/ising_domestic_fast_opt_stfusion_r9_x_mixed_noise/models/PreDecoderSTFusion_v2.0.{epoch}.pt",
    ),
]

STAGE_TASK_BY_EPOCH = {
    20: "t0_base",
    40: "t1_meas_1p5",
    60: "t2_cnot_1p5",
    80: "t3_idle_1p5",
    100: "t4_z_bias_1p5",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument("--output-dir", default="outputs/paired_inference_compare/r9x_epoch_matrix")
    parser.add_argument("--matrix-csv", default="outputs/analysis/r9x_epoch_matrix_ler.csv")
    parser.add_argument("--detail-csv", default="outputs/analysis/r9x_epoch_matrix_details.csv")
    parser.add_argument("--summary-md", default="outputs/analysis/r9x_epoch_matrix_analysis.md")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    return parser.parse_args()


def rel(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


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


def fmt(value: Any, digits: int = 6) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if math.isnan(number) or math.isinf(number):
        return ""
    return f"{number:.{digits}f}"


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def checkpoint_specs() -> list[dict[str, Any]]:
    specs = []
    for group_key, group_label, epochs, pattern in GROUPS:
        for epoch in epochs:
            checkpoint = rel(pattern.format(epoch=epoch))
            method = f"{group_key}_e{epoch}"
            specs.append(
                {
                    "method": method,
                    "group": group_key,
                    "label": group_label,
                    "epoch": epoch,
                    "model_id": 111,
                    "checkpoint": checkpoint,
                }
            )
    return specs


def existing_inputs_or_raise() -> None:
    missing = []
    if not PAIRED_SCRIPT.exists():
        missing.append(PAIRED_SCRIPT)
    for _, _, config_name in TASKS:
        path = REPO_ROOT / "conf" / f"{config_name}.yaml"
        if not path.exists():
            missing.append(path)
    for spec in checkpoint_specs():
        if not spec["checkpoint"].exists():
            missing.append(spec["checkpoint"])
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Required inputs are missing:\n{formatted}")


def task_output_path(output_dir: Path, task_key: str) -> Path:
    return output_dir / f"r9x_epoch_matrix_{task_key}.json"


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
    for spec in checkpoint_specs():
        cmd.extend(
            [
                "--model",
                f"{spec['method']}:{spec['model_id']}:{spec['checkpoint']}",
            ]
        )
    return cmd


def run_inference(args: argparse.Namespace, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for task_key, task_name, config_name in TASKS:
        output_path = task_output_path(output_dir, task_key)
        cmd = build_command(args, config_name, output_path)
        print(f"[task] {task_name} -> {output_path}")
        print("[cmd] " + " ".join(cmd))
        if not args.dry_run:
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
        raise FileNotFoundError(f"Missing per-task outputs:\n{formatted}")
    return payloads


def aggregate(payloads: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    spec_by_method = {spec["method"]: spec for spec in checkpoint_specs()}
    detail_rows = []
    values: dict[str, dict[str, Any]] = {}

    for payload in payloads:
        task_key = payload["task_key"]
        task_name = payload["task_name"]
        for row in payload["rows"]:
            method = row["method"]
            if method == "pymatching":
                continue
            spec = spec_by_method[method]
            detail_rows.append(
                {
                    "task_key": task_key,
                    "task_name": task_name,
                    "basis": row.get("basis"),
                    "method": method,
                    "group": spec["group"],
                    "label": spec["label"],
                    "epoch": spec["epoch"],
                    "ler": row.get("ler"),
                    "latency_us_per_round": row.get("latency_us_per_round"),
                    "speedup_vs_pymatching": row.get("speedup_vs_pymatching"),
                    "logical_errors": row.get("logical_errors"),
                    "samples": row.get("samples"),
                    "checkpoint": row.get("checkpoint", str(spec["checkpoint"])),
                }
            )

        for item in payload["summary"]:
            method = item["method"]
            if method == "pymatching":
                continue
            spec = spec_by_method[method]
            entry = values.setdefault(
                method,
                {
                    "method": method,
                    "group": spec["group"],
                    "label": spec["label"],
                    "epoch": spec["epoch"],
                    "checkpoint": str(spec["checkpoint"]),
                    "task_lers": {},
                    "task_latencies": {},
                    "task_speedups": {},
                },
            )
            entry["task_lers"][task_key] = float(item["ler_avg"])
            entry["task_latencies"][task_key] = float(item["latency_us_per_round_avg"])
            entry["task_speedups"][task_key] = float(item["speedup_vs_pymatching_avg"])

    matrix_rows = []
    order = {spec["method"]: idx for idx, spec in enumerate(checkpoint_specs())}
    for method, entry in sorted(values.items(), key=lambda kv: order[kv[0]]):
        task_lers = [entry["task_lers"].get(task_key, float("nan")) for task_key, _, _ in TASKS]
        task_latencies = [entry["task_latencies"].get(task_key, float("nan")) for task_key, _, _ in TASKS]
        task_speedups = [entry["task_speedups"].get(task_key, float("nan")) for task_key, _, _ in TASKS]
        row = {
            "method": method,
            "group": entry["group"],
            "label": entry["label"],
            "epoch": entry["epoch"],
            "ler_avg_5task": mean(task_lers),
            "ler_worst_task": max(task_lers),
            "latency_us_per_round_avg": mean(task_latencies),
            "speedup_vs_pymatching_avg": mean(task_speedups),
            "checkpoint": entry["checkpoint"],
        }
        for task_key, _, _ in TASKS:
            row[f"ler_{task_key}"] = entry["task_lers"].get(task_key, float("nan"))
        matrix_rows.append(row)
    return detail_rows, matrix_rows


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


def group_rows(matrix_rows: list[dict[str, Any]], group: str) -> list[dict[str, Any]]:
    return [row for row in matrix_rows if row["group"] == group]


def task_matrix_table(rows: list[dict[str, Any]]) -> str:
    headers = ["epoch"] + [task_name for _, task_name, _ in TASKS] + ["5-task avg", "worst"]
    body = []
    for row in rows:
        body.append(
            [str(row["epoch"])]
            + [fmt(row.get(f"ler_{task_key}")) for task_key, _, _ in TASKS]
            + [fmt(row["ler_avg_5task"]), fmt(row["ler_worst_task"])]
        )
    return markdown_table(headers, body)


def forgetting_rows(matrix_rows: list[dict[str, Any]], group: str) -> list[list[str]]:
    rows = group_rows(matrix_rows, group)
    by_epoch = {int(row["epoch"]): row for row in rows}
    final = by_epoch.get(100)
    if final is None:
        return []
    out = []
    for task_idx, (task_key, task_name, _) in enumerate(TASKS):
        learned_epoch = (task_idx + 1) * 20
        candidates = [
            by_epoch[epoch].get(f"ler_{task_key}")
            for epoch in [20, 40, 60, 80, 100]
            if epoch in by_epoch and epoch >= learned_epoch
        ]
        candidates = [float(v) for v in candidates if v is not None and not math.isnan(float(v))]
        best_after_learned = min(candidates) if candidates else float("nan")
        final_ler = float(final.get(f"ler_{task_key}", float("nan")))
        out.append(
            [
                task_name,
                str(learned_epoch),
                fmt(best_after_learned),
                fmt(final_ler),
                fmt(final_ler - best_after_learned),
            ]
        )
    return out


def write_markdown(path: Path, args: argparse.Namespace, matrix_rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    summary_rows = [
        [
            row["label"],
            str(row["epoch"]),
            fmt(row["ler_avg_5task"]),
            fmt(row["ler_worst_task"]),
            fmt(row["latency_us_per_round_avg"], digits=3),
        ]
        for row in matrix_rows
    ]
    summary_table = markdown_table(
        ["模型", "epoch", "5-task LER-Avg", "worst task LER", "latency Avg"],
        summary_rows,
    )

    forgetting_table_seq = markdown_table(
        ["任务", "learned epoch", "best LER after learned", "final LER@100", "forgetting"],
        forgetting_rows(matrix_rows, "r9x_seq_noewc"),
    )
    forgetting_table_ewc = markdown_table(
        ["任务", "learned epoch", "best LER after learned", "final LER@100", "forgetting"],
        forgetting_rows(matrix_rows, "r9x_seq_ewc"),
    )

    text = f"""# R9-X Epoch Matrix Paired Inference

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
- 聚合脚本: `code/scripts/run_r9x_epoch_matrix_compare.py`

## 2. 总览

{summary_table}

## 3. R9-X domestic-only epoch 20

{task_matrix_table(group_rows(matrix_rows, "r9x_domestic"))}

## 4. R9-X sequential no-EWC

{task_matrix_table(group_rows(matrix_rows, "r9x_seq_noewc"))}

## 5. R9-X sequential + EWC

{task_matrix_table(group_rows(matrix_rows, "r9x_seq_ewc"))}

## 6. R9-X mixed-noise

{task_matrix_table(group_rows(matrix_rows, "r9x_mixed"))}

## 7. Forgetting 分析

### sequential no-EWC

{forgetting_table_seq}

### sequential + EWC

{forgetting_table_ewc}

## 8. 结果文件

- Matrix CSV: `{rel(args.matrix_csv).relative_to(REPO_ROOT)}`
- Detail CSV: `{rel(args.detail_csv).relative_to(REPO_ROOT)}`
- Markdown: `{path.relative_to(REPO_ROOT)}`
- Per-task JSON/CSV: `{rel(args.output_dir).relative_to(REPO_ROOT)}/r9x_epoch_matrix_<task>.json|csv`
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = rel(args.output_dir)
    matrix_csv = rel(args.matrix_csv)
    detail_csv = rel(args.detail_csv)
    summary_md = rel(args.summary_md)

    existing_inputs_or_raise()
    if not args.aggregate_only:
        run_inference(args, output_dir)
    if args.dry_run:
        return

    payloads = load_payloads(output_dir)
    detail_rows, matrix_rows = aggregate(payloads)
    write_csv(detail_csv, detail_rows)
    write_csv(matrix_csv, matrix_rows)
    write_markdown(summary_md, args, matrix_rows)
    print(f"[write] {detail_csv}")
    print(f"[write] {matrix_csv}")
    print(f"[write] {summary_md}")


if __name__ == "__main__":
    main()
