#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run and aggregate seq-favoring composite OOD paired inference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from scripts.config_paths import config_path  # noqa: E402
from scripts.run_unknown_t0_random_compare import (  # noqa: E402
    BASELINE_METHODS,
    MODEL_SPECS,
    PAIRED_SCRIPT,
    aggregate_rows,
    build_command,
    display_path,
    fmt_float,
    fmt_latency,
    markdown_table,
    rel,
    write_csv,
)


DEFAULT_NUM_ENVS = 5
DEFAULT_PREFIX = "config_unknown_seqfav_composite_v1"
DEFAULT_CONFIG_DIR = "experiments/unknown_seqfav_composite_v1"
DESIGN_LABEL = "seq-favoring OOD stress test"


def default_config_names(*, num_envs: int = DEFAULT_NUM_ENVS, prefix: str = DEFAULT_PREFIX) -> list[str]:
    return [f"{DEFAULT_CONFIG_DIR}/{prefix}_e{i:02d}" for i in range(int(num_envs))]


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
    parser.add_argument("--num-envs", type=int, default=DEFAULT_NUM_ENVS)
    parser.add_argument("--config-prefix", default=DEFAULT_PREFIX)
    parser.add_argument(
        "--config-name",
        action="append",
        default=None,
        help="Repeat to override the default unknown environment config names.",
    )
    parser.add_argument("--output-dir", default="outputs/paired_inference_compare/unknown_seqfav_composite_v1")
    parser.add_argument(
        "--summary-csv",
        default="outputs/analysis/unknown_seqfav_composite_v1_summary.csv",
    )
    parser.add_argument(
        "--detail-csv",
        default="outputs/analysis/unknown_seqfav_composite_v1_details.csv",
    )
    parser.add_argument(
        "--summary-md",
        default="outputs/analysis/unknown_seqfav_composite_v1_comparison.md",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    return parser.parse_args()


def config_names_from_args(args: argparse.Namespace) -> list[str]:
    if args.config_name:
        return list(args.config_name)
    return default_config_names(num_envs=args.num_envs, prefix=args.config_prefix)


def task_output_path(output_dir: Path, env_index: int) -> Path:
    return output_dir / f"unknown_seqfav_composite_v1_e{env_index:02d}.json"


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


def run_inference(args: argparse.Namespace, config_names: list[str], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for env_index, config_name in enumerate(config_names):
        output_path = task_output_path(output_dir, env_index)
        cmd = build_command(args, config_name, output_path)
        print(f"[env] e{env_index:02d} config={config_name} -> {output_path}")
        print("[cmd] " + " ".join(cmd))
        if not args.dry_run:
            import subprocess

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


def seqfav_success_status(summary_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_method = {row["method"]: row for row in summary_rows}
    noewc = by_method.get("stfusion_seq_noewc_e100")
    domestic100 = by_method.get("stfusion_domestic_e100")
    if not noewc or not domestic100:
        return {
            "avg_delta_vs_domestic100": float("nan"),
            "avg_beats_domestic100": False,
            "win_count_vs_domestic100": 0,
            "env_count": 0,
            "required_win_count": 3,
            "worst_beats_domestic100": False,
            "passed": False,
        }

    env_keys = sorted(key[4:] for key in noewc if key.startswith("ler_e"))
    win_count = 0
    for env_key in env_keys:
        if float(noewc[f"ler_{env_key}"]) < float(domestic100[f"ler_{env_key}"]):
            win_count += 1

    avg_delta = float(noewc["ler_avg_unknown_envs"]) - float(domestic100["ler_avg_unknown_envs"])
    env_count = len(env_keys)
    required_win_count = min(3, env_count) if env_count else 3
    worst_beats = float(noewc["ler_worst_env"]) <= float(domestic100["ler_worst_env"])
    avg_beats = avg_delta < 0
    return {
        "avg_delta_vs_domestic100": avg_delta,
        "avg_beats_domestic100": avg_beats,
        "win_count_vs_domestic100": win_count,
        "env_count": env_count,
        "required_win_count": required_win_count,
        "worst_beats_domestic100": worst_beats,
        "passed": avg_beats and win_count >= required_win_count and worst_beats,
    }


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


def _status_text(value: bool) -> str:
    return "PASS" if value else "FAIL"


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
    status = seqfav_success_status(summary_rows)

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
    gate_table = markdown_table(
        ["条件", "结果"],
        [
            [
                "seq no-EWC e100 average LER < domestic-only epoch 100",
                f"{_status_text(status['avg_beats_domestic100'])} "
                f"(delta={fmt_float(status['avg_delta_vs_domestic100'])})",
            ],
            [
                "seq no-EWC e100 wins enough environments",
                f"{_status_text(status['win_count_vs_domestic100'] >= status['required_win_count'])} "
                f"({status['win_count_vs_domestic100']}/{status['env_count']}, "
                f"required >= {status['required_win_count']})",
            ],
            [
                "seq no-EWC e100 worst-env LER <= domestic-only epoch 100",
                _status_text(status["worst_beats_domestic100"]),
            ],
            ["overall seq-favoring gate", _status_text(status["passed"])],
        ],
    )
    config_lines = "\n".join(f"- `{name}`" for name in config_names)

    text = f"""# Seq-Favoring Composite OOD Stress Test

## 1. 推理口径

- design: `{DESIGN_LABEL}`
- distance: `{distance}`
- n_rounds: `{n_rounds}`
- num_samples: `{num_samples}`
- latency_num_samples: `{latency_num_samples}`
- seed: `{seed}`
- basis: `{basis}`
- unknown environments: `{len(config_names)}`
- 评估脚本: `code/scripts/paired_inference_compare.py`
- 聚合脚本: `code/scripts/run_unknown_seqfav_composite_compare.py`

注意：本实验是有意偏向 sequential 任务流形的 OOD stress test，不应写成中性随机未知噪声。

## 2. 未知环境 configs

{config_lines}

## 3. 汇总结果

{summary_table}

## 4. 分环境 LER-Avg

{env_table}

## 5. Sequential 对比

负值表示 sequential 方法在 unknown-env LER-Avg 上更低。主结论只看 `seq no-EWC e100 vs domestic-only epoch 100`。

{comparison_table}

## 6. Success Gate

{gate_table}

## 7. 解释边界

- 若 gate PASS，可表述为：在 seq-favoring composite OOD stress test 中，seq no-EWC e100 优于 domestic-only epoch 100。
- 若 gate FAIL，不应强写 seq 优于 domestic-only epoch 100；应表述为当前 stress set 未证明该结论。
- EWC 不应被写成最优，除非 sequential + EWC 的 LER 明确低于 sequential no-EWC。

## 8. 结果文件

- Per-env JSON/CSV: `{display_path(output_dir)}/unknown_seqfav_composite_v1_e<idx>.json|csv`
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
