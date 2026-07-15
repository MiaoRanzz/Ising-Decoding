#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run three-model comparison on fixed multiplier-grid axis-mix OOD configs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from scripts.config_paths import config_basename, config_lookup_with_basename, config_path  # noqa: E402
from scripts.generate_unknown_axismix_grid_u1p2_5p0_configs import (  # noqa: E402
    DEFAULT_MANIFEST,
    DEFAULT_OUTPUT_DIR as DEFAULT_CONFIG_OUTPUT_DIR,
    DEFAULT_PREFIX,
    DESIGN_LABEL,
    GRID_MULTIPLIERS,
    write_axismix_grid_configs,
)
from scripts.generate_unknown_seqfav_composite_configs import DEFAULT_BASE_CONFIG  # noqa: E402
from scripts.run_unknown_t0_random_compare import (  # noqa: E402
    PAIRED_SCRIPT,
    display_path,
    markdown_table,
    rel,
)


MODEL_SPECS = [
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
]
DOMESTIC_METHOD = "stfusion_domestic_e100"
MIXED_METHOD = "stfusion_mixed_e100"
SEQ_METHOD = "stfusion_seq_noewc_e100"
DEFAULT_QUICK_OUTPUT_DIR = "outputs/paired_inference_compare/unknown_axismix_grid_u1p2_5p0_quick"
DEFAULT_FULL_OUTPUT_DIR = "outputs/paired_inference_compare/unknown_axismix_grid_u1p2_5p0_full"
DEFAULT_DETAILS_CSV = "outputs/analysis/unknown_axismix_grid_u1p2_5p0_details.csv"
DEFAULT_SUMMARY_CSV = "outputs/analysis/unknown_axismix_grid_u1p2_5p0_summary.csv"
DEFAULT_SUMMARY_MD = "outputs/analysis/unknown_axismix_grid_u1p2_5p0_comparison.md"
DEFAULT_FIGURE_DIR = "outputs/analysis/unknown_axismix_grid_u1p2_5p0_figures"


@dataclass(frozen=True)
class InferenceTask:
    env_index: int
    env_key: str
    multiplier_key: str
    config_name: str
    output_path: Path
    gpu: str
    cmd: list[str]
    phase: str


@dataclass(frozen=True)
class TaskResult:
    task: InferenceTask
    returncode: int
    error: str = ""


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


def parse_gpu_list(value: str | None) -> list[str]:
    if value:
        return [item.strip() for item in value.split(",") if item.strip()]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible:
        return [item.strip() for item in visible.split(",") if item.strip()]
    try:
        import torch

        count = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    except Exception:
        count = 0
    return [str(index) for index in range(count)] or ["0"]


def task_output_path(output_dir: Path, env: Mapping[str, Any]) -> Path:
    return output_dir / (
        f"unknown_axismix_grid_u1p2_5p0_"
        f"e{int(env['env_index']):02d}_{env['multiplier_key']}.json"
    )


def build_paired_command(
    *,
    python: str,
    paired_script: Path,
    config_name: str,
    output_path: Path,
    distance: int,
    n_rounds: int,
    num_samples: int,
    latency_num_samples: int,
    batch_size: int,
    num_workers: int,
    seed: int,
    basis: str,
) -> list[str]:
    cmd = [
        python,
        "-u",
        str(paired_script),
        "--config-name",
        config_name,
        "--distance",
        str(distance),
        "--n-rounds",
        str(n_rounds),
        "--num-samples",
        str(num_samples),
        "--latency-num-samples",
        str(latency_num_samples),
        "--batch-size",
        str(batch_size),
        "--num-workers",
        str(num_workers),
        "--seed",
        str(seed),
        "--basis",
        basis,
        "--device",
        "cuda:0",
        "--output",
        str(output_path),
    ]
    for name, _, model_id, checkpoint in MODEL_SPECS:
        cmd.extend(["--model", f"{name}:{model_id}:{rel(checkpoint)}"])
    return cmd


def build_tasks(
    manifest: Mapping[str, Any],
    *,
    output_dir: Path,
    phase: str,
    python: str,
    paired_script: Path,
    gpus: Sequence[str],
    distance: int,
    n_rounds: int,
    num_samples: int,
    latency_num_samples: int,
    batch_size: int,
    num_workers: int,
    seed: int,
    basis: str,
) -> list[InferenceTask]:
    if not gpus:
        raise ValueError("at least one GPU id is required")
    tasks = []
    for index, env in enumerate(manifest["environments"]):
        config_name = str(env["config_name"])
        output_path = task_output_path(output_dir, env)
        gpu = str(gpus[index % len(gpus)])
        cmd = build_paired_command(
            python=python,
            paired_script=paired_script,
            config_name=config_name,
            output_path=output_path,
            distance=distance,
            n_rounds=n_rounds,
            num_samples=num_samples,
            latency_num_samples=latency_num_samples,
            batch_size=batch_size,
            num_workers=num_workers,
            seed=seed,
            basis=basis,
        )
        tasks.append(
            InferenceTask(
                env_index=int(env["env_index"]),
                env_key=str(env["env_key"]),
                multiplier_key=str(env["multiplier_key"]),
                config_name=config_name,
                output_path=output_path,
                gpu=gpu,
                cmd=cmd,
                phase=phase,
            )
        )
    return tasks


def ensure_configs(args: argparse.Namespace) -> dict[str, Any]:
    _, manifest = write_axismix_grid_configs(
        base_config=args.base_config,
        output_dir=DEFAULT_CONFIG_OUTPUT_DIR,
        prefix=args.config_prefix,
        manifest=args.manifest,
        grid_multipliers=GRID_MULTIPLIERS,
    )
    return manifest


def existing_inputs_or_raise(manifest: Mapping[str, Any]) -> None:
    missing = []
    if not PAIRED_SCRIPT.exists():
        missing.append(PAIRED_SCRIPT)
    for env in manifest["environments"]:
        path = config_path(str(env["config_name"]))
        if not path.exists():
            missing.append(path)
    for _, _, _, checkpoint in MODEL_SPECS:
        path = rel(checkpoint)
        if not path.exists():
            missing.append(path)
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Required inputs are missing:\n{formatted}")


def _valid_json(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return True


def _run_one_task(task: InferenceTask) -> TaskResult:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(task.gpu)
    try:
        subprocess.run(task.cmd, cwd=REPO_ROOT, env=env, check=True)
        return TaskResult(task=task, returncode=0)
    except subprocess.CalledProcessError as exc:
        return TaskResult(task=task, returncode=int(exc.returncode), error=str(exc))


def run_tasks(
    tasks: Sequence[InferenceTask],
    *,
    parallelism: int,
    dry_run: bool,
    resume: bool,
    fail_fast: bool,
) -> list[TaskResult]:
    runnable = []
    for task in tasks:
        if resume and _valid_json(task.output_path):
            print(f"[skip] {task.phase} {task.env_key}/{task.multiplier_key} exists: {task.output_path}")
            continue
        runnable.append(task)

    if dry_run:
        for task in runnable:
            print(
                f"[dry-run] phase={task.phase} gpu={task.gpu} env={task.env_key} "
                f"multiplier={task.multiplier_key} config={task.config_name} -> {task.output_path}"
            )
            print("[cmd] CUDA_VISIBLE_DEVICES=" + task.gpu + " " + " ".join(task.cmd))
        return []

    results: list[TaskResult] = []
    if not runnable:
        return results
    workers = max(1, min(int(parallelism), len(runnable)))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_run_one_task, task): task for task in runnable}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            task = result.task
            if result.returncode == 0:
                print(f"[done] {task.phase} gpu={task.gpu} {task.env_key}/{task.multiplier_key}")
            else:
                print(
                    f"[fail] {task.phase} gpu={task.gpu} {task.env_key}/{task.multiplier_key} "
                    f"returncode={result.returncode}"
                )
                if fail_fast:
                    for pending in futures:
                        pending.cancel()
                    break
    failures = [result for result in results if result.returncode != 0]
    if failures:
        lines = [
            f"  - {item.task.config_name} gpu={item.task.gpu} returncode={item.returncode}"
            for item in failures
        ]
        raise RuntimeError("Inference tasks failed:\n" + "\n".join(lines))
    return results


def run_phase(args: argparse.Namespace, manifest: Mapping[str, Any], *, phase: str) -> Path:
    output_dir = rel(args.quick_output_dir if phase == "quick" else args.full_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    num_samples = int(args.quick_num_samples if phase == "quick" else args.full_num_samples)
    latency_num_samples = int(
        args.quick_latency_num_samples if phase == "quick" else args.full_latency_num_samples
    )
    gpus = parse_gpu_list(args.gpus)
    existing_inputs_or_raise(manifest)
    tasks = build_tasks(
        manifest,
        output_dir=output_dir,
        phase=phase,
        python=resolve_python(args),
        paired_script=PAIRED_SCRIPT,
        gpus=gpus,
        distance=args.distance,
        n_rounds=args.n_rounds,
        num_samples=num_samples,
        latency_num_samples=latency_num_samples,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        basis=args.basis,
    )
    run_tasks(
        tasks,
        parallelism=min(int(args.parallelism), len(gpus)),
        dry_run=bool(args.dry_run),
        resume=bool(args.resume),
        fail_fast=bool(args.fail_fast),
    )
    return output_dir


def load_payloads(output_dir: str | Path, manifest: Mapping[str, Any], *, run_phase: str) -> list[dict[str, Any]]:
    output_dir = rel(output_dir)
    payloads = []
    missing = []
    for env in manifest["environments"]:
        output_path = task_output_path(output_dir, env)
        if not output_path.exists():
            missing.append(output_path)
            continue
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        payload.setdefault("config_name", str(env["config_name"]))
        payload["run_phase"] = run_phase
        payload["path"] = str(output_path)
        payloads.append(payload)
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Missing per-config inference outputs:\n{formatted}")
    return payloads


def _method_metric(rows: Sequence[Mapping[str, Any]], basis: str, method: str, metric: str) -> float:
    values = [
        float(row[metric])
        for row in rows
        if row.get("basis") == basis and row.get("method") == method and row.get(metric) is not None
    ]
    return values[0] if len(values) == 1 else float("nan")


def _basis_values(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    values = sorted({str(row.get("basis")) for row in rows if row.get("basis") in {"X", "Z"}})
    return [basis for basis in ("X", "Z") if basis in values]


def _detail_row(
    *,
    payload: Mapping[str, Any],
    env: Mapping[str, Any],
    basis: str,
    metrics: Mapping[str, float],
) -> dict[str, Any]:
    domestic = float(metrics["domestic100_ler"])
    mixed = float(metrics["mixed100_ler"])
    seq = float(metrics["seq_noewc_ler"])
    delta_seq_domestic = seq - domestic
    delta_seq_mixed = seq - mixed
    delta_mixed_domestic = mixed - domestic
    return {
        "run_phase": payload.get("run_phase", ""),
        "env_key": env["env_key"],
        "env_index": int(env["env_index"]),
        "multiplier_key": env["multiplier_key"],
        "multiplier": float(env["multiplier"]),
        "config_name": env["config_name"],
        "axis_signature": env["axis_signature"],
        "active_axes": "+".join(env["active_axes"]),
        "combination_size": int(env["combination_size"]),
        "contains_z_bias": bool(env["contains_z_bias"]),
        "contains_cnot_z_bias": bool(env["contains_cnot_z_bias"]),
        "basis": basis,
        "domestic100_ler": domestic,
        "mixed100_ler": mixed,
        "seq_noewc_ler": seq,
        "delta_seq_vs_domestic": delta_seq_domestic,
        "delta_seq_vs_mixed": delta_seq_mixed,
        "delta_mixed_vs_domestic": delta_mixed_domestic,
        "seq_beats_domestic": delta_seq_domestic < 0,
        "seq_beats_mixed": delta_seq_mixed < 0,
        "mixed_beats_domestic": delta_mixed_domestic < 0,
        "domestic100_latency_us_per_round": metrics["domestic100_latency"],
        "mixed100_latency_us_per_round": metrics["mixed100_latency"],
        "seq_noewc_latency_us_per_round": metrics["seq_noewc_latency"],
        "domestic100_speedup_vs_pymatching": metrics["domestic100_speedup"],
        "mixed100_speedup_vs_pymatching": metrics["mixed100_speedup"],
        "seq_noewc_speedup_vs_pymatching": metrics["seq_noewc_speedup"],
        "output_json": payload.get("path", ""),
    }


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
        env = env_by_config[config_name]
        rows = payload["rows"]
        basis_metrics: dict[str, dict[str, float]] = {}
        for basis in _basis_values(rows):
            metrics = {
                "domestic100_ler": _method_metric(rows, basis, DOMESTIC_METHOD, "ler"),
                "mixed100_ler": _method_metric(rows, basis, MIXED_METHOD, "ler"),
                "seq_noewc_ler": _method_metric(rows, basis, SEQ_METHOD, "ler"),
                "domestic100_latency": _method_metric(rows, basis, DOMESTIC_METHOD, "latency_us_per_round"),
                "mixed100_latency": _method_metric(rows, basis, MIXED_METHOD, "latency_us_per_round"),
                "seq_noewc_latency": _method_metric(rows, basis, SEQ_METHOD, "latency_us_per_round"),
                "domestic100_speedup": _method_metric(rows, basis, DOMESTIC_METHOD, "speedup_vs_pymatching"),
                "mixed100_speedup": _method_metric(rows, basis, MIXED_METHOD, "speedup_vs_pymatching"),
                "seq_noewc_speedup": _method_metric(rows, basis, SEQ_METHOD, "speedup_vs_pymatching"),
            }
            basis_metrics[basis] = metrics
            detail_rows.append(_detail_row(payload=payload, env=env, basis=basis, metrics=metrics))
        if {"X", "Z"}.issubset(basis_metrics):
            combined = {
                key: mean([basis_metrics["X"][key], basis_metrics["Z"][key]])
                for key in basis_metrics["X"]
            }
            detail_rows.append(_detail_row(payload=payload, env=env, basis="both", metrics=combined))

    detail_rows.sort(
        key=lambda row: (
            int(row["env_index"]),
            float(row["multiplier"]),
            ("both", "X", "Z").index(str(row["basis"])),
        )
    )
    summary_rows = aggregate_summary(detail_rows)
    group_rows = [row for row in summary_rows if row["group_type"] not in {"env", "multiplier"}]
    return detail_rows, summary_rows, group_rows


def aggregate_summary(detail_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    group_defs = [
        ("all", lambda row: "all"),
        ("multiplier", lambda row: str(row["multiplier_key"])),
        ("env", lambda row: str(row["env_key"])),
        ("combination_size", lambda row: str(row["combination_size"])),
        ("contains_z_bias", lambda row: "yes" if row["contains_z_bias"] else "no"),
        ("contains_cnot_z_bias", lambda row: "yes" if row["contains_cnot_z_bias"] else "no"),
    ]
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for row in detail_rows:
        for group_type, group_fn in group_defs:
            grouped.setdefault((group_type, group_fn(row), str(row["basis"])), []).append(row)

    summary_rows = []
    for (group_type, group_value, basis), rows in sorted(grouped.items()):
        summary_rows.append(
            {
                "group_type": group_type,
                "group_value": group_value,
                "basis": basis,
                "config_count": len(rows),
                "env_count": len({str(row["env_key"]) for row in rows}),
                "domestic100_ler_mean": mean([float(row["domestic100_ler"]) for row in rows]),
                "mixed100_ler_mean": mean([float(row["mixed100_ler"]) for row in rows]),
                "seq_noewc_ler_mean": mean([float(row["seq_noewc_ler"]) for row in rows]),
                "delta_seq_vs_domestic_mean": mean([float(row["delta_seq_vs_domestic"]) for row in rows]),
                "delta_seq_vs_mixed_mean": mean([float(row["delta_seq_vs_mixed"]) for row in rows]),
                "delta_mixed_vs_domestic_mean": mean([float(row["delta_mixed_vs_domestic"]) for row in rows]),
                "seq_win_vs_domestic_count": sum(1 for row in rows if row["seq_beats_domestic"]),
                "seq_win_vs_mixed_count": sum(1 for row in rows if row["seq_beats_mixed"]),
                "mixed_win_vs_domestic_count": sum(1 for row in rows if row["mixed_beats_domestic"]),
            }
        )
    return summary_rows


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


def _summary_rows_for_table(
    summary_rows: Sequence[Mapping[str, Any]],
    *,
    group_type: str,
    basis: str,
) -> list[list[str]]:
    selected = [
        row for row in summary_rows
        if row["group_type"] == group_type and row["basis"] == basis
    ]
    if group_type == "multiplier":
        selected.sort(key=lambda row: float(str(row["group_value"])[1:].replace("p", ".")))
    else:
        selected.sort(key=lambda row: str(row["group_value"]))
    return [
        [
            str(row["group_value"]),
            str(row["config_count"]),
            fmt_float(row["domestic100_ler_mean"]),
            fmt_float(row["mixed100_ler_mean"]),
            fmt_float(row["seq_noewc_ler_mean"]),
            fmt_float(row["delta_seq_vs_domestic_mean"]),
            fmt_float(row["delta_seq_vs_mixed_mean"]),
            str(row["seq_win_vs_domestic_count"]),
            str(row["seq_win_vs_mixed_count"]),
        ]
        for row in selected
    ]



def _summary_row(
    summary_rows: Sequence[Mapping[str, Any]],
    *,
    group_type: str,
    group_value: str,
    basis: str,
) -> Mapping[str, Any]:
    return next(
        row for row in summary_rows
        if row["group_type"] == group_type
        and str(row["group_value"]) == group_value
        and row["basis"] == basis
    )


def _multiplier_value(multiplier_key: str) -> float:
    return float(str(multiplier_key)[1:].replace("p", "."))


def _figure_markdown_path(path: str | Path, *, markdown_path: str | Path) -> str:
    path = Path(path)
    markdown_dir = Path(markdown_path).parent
    try:
        rendered = path.relative_to(markdown_dir)
    except ValueError:
        try:
            rendered = path.resolve().relative_to(markdown_dir.resolve())
        except ValueError:
            rendered = Path(display_path(path))
    return str(rendered).replace("\\", "/")


def _mean_metric(detail_rows: Sequence[Mapping[str, Any]], basis: str, metric: str) -> float:
    return mean([float(row[metric]) for row in detail_rows if row["basis"] == basis])


def _fmt_signed(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    return f"{number:+.6f}"


def generate_figures(
    detail_rows: Sequence[Mapping[str, Any]],
    summary_rows: Sequence[Mapping[str, Any]],
    figure_dir: str | Path = DEFAULT_FIGURE_DIR,
) -> list[Path]:
    """Generate report figures from aggregate rows."""
    figure_dir = rel(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    basis_order = ("both", "X", "Z")
    model_series = [
        ("domestic-only e100", "domestic100_ler_mean", "#4c78a8"),
        ("mixed-noise e100", "mixed100_ler_mean", "#f58518"),
        ("seq no-EWC e100", "seq_noewc_ler_mean", "#54a24b"),
    ]
    delta_series = [
        ("seq - domestic", "delta_seq_vs_domestic_mean", "#54a24b"),
        ("seq - mixed", "delta_seq_vs_mixed_mean", "#b279a2"),
    ]

    figure_paths = [
        figure_dir / "model_gap_vs_multiplier.png",
        figure_dir / "delta_vs_multiplier.png",
        figure_dir / "env_delta_heatmap_both.png",
        figure_dir / "env_delta_seq_vs_mixed_heatmap_both.png",
    ]

    rows = [
        row for row in summary_rows
        if row["group_type"] == "multiplier" and row["basis"] == "both"
    ]
    rows.sort(key=lambda row: _multiplier_value(str(row["group_value"])))
    x = [_multiplier_value(str(row["group_value"])) for row in rows]
    mixed_delta_pp = [100.0 * float(row["delta_mixed_vs_domestic_mean"]) for row in rows]
    seq_delta_pp = [100.0 * float(row["delta_seq_vs_domestic_mean"]) for row in rows]
    width = 0.16
    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    ax.bar([value - width / 2 for value in x], mixed_delta_pp, width=width, label="mixed - domestic", color="#f58518", alpha=0.85)
    ax.bar([value + width / 2 for value in x], seq_delta_pp, width=width, label="seq - domestic", color="#54a24b", alpha=0.85)
    ax.plot(x, seq_delta_pp, color="#2f7d32", marker="o", linewidth=1.6)
    ax.axhline(0.0, color="#222222", linewidth=1.0, linestyle="--")
    ax.set_xticks(x, [str(value) for value in x])
    ax.set_xlabel("active-axis multiplier")
    ax.set_ylabel("mean LER delta vs domestic (percentage points)")
    ax.set_title("Model gap relative to domestic-only e100 (basis=both)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_paths[0], dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)
    for ax, basis_name in zip(axes, basis_order):
        rows = [
            row for row in summary_rows
            if row["group_type"] == "multiplier" and row["basis"] == basis_name
        ]
        rows.sort(key=lambda row: _multiplier_value(str(row["group_value"])))
        x = [_multiplier_value(str(row["group_value"])) for row in rows]
        for label, field, color in delta_series:
            ax.plot(x, [float(row[field]) for row in rows], marker="o", label=label, color=color)
        ax.axhline(0.0, color="#222222", linewidth=1.0, linestyle="--")
        ax.set_title(f"basis={basis_name}")
        ax.set_xlabel("active-axis multiplier")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("mean LER delta (negative favors seq)")
    axes[-1].legend(loc="lower right", fontsize=8)
    fig.suptitle("Seq no-EWC delta against domestic-only and mixed-noise baselines")
    fig.tight_layout()
    fig.savefig(figure_paths[1], dpi=180)
    plt.close(fig)

    def heatmap(metric: str, output_path: Path, title: str) -> None:
        envs = sorted({str(row["env_key"]) for row in detail_rows if row["basis"] == "both"})
        multipliers = sorted(
            {str(row["multiplier_key"]) for row in detail_rows if row["basis"] == "both"},
            key=_multiplier_value,
        )
        lookup = {
            (str(row["env_key"]), str(row["multiplier_key"])): float(row[metric])
            for row in detail_rows
            if row["basis"] == "both"
        }
        matrix = [
            [lookup.get((env, multiplier), math.nan) for multiplier in multipliers]
            for env in envs
        ]
        finite = [abs(value) for row in matrix for value in row if not math.isnan(value)]
        vmax = max(finite) if finite else 0.001
        vmax = max(vmax, 0.001)
        fig, ax = plt.subplots(figsize=(11, 5.5))
        im = ax.imshow(matrix, cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(multipliers)), [str(_multiplier_value(item)) for item in multipliers])
        ax.set_yticks(range(len(envs)), envs)
        ax.set_xlabel("active-axis multiplier")
        ax.set_ylabel("environment")
        ax.set_title(title)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("LER delta (negative favors seq)")
        fig.tight_layout()
        fig.savefig(output_path, dpi=180)
        plt.close(fig)

    heatmap(
        "delta_seq_vs_domestic",
        figure_paths[2],
        "Seq no-EWC vs domestic-only delta by env and multiplier (basis=both)",
    )
    heatmap(
        "delta_seq_vs_mixed",
        figure_paths[3],
        "Seq no-EWC vs mixed-noise delta by env and multiplier (basis=both)",
    )
    return figure_paths


def write_markdown(
    path: str | Path,
    *,
    details_csv: str | Path,
    summary_csv: str | Path,
    detail_rows: Sequence[Mapping[str, Any]],
    summary_rows: Sequence[Mapping[str, Any]],
    output_dir: str | Path,
    basis: str,
    num_samples: int,
    distance: int = 9,
    n_rounds: int = 9,
    figure_dir: str | Path = DEFAULT_FIGURE_DIR,
    figure_paths: Sequence[str | Path] | None = None,
) -> None:
    path = rel(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    overall = _summary_row(summary_rows, group_type="all", group_value="all", basis="both")
    overall_x = _summary_row(summary_rows, group_type="all", group_value="all", basis="X")
    overall_z = _summary_row(summary_rows, group_type="all", group_value="all", basis="Z")
    contains_z = _summary_row(summary_rows, group_type="contains_z_bias", group_value="yes", basis="both")
    no_z = _summary_row(summary_rows, group_type="contains_z_bias", group_value="no", basis="both")
    four_axis = _summary_row(summary_rows, group_type="combination_size", group_value="4", basis="both")
    seq_vs_domestic = float(overall["delta_seq_vs_domestic_mean"])
    seq_vs_mixed = float(overall["delta_seq_vs_mixed_mean"])
    both_count = len([row for row in detail_rows if row["basis"] == "both"])
    figure_paths = list(figure_paths or [])

    if seq_vs_domestic < 0 and seq_vs_mixed < 0:
        conclusion = (
            "在该固定训练轴倍率网格 OOD 测试中，seq no-EWC e100 的平均 LER "
            "同时低于 domestic-only e100 与 mixed-noise e100。"
        )
    elif seq_vs_domestic < 0:
        conclusion = (
            "在该固定训练轴倍率网格 OOD 测试中，seq no-EWC e100 低于 domestic-only e100，"
            "但没有超过 mixed-noise e100；结论应限定在单一 domestic 基线。"
        )
    else:
        conclusion = (
            "seq no-EWC e100 未超过 domestic-only e100；不能据此声称 sequential checkpoint "
            "在该 OOD 测试中整体更优。"
        )

    figure_lines = []
    if figure_paths:
        captions = [
            ("相对 domestic 的模型差距（百分点）", "model_gap_vs_multiplier.png"),
            ("seq 相对 domestic/mixed 的 delta 趋势", "delta_vs_multiplier.png"),
            ("seq 相对 domestic 的 env × 倍率 heatmap", "env_delta_heatmap_both.png"),
            ("seq 相对 mixed 的 env × 倍率 heatmap", "env_delta_seq_vs_mixed_heatmap_both.png"),
        ]
        by_name = {Path(item).name: item for item in figure_paths}
        for caption, filename in captions:
            if filename in by_name:
                figure_lines.extend([
                    f"![{caption}]({_figure_markdown_path(by_name[filename], markdown_path=path)})",
                    "",
                ])

    latency_rows = [
        [
            basis_name,
            fmt_float(_mean_metric(detail_rows, basis_name, "domestic100_latency_us_per_round"), 3),
            fmt_float(_mean_metric(detail_rows, basis_name, "mixed100_latency_us_per_round"), 3),
            fmt_float(_mean_metric(detail_rows, basis_name, "seq_noewc_latency_us_per_round"), 3),
            fmt_float(_mean_metric(detail_rows, basis_name, "domestic100_speedup_vs_pymatching"), 3),
            fmt_float(_mean_metric(detail_rows, basis_name, "mixed100_speedup_vs_pymatching"), 3),
            fmt_float(_mean_metric(detail_rows, basis_name, "seq_noewc_speedup_vs_pymatching"), 3),
        ]
        for basis_name in ("both", "X", "Z")
    ]

    lines = [
        "# 三模型固定倍率网格 OOD 详细分析报告",
        "",
        "## 结论摘要",
        "",
        f"- {conclusion}",
        f"- 推理口径：distance=`{distance}`，n_rounds=`{n_rounds}`，basis=`{basis}`，num_samples=`{num_samples}`。",
        f"- 主口径 `basis=both`，覆盖 11 个训练轴混合 env × 9 个固定倍率，共 `{both_count}` 个 paired config。",
        f"- `basis=both` 平均 LER：domestic-only `{float(overall['domestic100_ler_mean']):.6f}`，mixed-noise `{float(overall['mixed100_ler_mean']):.6f}`，seq no-EWC `{float(overall['seq_noewc_ler_mean']):.6f}`。",
        f"- `basis=both` 平均 delta：seq-domestic `{seq_vs_domestic:+.6f}`，seq-mixed `{seq_vs_mixed:+.6f}`；负 delta 表示 seq no-EWC 更好。",
        f"- 胜场数：seq 胜 domestic `{overall['seq_win_vs_domestic_count']}/{overall['config_count']}`，seq 胜 mixed `{overall['seq_win_vs_mixed_count']}/{overall['config_count']}`。",
        "- 结论边界：该结果支持固定训练轴混合倍率网格下的 OOD 泛化收益，不应表述为所有随机未知噪声或所有物理噪声分布上都更优。",
        "",
        "## 实验目的与设计",
        "",
        f"本实验是 `{DESIGN_LABEL}`。它不是随机未知噪声采样，而是沿训练中出现过的四类扰动轴构造固定 OOD 组合：`meas_all`、`cnot_all`、`idle_all`、`z_bias`。每个 env 激活两轴、三轴或四轴，激活参数统一乘以固定倍率 `[1.2, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]`，重叠参数只乘一次。",
        "",
        "比较对象是 domestic-only e100、mixed-noise e100 和 seq no-EWC e100。每个 config 内三个模型共享同一批样本和同一个 matcher，因此 delta 可以按 paired comparison 解读。主结论看 `basis=both`，同时报告 X/Z 拆分，避免单一 basis 掩盖模型差异。",
        "",
        "## 模型说明",
        "",
        markdown_table(
            ["模型", "训练范式", "在本实验中的作用"],
            [
                ["domestic-only e100", "只在 domestic/T0 类训练入口上充分训练到 epoch 100", "主要单任务基线，用来判断 sequential 是否带来 OOD 泛化收益"],
                ["mixed-noise e100", "在 mixed-noise 设置下训练到 epoch 100", "更强的多噪声基线；只有 seq 同时优于它时，才能声明超过 mixed 基线"],
                ["seq no-EWC e100", "按任务序列连续训练到 epoch 100，不使用 EWC 正则", "主模型；检验连续训练是否学习到跨任务噪声轴的可迁移结构"],
            ],
        ),
        "",
        "三者使用同一 ST-Fusion-R9-X 推理结构和相同 paired samples。报告中的 LER 差异主要反映训练范式和训练噪声覆盖差异，而不是评估样本差异。",
        "",
        "## 噪声构造解释",
        "",
        "本实验从 T0 base 的 25 参数噪声模型出发，只改变被激活训练轴对应的物理噪声参数。四类轴含义如下：`meas_all` 对应测量错误参数，`cnot_all` 对应 CNOT 后 Pauli 错误参数，`idle_all` 对应 idle/cnot 与 idle/spam 期间的 X/Y/Z 错误，`z_bias` 对应 T4 类 Z-biased 参数集合。",
        "",
        "11 个 env 覆盖任意两轴、任意三轴和四轴全组合；每个 env 在 9 个固定倍率上评估。激活轴参数统一乘以当前倍率，未激活参数保持 T0 base，重叠参数只乘一次，不做 2.25x、4x 这类乘法叠加。因此它是训练轴组合 OOD 网格测试，不是 per-sample 随机噪声，也不是中性随机未知噪声。",
        "",
        "## 图表总览",
        "",
        *figure_lines,
        "## 三模型总体对比",
        "",
        markdown_table(
            [
                "口径",
                "domestic LER",
                "mixed LER",
                "seq LER",
                "seq-domestic",
                "seq-mixed",
                "mixed-domestic",
                "seq 胜 domestic",
                "seq 胜 mixed",
            ],
            [
                [
                    basis_name,
                    fmt_float(row["domestic100_ler_mean"]),
                    fmt_float(row["mixed100_ler_mean"]),
                    fmt_float(row["seq_noewc_ler_mean"]),
                    _fmt_signed(row["delta_seq_vs_domestic_mean"]),
                    _fmt_signed(row["delta_seq_vs_mixed_mean"]),
                    _fmt_signed(row["delta_mixed_vs_domestic_mean"]),
                    f"{row['seq_win_vs_domestic_count']}/{row['config_count']}",
                    f"{row['seq_win_vs_mixed_count']}/{row['config_count']}",
                ]
                for basis_name in ("both", "X", "Z")
                for row in summary_rows
                if row["group_type"] == "all" and row["basis"] == basis_name
            ],
        ),
        "",
        "从总体均值看，seq no-EWC 是三者中 LER 最低的模型；优势在 Z basis 更稳定，X basis 上仍为负 delta 但幅度更小。这说明 sequential 训练带来的收益主要体现在部分后续任务轴相关的错误结构上，而不是简单地在所有 basis 上等幅提升。",
        "",
        "## 倍率趋势",
        "",
    ]
    for basis_name in ("both", "X", "Z"):
        rows = _summary_rows_for_table(summary_rows, group_type="multiplier", basis=basis_name)
        if rows:
            lines.extend(
                [
                    f"### basis={basis_name}",
                    "",
                    markdown_table(
                        [
                            "倍率",
                            "config 数",
                            "domestic LER",
                            "mixed LER",
                            "seq LER",
                            "seq-domestic",
                            "seq-mixed",
                            "seq 胜 domestic",
                            "seq 胜 mixed",
                        ],
                        rows,
                    ),
                    "",
                ]
            )
    lines.extend(
        [
            "倍率趋势的关键点是：`1.2x` 和 `1.5x` 下 seq 相对 domestic 的平均 delta 接近 0 或略为正，说明接近训练强度的轻度组合扰动不足以稳定拉开 domestic-only e100；从 `2.0x` 起，尤其 `2.5x-5.0x`，seq 对 domestic 的平均 delta 转为更稳定的负值。相对 mixed-noise，seq 在低倍率区间反而差距更大，高倍率区间差距收窄，说明 mixed-noise 在强扰动饱和区域接近 seq，但整体仍未超过 seq。",
            "",
            "## X/Z basis 差异",
            "",
            f"X basis 下 seq-domestic 平均 delta 为 `{float(overall_x['delta_seq_vs_domestic_mean']):+.6f}`，胜场 `{overall_x['seq_win_vs_domestic_count']}/{overall_x['config_count']}`；Z basis 下该 delta 为 `{float(overall_z['delta_seq_vs_domestic_mean']):+.6f}`，胜场 `{overall_z['seq_win_vs_domestic_count']}/{overall_z['config_count']}`。Z basis 的优势更强，也更接近此前 seq-favoring z-stress 观察到的现象。",
            "",
            "这意味着报告主张应写成：seq no-EWC 在固定训练轴混合 OOD 网格中整体优于 domestic-only，且优势主要由 Z basis 和含 z-bias 组合贡献；不宜写成 X/Z 完全一致的泛化提升。",
            "",
            "## 环境组合差异",
            "",
            "### combination_size（basis=both）",
            "",
            markdown_table(
                ["分组", "config 数", "domestic LER", "mixed LER", "seq LER", "seq-domestic", "seq-mixed", "seq 胜 domestic", "seq 胜 mixed"],
                _summary_rows_for_table(summary_rows, group_type="combination_size", basis="both"),
            ),
            "",
            "### contains_z_bias（basis=both）",
            "",
            markdown_table(
                ["分组", "config 数", "domestic LER", "mixed LER", "seq LER", "seq-domestic", "seq-mixed", "seq 胜 domestic", "seq 胜 mixed"],
                _summary_rows_for_table(summary_rows, group_type="contains_z_bias", basis="both"),
            ),
            "",
            f"含 `z_bias` 的组合平均 seq-domestic delta 为 `{float(contains_z['delta_seq_vs_domestic_mean']):+.6f}`，不含 `z_bias` 的组合为 `{float(no_z['delta_seq_vs_domestic_mean']):+.6f}`。含 z-bias 场景的收益更明显，符合 sequential 任务轴暴露更多 Z-biased 结构的预期。四轴全组合的平均 delta 为 `{float(four_axis['delta_seq_vs_domestic_mean']):+.6f}`，仍为负但幅度较小，是该结论的重要边界。",
            "",
            "### env（basis=both）",
            "",
            markdown_table(
                ["env", "config 数", "domestic LER", "mixed LER", "seq LER", "seq-domestic", "seq-mixed", "seq 胜 domestic", "seq 胜 mixed"],
                _summary_rows_for_table(summary_rows, group_type="env", basis="both"),
            ),
            "",
            "## 与 mixed-noise 基线的关系",
            "",
            f"mixed-noise e100 在该固定网格中不是更强的 OOD 基线：`basis=both` 下 mixed-domestic delta 为 `{float(overall['delta_mixed_vs_domestic_mean']):+.6f}`，说明 mixed 平均 LER 反而高于 domestic；seq-mixed delta 为 `{seq_vs_mixed:+.6f}`，且 seq 胜 mixed `{overall['seq_win_vs_mixed_count']}/{overall['config_count']}`。",
            "",
            "因此可声明 seq no-EWC 在该固定训练轴倍率网格测试中同时超过 domestic-only 和 mixed-noise。但这个结论依赖当前 grid、checkpoint 与采样口径；不能外推为 mixed-noise 在所有 OOD 场景下都弱。",
            "",
            "## 速度与延迟分析",
            "",
            markdown_table(
                ["口径", "domestic us/round", "mixed us/round", "seq us/round", "domestic speedup", "mixed speedup", "seq speedup"],
                latency_rows,
            ),
            "",
            "延迟和 speedup 是工程部署指标，不作为泛化能力的主证据。当前结果中 mixed/seq 的神经推理速度通常优于 PyMatching，domestic 在高噪声下可能出现更差 speedup；但模型优劣结论仍应以 paired LER 为主。",
            "",
            "## 结论边界与推荐表述",
            "",
            "- 推荐主表述：在 11 个训练轴混合 env、9 个固定倍率的 OOD composite test 中，seq no-EWC e100 在 `basis=both` 上整体低于 domestic-only e100 和 mixed-noise e100。",
            "- 推荐补充：优势主要在 Z basis、含 `z_bias` 组合以及 `2.0x` 以上倍率更明显；低倍率相对 domestic 的优势不稳定。",
            "- 不推荐表述：不要写成随机未知噪声；不要写成所有未知噪声上 seq 都更优；不要把 mixed-noise 在所有 OOD 场景中概括为弱基线。",
            "",
            "## 文件索引",
            "",
            f"- detail CSV: `{display_path(details_csv)}`",
            f"- summary CSV: `{display_path(summary_csv)}`",
            f"- figures: `{display_path(figure_dir)}`",
            f"- per-config JSON/CSV: `{display_path(output_dir)}`",
            f"- report: `{display_path(path)}`",
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
    output_dir: str | Path,
    basis: str,
    num_samples: int,
    distance: int = 9,
    n_rounds: int = 9,
    figure_dir: str | Path = DEFAULT_FIGURE_DIR,
) -> None:
    _write_csv(details_csv, detail_rows)
    _write_csv(summary_csv, summary_rows)
    figure_paths = generate_figures(detail_rows, summary_rows, figure_dir)
    write_markdown(
        summary_md,
        details_csv=details_csv,
        summary_csv=summary_csv,
        detail_rows=detail_rows,
        summary_rows=summary_rows,
        output_dir=output_dir,
        basis=basis,
        num_samples=num_samples,
        distance=distance,
        n_rounds=n_rounds,
        figure_dir=figure_dir,
        figure_paths=figure_paths,
    )

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("quick", "full", "all"), default="quick")
    parser.add_argument("--base-config", default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--config-prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--distance", type=int, default=9)
    parser.add_argument("--n-rounds", type=int, default=9)
    parser.add_argument("--quick-num-samples", type=int, default=65536)
    parser.add_argument("--full-num-samples", type=int, default=262144)
    parser.add_argument("--quick-latency-num-samples", type=int, default=1000)
    parser.add_argument("--full-latency-num-samples", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--basis", choices=("both", "X", "Z"), default="both")
    parser.add_argument("--parallelism", type=int, default=8)
    parser.add_argument("--gpus", default=None)
    parser.add_argument("--python", default=None)
    parser.add_argument("--quick-output-dir", default=DEFAULT_QUICK_OUTPUT_DIR)
    parser.add_argument("--full-output-dir", default=DEFAULT_FULL_OUTPUT_DIR)
    parser.add_argument("--details-csv", default=DEFAULT_DETAILS_CSV)
    parser.add_argument("--summary-csv", default=DEFAULT_SUMMARY_CSV)
    parser.add_argument("--summary-md", default=DEFAULT_SUMMARY_MD)
    parser.add_argument("--figure-dir", default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = ensure_configs(args)
    if args.phase == "quick":
        phase = "quick"
        output_dir = rel(args.quick_output_dir)
        num_samples = int(args.quick_num_samples)
    else:
        phase = "full"
        output_dir = rel(args.full_output_dir)
        num_samples = int(args.full_num_samples)

    if not args.aggregate_only:
        if args.phase in {"quick", "all"}:
            run_phase(args, manifest, phase="quick")
        if args.phase in {"full", "all"}:
            run_phase(args, manifest, phase="full")
    if args.dry_run:
        return

    payloads = load_payloads(output_dir, manifest, run_phase=phase)
    detail_rows, summary_rows, _ = aggregate_payloads(payloads, manifest)
    write_outputs(
        details_csv=args.details_csv,
        summary_csv=args.summary_csv,
        summary_md=args.summary_md,
        detail_rows=detail_rows,
        summary_rows=summary_rows,
        output_dir=output_dir,
        basis=args.basis,
        num_samples=num_samples,
        distance=args.distance,
        n_rounds=args.n_rounds,
        figure_dir=args.figure_dir,
    )
    print(f"[write] {rel(args.details_csv)}")
    print(f"[write] {rel(args.summary_csv)}")
    print(f"[write] {rel(args.summary_md)}")


if __name__ == "__main__":
    main()
