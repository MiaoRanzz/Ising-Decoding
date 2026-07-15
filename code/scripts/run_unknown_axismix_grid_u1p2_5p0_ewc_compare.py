#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run seq+EWC supplemental inference on fixed multiplier-grid axis-mix OOD configs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from scripts.config_paths import config_lookup_with_basename, config_path  # noqa: E402
from scripts.generate_unknown_axismix_grid_u1p2_5p0_configs import (  # noqa: E402
    DEFAULT_MANIFEST,
    DEFAULT_OUTPUT_DIR as DEFAULT_CONFIG_OUTPUT_DIR,
    DEFAULT_PREFIX,
    GRID_MULTIPLIERS,
    write_axismix_grid_configs,
)
from scripts.generate_unknown_seqfav_composite_configs import DEFAULT_BASE_CONFIG  # noqa: E402
from scripts.run_unknown_axismix_grid_u1p2_5p0_compare import (  # noqa: E402
    InferenceTask,
    PAIRED_SCRIPT,
    parse_gpu_list,
    resolve_python,
    run_tasks,
)
from scripts.run_unknown_t0_random_compare import rel  # noqa: E402

SEQ_EWC_METHOD = "stfusion_seq_ewc_e100"
SEQ_EWC_MODEL_ID = 111
SEQ_EWC_CHECKPOINT = "outputs/ising_domestic_fast_opt_stfusion_r9_x_seq_ewc/models/PreDecoderSTFusion_v2.0.100.pt"
DEFAULT_FULL_OUTPUT_DIR = "outputs/paired_inference_compare/unknown_axismix_grid_u1p2_5p0_seq_ewc_d9_full"
DEFAULT_DETAILS_CSV = "outputs/analysis/unknown_axismix_grid_u1p2_5p0_seq_ewc_d9_details.csv"
DEFAULT_SUMMARY_CSV = "outputs/analysis/unknown_axismix_grid_u1p2_5p0_seq_ewc_d9_summary.csv"


def mean(values: Sequence[float]) -> float:
    values = [float(value) for value in values]
    return sum(values) / len(values) if values else float("nan")


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
    return [
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
        "--model",
        f"{SEQ_EWC_METHOD}:{SEQ_EWC_MODEL_ID}:{rel(SEQ_EWC_CHECKPOINT)}",
    ]


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
        output_path = task_output_path(output_dir, env)
        gpu = str(gpus[index % len(gpus)])
        cmd = build_paired_command(
            python=python,
            paired_script=paired_script,
            config_name=str(env["config_name"]),
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
                config_name=str(env["config_name"]),
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
    missing: list[Path] = []
    if not PAIRED_SCRIPT.exists():
        missing.append(PAIRED_SCRIPT)
    checkpoint = rel(SEQ_EWC_CHECKPOINT)
    if not checkpoint.exists():
        missing.append(checkpoint)
    for env in manifest["environments"]:
        path = config_path(str(env["config_name"]))
        if not path.exists():
            missing.append(path)
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Required inputs are missing:\n{formatted}")


def _basis_values(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    values = sorted({str(row.get("basis")) for row in rows if row.get("basis") in {"X", "Z"}})
    return [basis for basis in ("X", "Z") if basis in values]


def _method_metric(rows: Sequence[Mapping[str, Any]], basis: str, metric: str) -> float:
    values = [
        float(row[metric])
        for row in rows
        if row.get("basis") == basis and row.get("method") == SEQ_EWC_METHOD and row.get(metric) is not None
    ]
    return values[0] if len(values) == 1 else float("nan")


def _detail_row(
    *,
    distance: int,
    run_phase: str,
    payload: Mapping[str, Any],
    env: Mapping[str, Any],
    basis: str,
    metrics: Mapping[str, float],
) -> dict[str, Any]:
    return {
        "distance": int(distance),
        "run_phase": run_phase,
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
        "seq_ewc_ler": float(metrics["seq_ewc_ler"]),
        "seq_ewc_latency_us_per_round": float(metrics["seq_ewc_latency"]),
        "seq_ewc_speedup_vs_pymatching": float(metrics["seq_ewc_speedup"]),
        "output_json": payload.get("path", ""),
    }


def aggregate_ewc_payloads(
    payloads: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    *,
    distance: int,
    run_phase: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    env_by_config = config_lookup_with_basename(manifest["environments"])
    details: list[dict[str, Any]] = []
    for payload in payloads:
        config_name = str(payload["config_name"])
        if config_name not in env_by_config:
            raise KeyError(f"payload config not found in manifest: {config_name}")
        env = env_by_config[config_name]
        rows = payload["rows"]
        basis_metrics: dict[str, dict[str, float]] = {}
        for basis in _basis_values(rows):
            metrics = {
                "seq_ewc_ler": _method_metric(rows, basis, "ler"),
                "seq_ewc_latency": _method_metric(rows, basis, "latency_us_per_round"),
                "seq_ewc_speedup": _method_metric(rows, basis, "speedup_vs_pymatching"),
            }
            basis_metrics[basis] = metrics
            details.append(_detail_row(distance=distance, run_phase=run_phase, payload=payload, env=env, basis=basis, metrics=metrics))
        if {"X", "Z"}.issubset(basis_metrics):
            combined = {key: mean([basis_metrics["X"][key], basis_metrics["Z"][key]]) for key in basis_metrics["X"]}
            details.append(_detail_row(distance=distance, run_phase=run_phase, payload=payload, env=env, basis="both", metrics=combined))
    details.sort(key=lambda row: (int(row["env_index"]), float(row["multiplier"]), ("both", "X", "Z").index(str(row["basis"]))))
    return details, aggregate_summary(details)


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
    summary_rows: list[dict[str, Any]] = []
    for (group_type, group_value, basis), rows in sorted(grouped.items()):
        summary_rows.append(
            {
                "distance": int(rows[0]["distance"]),
                "group_type": group_type,
                "group_value": group_value,
                "basis": basis,
                "config_count": len(rows),
                "env_count": len({str(row["env_key"]) for row in rows}),
                "seq_ewc_ler_mean": mean([float(row["seq_ewc_ler"]) for row in rows]),
                "seq_ewc_latency_us_per_round_mean": mean([float(row["seq_ewc_latency_us_per_round"]) for row in rows]),
                "seq_ewc_speedup_vs_pymatching_mean": mean([float(row["seq_ewc_speedup_vs_pymatching"]) for row in rows]),
            }
        )
    return summary_rows


def _write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path = rel(path)
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


def run_phase(args: argparse.Namespace, manifest: Mapping[str, Any]) -> Path:
    output_dir = rel(args.full_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    existing_inputs_or_raise(manifest)
    tasks = build_tasks(
        manifest,
        output_dir=output_dir,
        phase="full",
        python=resolve_python(args),
        paired_script=PAIRED_SCRIPT,
        gpus=parse_gpu_list(args.gpus),
        distance=args.distance,
        n_rounds=args.n_rounds,
        num_samples=args.full_num_samples,
        latency_num_samples=args.full_latency_num_samples,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        basis=args.basis,
    )
    gpus = parse_gpu_list(args.gpus)
    run_tasks(
        tasks,
        parallelism=min(int(args.parallelism), len(gpus)),
        dry_run=bool(args.dry_run),
        resume=bool(args.resume),
        fail_fast=bool(args.fail_fast),
    )
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--config-prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--distance", type=int, default=9)
    parser.add_argument("--n-rounds", type=int, default=9)
    parser.add_argument("--full-num-samples", type=int, default=262144)
    parser.add_argument("--full-latency-num-samples", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--basis", choices=("both", "X", "Z"), default="both")
    parser.add_argument("--parallelism", type=int, default=8)
    parser.add_argument("--gpus", default=None)
    parser.add_argument("--python", default=None)
    parser.add_argument("--full-output-dir", default=DEFAULT_FULL_OUTPUT_DIR)
    parser.add_argument("--details-csv", default=DEFAULT_DETAILS_CSV)
    parser.add_argument("--summary-csv", default=DEFAULT_SUMMARY_CSV)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = ensure_configs(args)
    output_dir = rel(args.full_output_dir)
    if not args.aggregate_only:
        run_phase(args, manifest)
    if args.dry_run:
        return
    payloads = load_payloads(output_dir, manifest, run_phase="full")
    details, summary = aggregate_ewc_payloads(payloads, manifest, distance=args.distance, run_phase="full")
    _write_csv(args.details_csv, details)
    _write_csv(args.summary_csv, summary)
    print(f"[write] {rel(args.details_csv)}")
    print(f"[write] {rel(args.summary_csv)}")


if __name__ == "__main__":
    main()
