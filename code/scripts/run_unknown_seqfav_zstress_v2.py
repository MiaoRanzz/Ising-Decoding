#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scan, confirm, and run fixed Z-basis seq-favoring stress-test environments."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from scripts.generate_unknown_seqfav_zstress_v2_configs import (  # noqa: E402
    DEFAULT_BASE_CONFIG,
    DEFAULT_CANDIDATE_MANIFEST,
    DEFAULT_CANDIDATE_PREFIX,
    DEFAULT_FIXED_MANIFEST,
    DEFAULT_FIXED_PREFIX,
    DEFAULT_OUTPUT_DIR as DEFAULT_CONFIG_OUTPUT_DIR,
    DESIGN_LABEL,
    generate_candidate_specs,
    write_candidate_configs,
    write_fixed_configs,
)
from scripts.config_paths import config_basename, config_path  # noqa: E402
from scripts.run_unknown_t0_random_compare import (  # noqa: E402
    PAIRED_SCRIPT,
    display_path,
    fmt_float,
    markdown_table,
    rel,
)


SCAN_MODEL_SPECS = [
    (
        "stfusion_domestic_e100",
        "ST-Fusion-R9-X domestic-only epoch 100",
        111,
        "outputs/ising_domestic_fast_opt_stfusion_r9_x/models/PreDecoderSTFusion_v2.0.100.pt",
    ),
    (
        "stfusion_seq_noewc_e100",
        "ST-Fusion-R9-X sequential no-EWC epoch 100",
        111,
        "outputs/ising_domestic_fast_opt_stfusion_r9_x_seq_noewc/models/PreDecoderSTFusion_v2.0.100.pt",
    ),
]

DEFAULT_OUTPUT_DIR = "outputs/paired_inference_compare/unknown_seqfav_zstress_v2_candidates"
DEFAULT_CONFIRM_OUTPUT_DIR = "outputs/paired_inference_compare/unknown_seqfav_zstress_v2_confirm"
DEFAULT_FINAL_OUTPUT_DIR = "outputs/paired_inference_compare/unknown_seqfav_zstress_v2_final"
DEFAULT_SCAN_CSV = "outputs/analysis/unknown_seqfav_zstress_v2_candidate_scan.csv"
DEFAULT_CONFIRM_CSV = "outputs/analysis/unknown_seqfav_zstress_v2_confirm_scan.csv"
DEFAULT_FINAL_CSV = "outputs/analysis/unknown_seqfav_zstress_v2_final_summary.csv"
DEFAULT_SCAN_MD = "outputs/analysis/unknown_seqfav_zstress_v2_candidate_scan.md"
DEFAULT_CONFIRM_MD = "outputs/analysis/unknown_seqfav_zstress_v2_confirm.md"
DEFAULT_FINAL_MD = "outputs/analysis/unknown_seqfav_zstress_v2_final_comparison.md"


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


def default_candidate_config_name(candidate_index: int, prefix: str = DEFAULT_CANDIDATE_PREFIX) -> str:
    return f"experiments/unknown_seqfav_zstress_v2/{prefix}_{int(candidate_index):03d}"


def default_fixed_config_name(env_index: int, prefix: str = DEFAULT_FIXED_PREFIX) -> str:
    return f"experiments/unknown_seqfav_zstress_v2/{prefix}_e{int(env_index):02d}"


def paired_output_path(output_dir: Path, config_name: str) -> Path:
    return output_dir / f"{config_basename(config_name)}.json"


def build_paired_command(
    args: argparse.Namespace,
    *,
    config_name: str,
    output_path: Path,
    seed: int,
    num_samples: int,
    latency_num_samples: int,
    basis: str,
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
        str(num_samples),
        "--latency-num-samples",
        str(latency_num_samples),
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--seed",
        str(seed),
        "--basis",
        basis,
        "--output",
        str(output_path),
    ]
    if args.device:
        cmd.extend(["--device", args.device])
    for name, _, model_id, checkpoint in SCAN_MODEL_SPECS:
        cmd.extend(["--model", f"{name}:{model_id}:{rel(checkpoint)}"])
    return cmd


def existing_inputs_or_raise(config_names: Sequence[str]) -> None:
    missing = []
    if not PAIRED_SCRIPT.exists():
        missing.append(PAIRED_SCRIPT)
    for config_name in config_names:
        path = config_path(config_name)
        if not path.exists():
            missing.append(path)
    for _, _, _, checkpoint in SCAN_MODEL_SPECS:
        path = rel(checkpoint)
        if not path.exists():
            missing.append(path)
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Required inputs are missing:\n{formatted}")


def _summary_ler(payload: Mapping[str, Any], method: str) -> float:
    for row in payload.get("summary", []):
        if row.get("method") == method:
            return float(row["ler_avg"])
    values = [float(row["ler"]) for row in payload.get("rows", []) if row.get("method") == method]
    return sum(values) / len(values) if values else float("nan")


def row_from_payload(
    payload_path: str | Path,
    spec: Mapping[str, Any],
    *,
    config_name: str,
) -> dict[str, Any]:
    path = rel(payload_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    domestic_ler = _summary_ler(payload, "stfusion_domestic_e100")
    seq_ler = _summary_ler(payload, "stfusion_seq_noewc_e100")
    delta = seq_ler - domestic_ler
    axis_multipliers = dict(spec["axis_multipliers"])
    return {
        "candidate_index": int(spec.get("candidate_index", 0)),
        "candidate_key": str(spec.get("candidate_key", f"candidate_{int(spec.get('candidate_index', 0)):03d}")),
        "config_name": config_name,
        "axis_signature": str(spec.get("axis_signature", "")),
        "axis_multipliers": json.dumps(axis_multipliers, sort_keys=True),
        "domestic100_ler": domestic_ler,
        "seq_noewc_ler": seq_ler,
        "ler_delta": delta,
        "delta": delta,
        "rank": "",
        "selected": False,
        "output_json": str(path),
    }


def select_top_candidates(
    rows: Sequence[Mapping[str, Any]],
    *,
    count: int = 5,
    threshold: float = -0.0015,
) -> list[dict[str, Any]]:
    eligible = [
        dict(row)
        for row in rows
        if float(row.get("ler_delta", row.get("delta", float("nan")))) < float(threshold)
    ]
    eligible.sort(key=lambda row: float(row.get("ler_delta", row.get("delta"))))
    selected = []
    for rank, row in enumerate(eligible[: int(count)], start=1):
        row["rank"] = rank
        row["selected"] = True
        selected.append(row)
    return selected


def select_confirmed_fixed_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    count: int = 5,
    threshold: float = 0.0,
) -> list[dict[str, Any]]:
    """Pick fixed environments from independently confirmed no-EWC winners."""
    return select_top_candidates(rows, count=count, threshold=threshold)


def select_named_confirmed_fixed_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    candidate_keys: Sequence[str],
    threshold: float = 0.0,
) -> list[dict[str, Any]]:
    by_key = {str(row["candidate_key"]): row for row in rows}
    missing = [key for key in candidate_keys if key not in by_key]
    if missing:
        raise ValueError(f"fixed candidate keys were not confirmed: {missing}")

    selected = []
    for rank, key in enumerate(candidate_keys, start=1):
        row = dict(by_key[key])
        delta = float(row.get("ler_delta", row.get("delta", float("nan"))))
        if not delta < float(threshold):
            raise ValueError(
                f"fixed candidate {key} has delta {delta}; expected < {threshold}"
            )
        row["rank"] = rank
        row["selected"] = True
        selected.append(row)
    return selected


def parse_candidate_key_list(value: str | None) -> list[str]:
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def _fieldnames(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    preferred = [
        "candidate_index",
        "candidate_key",
        "config_name",
        "axis_signature",
        "axis_multipliers",
        "domestic100_ler",
        "seq_noewc_ler",
        "ler_delta",
        "delta",
        "rank",
        "selected",
        "output_json",
    ]
    seen = set()
    fields = []
    for name in preferred:
        if any(name in row for row in rows):
            fields.append(name)
            seen.add(name)
    for row in rows:
        for name in row:
            if name not in seen:
                fields.append(name)
                seen.add(name)
    return fields


def write_scan_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path = rel(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    normalized = []
    for row in rows:
        item = dict(row)
        if "delta" not in item and "ler_delta" in item:
            item["delta"] = item["ler_delta"]
        if "ler_delta" not in item and "delta" in item:
            item["ler_delta"] = item["delta"]
        normalized.append(item)
    fieldnames = _fieldnames(normalized)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(normalized)


def read_scan_csv(path: str | Path) -> list[dict[str, Any]]:
    path = rel(path)
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            item = dict(row)
            for key in ("candidate_index", "rank"):
                if item.get(key) not in ("", None):
                    item[key] = int(item[key])
            for key in ("domestic100_ler", "seq_noewc_ler", "ler_delta", "delta"):
                if item.get(key) not in ("", None):
                    item[key] = float(item[key])
            if isinstance(item.get("selected"), str):
                item["selected"] = item["selected"].lower() in {"1", "true", "yes"}
            rows.append(item)
    return rows


def zstress_success_status(
    rows: Sequence[Mapping[str, Any]],
    *,
    require_all_individual_win: bool = True,
) -> dict[str, Any]:
    if not rows:
        return {
            "env_count": 0,
            "avg_delta": float("nan"),
            "individual_win_count": 0,
            "avg_beats_domestic100": False,
            "all_individual_win": False,
            "worst_beats_domestic100": False,
            "requires_all_individual_win": bool(require_all_individual_win),
            "passed": False,
        }
    deltas = [float(row.get("ler_delta", row.get("delta"))) for row in rows]
    domestic_values = [float(row["domestic100_ler"]) for row in rows]
    seq_values = [float(row["seq_noewc_ler"]) for row in rows]
    avg_delta = sum(deltas) / len(deltas)
    win_count = sum(1 for delta in deltas if delta < 0)
    avg_beats = avg_delta < 0
    all_win = win_count == len(rows)
    worst_beats = max(seq_values) <= max(domestic_values)
    return {
        "env_count": len(rows),
        "avg_delta": avg_delta,
        "individual_win_count": win_count,
        "avg_beats_domestic100": avg_beats,
        "all_individual_win": all_win,
        "worst_beats_domestic100": worst_beats,
        "requires_all_individual_win": bool(require_all_individual_win),
        "passed": avg_beats and worst_beats and (all_win if require_all_individual_win else True),
    }


def _status_text(value: bool) -> str:
    return "PASS" if value else "FAIL"


def write_zstress_markdown(
    path: str | Path,
    *,
    title: str,
    rows: Sequence[Mapping[str, Any]],
    selected_rows: Sequence[Mapping[str, Any]],
    status: Mapping[str, Any],
    basis: str,
    seed: int,
    num_samples: int,
    output_dir: str | Path,
    scan_csv: str | Path,
) -> None:
    path = rel(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    selected_table = markdown_table(
        [
            "rank",
            "candidate",
            "axis signature",
            "domestic-only epoch 100 LER",
            "seq no-EWC epoch 100 LER",
            "delta",
        ],
        [
            [
                str(row.get("rank", "")),
                str(row.get("candidate_key", "")),
                str(row.get("axis_signature", "")),
                fmt_float(row.get("domestic100_ler")),
                fmt_float(row.get("seq_noewc_ler")),
                fmt_float(row.get("ler_delta", row.get("delta"))),
            ]
            for row in selected_rows
        ],
    )
    individual_gate_label = "all selected/fixed environments individually win"
    if not bool(status.get("requires_all_individual_win", True)):
        individual_gate_label += " (reported, not required for final gate)"
    gate_table = markdown_table(
        ["条件", "结果"],
        [
            [
                "average seq no-EWC LER < domestic-only epoch 100",
                f"{_status_text(bool(status['avg_beats_domestic100']))} "
                f"(avg delta={fmt_float(status['avg_delta'])})",
            ],
            [
                individual_gate_label,
                f"{_status_text(bool(status['all_individual_win']))} "
                f"({status['individual_win_count']}/{status['env_count']})",
            ],
            [
                "worst-env seq no-EWC LER <= domestic-only epoch 100",
                _status_text(bool(status["worst_beats_domestic100"])),
            ],
            ["overall gate", _status_text(bool(status["passed"]))],
        ],
    )
    text = f"""# {title}

## 1. 推理口径

- design: `{DESIGN_LABEL}`
- basis: `{basis}`
- seed: `{seed}`
- num_samples: `{num_samples}`
- candidate/fixed rows: `{len(rows)}`
- 评估脚本: `code/scripts/paired_inference_compare.py`
- 扫描脚本: `code/scripts/run_unknown_seqfav_zstress_v2.py`

注意：本实验是 Z-basis seq-favoring stress test，不应写成中性随机未知噪声。

## 2. Selected / Fixed Environments

{selected_table}

## 3. Gate

{gate_table}

## 4. 结果文件

- Per-env JSON/CSV: `{display_path(output_dir)}`
- CSV: `{display_path(scan_csv)}`
- Markdown: `{display_path(path)}`
"""
    path.write_text(text, encoding="utf-8")


def spec_from_scan_row(row: Mapping[str, Any], *, env_index: int | None = None) -> dict[str, Any]:
    axis_multipliers = json.loads(str(row["axis_multipliers"]))
    candidate_index = int(row.get("candidate_index", env_index or 0))
    spec = {
        "candidate_index": candidate_index,
        "candidate_key": str(row.get("candidate_key", f"candidate_{candidate_index:03d}")),
        "axis_multipliers": {axis: float(value) for axis, value in axis_multipliers.items()},
        "axis_signature": str(row.get("axis_signature", "")),
        "purpose": "selected Z-basis seq-favoring stress-test environment",
    }
    if env_index is not None:
        spec["env_index"] = env_index
        spec["env_key"] = f"e{env_index:02d}"
    return spec


def run_rows(
    args: argparse.Namespace,
    *,
    specs: Sequence[Mapping[str, Any]],
    config_names: Sequence[str],
    output_dir: str | Path,
    seed: int,
    num_samples: int,
    latency_num_samples: int,
    basis: str,
    existing_csv: str | Path | None = None,
) -> list[dict[str, Any]]:
    output_dir = rel(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    existing_inputs_or_raise(config_names)
    existing_by_config: dict[str, dict[str, Any]] = {}
    if existing_csv and rel(existing_csv).exists():
        for row in read_scan_csv(existing_csv):
            if row.get("config_name"):
                existing_by_config[str(row["config_name"])] = row

    rows = []
    for spec, config_name in zip(specs, config_names):
        if args.resume and config_name in existing_by_config:
            rows.append(existing_by_config[config_name])
            continue
        output_path = paired_output_path(output_dir, config_name)
        cmd = build_paired_command(
            args,
            config_name=config_name,
            output_path=output_path,
            seed=seed,
            num_samples=num_samples,
            latency_num_samples=latency_num_samples,
            basis=basis,
        )
        print(f"[run] {config_name} -> {output_path}")
        print("[cmd] " + " ".join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, cwd=REPO_ROOT, check=True)
            rows.append(row_from_payload(output_path, spec, config_name=config_name))
    return rows


def annotate_selection(rows: Sequence[Mapping[str, Any]], selected_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    selected_by_key = {row["candidate_key"]: row for row in selected_rows}
    annotated = []
    for row in rows:
        item = dict(row)
        selected = selected_by_key.get(item.get("candidate_key"))
        if selected:
            item["rank"] = selected["rank"]
            item["selected"] = True
        else:
            item["rank"] = ""
            item["selected"] = False
        annotated.append(item)
    return annotated


def phase_generate(args: argparse.Namespace) -> list[dict[str, Any]]:
    specs = generate_candidate_specs()
    if args.max_candidates:
        specs = specs[: int(args.max_candidates)]
    paths, manifest = write_candidate_configs(
        base_config=args.base_config,
        output_dir=DEFAULT_CONFIG_OUTPUT_DIR,
        manifest=args.candidate_manifest,
        candidate_specs=specs,
    )
    print(f"[write] {manifest['manifest_path']}")
    print(f"[count] candidates={len(paths)}")
    return specs


def phase_scan(args: argparse.Namespace) -> list[dict[str, Any]]:
    specs = phase_generate(args)
    config_names = [default_candidate_config_name(spec["candidate_index"]) for spec in specs]
    rows = run_rows(
        args,
        specs=specs,
        config_names=config_names,
        output_dir=args.output_dir,
        seed=args.seed,
        num_samples=args.num_samples,
        latency_num_samples=args.latency_num_samples,
        basis=args.basis,
        existing_csv=args.scan_csv,
    )
    selected = select_top_candidates(rows, count=args.select_count, threshold=args.delta_threshold)
    annotated = annotate_selection(rows, selected)
    write_scan_csv(args.scan_csv, annotated)
    write_zstress_markdown(
        args.scan_md,
        title="Z-basis seq-favoring stress test candidate scan",
        rows=annotated,
        selected_rows=selected,
        status=zstress_success_status(selected),
        basis=args.basis,
        seed=args.seed,
        num_samples=args.num_samples,
        output_dir=args.output_dir,
        scan_csv=args.scan_csv,
    )
    print(f"[write] {rel(args.scan_csv)}")
    print(f"[write] {rel(args.scan_md)}")
    return annotated


def phase_confirm(args: argparse.Namespace) -> list[dict[str, Any]]:
    scan_rows = read_scan_csv(args.scan_csv)
    selected = select_top_candidates(scan_rows, count=args.select_count, threshold=args.delta_threshold)
    if len(selected) < args.select_count:
        raise RuntimeError(
            f"Only {len(selected)} candidates pass delta threshold {args.delta_threshold}; "
            f"need {args.select_count}."
        )
    specs = [spec_from_scan_row(row, env_index=index) for index, row in enumerate(selected)]
    candidate_config_names = [str(row["config_name"]) for row in selected]
    rows = run_rows(
        args,
        specs=specs,
        config_names=candidate_config_names,
        output_dir=args.confirm_output_dir,
        seed=args.confirm_seed,
        num_samples=args.confirm_num_samples,
        latency_num_samples=args.latency_num_samples,
        basis=args.basis,
        existing_csv=args.confirm_csv,
    )
    fixed_candidate_keys = parse_candidate_key_list(args.fixed_candidate_keys)
    if fixed_candidate_keys:
        if len(fixed_candidate_keys) != args.fixed_count:
            raise RuntimeError(
                f"--fixed-candidate-keys contains {len(fixed_candidate_keys)} keys; "
                f"expected --fixed-count={args.fixed_count}."
            )
        confirmed = select_named_confirmed_fixed_rows(
            rows,
            candidate_keys=fixed_candidate_keys,
            threshold=args.confirm_delta_threshold,
        )
    else:
        confirmed = select_confirmed_fixed_rows(
            rows,
            count=args.fixed_count,
            threshold=args.confirm_delta_threshold,
        )
    annotated = annotate_selection(rows, confirmed)
    write_scan_csv(args.confirm_csv, annotated)
    status = zstress_success_status(confirmed)
    write_zstress_markdown(
        args.confirm_md,
        title="Z-basis seq-favoring stress test confirmation",
        rows=annotated,
        selected_rows=confirmed,
        status=status,
        basis=args.basis,
        seed=args.confirm_seed,
        num_samples=args.confirm_num_samples,
        output_dir=args.confirm_output_dir,
        scan_csv=args.confirm_csv,
    )
    print(f"[write] {rel(args.confirm_csv)}")
    print(f"[write] {rel(args.confirm_md)}")
    if status["passed"] and len(confirmed) == args.fixed_count:
        fixed_specs = [spec_from_scan_row(row, env_index=index) for index, row in enumerate(confirmed)]
        write_fixed_configs(
            base_config=args.base_config,
            output_dir=DEFAULT_CONFIG_OUTPUT_DIR,
            manifest=args.fixed_manifest,
            selected_specs=fixed_specs,
        )
        print(f"[write] {rel(args.fixed_manifest)}")
    return annotated


def phase_final(args: argparse.Namespace) -> list[dict[str, Any]]:
    manifest_path = rel(args.fixed_manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    envs = manifest["environments"]
    specs = [
        {
            "candidate_index": int(env["candidate_index"]),
            "candidate_key": str(env["candidate_key"]),
            "axis_multipliers": env["axis_multipliers"],
            "axis_signature": str(env["axis_signature"]),
            "env_index": int(env["env_index"]),
            "env_key": str(env["env_key"]),
        }
        for env in envs
    ]
    config_names = [str(env["config_name"]) for env in envs]
    rows = run_rows(
        args,
        specs=specs,
        config_names=config_names,
        output_dir=args.final_output_dir,
        seed=args.final_seed,
        num_samples=args.final_num_samples,
        latency_num_samples=args.final_latency_num_samples,
        basis=args.basis,
        existing_csv=args.final_csv,
    )
    selected = []
    for rank, row in enumerate(rows, start=1):
        item = dict(row)
        item["rank"] = rank
        item["selected"] = True
        selected.append(item)
    write_scan_csv(args.final_csv, selected)
    status = zstress_success_status(selected, require_all_individual_win=False)
    write_zstress_markdown(
        args.final_md,
        title="Z-basis seq-favoring stress test final fixed run",
        rows=selected,
        selected_rows=selected,
        status=status,
        basis=args.basis,
        seed=args.final_seed,
        num_samples=args.final_num_samples,
        output_dir=args.final_output_dir,
        scan_csv=args.final_csv,
    )
    print(f"[write] {rel(args.final_csv)}")
    print(f"[write] {rel(args.final_md)}")
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("generate", "scan", "confirm", "final", "all"),
        default="scan",
    )
    parser.add_argument("--base-config", default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--distance", type=int, default=9)
    parser.add_argument("--n-rounds", type=int, default=9)
    parser.add_argument("--num-samples", type=int, default=65536)
    parser.add_argument("--confirm-num-samples", type=int, default=65536)
    parser.add_argument("--final-num-samples", type=int, default=262144)
    parser.add_argument("--latency-num-samples", type=int, default=10000)
    parser.add_argument("--final-latency-num-samples", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--confirm-seed", type=int, default=22345)
    parser.add_argument("--final-seed", type=int, default=12345)
    parser.add_argument("--basis", choices=("Z", "X", "both"), default="Z")
    parser.add_argument("--device", default=None)
    parser.add_argument("--python", default=None)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--confirm-output-dir", default=DEFAULT_CONFIRM_OUTPUT_DIR)
    parser.add_argument("--final-output-dir", default=DEFAULT_FINAL_OUTPUT_DIR)
    parser.add_argument("--scan-csv", default=DEFAULT_SCAN_CSV)
    parser.add_argument("--confirm-csv", default=DEFAULT_CONFIRM_CSV)
    parser.add_argument("--final-csv", default=DEFAULT_FINAL_CSV)
    parser.add_argument("--scan-md", default=DEFAULT_SCAN_MD)
    parser.add_argument("--confirm-md", default=DEFAULT_CONFIRM_MD)
    parser.add_argument("--final-md", default=DEFAULT_FINAL_MD)
    parser.add_argument("--candidate-manifest", default=DEFAULT_CANDIDATE_MANIFEST)
    parser.add_argument("--fixed-manifest", default=DEFAULT_FIXED_MANIFEST)
    parser.add_argument("--select-count", type=int, default=5)
    parser.add_argument("--fixed-count", type=int, default=5)
    parser.add_argument("--fixed-candidate-keys", default="")
    parser.add_argument("--delta-threshold", type=float, default=-0.0015)
    parser.add_argument("--confirm-delta-threshold", type=float, default=0.0)
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.phase == "generate":
        phase_generate(args)
    elif args.phase == "scan":
        phase_scan(args)
    elif args.phase == "confirm":
        phase_confirm(args)
    elif args.phase == "final":
        phase_final(args)
    elif args.phase == "all":
        phase_scan(args)
        confirmed = phase_confirm(args)
        fixed = [row for row in confirmed if row.get("selected")]
        if zstress_success_status(fixed)["passed"]:
            phase_final(args)
        else:
            print("[skip] confirmation gate failed; final fixed run not started")


if __name__ == "__main__":
    main()
