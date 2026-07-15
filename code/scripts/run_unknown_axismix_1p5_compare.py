#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run domestic-100 vs seq no-EWC on 1.5x training-axis mixed OOD configs."""

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

from scripts.generate_unknown_axismix_1p5_configs import (  # noqa: E402
    DEFAULT_MANIFEST,
    DEFAULT_OUTPUT_DIR as DEFAULT_CONFIG_OUTPUT_DIR,
    DEFAULT_PREFIX,
    DESIGN_LABEL,
    write_axismix_configs,
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

DOMESTIC_METHOD = "stfusion_domestic_e100"
SEQ_METHOD = "stfusion_seq_noewc_e100"
DEFAULT_QUICK_OUTPUT_DIR = "outputs/paired_inference_compare/unknown_axismix_1p5_quick"
DEFAULT_FULL_OUTPUT_DIR = "outputs/paired_inference_compare/unknown_axismix_1p5_full"
DEFAULT_SUMMARY_CSV = "outputs/analysis/unknown_axismix_1p5_summary.csv"
DEFAULT_SUMMARY_MD = "outputs/analysis/unknown_axismix_1p5_comparison.md"


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


def default_config_name(env_index: int, prefix: str = DEFAULT_PREFIX) -> str:
    return f"experiments/unknown_axismix_1p5/{prefix}_e{int(env_index):02d}"


def task_output_path(output_dir: Path, env_index: int) -> Path:
    return output_dir / f"unknown_axismix_1p5_e{int(env_index):02d}.json"


def existing_inputs_or_raise(config_names: Sequence[str]) -> None:
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


def build_command(
    args: argparse.Namespace,
    *,
    config_name: str,
    output_path: Path,
    num_samples: int,
    latency_num_samples: int,
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


def ensure_configs(args: argparse.Namespace) -> dict[str, Any]:
    _, manifest = write_axismix_configs(
        base_config=args.base_config,
        output_dir=DEFAULT_CONFIG_OUTPUT_DIR,
        prefix=args.config_prefix,
        manifest=args.manifest,
    )
    return manifest


def load_manifest(path: str | Path) -> dict[str, Any]:
    return json.loads(rel(path).read_text(encoding="utf-8"))


def config_names_from_manifest(manifest: Mapping[str, Any]) -> list[str]:
    return [str(env["config_name"]) for env in manifest["environments"]]


def run_phase(
    args: argparse.Namespace,
    *,
    phase: str,
    manifest: Mapping[str, Any],
) -> Path:
    output_dir = rel(args.quick_output_dir if phase == "quick" else args.full_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    num_samples = int(args.quick_num_samples if phase == "quick" else args.full_num_samples)
    latency_num_samples = int(args.quick_latency_num_samples if phase == "quick" else args.full_latency_num_samples)
    config_names = config_names_from_manifest(manifest)
    existing_inputs_or_raise(config_names)
    for env in manifest["environments"]:
        env_index = int(env["env_index"])
        config_name = str(env["config_name"])
        output_path = task_output_path(output_dir, env_index)
        if args.resume and output_path.exists():
            print(f"[skip] {phase} {env['env_key']} exists: {output_path}")
            continue
        cmd = build_command(
            args,
            config_name=config_name,
            output_path=output_path,
            num_samples=num_samples,
            latency_num_samples=latency_num_samples,
        )
        print(f"[run] phase={phase} env={env['env_key']} config={config_name} -> {output_path}")
        print("[cmd] " + " ".join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    return output_dir


def load_payloads(output_dir: str | Path, manifest: Mapping[str, Any], *, run_phase: str) -> list[dict[str, Any]]:
    output_dir = rel(output_dir)
    payloads = []
    missing = []
    for env in manifest["environments"]:
        env_index = int(env["env_index"])
        output_path = task_output_path(output_dir, env_index)
        if not output_path.exists():
            missing.append(output_path)
            continue
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        payload["env_index"] = env_index
        payload["env_key"] = str(env["env_key"])
        payload["config_name"] = str(env["config_name"])
        payload["run_phase"] = run_phase
        payload["path"] = str(output_path)
        payloads.append(payload)
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Missing per-environment inference outputs:\n{formatted}")
    return payloads


def _env_lookup(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return config_lookup_with_basename(manifest["environments"])


def _method_ler(rows: Sequence[Mapping[str, Any]], basis: str, method: str) -> float:
    values = [
        float(row["ler"])
        for row in rows
        if row.get("basis") == basis and row.get("method") == method
    ]
    if len(values) != 1:
        return float("nan")
    return values[0]


def _basis_values(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    values = sorted({str(row.get("basis")) for row in rows if row.get("basis") in {"X", "Z"}})
    return [basis for basis in ("X", "Z") if basis in values]


def _comparison_row(
    *,
    payload: Mapping[str, Any],
    env: Mapping[str, Any],
    basis: str,
    domestic_ler: float,
    seq_ler: float,
) -> dict[str, Any]:
    delta = seq_ler - domestic_ler
    relative_delta = delta / domestic_ler * 100.0 if domestic_ler else float("nan")
    return {
        "run_phase": payload.get("run_phase", ""),
        "env_key": env["env_key"],
        "env_index": int(env["env_index"]),
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
        "output_json": payload.get("path", ""),
    }


def aggregate_payloads(
    payloads: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    env_by_config = _env_lookup(manifest)
    env_rows: list[dict[str, Any]] = []
    for payload in payloads:
        config_name = str(payload["config_name"])
        env = env_by_config[config_name]
        basis_values = _basis_values(payload["rows"])
        per_basis_lers = {}
        for basis in basis_values:
            domestic_ler = _method_ler(payload["rows"], basis, DOMESTIC_METHOD)
            seq_ler = _method_ler(payload["rows"], basis, SEQ_METHOD)
            per_basis_lers[basis] = (domestic_ler, seq_ler)
            env_rows.append(
                _comparison_row(
                    payload=payload,
                    env=env,
                    basis=basis,
                    domestic_ler=domestic_ler,
                    seq_ler=seq_ler,
                )
            )
        if {"X", "Z"}.issubset(per_basis_lers):
            domestic_both = mean([per_basis_lers["X"][0], per_basis_lers["Z"][0]])
            seq_both = mean([per_basis_lers["X"][1], per_basis_lers["Z"][1]])
            env_rows.append(
                _comparison_row(
                    payload=payload,
                    env=env,
                    basis="both",
                    domestic_ler=domestic_both,
                    seq_ler=seq_both,
                )
            )

    group_rows = aggregate_groups(env_rows)
    return env_rows, group_rows


def aggregate_groups(env_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    group_defs = [
        ("combination_size", lambda row: str(row["combination_size"])),
        ("contains_z_bias", lambda row: "yes" if row["contains_z_bias"] else "no"),
        ("contains_cnot_z_bias", lambda row: "yes" if row["contains_cnot_z_bias"] else "no"),
    ]
    groups: dict[tuple[str, str, str, str], list[Mapping[str, Any]]] = {}
    for row in env_rows:
        phase = str(row.get("run_phase", ""))
        basis = str(row["basis"])
        for group_type, group_fn in group_defs:
            key = (phase, group_type, group_fn(row), basis)
            groups.setdefault(key, []).append(row)

    group_rows = []
    for (phase, group_type, group_value, basis), rows in sorted(groups.items()):
        domestic_values = [float(row["domestic100_ler"]) for row in rows]
        seq_values = [float(row["seq_noewc_ler"]) for row in rows]
        deltas = [float(row["delta"]) for row in rows]
        group_rows.append(
            {
                "run_phase": phase,
                "group_type": group_type,
                "group_value": group_value,
                "basis": basis,
                "env_count": len(rows),
                "domestic100_ler_avg": mean(domestic_values),
                "seq_noewc_ler_avg": mean(seq_values),
                "delta_avg": mean(deltas),
                "relative_delta_pct_avg": mean([float(row["relative_delta_pct"]) for row in rows]),
                "seq_win_count": sum(1 for row in rows if float(row["delta"]) < 0),
            }
        )
    return group_rows


def _fieldnames(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    preferred = [
        "run_phase",
        "env_key",
        "env_index",
        "config_name",
        "axis_signature",
        "active_axes",
        "combination_size",
        "contains_z_bias",
        "contains_cnot_z_bias",
        "basis",
        "domestic100_ler",
        "seq_noewc_ler",
        "delta",
        "relative_delta_pct",
        "winner",
        "output_json",
    ]
    seen = set()
    fields = []
    for field in preferred:
        if any(field in row for row in rows):
            fields.append(field)
            seen.add(field)
    for row in rows:
        for field in row:
            if field not in seen:
                fields.append(field)
                seen.add(field)
    return fields


def write_env_csv(path: str | Path, env_rows: Sequence[Mapping[str, Any]]) -> None:
    path = rel(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_fieldnames(env_rows))
        writer.writeheader()
        writer.writerows(env_rows)


def _table_rows(rows: Sequence[Mapping[str, Any]], *, phase: str, basis: str) -> list[list[str]]:
    selected = [
        row for row in rows
        if str(row.get("run_phase", "")) == phase and row.get("basis") == basis
    ]
    selected.sort(key=lambda row: int(row["env_index"]))
    return [
        [
            str(row["env_key"]),
            str(row["axis_signature"]),
            str(row["combination_size"]),
            fmt_float(row["domestic100_ler"]),
            fmt_float(row["seq_noewc_ler"]),
            fmt_float(row["delta"]),
            fmt_float(row["relative_delta_pct"], digits=3),
            str(row["winner"]),
        ]
        for row in selected
    ]


def _group_table_rows(rows: Sequence[Mapping[str, Any]], *, phase: str, basis: str) -> list[list[str]]:
    selected = [
        row for row in rows
        if str(row.get("run_phase", "")) == phase and row.get("basis") == basis
    ]
    selected.sort(key=lambda row: (str(row["group_type"]), str(row["group_value"])))
    return [
        [
            str(row["group_type"]),
            str(row["group_value"]),
            str(row["env_count"]),
            fmt_float(row["domestic100_ler_avg"]),
            fmt_float(row["seq_noewc_ler_avg"]),
            fmt_float(row["delta_avg"]),
            str(row["seq_win_count"]),
        ]
        for row in selected
    ]


def write_markdown(
    path: str | Path,
    *,
    summary_csv: str | Path,
    env_rows: Sequence[Mapping[str, Any]],
    group_rows: Sequence[Mapping[str, Any]],
    output_dirs: Sequence[str | Path],
    basis: str,
    num_samples: int,
) -> None:
    path = rel(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    phases = []
    for row in env_rows:
        phase = str(row.get("run_phase", "")) or "run"
        if phase not in phases:
            phases.append(phase)
    basis_order = [item for item in ("both", "X", "Z") if any(row["basis"] == item for row in env_rows)]

    lines = [
        "# Training-axis mixed 1.5x OOD composite comparison",
        "",
        "## 1. Protocol",
        "",
        f"- design: `{DESIGN_LABEL}`",
        f"- basis request: `{basis}`",
        f"- num_samples shown for the latest requested phase: `{num_samples}`",
        "- compared methods: `stfusion_domestic_e100` vs `stfusion_seq_noewc_e100`",
        "- Negative delta means seq no-EWC is better.",
        "- This is a training-axis mixed 1.5x OOD composite test, not random unknown noise.",
        "",
        "## 2. Per-environment results",
        "",
    ]
    for phase in phases:
        lines.extend([f"### {phase}", ""])
        for basis_name in basis_order:
            rows = _table_rows(env_rows, phase=phase, basis=basis_name)
            if not rows:
                continue
            lines.extend(
                [
                    f"#### basis={basis_name}",
                    "",
                    markdown_table(
                        [
                            "env",
                            "axis signature",
                            "size",
                            "domestic100 LER",
                            "seq no-EWC LER",
                            "delta",
                            "relative delta %",
                            "winner",
                        ],
                        rows,
                    ),
                    "",
                ]
            )

    lines.extend(["## 3. Grouped results", ""])
    for phase in phases:
        lines.extend([f"### {phase}", ""])
        for basis_name in basis_order:
            rows = _group_table_rows(group_rows, phase=phase, basis=basis_name)
            if not rows:
                continue
            lines.extend(
                [
                    f"#### basis={basis_name}",
                    "",
                    markdown_table(
                        [
                            "group",
                            "value",
                            "envs",
                            "domestic100 avg",
                            "seq no-EWC avg",
                            "delta avg",
                            "seq wins",
                        ],
                        rows,
                    ),
                    "",
                ]
            )

    lines.extend(
        [
            "## 4. Files",
            "",
            f"- summary CSV: `{display_path(summary_csv)}`",
        ]
    )
    for output_dir in output_dirs:
        lines.append(f"- per-env JSON/CSV: `{display_path(output_dir)}`")
    lines.append(f"- Markdown: `{display_path(path)}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(
    *,
    summary_csv: str | Path,
    summary_md: str | Path,
    env_rows: Sequence[Mapping[str, Any]],
    group_rows: Sequence[Mapping[str, Any]],
    output_dirs: Sequence[str | Path],
    basis: str,
    num_samples: int,
) -> None:
    write_env_csv(summary_csv, env_rows)
    write_markdown(
        summary_md,
        summary_csv=summary_csv,
        env_rows=env_rows,
        group_rows=group_rows,
        output_dirs=output_dirs,
        basis=basis,
        num_samples=num_samples,
    )


def phase_output_dirs(args: argparse.Namespace) -> list[tuple[str, Path]]:
    if args.phase == "quick":
        return [("quick", rel(args.quick_output_dir))]
    if args.phase == "full":
        return [("full", rel(args.full_output_dir))]
    return [("quick", rel(args.quick_output_dir)), ("full", rel(args.full_output_dir))]


def aggregate_existing(args: argparse.Namespace, manifest: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    payloads = []
    for phase, output_dir in phase_output_dirs(args):
        payloads.extend(load_payloads(output_dir, manifest, run_phase=phase))
    return aggregate_payloads(payloads, manifest)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("quick", "full", "all"), default="all")
    parser.add_argument("--base-config", default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--config-prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--distance", type=int, default=9)
    parser.add_argument("--n-rounds", type=int, default=9)
    parser.add_argument("--quick-num-samples", type=int, default=65536)
    parser.add_argument("--full-num-samples", type=int, default=262144)
    parser.add_argument("--quick-latency-num-samples", type=int, default=10000)
    parser.add_argument("--full-latency-num-samples", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--basis", choices=("both", "X", "Z"), default="both")
    parser.add_argument("--device", default=None)
    parser.add_argument("--python", default=None)
    parser.add_argument("--quick-output-dir", default=DEFAULT_QUICK_OUTPUT_DIR)
    parser.add_argument("--full-output-dir", default=DEFAULT_FULL_OUTPUT_DIR)
    parser.add_argument("--summary-csv", default=DEFAULT_SUMMARY_CSV)
    parser.add_argument("--summary-md", default=DEFAULT_SUMMARY_MD)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = ensure_configs(args)
    output_dirs: list[Path] = []
    if not args.aggregate_only:
        if args.phase in {"quick", "all"}:
            output_dirs.append(run_phase(args, phase="quick", manifest=manifest))
        if args.phase in {"full", "all"}:
            output_dirs.append(run_phase(args, phase="full", manifest=manifest))
    else:
        output_dirs = [output_dir for _, output_dir in phase_output_dirs(args)]

    env_rows, group_rows = aggregate_existing(args, manifest)
    latest_num_samples = args.full_num_samples if args.phase in {"full", "all"} else args.quick_num_samples
    write_outputs(
        summary_csv=args.summary_csv,
        summary_md=args.summary_md,
        env_rows=env_rows,
        group_rows=group_rows,
        output_dirs=output_dirs,
        basis=args.basis,
        num_samples=latest_num_samples,
    )
    print(f"[write] {rel(args.summary_csv)}")
    print(f"[write] {rel(args.summary_md)}")


if __name__ == "__main__":
    main()
