#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the 2^16-shot screening protocol on the already qualified smoke engines."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[3]
REPO_ROOT = CODE_ROOT.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from benchmarks.quant_t0_common import combine_basis_results, sha256_file, write_json  # noqa: E402
from scripts.experiments.quant_t0_compare import run as workflow  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan", type=Path, default=Path(__file__).with_name("experiment.yaml")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=REPO_ROOT / "outputs/quant_t0_compare_v1"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--only-model", choices=("htnet", "ising_fast"))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run_job(job, plan, output_dir, device, dry_run):
    engine = workflow.ensure_engine(job, plan, output_dir, smoke=True, dry_run=dry_run)
    root = workflow.artifact_root(output_dir, job) / job.variant
    report = root / "benchmark_screen.json"
    if not dry_run and report.is_file():
        payload = json.loads(report.read_text(encoding="utf-8"))
        if payload.get("engine_sha256") == sha256_file(engine):
            return report
        raise FileExistsError(f"screening report engine mismatch: {report}")
    seed = int(plan["evaluation_seed"]) + (0 if job.basis == "X" else 1)
    command = [
        sys.executable,
        str(CODE_ROOT / "benchmarks/bench_quantized_predecoder.py"),
        "--checkpoint",
        str(job.checkpoint),
        "--engine",
        str(engine),
        "--report",
        str(report),
        "--errors-output",
        str(root / "errors_screen.npz"),
        "--layers-output",
        str(root / "runtime_layers.json"),
        "--model-id",
        str(job.model_id),
        "--config-name",
        str(plan["config_name"]),
        "--distance",
        str(plan["distance"]),
        "--n-rounds",
        str(plan["n_rounds"]),
        "--basis",
        job.basis,
        "--expected-precision",
        job.precision,
        "--num-samples",
        str(plan["screening_shots"]),
        "--seed",
        str(seed),
        "--eval-batch-size",
        "256",
        "--batch-sizes",
        "1,8,32,128,256",
        "--warmup-iterations",
        str(plan["warmup_iterations"]),
        "--latency-iterations",
        str(plan["latency_iterations"]),
        "--absolute-ler-margin",
        str(plan["absolute_ler_margin"]),
        "--relative-ler-margin",
        str(plan["relative_ler_margin"]),
        "--alpha",
        str(plan["alpha"]),
        "--device",
        device,
    ]
    workflow.run_command(
        command,
        output_dir / "logs" / f"{job.model}_{job.basis}_{job.variant}_screen.log",
        dry_run=dry_run,
    )
    return report


def write_summary(output_dir: Path) -> None:
    reports = sorted((output_dir / "artifacts").glob("**/benchmark_screen.json"))
    payloads = [(path, json.loads(path.read_text(encoding="utf-8"))) for path in reports]
    groups = {}
    for path, payload in payloads:
        relative = path.relative_to(output_dir / "artifacts")
        key = (relative.parts[0], relative.parts[3])
        groups.setdefault(key, {})[payload["basis"]] = payload["noninferiority"]
    statuses = {
        key: combine_basis_results(basis)["status"]
        for key, basis in groups.items()
        if set(basis) == {"X", "Z"}
    }
    rows = []
    for path, payload in payloads:
        relative = path.relative_to(output_dir / "artifacts")
        key = (relative.parts[0], relative.parts[3])
        backend_batch1 = next(row for row in payload["backend"] if row["batch_size"] == 1)
        decoder_batch1 = next(row for row in payload["decoder"] if row["batch_size"] == 1)
        rows.append(
            {
                "model": key[0],
                "variant": key[1],
                "basis": payload["basis"],
                "status_holm": statuses.get(key, "INCOMPLETE"),
                "samples": payload["samples"],
                "baseline_ler": payload["noninferiority"]["baseline_ler"],
                "candidate_ler": payload["noninferiority"]["candidate_ler"],
                "delta_ler": payload["noninferiority"]["delta_ler"],
                "upper_bound": payload["noninferiority"]["one_sided_upper_bound"],
                "margin": payload["noninferiority"]["effective_margin"],
                "backend_p50_ms_batch1": backend_batch1["request_latency"]["p50_ms"],
                "decoder_p99_ms_batch1": decoder_batch1["request_latency"]["p99_ms"],
                "max_backend_shots_per_s_batch_le_256": max(
                    row["shots_per_second"] for row in payload["backend"]
                ),
                "simulation_shots_per_s": payload["simulation"]["steady_state_shots_per_s"],
                "report": str(path),
            }
        )
    results = output_dir / "results"
    results.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with (results / "screen_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if fields:
            writer.writeheader()
            writer.writerows(rows)
    write_json(results / "screen_summary.json", {"rows": rows})
    lines = ["# T0 quantization screening (2^16 shots/basis)", ""]
    if rows:
        lines.extend(
            [
                "| Model | Variant | Basis | Status | LER | ΔLER | Upper | Decoder p99 ms | Max shots/s (B≤256) |",
                "|---|---|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in rows:
            lines.append(
                f"| {row['model']} | {row['variant']} | {row['basis']} | {row['status_holm']} "
                f"| {row['candidate_ler']:.6g} | {row['delta_ler']:.6g} | {row['upper_bound']:.6g} "
                f"| {row['decoder_p99_ms_batch1']:.6g} | {row['max_backend_shots_per_s_batch_le_256']:.6g} |"
            )
    (results / "screen_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    plan = workflow.load_plan(args.plan)
    jobs = workflow.expand_jobs(plan, smoke=True, only_model=args.only_model)
    random.Random(int(plan["schedule_seed"])).shuffle(jobs)
    for job in jobs:
        run_job(job, plan, args.output_dir, args.device, args.dry_run)
    if not args.dry_run:
        write_summary(args.output_dir)


if __name__ == "__main__":
    main()
