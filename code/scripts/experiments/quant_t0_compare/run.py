#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build and benchmark the paired HTnet/Ising-Fast T0 quantization matrix."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

CODE_ROOT = Path(__file__).resolve().parents[3]
REPO_ROOT = CODE_ROOT.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from benchmarks.quant_t0_common import (  # noqa: E402
    combine_basis_results,
    sha256_file,
    write_json,
)


@dataclass(frozen=True)
class Job:
    model: str
    model_id: int
    epoch: int
    checkpoint: Path
    reference_checkpoint: Path | None
    basis: str
    precision: str
    calibration_samples: int | None
    calibration_seed: int

    @property
    def variant(self) -> str:
        if self.precision != "int8":
            return self.precision
        return f"int8_calib{self.calibration_samples}_seed{self.calibration_seed}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan",
        type=Path,
        default=Path(__file__).with_name("experiment.yaml"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs/quant_t0_compare_v1",
    )
    parser.add_argument(
        "--stage", choices=("all", "export", "build", "benchmark", "report"), default="all"
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--only-model", choices=("htnet", "ising_fast"))
    return parser.parse_args()


def resolve_repo_path(value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def load_plan(path: Path) -> dict[str, Any]:
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)


def expand_jobs(plan: dict[str, Any], *, smoke: bool, only_model: str | None = None) -> list[Job]:
    calibrations = [int(plan["calibration_samples"][0])] if smoke else [
        int(value) for value in plan["calibration_samples"]
    ]
    jobs = []
    for model_name, model in plan["models"].items():
        if only_model is not None and model_name != only_model:
            continue
        checkpoint = resolve_repo_path(model["checkpoint"])
        reference = resolve_repo_path(model.get("reference_checkpoint"))
        for basis in plan["bases"]:
            for precision in plan["precisions"]:
                sizes = calibrations if precision == "int8" else [None]
                for size in sizes:
                    jobs.append(
                        Job(
                            model=model_name,
                            model_id=int(model["model_id"]),
                            epoch=int(model["epoch"]),
                            checkpoint=checkpoint,
                            reference_checkpoint=reference,
                            basis=str(basis),
                            precision=str(precision),
                            calibration_samples=size,
                            calibration_seed=int(plan["calibration_seed"]),
                        )
                    )
    return jobs


def run_command(command: list[str], log_path: Path, *, dry_run: bool) -> None:
    print("[command]", " ".join(command))
    if dry_run:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(CODE_ROOT)
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log_path.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"command failed ({completed.returncode}); see {log_path}")


def artifact_root(output_dir: Path, job: Job) -> Path:
    return output_dir / "artifacts" / job.model / f"e{job.epoch}" / job.basis


def ensure_reference_checkpoint(
    job: Job, output_dir: Path, *, dry_run: bool
) -> Path:
    if job.reference_checkpoint is not None:
        if not dry_run and not job.reference_checkpoint.is_file():
            raise FileNotFoundError(job.reference_checkpoint)
        return job.reference_checkpoint
    output = output_dir / "artifacts" / job.model / f"e{job.epoch}" / "model_fp32.safetensors"
    if output.is_file() or dry_run:
        return output
    command = [
        sys.executable,
        str(CODE_ROOT / "export/checkpoint_to_safetensors.py"),
        "--checkpoint",
        str(job.checkpoint),
        "--model-id",
        str(job.model_id),
        "--output",
        str(output),
    ]
    run_command(command, output_dir / "logs" / f"{job.model}_safetensors.log", dry_run=dry_run)
    return output


def ensure_export(job: Job, plan: dict[str, Any], output_dir: Path, *, smoke: bool, dry_run: bool) -> Path:
    root = artifact_root(output_dir, job) / "export"
    onnx_path = root / "model_fp32.onnx"
    report = root / "parity.json"
    manifest = root / "manifest.json"
    reference = ensure_reference_checkpoint(job, output_dir, dry_run=dry_run)
    if not dry_run and manifest.is_file() and onnx_path.is_file():
        saved = json.loads(manifest.read_text(encoding="utf-8"))
        if (
            saved.get("checkpoint_sha256") == sha256_file(job.checkpoint)
            and saved.get("reference_checkpoint_sha256") == sha256_file(reference)
            and saved.get("onnx_sha256") == sha256_file(onnx_path)
        ):
            return onnx_path
        raise FileExistsError(f"export manifest mismatch: {manifest}")
    command = [
        sys.executable,
        str(CODE_ROOT / "scripts/validate_deployment_parity.py"),
        "--checkpoint",
        str(job.checkpoint),
        "--safetensors",
        str(reference),
        "--onnx-output",
        str(onnx_path),
        "--report",
        str(report),
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
        "--num-samples",
        "256" if smoke else "1024",
        "--batch-size",
        "256",
        "--seed",
        str(plan["calibration_seed"]),
        "--device",
        "cuda:0",
    ]
    run_command(command, output_dir / "logs" / f"{job.model}_{job.basis}_export.log", dry_run=dry_run)
    if not dry_run:
        write_json(
            manifest,
            {
                "status": "PASS",
                "model": job.model,
                "epoch": job.epoch,
                "basis": job.basis,
                "checkpoint_sha256": sha256_file(job.checkpoint),
                "reference_checkpoint_sha256": sha256_file(reference),
                "onnx_sha256": sha256_file(onnx_path),
            },
        )
    return onnx_path


def ensure_engine(job: Job, plan: dict[str, Any], output_dir: Path, *, smoke: bool, dry_run: bool) -> Path:
    fp32_onnx = ensure_export(job, plan, output_dir, smoke=smoke, dry_run=dry_run)
    root = artifact_root(output_dir, job) / job.variant
    source_onnx = fp32_onnx
    if job.precision == "int8":
        source_onnx = root / "model_int8.onnx"
        quant_report = root / "quantize.json"
        if not source_onnx.is_file() or not quant_report.is_file():
            command = [
                sys.executable,
                str(CODE_ROOT / "scripts/quantize_deployment_onnx.py"),
                "--onnx-input",
                str(fp32_onnx),
                "--onnx-output",
                str(source_onnx),
                "--report",
                str(quant_report),
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
                "--num-samples",
                str(job.calibration_samples),
                "--batch-size",
                str(min(256, int(job.calibration_samples))),
                "--seed",
                str(job.calibration_seed),
            ]
            run_command(
                command,
                output_dir / "logs" / f"{job.model}_{job.basis}_{job.variant}_quantize.log",
                dry_run=dry_run,
            )
    engine = root / "model.engine"
    command = [
        sys.executable,
        str(CODE_ROOT / "scripts/build_tensorrt_engine.py"),
        "--onnx",
        str(source_onnx),
        "--engine",
        str(engine),
        "--report",
        str(root / "engine.json"),
        "--layer-info",
        str(root / "layers.json"),
        "--build-log",
        str(root / "build.log"),
        "--precision",
        job.precision,
        "--input-width",
        "720",
        "--min-batch",
        "1",
        "--opt-batch",
        "256" if smoke else "2048",
        "--max-batch",
        "256" if smoke else "8192",
    ]
    run_command(
        command,
        output_dir / "logs" / f"{job.model}_{job.basis}_{job.variant}_build_wrapper.log",
        dry_run=dry_run,
    )
    return engine


def run_benchmark(job: Job, plan: dict[str, Any], output_dir: Path, *, smoke: bool, device: str, dry_run: bool) -> Path:
    engine = ensure_engine(job, plan, output_dir, smoke=smoke, dry_run=dry_run)
    root = artifact_root(output_dir, job) / job.variant
    report = root / ("benchmark_smoke.json" if smoke else "benchmark_screen.json")
    if not dry_run and report.is_file():
        saved = json.loads(report.read_text(encoding="utf-8"))
        if saved.get("engine_sha256") == sha256_file(engine):
            return report
        raise FileExistsError(f"benchmark report engine mismatch: {report}")
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
        str(root / ("errors_smoke.npz" if smoke else "errors_screen.npz")),
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
        "4096" if smoke else str(plan["screening_shots"]),
        "--seed",
        str(seed),
        "--eval-batch-size",
        "256" if smoke else "2048",
        "--batch-sizes",
        "1,8,32,256" if smoke else ",".join(str(v) for v in plan["batch_sizes"]),
        "--warmup-iterations",
        "10" if smoke else str(plan["warmup_iterations"]),
        "--latency-iterations",
        "20" if smoke else str(plan["latency_iterations"]),
        "--absolute-ler-margin",
        str(plan["absolute_ler_margin"]),
        "--relative-ler-margin",
        str(plan["relative_ler_margin"]),
        "--alpha",
        str(plan["alpha"]),
        "--device",
        device,
    ]
    run_command(
        command,
        output_dir / "logs" / f"{job.model}_{job.basis}_{job.variant}_benchmark.log",
        dry_run=dry_run,
    )
    return report


def build_report(output_dir: Path) -> None:
    reports = sorted((output_dir / "artifacts").glob("**/benchmark_*.json"))
    rows = []
    for path in reports:
        payload = json.loads(path.read_text(encoding="utf-8"))
        relative = path.relative_to(output_dir / "artifacts")
        rows.append(
            {
                "model": relative.parts[0],
                "epoch": relative.parts[1],
                "basis": payload["basis"],
                "precision": payload["expected_precision"],
                "variant": relative.parts[3],
                "status_unadjusted": payload["status"],
                "samples": payload["samples"],
                "baseline_ler": payload["noninferiority"]["baseline_ler"],
                "candidate_ler": payload["noninferiority"]["candidate_ler"],
                "delta_ler": payload["noninferiority"]["delta_ler"],
                "upper_bound": payload["noninferiority"]["one_sided_upper_bound"],
                "margin": payload["noninferiority"]["effective_margin"],
                "p_value": payload["noninferiority"]["noninferiority_p_value"],
                "backend_p50_ms_batch1": payload["backend"][0]["request_latency"]["p50_ms"],
                "decoder_p99_ms_batch1": payload["decoder"][0]["request_latency"]["p99_ms"],
                "max_backend_shots_per_s": max(row["shots_per_second"] for row in payload["backend"]),
                "simulation_shots_per_s": payload["simulation"]["steady_state_shots_per_s"],
                "report": str(path),
            }
        )

    combined_status: dict[tuple[str, str], str] = {}
    grouped: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for row in rows:
        key = (row["model"], row["variant"])
        grouped.setdefault(key, {})[row["basis"]] = {
            "noninferiority_p_value": row["p_value"]
        }
    for key, basis in grouped.items():
        if set(basis) == {"X", "Z"}:
            combined_status[key] = combine_basis_results(basis)["status"]
    for row in rows:
        row["status_holm"] = combined_status.get((row["model"], row["variant"]), "INCOMPLETE")

    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with (results_dir / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if fields:
            writer.writeheader()
            writer.writerows(rows)
    lines = ["# T0 quantization benchmark", "", f"Reports: {len(rows)}", ""]
    if rows:
        lines.extend(
            [
                "| Model | Variant | Basis | Status | LER | ΔLER | Decoder p99 ms | Max shots/s |",
                "|---|---|---|---|---:|---:|---:|---:|",
            ]
        )
        for row in rows:
            lines.append(
                f"| {row['model']} | {row['variant']} | {row['basis']} | {row['status_holm']} "
                f"| {row['candidate_ler']:.6g} | {row['delta_ler']:.6g} "
                f"| {row['decoder_p99_ms_batch1']:.6g} | {row['max_backend_shots_per_s']:.6g} |"
            )
    (results_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(results_dir / "summary.json", {"rows": rows})


def main() -> None:
    args = parse_args()
    plan = load_plan(args.plan)
    jobs = expand_jobs(plan, smoke=args.smoke, only_model=args.only_model)
    if not args.dry_run:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "plan.resolved.yaml").write_text(
            OmegaConf.to_yaml(OmegaConf.create(plan), resolve=True), encoding="utf-8"
        )
    stages = {args.stage} if args.stage != "all" else {"export", "build", "benchmark", "report"}
    if "export" in stages:
        for job in jobs:
            ensure_export(job, plan, args.output_dir, smoke=args.smoke, dry_run=args.dry_run)
    if "build" in stages:
        for job in jobs:
            ensure_engine(job, plan, args.output_dir, smoke=args.smoke, dry_run=args.dry_run)
    if "benchmark" in stages:
        scheduled = list(jobs)
        random.Random(int(plan["schedule_seed"])).shuffle(scheduled)
        for job in scheduled:
            run_benchmark(
                job,
                plan,
                args.output_dir,
                smoke=args.smoke,
                device=args.device,
                dry_run=args.dry_run,
            )
    if "report" in stages and not args.dry_run:
        build_report(args.output_dir)


if __name__ == "__main__":
    main()
