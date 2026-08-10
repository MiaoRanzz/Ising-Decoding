# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared command construction and execution for the public QAdapt examples."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
PAIRED_INFERENCE_SCRIPT = REPO_ROOT / "code" / "scripts" / "paired_inference_compare.py"
TASK_CONFIGS = (
    ("t0_base", "examples/qadapt/config_qadapt_t0_base"),
    ("t1_meas_1p5", "examples/qadapt/config_qadapt_t1_meas_1p5"),
    ("t2_cnot_1p5", "examples/qadapt/config_qadapt_t2_cnot_1p5"),
    ("t3_idle_1p5", "examples/qadapt/config_qadapt_t3_idle_1p5"),
    ("t4_z_bias_1p5", "examples/qadapt/config_qadapt_t4_z_bias_1p5"),
)


@dataclass(frozen=True)
class ModelArgument:
    name: str
    model_id: int
    checkpoint: Path


@dataclass(frozen=True)
class InferenceJob:
    label: str
    command: tuple[str, ...]
    output_path: Path


def parse_model_argument(value: str) -> ModelArgument:
    parts = value.split(":", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "--model must be formatted as name:model_id:/path/to/checkpoint"
        )
    name, model_id_raw, checkpoint_raw = (part.strip() for part in parts)
    if not name or not checkpoint_raw:
        raise argparse.ArgumentTypeError("model name and checkpoint must not be empty")
    try:
        model_id = int(model_id_raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"invalid model_id: {model_id_raw}"
        ) from exc
    checkpoint = Path(checkpoint_raw).expanduser()
    if not checkpoint.is_absolute():
        checkpoint = REPO_ROOT / checkpoint
    return ModelArgument(name=name, model_id=model_id, checkpoint=checkpoint)


def _default_gpus() -> str:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    return visible or "0"


def add_common_inference_args(
    parser: argparse.ArgumentParser,
    *,
    default_output_dir: Path,
) -> None:
    parser.add_argument(
        "--model",
        action="append",
        type=parse_model_argument,
        required=True,
        help=(
            "Repeat for each released model: name:model_id:/path/to/checkpoint. "
            "Both .pt and .safetensors are supported."
        ),
    )
    parser.add_argument("--num-samples", type=int, default=262144)
    parser.add_argument("--latency-num-samples", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--basis", choices=("both", "X", "Z"), default="both")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--gpus", default=_default_gpus())
    parser.add_argument("--parallelism", type=int, default=1)
    parser.add_argument(
        "--python",
        default=os.environ.get("PREDECODER_PYTHON", sys.executable),
    )
    parser.add_argument("--output-dir", type=Path, default=default_output_dir)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")


def checkpoint_specs(args: argparse.Namespace) -> tuple[ModelArgument, ...]:
    specs = tuple(args.model)
    names = [spec.name for spec in specs]
    if len(names) != len(set(names)):
        raise ValueError(f"model names must be unique: {names}")
    return specs


def parse_gpus(value: str | Sequence[str]) -> list[str]:
    raw = value.split(",") if isinstance(value, str) else value
    result = [str(item).strip() for item in raw if str(item).strip()]
    if not result:
        raise ValueError("at least one GPU must be selected")
    return result


def build_paired_command(
    args: argparse.Namespace,
    *,
    output_path: Path,
    distance: int,
    n_rounds: int,
    config_name: str | None = None,
    config_file: Path | None = None,
) -> tuple[str, ...]:
    if (config_name is None) == (config_file is None):
        raise ValueError("provide exactly one of config_name or config_file")
    command = [
        str(args.python),
        "-u",
        str(PAIRED_INFERENCE_SCRIPT),
    ]
    if config_name is not None:
        command.extend(("--config-name", config_name))
    else:
        command.extend(("--config-file", str(Path(config_file))))
    command.extend(
        (
            "--distance",
            str(distance),
            "--n-rounds",
            str(n_rounds),
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
            str(args.basis),
            "--device",
            "cuda:0",
            "--output",
            str(output_path),
        )
    )
    for spec in checkpoint_specs(args):
        command.extend(
            ("--model", f"{spec.name}:{spec.model_id}:{spec.checkpoint}")
        )
    return tuple(command)


def _run_one(job: InferenceJob, gpu: str) -> tuple[InferenceJob, int, Path]:
    job.output_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = job.output_path.with_suffix(".log")
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = gpu
    with log_path.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(
            job.command,
            cwd=REPO_ROOT,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return job, int(completed.returncode), log_path


def run_jobs(
    jobs: Sequence[InferenceJob],
    *,
    gpus: Sequence[str],
    parallelism: int,
    resume: bool,
    dry_run: bool,
) -> None:
    selected_gpus = parse_gpus(gpus)
    workers = max(1, min(int(parallelism), len(selected_gpus)))
    pending = [
        job for job in jobs
        if not (resume and job.output_path.is_file())
    ]
    skipped = len(jobs) - len(pending)
    if skipped:
        print(f"[resume] skipped {skipped} existing outputs")
    if dry_run:
        for index, job in enumerate(pending):
            gpu = selected_gpus[index % workers]
            print(
                f"[dry-run] gpu={gpu} label={job.label} "
                + shlex.join(job.command)
            )
        return
    failures = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_run_one, job, selected_gpus[index % workers]): job
            for index, job in enumerate(pending)
        }
        for future in as_completed(futures):
            job, returncode, log_path = future.result()
            if returncode:
                failures.append((job, returncode, log_path))
                print(f"[fail] {job.label} log={log_path}")
            else:
                print(f"[done] {job.label} output={job.output_path}")
    if failures:
        details = "\n".join(
            f"  - {job.label}: exit={returncode}, log={log_path}"
            for job, returncode, log_path in failures
        )
        raise RuntimeError(f"Released-model inference jobs failed:\n{details}")
