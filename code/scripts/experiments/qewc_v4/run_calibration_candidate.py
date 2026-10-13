#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Train and ID-evaluate one full-sequence QEWC-v4 calibration candidate."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
RUN_CONTINUAL = REPO_ROOT / "code" / "scripts" / "experiments" / "qewc_v4" / "run_continual.py"
EVALUATOR = REPO_ROOT / "code" / "scripts" / "experiments" / "qewc_v3" / "evaluate_realistic_matrix.py"
EXPERIMENT = REPO_ROOT / "experiment" / "qewc-v4" / "experiment_manifest.yaml"
BUDGET = REPO_ROOT / "experiment" / "qewc-v4" / "budget_decision.json"
CALIBRATION_ROOT = REPO_ROOT / "outputs" / "qewc_v4" / "calibration"
SAMPLE_ROOT = REPO_ROOT / "outputs" / "qewc_v4" / "evaluation_streams" / "calibration_seed20260914" / "stress"
PYTHON = Path("/root/miniconda3/envs/ising-decoding/bin/python")
FORBIDDEN_LAMBDA = 43.87539484535376
ORDER = ("T1", "T3", "T2", "T4")


@dataclass(frozen=True)
class Architecture:
    config_dir: str
    base_config: str
    checkpoint_prefix: str
    stage0: str
    model_id: int


ARCHITECTURES = {
    "htnet_a101": Architecture(
        "experiments/qewc_v4/htnet", "experiments/qewc_v4/htnet/T0", "HTnet",
        "outputs/htnet_ablation_t0_a101_s4m_e100_seed12345_a100_skipup_v1/models/HTnet.0.100.pt",
        111,
    ),
    "ising_fast": Architecture(
        "experiments/qewc_v4/ising", "experiments/qewc_v4/ising/t0_base", "PreDecoderModelMemory_v1",
        "outputs/ising_fast_qadapt_t0_s4m_e100_seed12345_a100_skipup_v1/models/PreDecoderModelMemory_v1.0.100.pt",
        1,
    ),
}


def load_selected_budget(path: Path) -> int:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "selected" or payload.get("selected_B") is None:
        raise RuntimeError(f"budget is not selected: {path}")
    return int(payload["selected_B"])


def selected_fisher(architecture: str) -> tuple[int, Path]:
    root = CALIBRATION_ROOT / architecture / "fisher"
    payload = json.loads((root / "selection.json").read_text(encoding="utf-8"))
    count = int(payload["selected_fisher_samples"])
    path = root / f"T0_n{count}.pt"
    if not path.is_file():
        raise FileNotFoundError(path)
    return count, path


def replay_capacity(architecture: str) -> dict:
    path = CALIBRATION_ROOT / architecture / "replay_capacity" / "capacity_selection.json"
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", choices=tuple(ARCHITECTURES), required=True)
    parser.add_argument("--kind", choices=("seq", "ewc", "replay"), required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--value", type=float)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--budget-selection", type=Path, default=BUDGET)
    parser.add_argument("--python", type=Path, default=PYTHON)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.kind == "seq" and args.value is not None:
        parser.error("SEQ has no candidate value")
    if args.kind != "seq" and args.value is None:
        parser.error("EWC and Replay require --value")
    if args.kind == "ewc" and float(args.value) == FORBIDDEN_LAMBDA:
        parser.error("the V3 lambda 43.87539484535376 is forbidden")
    if args.kind == "replay" and float(args.value) not in (0.10, 0.25, 0.50):
        parser.error("Replay ratio must be one of 0.10, 0.25, 0.50")
    return args


def main() -> None:
    args = parse_args()
    if Path(sys.prefix).resolve() != PYTHON.parent.parent.resolve():
        raise RuntimeError(f"wrong Python environment: {sys.prefix}")
    architecture = ARCHITECTURES[args.architecture]
    budget_path = args.budget_selection if args.budget_selection.is_absolute() else REPO_ROOT / args.budget_selection
    budget = load_selected_budget(budget_path)
    run_name = f"qewc_v4/calibration/{args.architecture}/{args.kind}/{args.name}/training"
    run_root = REPO_ROOT / "outputs" / run_name
    command = [
        str(args.python), "-u", str(RUN_CONTINUAL),
        "--method", args.kind, "--run-name", run_name,
        "--base-task", f"T0:{architecture.base_config}",
    ]
    for task in ORDER:
        command.extend(("--task", f"{task}:{architecture.config_dir}/{task}"))
    command.extend((
        "--order", ",".join(ORDER), "--expected-stages", "4",
        "--stage0", str(REPO_ROOT / architecture.stage0),
        "--epochs-per-task", str(budget),
        "--requested-samples-per-epoch", "4194304",
        "--batch-size", "2048", "--accumulate-steps", "2",
        "--checkpoint-prefix", architecture.checkpoint_prefix,
        "--validation-samples", "65536", "--seed", "20260907",
        "--gpu", args.gpu, "--python", str(args.python),
    ))
    if args.kind == "ewc":
        fisher_samples, fisher_path = selected_fisher(args.architecture)
        command.extend((
            "--positive-weight", "1.0", "--fisher-samples", str(fisher_samples),
            "--fisher-batch-size", "2048", "--fisher-per-sample-chunk-size", "16",
            "--fisher-seed", "20260912", "--initial-fisher", str(fisher_path),
            "--ewc-lambda", str(args.value), "--ewc-gamma", "1.0",
        ))
    elif args.kind == "replay":
        capacity = replay_capacity(args.architecture)
        command.extend((
            "--replay-dir", str(run_root / "replay"),
            "--replay-capacity", str(capacity["capacity_samples"]),
            "--replay-ratio", str(args.value),
            "--replay-target-bytes", str(capacity["target_bytes"]),
            "--replay-byte-tolerance", str(capacity["byte_tolerance"]),
            "--replay-seed", "20260913",
        ))
    if args.resume:
        command.append("--resume")
    print("[QEWC v4 calibration train] " + " ".join(command), flush=True)
    if not args.dry_run:
        subprocess.run(command, cwd=REPO_ROOT, check=True)

    training_path = run_root / "run_manifest.json"
    if args.dry_run:
        checkpoints = {
            "C0": REPO_ROOT / architecture.stage0,
            **{
                f"C{stage}": run_root / f"stage_{stage}_{task}" / "models" / f"{architecture.checkpoint_prefix}.0.{budget}.pt"
                for stage, task in enumerate(ORDER, 1)
            },
        }
    else:
        training = json.loads(training_path.read_text(encoding="utf-8"))
        checkpoints = {name: Path(item["path"]) for name, item in training["endpoints"].items()}
    output = run_root.parent / "evaluation"
    evaluate = [
        str(args.python), "-u", str(EVALUATOR),
        "--track", f"calibration_{args.architecture}_{args.kind}_{args.name}",
        "--sample-root", str(SAMPLE_ROOT), "--num-samples", "65536",
        "--latency-num-samples", "1000", "--batch-size", "2048",
        "--model-id", str(architecture.model_id),
        "--seed", "20260914", "--gpus", args.gpu, "--parallelism", "1",
        "--python", str(args.python), "--output-dir", str(output),
    ]
    for name, path in checkpoints.items():
        evaluate.extend(("--checkpoint", f"{name}:{path}"))
    task_configs = {
        "T0": REPO_ROOT / "conf" / f"{architecture.base_config}.yaml",
        **{
            task: REPO_ROOT / "conf" / architecture.config_dir / f"{task}.yaml"
            for task in ("T1", "T2", "T3", "T4")
        },
    }
    for task, config in task_configs.items():
        evaluate.extend(("--task", f"{task}:{config}"))
    if args.resume:
        evaluate.append("--resume")
    print("[QEWC v4 calibration eval] " + " ".join(evaluate), flush=True)
    if not args.dry_run:
        subprocess.run(evaluate, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
