#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluate HTNet budget pilots, extend only needed tasks to 100, and select B."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
PYTHON = Path("/root/miniconda3/envs/ising-decoding/bin/python")
RUN = REPO_ROOT / "code" / "scripts" / "experiments" / "qewc_v4" / "run_continual.py"
EVALUATE = REPO_ROOT / "code" / "scripts" / "experiments" / "qewc_v3" / "evaluate_realistic_matrix.py"
SELECT = REPO_ROOT / "code" / "scripts" / "experiments" / "qewc_v4" / "select_budget.py"
STAGE0 = REPO_ROOT / "outputs" / "htnet_ablation_t0_a101_s4m_e100_seed12345_a100_skipup_v1" / "models" / "HTnet.0.100.pt"
SAMPLE_ROOT = REPO_ROOT / "outputs" / "qewc_v4" / "evaluation_streams" / "calibration_seed20260914" / "stress"
TASKS = ("T1", "T2", "T3", "T4")
BASE_EPOCHS = (1, 5, 20, 50)


def checkpoint(task: str, epochs: int) -> Path:
    return REPO_ROOT / "outputs" / "qewc_v4" / "budget_pilot" / "htnet" / f"{task}_e{epochs}" / f"stage_1_{task}" / "models" / f"HTnet.0.{epochs}.pt"


def evaluate_task(args, task: str, *, include_100: bool, resume: bool) -> None:
    epochs = (*BASE_EPOCHS, 100) if include_100 else BASE_EPOCHS
    command = [
        str(args.python), "-u", str(EVALUATE),
        "--checkpoint", f"C0:{STAGE0}",
        "--track", f"budget_{task}",
        "--sample-root", str(SAMPLE_ROOT), "--num-samples", "65536",
        "--latency-num-samples", "1000", "--batch-size", "2048",
        "--model-id", "111",
        "--seed", "20260914", "--gpus", args.gpus, "--parallelism", "2",
        "--python", str(args.python),
        "--output-dir", str(REPO_ROOT / "outputs" / "qewc_v4" / "budget_pilot" / "evaluations" / task),
    ]
    for epoch in epochs:
        command.extend(("--checkpoint", f"E{epoch}:{checkpoint(task, epoch)}"))
    for name in ("T0", *TASKS):
        command.extend(("--task", f"{name}:{REPO_ROOT / 'conf' / 'experiments' / 'qewc_v4' / 'htnet' / (name + '.yaml')}"))
    if resume:
        command.append("--resume")
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def select_budget(args, extended_tasks: list[str]) -> dict:
    output = REPO_ROOT / "outputs" / "qewc_v4" / "budget_pilot" / "selection.json"
    command = [
        str(args.python), "-u", str(SELECT),
        "--root", "outputs/qewc_v4/budget_pilot",
        "--log-root", "logs/qewc_v4/budget_pilot/htnet",
        "--ler-threshold", "0.001", "--bce-threshold", "0.01",
        "--output", str(output),
    ]
    for task in extended_tasks:
        command.extend(("--include-100-task", task))
    subprocess.run(command, cwd=REPO_ROOT, check=True)
    return json.loads(output.read_text(encoding="utf-8"))


def train_e100(args, task: str, gpu: str) -> None:
    command = [
        str(args.python), "-u", str(RUN), "--method", "seq",
        "--run-name", f"qewc_v4/budget_pilot/htnet/{task}_e100",
        "--base-task", "T0:experiments/qewc_v4/htnet/T0",
        "--task", f"{task}:experiments/qewc_v4/htnet/{task}",
        "--order", task, "--expected-stages", "1", "--stage0", str(STAGE0),
        "--epochs-per-task", "100", "--requested-samples-per-epoch", "4194304",
        "--batch-size", "2048", "--accumulate-steps", "2",
        "--checkpoint-prefix", "HTnet", "--validation-samples", "65536",
        "--seed", "20260907", "--gpu", gpu, "--python", str(args.python), "--resume",
    ]
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--python", type=Path, default=PYTHON)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if Path(sys.prefix).resolve() != PYTHON.parent.parent.resolve():
        raise RuntimeError(f"wrong Python environment: {sys.prefix}")
    for task in TASKS:
        evaluate_task(args, task, include_100=False, resume=True)
    initial = select_budget(args, [])
    if initial["status"] == "selected":
        print(f"[QEWC v4] selected B={initial['selected_B']} without E100")
        return
    extended = [
        task for task in TASKS
        if initial["tasks"][task]["needs_extension_to_100"]
    ]
    if not extended:
        raise RuntimeError("budget requested extension but named no tasks")
    gpu_list = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if len(gpu_list) != 2:
        raise ValueError("budget extension requires exactly two GPU ids")
    assignments = {task: gpu_list[index % 2] for index, task in enumerate(extended)}
    def train_gpu(gpu: str) -> None:
        for task in extended:
            if assignments[task] == gpu:
                train_e100(args, task, gpu)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(train_gpu, gpu) for gpu in gpu_list]
        for future in futures:
            future.result()
    for task in TASKS:
        evaluate_task(args, task, include_100=task in extended, resume=task not in extended)
    final = select_budget(args, extended)
    if final["status"] != "selected":
        raise RuntimeError(f"budget did not terminate at E100: {final['status']}")
    print(f"[QEWC v4] selected B={final['selected_B']} after E100 tasks={extended}")


if __name__ == "__main__":
    main()
