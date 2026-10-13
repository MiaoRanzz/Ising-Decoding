#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the full V4 SEQ, EWC-grid, refinement, and Replay calibration."""

from __future__ import annotations

import argparse
import copy
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from queue import Queue, Empty
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
CANDIDATE = REPO_ROOT / "code" / "scripts" / "experiments" / "qewc_v4" / "run_calibration_candidate.py"
SELECTOR = REPO_ROOT / "code" / "scripts" / "experiments" / "qewc_v4" / "select_continual_calibration.py"
PYTHON = Path("/root/miniconda3/envs/ising-decoding/bin/python")
COARSE = (0.1, 1.0, 10.0, 100.0, 1000.0)
REPLAY = (0.10, 0.25, 0.50)


def value_name(prefix: str, value: float) -> str:
    raw = f"{value:g}".replace("-", "m").replace(".", "p").replace("+", "")
    return f"{prefix}_{raw}"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def candidate_paths(root: Path, kind: str, name: str) -> tuple[Path, Path]:
    base = root / kind / name
    return base / "evaluation" / "evaluation_manifest.json", base / "training" / "run_manifest.json"


def run_candidate(args, kind: str, name: str, value: float | None = None) -> None:
    root = REPO_ROOT / 'outputs/qewc_v4/calibration' / args.architecture
    evaluation, training = candidate_paths(root, kind, name)
    def completed():
        if not evaluation.is_file() or not training.is_file():
            return False
        record = json.loads(training.read_text())
        budget = json.loads((REPO_ROOT / 'experiment/qewc-v4/budget_decision.json').read_text())['selected_B']
        if record['legacy_epochs_per_task'] != budget:
            raise RuntimeError(f'Budget mismatch: {training}')
        for item in record['endpoints'].values():
            if sha256(Path(item['path'])) != item['sha256']:
                raise RuntimeError(f'Endpoint mismatch: {training}')
        json.loads(evaluation.read_text())
        return True
    adopt = getattr(args, 'adopt_pid', None)
    if adopt and kind == 'ewc' and name == 'lambda_0p1' and not args.dry_run:
        print(f'[QEWC v4] adopting existing HTNet lambda_0p1 PID={adopt}', flush=True)
        while Path(f'/proc/{adopt}/stat').exists():
            try:
                if Path(f'/proc/{adopt}/stat').read_text().split(') ')[1].split()[0] == 'Z':
                    break
                cmd = Path(f'/proc/{adopt}/cmdline').read_bytes().replace(b'\0', b' ').decode()
            except FileNotFoundError:
                break
            if 'run_calibration_candidate.py' not in cmd or 'lambda_0p1' not in cmd or 'htnet_a101' not in cmd:
                raise RuntimeError('Adopted PID no longer identifies the expected candidate')
            from scripts.experiments.qewc_v4.retain_checkpoints import prune
            prune(root / kind / name / 'training', REPO_ROOT / 'logs/qewc_v4/orchestration/htnet_adopt.pruned.jsonl')
            time.sleep(30)
        if not completed():
            raise RuntimeError('Adopted candidate exited without complete training/evaluation')
    if args.resume and not args.dry_run and completed():
        print(f'[QEWC v4] verified completed candidate: {kind}/{name}', flush=True)
        return
    command = [
        str(args.python), "-u", str(CANDIDATE),
        "--architecture", args.architecture, "--kind", kind,
        "--name", name, "--gpu", args.gpu, "--python", str(args.python),
    ]
    if value is not None:
        command.extend(("--value", str(value)))
    if args.resume:
        command.append("--resume")
    if args.dry_run:
        command.append("--dry-run")
    print("[QEWC v4 grid] " + " ".join(command), flush=True)
    if not getattr(args, 'gpus', None) or args.dry_run:
        subprocess.run(command, cwd=REPO_ROOT, check=True)
        return
    from scripts.experiments.qewc_v4.retain_checkpoints import prune
    folder = REPO_ROOT / 'logs/qewc_v4/orchestration/htnet_dualgpu'
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / f'{kind}_{name}.log').open('a') as log:
        process = subprocess.Popen(command, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT)
        while True:
            try:
                code = process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                code = None
            prune(root / kind / name / 'training', folder / f'{kind}_{name}.pruned.jsonl')
            if code is not None:
                if code:
                    raise subprocess.CalledProcessError(code, command)
                break


def run_jobs(args, jobs):
    gpus = getattr(args, 'gpus', None)
    if not gpus:
        for job in jobs:
            run_candidate(args, *job)
        return
    remaining = Queue()
    for job in jobs[len(gpus):]:
        remaining.put(job)
    def worker(gpu, first):
        local = copy.copy(args)
        local.gpu = gpu
        run_candidate(local, *first)
        while True:
            try:
                job = remaining.get_nowait()
            except Empty:
                return
            run_candidate(local, *job)
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures = [pool.submit(worker, gpu, first) for gpu, first in zip(gpus, jobs)]
        for future in futures:
            future.result()


def run_selector(
    args,
    root: Path,
    kind: str,
    candidates: list[tuple[str, float]],
    output: Path,
    *,
    phase: str = "coarse",
    boundary_extended: bool = False,
) -> dict:
    baseline_evaluation, _ = candidate_paths(root, "seq", "baseline")
    command = [
        str(args.python), "-u", str(SELECTOR), "--kind", kind,
        "--baseline-evaluation", str(baseline_evaluation),
        "--phase", phase, "--output", str(output),
    ]
    if boundary_extended:
        command.append("--boundary-already-extended")
    for name, value in candidates:
        evaluation, training = candidate_paths(root, kind, name)
        command.extend(("--candidate", f"{name}:{value}:{evaluation}:{training}"))
    print("[QEWC v4 grid] " + " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)
    return json.loads(output.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", choices=("htnet_a101", "ising_fast"), required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument('--gpus', help='Parallel candidate GPUs, e.g. 0,1 (HTNet only)')
    parser.add_argument('--adopt-pid', type=int, help='Existing HTNet lambda_0p1 candidate PID on first GPU')
    parser.add_argument("--python", type=Path, default=PYTHON)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.gpus:
        args.gpus = args.gpus.split(',')
        if args.architecture != 'htnet_a101' or len(set(args.gpus)) != len(args.gpus):
            parser.error('Parallel search requires HTNet and distinct GPUs')
        args.gpu = args.gpus[0]
    return args


def main() -> None:
    args = parse_args()
    if Path(sys.prefix).resolve() != PYTHON.parent.parent.resolve():
        raise RuntimeError(f"wrong Python environment: {sys.prefix}")
    root = REPO_ROOT / "outputs" / "qewc_v4" / "calibration" / args.architecture
    root.mkdir(parents=True, exist_ok=True)
    run_candidate(args, "seq", "baseline")
    ewc_candidates = []
    for value in COARSE:
        name = value_name("lambda", value)
        ewc_candidates.append((name, value))
    replay_candidates = []
    for value in REPLAY:
        name = value_name("ratio", value)
        replay_candidates.append((name, value))
    run_jobs(args, [('ewc', name, value) for name, value in ewc_candidates]
             + [('replay', name, value) for name, value in replay_candidates])
    if args.dry_run:
        print("[QEWC v4 grid] dry-run stops before data-dependent EWC refinement")
        return

    ewc_dir = root / "ewc"
    coarse_path = ewc_dir / "selection_coarse.json"
    selection = run_selector(args, root, "ewc", ewc_candidates, coarse_path)
    boundary_extended = False
    if selection["next_step"]["action"] == "extend_one_decade":
        value = float(selection["next_step"]["candidate_values"][0])
        name = value_name("lambda", value)
        run_candidate(args, "ewc", name, value)
        ewc_candidates.append((name, value))
        boundary_extended = True
        selection = run_selector(
            args, root, "ewc", ewc_candidates,
            ewc_dir / "selection_coarse_extended.json",
            boundary_extended=True,
        )
    if selection["next_step"]["action"] != "refine_geometric_midpoints":
        raise RuntimeError(f"unexpected EWC next step: {selection['next_step']}")
    refinement_jobs = []
    for value in selection["next_step"]["candidate_values"]:
        value = float(value)
        name = value_name("lambda", value)
        if not any(existing == name for existing, _ in ewc_candidates):
            refinement_jobs.append(('ewc', name, value))
            ewc_candidates.append((name, value))
    run_jobs(args, refinement_jobs)
    final_ewc_path = ewc_dir / "selection_final.json"
    final_ewc = run_selector(
        args, root, "ewc", ewc_candidates, final_ewc_path,
        phase="refined", boundary_extended=boundary_extended,
    )
    replay_path = root / "replay" / "selection_final.json"
    final_replay = run_selector(args, root, "replay", replay_candidates, replay_path)
    fisher_path = root / "fisher" / "selection.json"
    capacity_path = root / "replay_capacity" / "capacity_selection.json"
    fisher = json.loads(fisher_path.read_text(encoding="utf-8"))
    capacity = json.loads(capacity_path.read_text(encoding="utf-8"))
    selected_fisher = int(fisher["selected_fisher_samples"])
    initial_fisher = root / "fisher" / f"T0_n{selected_fisher}.pt"
    summary = {
        "architecture": args.architecture,
        "budget_selection": str(REPO_ROOT / "experiment/qewc-v4/budget_decision.json"),
        "budget_B": json.loads((REPO_ROOT / "experiment/qewc-v4/budget_decision.json").read_text())["selected_B"],
        "fisher_samples": selected_fisher,
        "initial_fisher_path": str(initial_fisher.relative_to(REPO_ROOT)),
        "initial_fisher_sha256": sha256(initial_fisher),
        "ewc_state_bytes": initial_fisher.stat().st_size,
        "selected_lambda": float(final_ewc["selected"]["value"]),
        "ewc_selection": str(final_ewc_path),
        "selected_replay_ratio": float(final_replay["selected"]["value"]),
        "replay_selection": str(replay_path),
        "replay_capacity_samples": int(capacity["capacity_samples"]),
        "replay_target_bytes": int(capacity["target_bytes"]),
        "replay_byte_tolerance": float(capacity["byte_tolerance"]),
        "coarse_grid": list(COARSE),
        "coarse_boundary_extended": boundary_extended,
        "ewc_candidates_evaluated": [
            {"name": name, "value": value} for name, value in ewc_candidates
        ],
        "replay_candidates_evaluated": [
            {"name": name, "value": value} for name, value in replay_candidates
        ],
    }
    (root / "calibration_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[QEWC v4] calibration complete: {root / 'calibration_summary.json'}")


if __name__ == "__main__":
    main()
