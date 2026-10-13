#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare the four single-task B-epoch Oracles for one V4 architecture."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[4]
CODE_ROOT = REPO_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from scripts.experiments.qewc_v4.evaluation_common import (  # noqa: E402
    ARCHITECTURES,
    TASKS,
)


RUN_CONTINUAL = REPO_ROOT / "code" / "scripts" / "experiments" / "qewc_v4" / "run_continual.py"
DEFAULT_MANIFEST = REPO_ROOT / "outputs" / "qewc_v4" / "frozen_experiment_manifest.yaml"
EXPECTED_PREFIX = Path("/root/miniconda3/envs/ising-decoding")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def htnet_source(task: str, budget: int) -> Path:
    return (
        REPO_ROOT
        / "outputs"
        / "qewc_v4"
        / "budget_pilot"
        / "htnet"
        / f"{task}_e{budget}"
        / "run_manifest.json"
    )


def oracle_root(architecture: str, task: str) -> Path:
    return REPO_ROOT / "outputs" / "qewc_v4" / "oracles" / architecture / task


def promote_htnet(task: str, budget: int, *, dry_run: bool) -> dict:
    source = htnet_source(task, budget)
    destination = oracle_root("htnet_a101", task) / "run_manifest.json"
    if dry_run:
        return {"task": task, "source": str(source), "destination": str(destination)}
    training = json.loads(source.read_text(encoding="utf-8"))
    if training.get("method") != "seq" or training.get("order") != [task]:
        raise RuntimeError(f"invalid HTNet budget-pilot Oracle source: {source}")
    endpoint = Path(training["endpoints"]["C1"]["path"])
    if not endpoint.is_file() or training["endpoints"]["C1"]["sha256"] != sha256(endpoint):
        raise RuntimeError(f"invalid HTNet Oracle endpoint: {endpoint}")
    training["oracle"] = {
        "task": task,
        "architecture": "htnet_a101",
        "budget_B": budget,
        "promoted_from_budget_pilot": True,
        "source_manifest": str(source),
        "source_manifest_sha256": sha256(source),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(training, indent=2) + "\n", encoding="utf-8")
    return {
        "task": task,
        "manifest": str(destination),
        "manifest_sha256": sha256(destination),
        "endpoint": str(endpoint),
        "endpoint_sha256": sha256(endpoint),
        "promoted": True,
    }


def train_ising(args: argparse.Namespace, task: str, budget: int) -> dict:
    spec = ARCHITECTURES["ising_fast"]
    run_name = f"qewc_v4/oracles/ising_fast/{task}"
    command = [
        str(args.python),
        "-u",
        str(RUN_CONTINUAL),
        "--method",
        "seq",
        "--run-name",
        run_name,
        "--base-task",
        f"T0:{spec.base_config}",
        "--task",
        f"{task}:{spec.task_config_dir}/{task}",
        "--order",
        task,
        "--expected-stages",
        "1",
        "--stage0",
        str(args.stage0),
        "--epochs-per-task",
        str(budget),
        "--requested-samples-per-epoch",
        "4194304",
        "--batch-size",
        "2048",
        "--accumulate-steps",
        "2",
        "--checkpoint-prefix",
        spec.checkpoint_prefix,
        "--validation-samples",
        "65536",
        "--seed",
        "20260907",
        "--gpu",
        args.gpu,
        "--python",
        str(args.python),
    ]
    if args.resume:
        command.append("--resume")
    print("[QEWC v4 Ising Oracle] " + " ".join(command), flush=True)
    if args.dry_run:
        return {"task": task, "command": command}
    subprocess.run(command, cwd=REPO_ROOT, check=True)
    path = oracle_root("ising_fast", task) / "run_manifest.json"
    training = json.loads(path.read_text(encoding="utf-8"))
    endpoint = Path(training["endpoints"]["C1"]["path"])
    training["oracle"] = {
        "task": task,
        "architecture": "ising_fast",
        "budget_B": budget,
        "promoted_from_budget_pilot": False,
    }
    path.write_text(json.dumps(training, indent=2) + "\n", encoding="utf-8")
    return {
        "task": task,
        "manifest": str(path),
        "manifest_sha256": sha256(path),
        "endpoint": str(endpoint),
        "endpoint_sha256": sha256(endpoint),
        "promoted": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", choices=tuple(ARCHITECTURES), required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--python", type=Path, default=EXPECTED_PREFIX / "bin" / "python")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if Path(sys.prefix).resolve() != EXPECTED_PREFIX.resolve():
        raise RuntimeError(f"wrong Python environment: {sys.prefix}")
    manifest = OmegaConf.load(args.manifest)
    budget = int(manifest.budget_pilot.selected_B)
    args.stage0 = REPO_ROOT / str(manifest.stage0[args.architecture].endpoint.path)
    records = []
    for task in TASKS[1:]:
        records.append(
            promote_htnet(task, budget, dry_run=args.dry_run)
            if args.architecture == "htnet_a101"
            else train_ising(args, task, budget)
        )
    if args.dry_run:
        return
    output = REPO_ROOT / "outputs" / "qewc_v4" / "oracles" / args.architecture / "oracle_manifest.json"
    output.write_text(
        json.dumps(
            {"architecture": args.architecture, "budget_B": budget, "oracles": records},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[QEWC v4] four {args.architecture} Oracles ready: {output}")


if __name__ == "__main__":
    main()
