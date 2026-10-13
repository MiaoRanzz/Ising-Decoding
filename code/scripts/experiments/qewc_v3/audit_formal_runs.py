#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Audit completed formal QEWC-v3 run manifests before evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[4]
CODE_ROOT = REPO_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from scripts.experiments.qewc_v3.run_formal import RUNS, load_track




def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_run(run_id: str, training: dict, experiment) -> dict:
    spec = RUNS[run_id]
    _, expected_order = load_track(spec)
    is_mix = spec.method in {"mix", "mix_exposure"}
    expected_method = "mix_total" if spec.method == "mix" else spec.method
    stages = 4 if is_mix else len(expected_order)
    step_multiplier = 5 if spec.method == "mix_exposure" else stages
    expected_steps = step_multiplier * int(experiment.pilot.selected_optimizer_steps_per_task)
    errors = []
    if training.get("method") != expected_method:
        errors.append(f"method: expected {expected_method}, got {training.get('method')}")
    if not is_mix and training.get("order") != expected_order:
        errors.append(f"order: expected {expected_order}, got {training.get('order')}")
    if training.get("total_optimizer_steps") != expected_steps:
        errors.append(f"optimizer steps: expected {expected_steps}, got {training.get('total_optimizer_steps')}")
    expected_samples = expected_steps * int(experiment.pilot.batch_size_per_gpu)
    if training.get("samples_processed") != expected_samples:
        errors.append(f"samples: expected {expected_samples}, got {training.get('samples_processed')}")

    entries = training.get("checkpoints" if is_mix else "endpoints", {})
    expected_names = [f"M{i}" for i in range(5)] if is_mix else [f"C{i}" for i in range(stages + 1)]
    if list(entries) != expected_names:
        errors.append(f"checkpoint names: expected {expected_names}, got {list(entries)}")
    for name, item in entries.items():
        path = Path(item["path"])
        if not path.is_file():
            errors.append(f"missing checkpoint {name}: {path}")
        elif sha256(path) != item["sha256"]:
            errors.append(f"checkpoint hash mismatch: {name}")

    parameters = training.get("method_parameters", {})
    if spec.method == "ewc" and float(parameters.get("lambda", -1)) != float(experiment.ewc.get("lambda")):
        errors.append("EWC lambda does not match frozen manifest")
    if spec.method == "replay":
        if float(parameters.get("ratio", -1)) != float(experiment.replay.get("ratio")):
            errors.append("Replay ratio does not match frozen manifest")
        if int(parameters.get("capacity", -1)) != int(experiment.replay.capacity_samples):
            errors.append("Replay capacity does not match frozen manifest")
    return {
        "run_id": run_id,
        "status": "passed" if not errors else "failed",
        "method": training.get("method"),
        "order": training.get("order", []),
        "stages": stages,
        "optimizer_steps": training.get("total_optimizer_steps"),
        "samples_processed": training.get("samples_processed"),
        "checkpoint_names": list(entries),
        "errors": errors,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", action="append", choices=tuple(RUNS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment = OmegaConf.load(REPO_ROOT / "experiment" / "qewc-v3" / "experiment_manifest.yaml")
    reports = []
    for run_id in args.run_id:
        path = REPO_ROOT / "outputs" / "qewc_v3" / "formal" / run_id / "run_manifest.json"
        training = json.loads(path.read_text(encoding="utf-8"))
        reports.append(audit_run(run_id, training, experiment))
    payload = {"status": "passed" if all(row["status"] == "passed" for row in reports) else "failed", "runs": reports}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"[QEWC v3] formal audit {payload['status']}: {args.output}")
    if payload["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
