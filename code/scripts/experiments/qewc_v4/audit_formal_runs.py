#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Audit QEWC-v4 formal training manifests before any result aggregation."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[4]
CODE_ROOT = REPO_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from scripts.experiments.qewc_v4.run_formal import (  # noqa: E402
    MANIFEST_PATH,
    RUNS,
    _calibration,
    _order,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_run(run_id: str, training: dict, experiment) -> dict:
    spec = RUNS[run_id]
    is_mix = spec.method in {"mix_total", "mix_exposure"}
    stages = 5 if spec.method == "mix_exposure" else 4
    updates_per_epoch = int(experiment.legacy_epoch.optimizer_updates)
    budget = int(experiment.budget_pilot.selected_B)
    expected_updates = stages * budget * updates_per_epoch
    expected_samples = expected_updates * int(experiment.legacy_epoch.effective_batch_size)
    errors: list[str] = []
    if training.get("method") != spec.method:
        errors.append(f"method: expected {spec.method}, got {training.get('method')}")
    if not is_mix:
        expected_order = _order(experiment, spec.order_kind)
        if training.get("order") != expected_order:
            errors.append(f"order: expected {expected_order}, got {training.get('order')}")
    if training.get("total_optimizer_steps") != expected_updates:
        errors.append(
            f"optimizer steps: expected {expected_updates}, got {training.get('total_optimizer_steps')}"
        )
    if training.get("samples_processed") != expected_samples:
        errors.append(f"samples: expected {expected_samples}, got {training.get('samples_processed')}")
    entries = training.get("checkpoints" if is_mix else "endpoints", {})
    expected_names = [f"M{i}" for i in range(stages + 1)] if is_mix else [f"C{i}" for i in range(5)]
    if list(entries) != expected_names:
        errors.append(f"checkpoint names: expected {expected_names}, got {list(entries)}")
    for name, item in entries.items():
        path = Path(item["path"])
        if not path.is_file():
            errors.append(f"missing checkpoint {name}: {path}")
        elif sha256(path) != item["sha256"]:
            errors.append(f"checkpoint hash mismatch: {name}")
    parameters = training.get("method_parameters", {})
    if spec.method == "ewc":
        frozen = _calibration(experiment, "ewc_calibration", spec.architecture)
        if float(parameters.get("lambda", -1)) != float(frozen.selected_lambda):
            errors.append("EWC lambda does not match frozen calibration")
        if int(parameters.get("fisher_samples", -1)) != int(frozen.fisher_samples):
            errors.append("EWC Fisher sample count does not match frozen calibration")
    if spec.method == "replay":
        frozen = _calibration(experiment, "replay_calibration", spec.architecture)
        if float(parameters.get("ratio", -1)) != float(frozen.selected_ratio):
            errors.append("Replay ratio does not match frozen calibration")
        if int(parameters.get("capacity", -1)) != int(frozen.capacity_samples):
            errors.append("Replay capacity does not match frozen calibration")
        if int(parameters.get("target_bytes", -1)) != int(frozen.target_bytes):
            errors.append("Replay byte target does not match frozen EWC state size")
    return {
        "run_id": run_id,
        "architecture": spec.architecture,
        "status": "passed" if not errors else "failed",
        "method": training.get("method"),
        "optimizer_steps": training.get("total_optimizer_steps"),
        "samples_processed": training.get("samples_processed"),
        "checkpoint_names": list(entries),
        "errors": errors,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", action="append", choices=tuple(RUNS), required=True)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment = OmegaConf.load(args.manifest)
    reports = []
    for run_id in args.run_id:
        path = REPO_ROOT / "outputs" / "qewc_v4" / "formal" / run_id / "run_manifest.json"
        reports.append(audit_run(run_id, json.loads(path.read_text(encoding="utf-8")), experiment))
    payload = {
        "status": "passed" if all(row["status"] == "passed" for row in reports) else "failed",
        "runs": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"[QEWC v4] formal audit {payload['status']}: {args.output}")
    if payload["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
