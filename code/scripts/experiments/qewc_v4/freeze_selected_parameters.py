#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Materialize a frozen V4 manifest after both architecture calibrations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[4]
SOURCE = REPO_ROOT / "experiment" / "qewc-v4" / "experiment_manifest.yaml"
OUTPUT = REPO_ROOT / "outputs" / "qewc_v4" / "frozen_experiment_manifest.yaml"
FORBIDDEN_LAMBDA = 43.87539484535376
ARCHITECTURES = ("htnet_a101", "ising_fast")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source if args.source.is_absolute() else REPO_ROOT / args.source
    output = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    manifest = OmegaConf.load(source)
    budget_path = REPO_ROOT / "experiment" / "qewc-v4" / "budget_decision.json"
    budget = json.loads(budget_path.read_text(encoding="utf-8"))
    if budget.get("status") != "selected" or budget.get("selected_B") is None:
        raise RuntimeError("cannot freeze before HTNet budget B is selected")
    ewc_architectures = {}
    replay_architectures = {}
    inputs = {"source_manifest": {"path": str(source), "sha256": sha256(source)},
              "budget": {"path": str(budget_path), "sha256": sha256(budget_path)}}
    for architecture in ARCHITECTURES:
        path = REPO_ROOT / "outputs" / "qewc_v4" / "calibration" / architecture / "calibration_summary.json"
        summary = json.loads(path.read_text(encoding="utf-8"))
        if summary.get("budget_B") != budget["selected_B"]:
            raise RuntimeError(f"calibration budget mismatch: {path}")
        if summary.get("architecture") != architecture:
            raise RuntimeError(f"calibration architecture mismatch: {path}")
        selected_lambda = float(summary["selected_lambda"])
        if selected_lambda == FORBIDDEN_LAMBDA:
            raise RuntimeError("forbidden V3 lambda cannot be frozen in V4")
        fisher_path = REPO_ROOT / summary["initial_fisher_path"]
        if not fisher_path.is_file() or sha256(fisher_path) != summary["initial_fisher_sha256"]:
            raise RuntimeError(f"missing or corrupt selected Fisher: {fisher_path}")
        ewc_architectures[architecture] = {
            "selected_lambda": selected_lambda,
            "fisher_samples": int(summary["fisher_samples"]),
            "initial_fisher_path": summary["initial_fisher_path"],
            "initial_fisher_sha256": summary["initial_fisher_sha256"],
            "state_bytes": int(summary["ewc_state_bytes"]),
            "positive_weight": 1.0,
            "fisher_batch_size": 2048,
            "per_sample_chunk_size": 16,
            "fisher_seed": 20260912,
            "selection": summary["ewc_selection"],
            "parameter_source_architecture": summary.get('parameter_source_architecture', architecture),
        }
        replay_architectures[architecture] = {
            "selected_ratio": float(summary["selected_replay_ratio"]),
            "capacity_samples": int(summary["replay_capacity_samples"]),
            "target_bytes": int(summary["replay_target_bytes"]),
            "byte_tolerance": float(summary["replay_byte_tolerance"]),
            "seed": 20260913,
            "selection": summary["replay_selection"],
            "parameter_source_architecture": summary.get('parameter_source_architecture', architecture),
        }
        inputs[architecture] = {"path": str(path), "sha256": sha256(path)}
    manifest.status = "calibration_completed_ready_for_formal"
    manifest.budget_pilot.status = "completed"
    manifest.budget_pilot.selected_B = int(budget["selected_B"])
    manifest.budget_pilot.selection = {"path": str(budget_path), "sha256": sha256(budget_path)}
    manifest.ewc_calibration.status = "completed"
    manifest.ewc_calibration.architectures = ewc_architectures
    manifest.replay_calibration.status = "completed"
    manifest.replay_calibration.architectures = replay_architectures
    manifest.formal_training.status = "ready"
    manifest.freeze.budget_B = True
    manifest.freeze.ewc_parameters = True
    manifest.freeze.replay_parameters = True
    manifest.freeze_materialization = {"inputs": inputs}
    output.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(manifest, output)
    sidecar = output.with_suffix(".json")
    sidecar.write_text(json.dumps({
        "manifest": {"path": str(output), "sha256": sha256(output)},
        "inputs": inputs,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"[QEWC v4] frozen manifest: {output}")


if __name__ == "__main__":
    main()
