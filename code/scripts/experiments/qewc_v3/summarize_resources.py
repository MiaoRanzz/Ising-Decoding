#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Aggregate compute and method-state storage for completed QEWC-v3 runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]


def resource_paths(training: dict) -> list[Path]:
    raw = training["resource_metrics"]
    values = raw.values() if isinstance(raw, dict) else (raw,)
    return [Path(value) for value in values]


def summarize_run(run_id: str, training: dict) -> dict:
    epochs = []
    for path in resource_paths(training):
        payload = json.loads(path.read_text(encoding="utf-8"))
        epochs.extend(payload["epochs"])
    parameters = training.get("method_parameters", {})
    fisher_seconds = sum(float(item["wall_seconds"]) for item in parameters.get("fisher_timings", []))
    replay_rebuild_seconds = sum(
        float(item["wall_seconds"]) for item in parameters.get("rebuild_timings", [])
    )
    if training["method"] == "ewc":
        extra_storage_bytes = int(parameters["online_state_bytes"])
    elif training["method"] == "replay":
        extra_storage_bytes = int(parameters["final_buffer_manifest"]["serialized_bytes"])
    else:
        extra_storage_bytes = 0
    training_gpu_hours = sum(float(item["gpu_hours"]) for item in epochs)
    auxiliary_gpu_hours = (fisher_seconds + replay_rebuild_seconds) / 3600.0
    return {
        "run_id": run_id,
        "method": training["method"],
        "stages": len(epochs),
        "optimizer_steps": int(training["total_optimizer_steps"]),
        "samples_processed": int(training["samples_processed"]),
        "training_gpu_hours": training_gpu_hours,
        "fisher_seconds": fisher_seconds,
        "replay_rebuild_seconds": replay_rebuild_seconds,
        "total_gpu_hours_including_auxiliary": training_gpu_hours + auxiliary_gpu_hours,
        "peak_allocated_bytes": max(int(item["peak_allocated_bytes_max_rank"]) for item in epochs),
        "peak_reserved_bytes": max(int(item["peak_reserved_bytes_max_rank"]) for item in epochs),
        "extra_method_storage_bytes": extra_storage_bytes,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", action="append")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    formal_root = REPO_ROOT / "outputs" / "qewc_v3" / "formal"
    run_ids = args.run_id or sorted(path.parent.name for path in formal_root.glob("*/run_manifest.json"))
    rows = []
    for run_id in run_ids:
        path = formal_root / run_id / "run_manifest.json"
        rows.append(summarize_run(run_id, json.loads(path.read_text(encoding="utf-8"))))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "resources.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    with (args.output_dir / "resources.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[QEWC v3] resource summaries: {args.output_dir}")


if __name__ == "__main__":
    main()
