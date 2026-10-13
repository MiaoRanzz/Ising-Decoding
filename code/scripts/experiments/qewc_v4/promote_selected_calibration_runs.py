#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Promote exact selected calibration training runs into six V4 formal runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
MAPPING = {
    "htnet_a101": {"seq": "H_SEQ", "ewc": "H_EWC", "replay": "H_REPLAY"},
    "ising_fast": {"seq": "I_SEQ", "ewc": "I_EWC", "replay": "I_REPLAY"},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--frozen-manifest", type=Path,
        default=Path("outputs/qewc_v4/frozen_experiment_manifest.yaml"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frozen = args.frozen_manifest if args.frozen_manifest.is_absolute() else REPO_ROOT / args.frozen_manifest
    if not frozen.is_file():
        raise FileNotFoundError(frozen)
    promoted = []
    for architecture, run_ids in MAPPING.items():
        root = REPO_ROOT / "outputs" / "qewc_v4" / "calibration" / architecture
        summary_path = root / "calibration_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        selected_names = {
            "seq": "baseline",
            "ewc": json.loads(Path(summary["ewc_selection"]).read_text(encoding="utf-8"))["selected"]["name"],
            "replay": json.loads(Path(summary["replay_selection"]).read_text(encoding="utf-8"))["selected"]["name"],
        }
        for kind, run_id in run_ids.items():
            if summary.get('parameter_source_architecture') == 'htnet_a101' and architecture == 'ising_fast' and kind != 'seq':
                # Ising search was explicitly stopped; train these formal runs
                # with transferred HTNet parameters, never promote unrelated candidates.
                continue
            source = root / kind / selected_names[kind] / "training" / "run_manifest.json"
            training = json.loads(source.read_text(encoding="utf-8"))
            expected_method = kind
            if training.get("method") != expected_method:
                raise RuntimeError(f"method mismatch for {source}")
            training["formal_promotion"] = {
                "source_training_manifest": str(source),
                "source_training_manifest_sha256": sha256(source),
                "selection_used_only_separate_id_validation": True,
                "calibration_stream_seeds": [20260914, 20260915],
                "formal_test_stream_seeds": [20260916, 20260917],
                "no_ood_or_willow_used": True,
                "weights_reused_without_retraining": True,
                "reason": "Exact selected train-seed run; retraining would duplicate the same deterministic training stream.",
            }
            destination = REPO_ROOT / "outputs" / "qewc_v4" / "formal" / run_id / "run_manifest.json"
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(training, indent=2) + "\n", encoding="utf-8")
            promoted.append({
                "run_id": run_id,
                "architecture": architecture,
                "method": kind,
                "source": str(source),
                "destination": str(destination),
                "destination_sha256": sha256(destination),
            })
    output = REPO_ROOT / "outputs" / "qewc_v4" / "formal" / "promotion_manifest.json"
    output.write_text(json.dumps({
        "frozen_manifest": {"path": str(frozen), "sha256": sha256(frozen)},
        "promoted_runs": promoted,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"[QEWC v4] promoted {len(promoted)} selected runs: {output}")


if __name__ == "__main__":
    main()
