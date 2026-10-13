#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build a Replay buffer whose serialized bytes match one EWC state."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
BUILDER = REPO_ROOT / "code" / "scripts" / "experiments" / "qewc_v3" / "build_balanced_replay.py"
PYTHON = Path("/root/miniconda3/envs/ising-decoding/bin/python")
REFERENCE_CAPACITY = 594
REFERENCE_BYTES = 5229158
CONFIGS = {
    "htnet_a101": "experiments/qewc_v4/htnet/T0",
    "ising_fast": "experiments/qewc_v4/ising/t0_base",
}


def estimated_capacity(target_bytes: int) -> int:
    if target_bytes <= 0:
        raise ValueError("target bytes must be positive")
    return max(1, round(target_bytes * REFERENCE_CAPACITY / REFERENCE_BYTES))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", choices=tuple(CONFIGS), required=True)
    parser.add_argument("--fisher-state", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--capacity", type=int)
    parser.add_argument("--byte-tolerance", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--python", type=Path, default=PYTHON)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fisher = args.fisher_state if args.fisher_state.is_absolute() else REPO_ROOT / args.fisher_state
    output = args.output_dir if args.output_dir.is_absolute() else REPO_ROOT / args.output_dir
    if not fisher.is_file():
        raise FileNotFoundError(fisher)
    target_bytes = fisher.stat().st_size
    capacity = args.capacity or estimated_capacity(target_bytes)
    command = [
        str(args.python), "-u", str(BUILDER),
        "--task", f"T0:{CONFIGS[args.architecture]}",
        "--capacity", str(capacity), "--output-dir", str(output),
        "--seed", str(args.seed), "--device", "cuda:0",
        "--target-bytes", str(target_bytes),
        "--byte-tolerance", str(args.byte_tolerance),
    ]
    print("[QEWC v4 Replay capacity] " + " ".join(command), flush=True)
    if args.dry_run:
        return
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["PYTHONPATH"] = str(REPO_ROOT / "code")
    subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)
    builder_manifest = output / "balanced_manifest.json"
    replay = json.loads(builder_manifest.read_text(encoding="utf-8"))
    actual_bytes = int(replay["serialized_bytes"])
    relative_error = abs(actual_bytes - target_bytes) / target_bytes
    if relative_error > args.byte_tolerance:
        raise RuntimeError(
            f"Replay/EWC byte mismatch {relative_error:.3%} exceeds {args.byte_tolerance:.3%}"
        )
    payload = {
        "architecture": args.architecture,
        "capacity_samples": capacity,
        "target_bytes": target_bytes,
        "serialized_bytes": actual_bytes,
        "relative_error": relative_error,
        "byte_tolerance": args.byte_tolerance,
        "fisher_state": {"path": str(fisher), "sha256": sha256(fisher)},
        "replay_manifest": {"path": str(builder_manifest), "sha256": sha256(builder_manifest)},
    }
    (output / "capacity_selection.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[QEWC v4] Replay capacity={capacity}, byte error={relative_error:.3%}")


if __name__ == "__main__":
    main()
