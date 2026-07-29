#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Download Google Quantum AI QEC benchmark archives through the public downloader."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DOWNLOADER = REPO_ROOT / "code" / "scripts" / "download_google_qec_benchmark.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/google_qec"),
    )
    parser.add_argument("--file", action="append", dest="files")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-space-check", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    command = [
        sys.executable,
        str(DOWNLOADER),
        "--output-dir",
        str(args.output_dir),
    ]
    for value in args.files or ():
        command.extend(("--file", value))
    for enabled, flag in (
        (args.all, "--all"),
        (args.list, "--list"),
        (args.manifest_only, "--manifest-only"),
        (args.extract, "--extract"),
        (args.force, "--force"),
        (args.skip_space_check, "--skip-space-check"),
    ):
        if enabled:
            command.append(flag)
    if args.dry_run:
        print("[dry-run] " + shlex.join(command))
        return 0
    return subprocess.run(command, cwd=REPO_ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
