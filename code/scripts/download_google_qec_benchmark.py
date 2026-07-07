#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Download Google Quantum AI QEC benchmark archives from Zenodo."""

from __future__ import annotations

import argparse
from pathlib import Path

from benchmarks.google_qec import (
    DEFAULT_BENCHMARK_KEY,
    GoogleQECBenchmarkStore,
    benchmark_keys,
    build_download_plan,
    fetch_zenodo_manifest,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/google_qec"),
        help="Directory for manifest and downloaded zip archives.",
    )
    parser.add_argument(
        "--file",
        action="append",
        dest="files",
        help=(
            "Zenodo file key to download. May be repeated. "
            f"Default: {DEFAULT_BENCHMARK_KEY}"
        ),
    )
    parser.add_argument("--all", action="store_true", help="Download all Google QEC archives.")
    parser.add_argument("--list", action="store_true", help="List available archives and exit.")
    parser.add_argument("--manifest-only", action="store_true", help="Only write manifest.json.")
    parser.add_argument("--extract", action="store_true", help="Extract downloaded zip archives.")
    parser.add_argument("--force", action="store_true", help="Re-download archives that already exist.")
    parser.add_argument("--skip-space-check", action="store_true", help="Skip free-space guard.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    manifest = fetch_zenodo_manifest()
    store = GoogleQECBenchmarkStore(args.output_dir)

    if args.list:
        for entry in manifest.files:
            gib = entry.size_bytes / (1024**3)
            distances = ",".join(str(d) for d in entry.distances) or "unknown"
            print(f"{entry.key}\t{gib:.2f} GiB\t{entry.code_family}\td={distances}")
        return 0

    if args.all:
        keys = benchmark_keys(manifest.files)
    else:
        keys = tuple(args.files) if args.files else (DEFAULT_BENCHMARK_KEY,)

    store.write_manifest(manifest)
    plan = build_download_plan(manifest, args.output_dir, keys)
    print(f"Google QEC Zenodo record: {manifest.record_url}")
    print(f"Output directory: {args.output_dir}")
    for item in plan.items:
        status = "exists" if item.exists else "download"
        gib = item.entry.size_bytes / (1024**3)
        print(f"  [{status}] {item.entry.key} ({gib:.2f} GiB, md5={item.entry.md5})")

    if args.manifest_only:
        print(f"Wrote manifest: {store.manifest_path}")
        return 0

    store.download(
        manifest,
        keys,
        force=args.force,
        extract=args.extract,
        check_space=not args.skip_space_check,
    )
    print("Download complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
