# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Google Quantum AI QEC benchmark dataset integration.

The source dataset is the Zenodo record for "Quantum error correction below the
surface code threshold". This module deliberately treats the data as an
external benchmark archive: the files are multi-GB zip archives with their own
README files and are not committed to this repository.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence


GOOGLE_QEC_RECORD_ID = 13273331
GOOGLE_QEC_RECORD_URL = f"https://zenodo.org/api/records/{GOOGLE_QEC_RECORD_ID}"
GOOGLE_QEC_RECORD_HTML = f"https://zenodo.org/records/{GOOGLE_QEC_RECORD_ID}"

DEFAULT_BENCHMARK_KEY = "google_105Q_surface_code_d3_d5_d7.zip"


@dataclass(frozen=True)
class GoogleQECFile:
    key: str
    size_bytes: int
    md5: str
    url: str
    code_family: str
    distances: tuple[int, ...]


@dataclass(frozen=True)
class GoogleQECManifest:
    record_id: int
    title: str
    license_id: str
    record_url: str
    files: tuple[GoogleQECFile, ...]

    def by_key(self) -> dict[str, GoogleQECFile]:
        return {entry.key: entry for entry in self.files}


@dataclass(frozen=True)
class DownloadItem:
    entry: GoogleQECFile
    path: Path
    exists: bool


@dataclass(frozen=True)
class DownloadPlan:
    root: Path
    items: tuple[DownloadItem, ...]
    required_bytes: int


@dataclass(frozen=True)
class GoogleQECIndex:
    root: Path
    manifest_path: Path | None
    archives: dict[str, Path]
    extracted_dirs: dict[str, Path]


def _infer_code_family(key: str) -> str:
    if "surface_code" in key:
        return "surface"
    if "repetition_code" in key:
        return "repetition"
    return "unknown"


def _infer_distances(key: str) -> tuple[int, ...]:
    stem = key.removesuffix(".zip")
    values = []
    for part in stem.split("_"):
        if len(part) > 1 and part[0] == "d" and part[1:].isdigit():
            values.append(int(part[1:]))
    return tuple(values)


def parse_zenodo_record(record: dict) -> GoogleQECManifest:
    """Parse the Zenodo API response into a stable local manifest."""

    files = []
    for file_info in record.get("files", []):
        checksum = str(file_info.get("checksum", ""))
        if not checksum.startswith("md5:"):
            raise ValueError(f"Unsupported checksum for {file_info.get('key')!r}: {checksum!r}")
        key = str(file_info["key"])
        files.append(
            GoogleQECFile(
                key=key,
                size_bytes=int(file_info["size"]),
                md5=checksum.split(":", 1)[1],
                url=str(file_info["links"]["self"]),
                code_family=_infer_code_family(key),
                distances=_infer_distances(key),
            )
        )

    metadata = record.get("metadata", {})
    license_info = metadata.get("license") or {}
    return GoogleQECManifest(
        record_id=int(record["id"]),
        title=str(metadata.get("title", record.get("title", ""))),
        license_id=str(license_info.get("id", "")),
        record_url=str(record.get("links", {}).get("self_html", GOOGLE_QEC_RECORD_HTML)),
        files=tuple(sorted(files, key=lambda entry: entry.size_bytes)),
    )


def fetch_zenodo_manifest(url: str = GOOGLE_QEC_RECORD_URL, timeout: float = 60.0) -> GoogleQECManifest:
    """Fetch and parse the official Zenodo record."""

    with urllib.request.urlopen(url, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return parse_zenodo_record(payload)


def build_download_plan(
    manifest: GoogleQECManifest,
    root: Path,
    keys: Sequence[str] | None = None,
) -> DownloadPlan:
    """Build a concrete download plan without performing network or disk writes."""

    selected_keys = tuple(keys) if keys else (DEFAULT_BENCHMARK_KEY,)
    by_key = manifest.by_key()
    missing = [key for key in selected_keys if key not in by_key]
    if missing:
        raise KeyError(f"Unknown Google QEC benchmark file(s): {missing}")

    root = Path(root)
    items = []
    required = 0
    for key in selected_keys:
        entry = by_key[key]
        path = root / entry.key
        exists = path.exists()
        items.append(DownloadItem(entry=entry, path=path, exists=exists))
        if not exists:
            required += entry.size_bytes
    return DownloadPlan(root=root, items=tuple(items), required_bytes=required)


def ensure_sufficient_space(path: Path, required_bytes: int, margin: float = 1.10) -> None:
    """Raise before starting a large download if the filesystem is too full."""

    if required_bytes <= 0:
        return
    usage = shutil.disk_usage(path)
    needed = int(required_bytes * float(margin))
    if usage.free < needed:
        raise RuntimeError(
            f"Not enough free space under {path}: need at least {needed:,} bytes "
            f"including margin, found {usage.free:,} bytes"
        )


def _md5_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.md5()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def verify_archive(path: Path, entry: GoogleQECFile) -> None:
    if path.stat().st_size != entry.size_bytes:
        raise RuntimeError(
            f"Size mismatch for {path}: expected {entry.size_bytes}, got {path.stat().st_size}"
        )
    got = _md5_file(path)
    if got != entry.md5:
        raise RuntimeError(f"MD5 mismatch for {path}: expected {entry.md5}, got {got}")


def build_download_request(entry: GoogleQECFile, resume_from: int = 0) -> urllib.request.Request:
    """Build a request for a benchmark archive, optionally using HTTP Range."""

    headers = {}
    if int(resume_from) > 0:
        headers["Range"] = f"bytes={int(resume_from)}-"
    return urllib.request.Request(entry.url, headers=headers)


def download_entry(entry: GoogleQECFile, path: Path, force: bool = False) -> Path:
    """Download one benchmark archive and verify size + md5."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        verify_archive(path, entry)
        return path

    tmp_path = path.with_suffix(path.suffix + ".part")
    if force and tmp_path.exists():
        tmp_path.unlink()

    resume_from = tmp_path.stat().st_size if tmp_path.exists() else 0
    if resume_from >= entry.size_bytes:
        tmp_path.replace(path)
        verify_archive(path, entry)
        return path

    request = build_download_request(entry, resume_from=resume_from)
    with urllib.request.urlopen(request, timeout=60.0) as response:
        status = getattr(response, "status", None) or response.getcode()
        mode = "ab" if resume_from > 0 and status == 206 else "wb"
        if mode == "wb":
            resume_from = 0
        with tmp_path.open(mode) as out:
            while True:
                chunk = response.read(16 * 1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
    tmp_path.replace(path)
    verify_archive(path, entry)
    return path


def extract_archive(path: Path, output_dir: Path | None = None) -> Path:
    """Extract a downloaded benchmark zip next to the archive by default."""

    target = output_dir or path.with_suffix("")
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as zf:
        zf.extractall(target)
    return target


class GoogleQECBenchmarkStore:
    """Local project store for Google QEC benchmark archives."""

    def __init__(self, root: Path | str = "benchmarks/google_qec"):
        self.root = Path(root)

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    def write_manifest(self, manifest: GoogleQECManifest) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        payload = asdict(manifest)
        self.manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return self.manifest_path

    def index(self) -> GoogleQECIndex:
        archives = {path.name: path for path in sorted(self.root.glob("*.zip"))}
        extracted_dirs = {
            path.name: path
            for path in sorted(self.root.iterdir()) if path.is_dir() and path.name != "__pycache__"
        } if self.root.exists() else {}
        manifest_path = self.manifest_path if self.manifest_path.exists() else None
        return GoogleQECIndex(
            root=self.root,
            manifest_path=manifest_path,
            archives=archives,
            extracted_dirs=extracted_dirs,
        )

    def download(
        self,
        manifest: GoogleQECManifest,
        keys: Sequence[str] | None = None,
        *,
        force: bool = False,
        extract: bool = False,
        check_space: bool = True,
    ) -> DownloadPlan:
        self.root.mkdir(parents=True, exist_ok=True)
        plan = build_download_plan(manifest, self.root, keys)
        if check_space:
            ensure_sufficient_space(self.root, plan.required_bytes)
        self.write_manifest(manifest)
        for item in plan.items:
            archive_path = download_entry(item.entry, item.path, force=force)
            if extract:
                extract_archive(archive_path)
        return plan


def benchmark_keys(files: Iterable[GoogleQECFile]) -> list[str]:
    return [entry.key for entry in sorted(files, key=lambda entry: (entry.code_family, entry.size_bytes))]

