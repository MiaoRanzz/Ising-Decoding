# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.google_qec import (
    DEFAULT_BENCHMARK_KEY,
    GOOGLE_QEC_RECORD_ID,
    GoogleQECBenchmarkStore,
    build_download_plan,
    build_download_request,
    parse_zenodo_record,
)


def _fake_record():
    return {
        "id": GOOGLE_QEC_RECORD_ID,
        "metadata": {
            "title": 'Data for "Quantum error correction below the surface code threshold"',
            "license": {
                "id": "cc-by-4.0"
            },
        },
        "files": [
            {
                "key": DEFAULT_BENCHMARK_KEY,
                "size": 5716907033,
                "checksum": "md5:21fa6ad35b395d838ebcdbc92e364a12",
                "links": {
                    "self":
                        "https://zenodo.org/api/records/13273331/files/"
                        "google_105Q_surface_code_d3_d5_d7.zip/content"
                },
            }
        ],
    }


class TestGoogleQECBenchmarkManifest(unittest.TestCase):

    def test_parse_zenodo_record_extracts_official_file_metadata(self):
        manifest = parse_zenodo_record(_fake_record())

        self.assertEqual(manifest.record_id, GOOGLE_QEC_RECORD_ID)
        self.assertEqual(manifest.license_id, "cc-by-4.0")
        self.assertEqual(len(manifest.files), 1)
        entry = manifest.files[0]
        self.assertEqual(entry.key, DEFAULT_BENCHMARK_KEY)
        self.assertEqual(entry.size_bytes, 5716907033)
        self.assertEqual(entry.md5, "21fa6ad35b395d838ebcdbc92e364a12")
        self.assertEqual(entry.code_family, "surface")
        self.assertEqual(entry.distances, (3, 5, 7))

    def test_build_download_plan_defaults_to_smallest_surface_archive(self):
        manifest = parse_zenodo_record(_fake_record())
        with tempfile.TemporaryDirectory() as tmp:
            plan = build_download_plan(manifest, Path(tmp), keys=None)

        self.assertEqual([item.entry.key for item in plan.items], [DEFAULT_BENCHMARK_KEY])
        self.assertEqual(plan.required_bytes, 5716907033)

    def test_build_download_request_uses_range_for_partial_file(self):
        manifest = parse_zenodo_record(_fake_record())
        entry = manifest.files[0]

        request = build_download_request(entry, resume_from=1024)

        self.assertEqual(request.full_url, entry.url)
        self.assertEqual(request.headers["Range"], "bytes=1024-")


class TestGoogleQECBenchmarkStore(unittest.TestCase):

    def test_store_writes_manifest_and_indexes_local_archive(self):
        manifest = parse_zenodo_record(_fake_record())
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / DEFAULT_BENCHMARK_KEY
            archive.write_bytes(b"placeholder")

            store = GoogleQECBenchmarkStore(root)
            store.write_manifest(manifest)
            index = store.index()

            manifest_json = json.loads((root / "manifest.json").read_text())
            self.assertEqual(manifest_json["record_id"], GOOGLE_QEC_RECORD_ID)
            self.assertIn(DEFAULT_BENCHMARK_KEY, index.archives)
            self.assertEqual(index.archives[DEFAULT_BENCHMARK_KEY], archive)


if __name__ == "__main__":
    unittest.main()
