#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare seeded FP16/INT8 error masks directly with TensorRT FP32."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[3]
REPO_ROOT = CODE_ROOT.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from benchmarks.quant_t0_common import (  # noqa: E402
    combine_basis_results,
    paired_noninferiority,
    write_json,
)


def unpack_error_mask(path: Path, key: str, samples: int) -> np.ndarray:
    with np.load(path) as payload:
        stored_samples = int(payload["samples"][0])
        if stored_samples != samples:
            raise ValueError(f"sample count mismatch in {path}: {stored_samples} != {samples}")
        return np.unpackbits(payload[key])[:samples].astype(np.bool_)


def load_report(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("stim_sampler_seeded") is not True:
        raise ValueError(f"report is not explicitly Stim-seeded: {path}")
    return payload


def performance_fields(report: dict, baseline: dict) -> dict:
    throughput = max(row["shots_per_second"] for row in report["backend"])
    baseline_throughput = max(row["shots_per_second"] for row in baseline["backend"])
    decoder_p99 = report["decoder"][0]["request_latency"]["p99_ms"]
    baseline_decoder_p99 = baseline["decoder"][0]["request_latency"]["p99_ms"]
    return {
        "backend_p50_ms_batch1": report["backend"][0]["request_latency"]["p50_ms"],
        "decoder_p99_ms_batch1": decoder_p99,
        "max_backend_shots_per_s_batch_le_256": throughput,
        "backend_throughput_speedup_vs_fp32": throughput / baseline_throughput,
        "decoder_p99_speedup_vs_fp32": baseline_decoder_p99 / decoder_p99,
        "dataset_generation_s": report["one_time_s"]["dataset_generation"],
        "matcher_build_s": report["one_time_s"]["matcher_build"],
        "tensorrt_decode_s": report["simulation"]["tensorrt_decode_s"],
        "simulation_steady_state_shots_per_s": report["simulation"]["steady_state_shots_per_s"],
        "simulation_generation_plus_decode_shots_per_s": report["simulation"][
            "generation_plus_decode_shots_per_s"
        ],
    }


def summarize(output_dir: Path, absolute_margin: float, relative_margin: float, alpha: float):
    artifact_dir = output_dir / "artifacts"
    report_paths = sorted(artifact_dir.glob("**/benchmark_screen_seeded.json"))
    reports = {path.parent: load_report(path) for path in report_paths}
    rows = []
    basis_results: dict[tuple[str, str], dict[str, dict]] = {}
    for root, report in reports.items():
        relative = root.relative_to(artifact_dir)
        model, _, basis, variant = relative.parts
        if variant == "fp32":
            continue
        fp32_root = root.parent / "fp32"
        fp32 = reports.get(fp32_root)
        if fp32 is None:
            raise FileNotFoundError(f"missing seeded FP32 baseline under {root.parent}")
        if report["dataset_sample_sha256"] != fp32["dataset_sample_sha256"]:
            raise AssertionError(f"candidate and FP32 samples differ: {root}")
        samples = int(report["samples"])
        candidate = unpack_error_mask(Path(report["errors_output"]), "tensorrt", samples)
        baseline = unpack_error_mask(Path(fp32["errors_output"]), "tensorrt", samples)
        comparison = paired_noninferiority(
            candidate,
            baseline,
            absolute_margin=absolute_margin,
            relative_margin=relative_margin,
            alpha=alpha,
        )
        basis_results.setdefault((model, variant), {})[basis] = comparison
        old_path = root / "benchmark_screen.json"
        old_fp32_path = fp32_root / "benchmark_screen.json"
        old = json.loads(old_path.read_text(encoding="utf-8"))
        old_fp32 = json.loads(old_fp32_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "model": model,
                "variant": variant,
                "basis": basis,
                "samples": samples,
                "sample_sha256": report["dataset_sample_sha256"],
                **comparison,
                **performance_fields(old, old_fp32),
                "accuracy_report": str(root / "benchmark_screen_seeded.json"),
                "performance_report": str(old_path),
            }
        )
    combined = {
        f"{model}/{variant}": combine_basis_results(results, alpha=alpha)
        for (model, variant), results in basis_results.items()
    }
    for row in rows:
        row["status_holm"] = combined[f"{row['model']}/{row['variant']}"]["status"]
    return rows, combined


def write_outputs(output_dir: Path, rows: list[dict], combined: dict) -> None:
    results = output_dir / "results"
    payload = {
        "accuracy_baseline": "TensorRT FP32 on identical seeded shots",
        "latency_protocol": "200 warmup iterations, 1000 measured iterations, A100 GPU 0",
        "rows": rows,
        "combined_basis_results": combined,
    }
    write_json(results / "quantization_effects_seeded.json", payload)
    lines = [
        "# T0 quantization effects versus TensorRT FP32",
        "",
        "LER deltas are paired on identical SHA-verified Stim shots. Positive speedup means faster than FP32.",
        "",
        "| Model | Variant | Basis | Status | ΔLER vs FP32 | Upper | Decoder p99 ms | Decoder speedup | Backend shots/s | Backend speedup | Sim end-to-end shots/s |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['variant']} | {row['basis']} | {row['status_holm']} "
            f"| {row['delta_ler']:.6g} | {row['one_sided_upper_bound']:.6g} "
            f"| {row['decoder_p99_ms_batch1']:.6g} | {row['decoder_p99_speedup_vs_fp32']:.3f}x "
            f"| {row['max_backend_shots_per_s_batch_le_256']:.6g} | {row['backend_throughput_speedup_vs_fp32']:.3f}x "
            f"| {row['simulation_generation_plus_decode_shots_per_s']:.6g} |"
        )
    (results / "quantization_effects_seeded.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=REPO_ROOT / "outputs/quant_t0_compare_v1"
    )
    parser.add_argument("--absolute-ler-margin", type=float, default=0.001)
    parser.add_argument("--relative-ler-margin", type=float, default=0.03)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()
    rows, combined = summarize(
        args.output_dir,
        args.absolute_ler_margin,
        args.relative_ler_margin,
        args.alpha,
    )
    write_outputs(args.output_dir, rows, combined)


if __name__ == "__main__":
    main()
