# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Paired topology-gating evaluation for an existing four-channel BCE model."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Iterable

import numpy as np
import torch

from benchmarks.patent_validation.common import (
    REPO_ROOT,
    atomic_json,
    load_yaml,
    paired_bootstrap_interval,
    sha256_file,
    wilson_interval,
    write_csv,
)
from benchmarks.patent_validation.topology_gating_v2_small import (
    _array_sha256,
    _detector_adjacency,
    _device,
    _gate_candidates,
    _geometry,
    _method_gate,
    _model,
    _predict_probabilities,
    _raw_method,
    _sample_stim,
    _seed_everything,
)
from evaluation.surface_topology_adapter import build_surface_action_adapter
from evaluation.topology_gating_v2 import (
    WorkloadGraph,
    build_workload_graph,
    config_from_mapping,
    workload_features,
)


DEFAULT_CONFIG = (
    REPO_ROOT / "conf/experiments/patent/topology_gating_v2_external_bce.yaml"
)
DEFAULT_OUTPUT = (
    REPO_ROOT / "outputs/patent_validation/topology_gating_v2_external_bce"
)
EVIDENCE_SCOPE = "external_bce_single_distance_paired_gate_validation"
METHODS = (
    "raw_pymatching",
    "bce_pointwise",
    "bce_whole_cluster_v1",
    "bce_combination_v2",
)


_PARALLEL_DETECTORS: np.ndarray | None = None
_PARALLEL_PROBABILITIES: np.ndarray | None = None
_PARALLEL_ADAPTER: Any = None
_PARALLEL_ADJACENCY: list[set[int]] | None = None
_PARALLEL_VALID: np.ndarray | None = None
_PARALLEL_WORKLOAD_GRAPHS: dict[int, WorkloadGraph] = {}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = json.loads(json.dumps(base))
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def resolved_config(path: Path, mode: str) -> dict[str, Any]:
    raw = load_yaml(path)
    modes = raw.pop("modes", {})
    if mode == "full":
        config = raw
    else:
        if mode not in modes:
            raise ValueError(f"mode {mode!r} is not declared in {path}")
        config = _deep_merge(raw, modes[mode])
    if config.get("evidence_scope") != EVIDENCE_SCOPE:
        raise ValueError("unexpected evidence_scope in external BCE config")
    return config


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _git_output(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, check=False, capture_output=True, text=True
    ).stdout.strip()


def _identity(
    config: dict[str, Any], mode: str, checkpoint: Path, **fields: Any
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "config_sha256": _canonical_hash(config),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "git_commit": _git_output("rev-parse", "HEAD"),
        "mode": mode,
        **fields,
    }


def _manifest(identity: dict[str, Any], status: str = "running") -> dict[str, Any]:
    return {
        **identity,
        "evidence_scope": EVIDENCE_SCOPE,
        "status": status,
        "git_status": _git_output("status", "--short"),
        "python": os.sys.version,
        "executable": os.sys.executable,
        "prefix": os.sys.prefix,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_devices": [
            torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
        ],
        "started_unix": time.time(),
        "completed_unix": None,
        "failures": [],
    }


def _artifact_hashes(paths: Iterable[Path]) -> dict[str, str]:
    return {str(path): sha256_file(path) for path in paths}


def _resume_complete(
    path: Path,
    identity: dict[str, Any],
    artifacts: Iterable[Path],
    resume: bool,
) -> bool:
    if not resume or not path.exists():
        return False
    previous = json.loads(path.read_text(encoding="utf-8"))
    if {key: previous.get(key) for key in identity} != identity:
        raise RuntimeError(f"resume identity mismatch in {path}")
    if previous.get("status") != "completed":
        return False
    recorded = previous.get("artifact_sha256", {})
    for artifact in artifacts:
        if not artifact.exists() or recorded.get(str(artifact)) != sha256_file(artifact):
            raise RuntimeError(f"resume artifact mismatch: {artifact}")
    return True


def _load_external_checkpoint(
    path: Path, config: dict[str, Any], device: torch.device
) -> torch.nn.Module:
    payload = torch.load(path, map_location=device, weights_only=False)
    if isinstance(payload, dict) and "model_state_dict" in payload:
        state = payload["model_state_dict"]
    elif isinstance(payload, dict) and payload and all(
        isinstance(value, torch.Tensor) for value in payload.values()
    ):
        state = payload
    else:
        raise ValueError("checkpoint must be a state_dict or contain model_state_dict")
    model = _model(config, device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def _parallel_gate_chunk(
    gate_method: str,
    gate_config: Any,
    start: int,
    stop: int,
) -> dict[str, np.ndarray]:
    """Evaluate a contiguous shot range in a forked CPU worker."""

    if (
        _PARALLEL_DETECTORS is None
        or _PARALLEL_PROBABILITIES is None
        or _PARALLEL_ADAPTER is None
        or _PARALLEL_ADJACENCY is None
        or _PARALLEL_VALID is None
    ):
        raise RuntimeError("parallel gate worker was not initialized")
    residuals = []
    frames = []
    action_counts = []
    candidate_counts = []
    combinations = []
    exact = []
    complexity = []
    density = []
    selection_source = []
    combination_score = []
    pointwise_score = []
    radius = int(gate_config.workload_pair_radius)
    graph = _PARALLEL_WORKLOAD_GRAPHS.get(radius)
    if graph is None:
        graph = build_workload_graph(_PARALLEL_ADJACENCY, radius)
        _PARALLEL_WORKLOAD_GRAPHS[radius] = graph
    for index in range(int(start), int(stop)):
        result = _method_gate(
            gate_method,
            _PARALLEL_DETECTORS[index],
            _PARALLEL_PROBABILITIES[index],
            _PARALLEL_ADAPTER,
            _PARALLEL_ADJACENCY,
            gate_config,
            _PARALLEL_VALID,
            workload_graph=graph,
        )
        residual = result.residual
        residuals.append(residual)
        frames.append(int(result.local_logical_frame[0]))
        action_counts.append(result.accepted_count)
        candidate_counts.append(result.candidate_count)
        combinations.append(result.combinations_evaluated)
        exact.append(result.exact_search)
        complexity.append(workload_features(residual, graph).component_square_sum)
        density.append(float(residual.mean()))
        selection_source.append(result.selection_source == "pointwise_fallback")
        combination_score.append(result.combination_workload_score)
        pointwise_score.append(result.pointwise_workload_score)
    return {
        "residual": np.stack(residuals).astype(np.uint8, copy=False),
        "frame": np.asarray(frames, dtype=np.uint8),
        "action_counts": np.asarray(action_counts, dtype=np.int16),
        "candidate_counts": np.asarray(candidate_counts, dtype=np.int16),
        "combinations": np.asarray(combinations, dtype=np.int32),
        "exact": np.asarray(exact, dtype=np.float32),
        "complexity": np.asarray(complexity, dtype=np.float32),
        "density": np.asarray(density, dtype=np.float32),
        "pointwise_fallback": np.asarray(selection_source, dtype=np.uint8),
        "combination_workload_score": np.asarray(
            [np.nan if value is None else value for value in combination_score],
            dtype=np.float32,
        ),
        "pointwise_workload_score": np.asarray(
            [np.nan if value is None else value for value in pointwise_score],
            dtype=np.float32,
        ),
    }


class _ParallelGatePool:
    """Fork workers after inference so large shot arrays remain copy-on-write."""

    def __init__(
        self,
        detectors: np.ndarray,
        probabilities: np.ndarray,
        adapter: Any,
        adjacency: list[set[int]],
        valid: np.ndarray,
        workers: int,
    ) -> None:
        global _PARALLEL_DETECTORS
        global _PARALLEL_PROBABILITIES
        global _PARALLEL_ADAPTER
        global _PARALLEL_ADJACENCY
        global _PARALLEL_VALID
        self.workers = max(1, min(int(workers), len(detectors)))
        _PARALLEL_DETECTORS = detectors
        _PARALLEL_PROBABILITIES = probabilities
        _PARALLEL_ADAPTER = adapter
        _PARALLEL_ADJACENCY = adjacency
        _PARALLEL_VALID = valid
        self.executor = ProcessPoolExecutor(
            max_workers=self.workers,
            mp_context=multiprocessing.get_context("fork"),
        )

    def run(self, gate_method: str, gate_config: Any) -> dict[str, np.ndarray]:
        if _PARALLEL_DETECTORS is None:
            raise RuntimeError("parallel gate pool is closed")
        shots = len(_PARALLEL_DETECTORS)
        chunk_size = max(1, math.ceil(shots / (self.workers * 4)))
        futures = [
            self.executor.submit(
                _parallel_gate_chunk,
                gate_method,
                gate_config,
                start,
                min(shots, start + chunk_size),
            )
            for start in range(0, shots, chunk_size)
        ]
        pieces = [future.result() for future in futures]
        return {
            key: np.concatenate([piece[key] for piece in pieces], axis=0)
            for key in pieces[0]
        }

    def close(self) -> None:
        global _PARALLEL_DETECTORS
        global _PARALLEL_PROBABILITIES
        global _PARALLEL_ADAPTER
        global _PARALLEL_ADJACENCY
        global _PARALLEL_VALID
        self.executor.shutdown(wait=True, cancel_futures=True)
        _PARALLEL_DETECTORS = None
        _PARALLEL_PROBABILITIES = None
        _PARALLEL_ADAPTER = None
        _PARALLEL_ADJACENCY = None
        _PARALLEL_VALID = None
        _PARALLEL_WORKLOAD_GRAPHS.clear()


def _evaluate_method_parallel(
    *,
    method_name: str,
    gate_method: str,
    observable: np.ndarray,
    matcher: Any,
    gate_config: Any,
    model_seconds: float,
    pool: _ParallelGatePool,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    started = time.perf_counter()
    values = pool.run(gate_method, gate_config)
    gate_seconds = time.perf_counter() - started
    decode_started = time.perf_counter()
    global_prediction = np.asarray(
        matcher.decode_batch(values["residual"]), dtype=np.uint8
    ).reshape(-1)
    decode_seconds = time.perf_counter() - decode_started
    prediction = values["frame"] ^ global_prediction
    errors = (prediction != observable).astype(np.uint8)
    low, high = wilson_interval(int(errors.sum()), len(errors))
    row = {
        "method": method_name,
        "errors": int(errors.sum()),
        "shots": len(errors),
        "ler": float(errors.mean()),
        "ler_ci_low": low,
        "ler_ci_high": high,
        "residual_density": float(values["density"].mean()),
        "topology_complexity": float(values["complexity"].mean()),
        "accepted_actions_per_shot": float(values["action_counts"].mean()),
        "candidate_count_per_shot": float(values["candidate_counts"].mean()),
        "combinations_evaluated_per_shot": float(values["combinations"].mean()),
        "exact_search_rate": float(values["exact"].mean()),
        "model_us_per_shot": model_seconds * 1e6 / len(errors),
        "gate_us_per_shot": gate_seconds * 1e6 / len(errors),
        "decode_us_per_shot": decode_seconds * 1e6 / len(errors),
        "end_to_end_us_per_shot": (
            model_seconds + gate_seconds + decode_seconds
        ) * 1e6 / len(errors),
    }
    return row, {
        f"{method_name}__errors": errors,
        f"{method_name}__action_counts": values["action_counts"],
        f"{method_name}__complexity": values["complexity"],
        f"{method_name}__density": values["density"],
    }


def _complete(path: Path, manifest: dict[str, Any], **values: Any) -> None:
    manifest.update(status="completed", completed_unix=time.time(), **values)
    atomic_json(path, manifest)


def evaluate(
    config: dict[str, Any], args: argparse.Namespace, output: Path, checkpoint: Path
) -> None:
    seed = int(args.seed)
    root = output / args.mode
    job = root / "evaluate" / f"seed_{seed}"
    manifest_path = job / "manifest.json"
    summary_path = job / "summary.csv"
    gate_path = job / "selected_gate.json"
    vector_paths = [job / "vectors" / f"t0_{basis}.npz" for basis in ("X", "Z")]
    cpu_workers = int(args.cpu_workers)
    if cpu_workers < 1:
        raise ValueError("cpu_workers must be positive")
    identity = _identity(
        config,
        args.mode,
        checkpoint,
        unit="evaluate",
        seed=seed,
        cpu_workers=cpu_workers,
    )
    if _resume_complete(
        manifest_path, identity, [summary_path, gate_path, *vector_paths], args.resume
    ):
        print(f"[resume] evaluate seed={seed}")
        return

    manifest = _manifest(identity)
    atomic_json(manifest_path, manifest)
    try:
        _seed_everything(seed)
        device = _device(args.device)
        model = _load_external_checkpoint(checkpoint, config, device)
        distance, rounds, rotation = _geometry(config)
        batch_size = int(config["evaluation"]["batch_size"])
        selected_gates: dict[str, Any] = {}
        data_sha256: dict[str, str] = {}
        data_seeds: dict[str, int] = {}

        for basis_index, basis in enumerate(("X", "Z")):
            data_seed = seed + int(config["seed_offsets"]["gate_validation"]) + basis_index * 10000
            detectors, observable, circuit = _sample_stim(
                config,
                task="t0",
                basis=basis,
                shots=int(config["gate_validation_shots_per_basis"]),
                seed=data_seed,
            )
            data_sha256[f"gate_validation_{basis}"] = _array_sha256(
                detectors, observable
            )
            data_seeds[f"gate_validation_{basis}"] = data_seed
            adapter = build_surface_action_adapter(distance, rounds, basis, rotation)
            probabilities, model_seconds = _predict_probabilities(
                model,
                detectors,
                adapter,
                device=device,
                batch_size=batch_size,
            )
            import pymatching

            matcher = pymatching.Matching.from_detector_error_model(
                circuit.detector_error_model(
                    decompose_errors=True, approximate_disjoint_errors=True
                )
            )
            adjacency = _detector_adjacency(circuit, detectors.shape[1])
            ranked = []
            pool = _ParallelGatePool(
                detectors,
                probabilities,
                adapter,
                adjacency,
                adapter.valid_actions.numpy(),
                cpu_workers,
            )
            try:
                for candidate in _gate_candidates(config):
                    row, _ = _evaluate_method_parallel(
                        method_name="bce_combination_v2",
                        gate_method="combination_v2",
                        observable=observable,
                        matcher=matcher,
                        gate_config=candidate,
                        model_seconds=model_seconds,
                        pool=pool,
                    )
                    ranked.append(
                        (
                            row["ler"],
                            row["topology_complexity"],
                            row["residual_density"],
                            candidate.data_threshold,
                            candidate.measurement_threshold,
                            candidate.logical_risk_weight,
                            candidate,
                        )
                    )
            finally:
                pool.close()
            selected_gates[basis] = asdict(min(ranked)[-1])
        atomic_json(gate_path, selected_gates)

        rows: list[dict[str, Any]] = []
        for basis_index, basis in enumerate(("X", "Z")):
            data_seed = seed + int(config["seed_offsets"]["test_t0"]) + basis_index * 10000
            detectors, observable, circuit = _sample_stim(
                config,
                task="t0",
                basis=basis,
                shots=int(config["test_shots_per_basis"]),
                seed=data_seed,
            )
            data_sha256[f"test_t0_{basis}"] = _array_sha256(detectors, observable)
            data_seeds[f"test_t0_{basis}"] = data_seed
            adapter = build_surface_action_adapter(distance, rounds, basis, rotation)
            probabilities, model_seconds = _predict_probabilities(
                model,
                detectors,
                adapter,
                device=device,
                batch_size=batch_size,
            )
            import pymatching

            matcher = pymatching.Matching.from_detector_error_model(
                circuit.detector_error_model(
                    decompose_errors=True, approximate_disjoint_errors=True
                )
            )
            adjacency = _detector_adjacency(circuit, detectors.shape[1])
            gate_config = config_from_mapping(selected_gates[basis])
            cell_rows: list[dict[str, Any]] = []
            vectors: dict[str, np.ndarray] = {}
            raw_row, raw_vectors = _raw_method(
                detectors, observable, adjacency, matcher
            )
            cell_rows.append(raw_row)
            vectors.update(raw_vectors)
            pool = _ParallelGatePool(
                detectors,
                probabilities,
                adapter,
                adjacency,
                adapter.valid_actions.numpy(),
                cpu_workers,
            )
            try:
                for method_name, gate_method in (
                    ("bce_pointwise", "pointwise"),
                    ("bce_whole_cluster_v1", "whole_cluster_v1"),
                    ("bce_combination_v2", "combination_v2"),
                ):
                    row, method_vectors = _evaluate_method_parallel(
                        method_name=method_name,
                        gate_method=gate_method,
                        observable=observable,
                        matcher=matcher,
                        gate_config=gate_config,
                        model_seconds=model_seconds,
                        pool=pool,
                    )
                    cell_rows.append(row)
                    vectors.update(method_vectors)
            finally:
                pool.close()
            for row in cell_rows:
                row.update(task="t0", basis=basis, seed=seed, scope="main")
                rows.append(row)
            whole_error = vectors["bce_whole_cluster_v1__errors"].astype(bool)
            v2_error = vectors["bce_combination_v2__errors"].astype(bool)
            whole_count = vectors["bce_whole_cluster_v1__action_counts"]
            v2_count = vectors["bce_combination_v2__action_counts"]
            vectors["harmful_whole_avoided"] = (
                whole_error & ~v2_error & (v2_count < whole_count)
            ).astype(np.uint8)
            vector_path = job / "vectors" / f"t0_{basis}.npz"
            vector_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(vector_path, **vectors)

        write_csv(summary_path, rows)
        _complete(
            manifest_path,
            manifest,
            selected_gate=selected_gates,
            data_sha256=data_sha256,
            data_seeds=data_seeds,
            artifact_sha256=_artifact_hashes([summary_path, gate_path, *vector_paths]),
        )
    except Exception as exc:
        manifest.update(status="failed", completed_unix=time.time())
        manifest["failures"].append(
            {"type": type(exc).__name__, "message": str(exc)}
        )
        atomic_json(manifest_path, manifest)
        raise


def _read_rows(path: Path) -> list[dict[str, str]]:
    import csv

    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def aggregate(
    config: dict[str, Any], args: argparse.Namespace, output: Path, checkpoint: Path
) -> None:
    root = output / args.mode
    aggregate_dir = root / "aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    summary_by_seed: list[dict[str, str]] = []
    missing = []
    for seed in config["seeds"]:
        job = root / "evaluate" / f"seed_{int(seed)}"
        manifest_path = job / "manifest.json"
        if not manifest_path.exists() or json.loads(
            manifest_path.read_text(encoding="utf-8")
        ).get("status") != "completed":
            missing.append(int(seed))
            continue
        summary_by_seed.extend(_read_rows(job / "summary.csv"))
    write_csv(aggregate_dir / "summary_by_seed.csv", summary_by_seed)
    if missing:
        decisions = {"status": "INCOMPLETE", "missing_seeds": missing}
        atomic_json(aggregate_dir / "decisions.json", decisions)
        (aggregate_dir / "results.md").write_text(
            "# ising_domestic_fast BCE 门控验证\n\n"
            f"结论：**INCOMPLETE**。缺少 seed：{missing}。\n",
            encoding="utf-8",
        )
        print("INCOMPLETE")
        return

    aggregate_rows: list[dict[str, Any]] = []
    for basis in ("X", "Z"):
        for method in METHODS:
            rows = [
                row
                for row in summary_by_seed
                if row["basis"] == basis and row["method"] == method
            ]
            shots = sum(int(row["shots"]) for row in rows)
            errors = sum(int(row["errors"]) for row in rows)
            low, high = wilson_interval(errors, shots)
            aggregate_rows.append(
                {
                    "basis": basis,
                    "method": method,
                    "shots": shots,
                    "errors": errors,
                    "ler": errors / shots,
                    "ler_ci_low": low,
                    "ler_ci_high": high,
                    **{
                        metric: sum(
                            float(row[metric]) * int(row["shots"]) for row in rows
                        )
                        / shots
                        for metric in (
                            "residual_density",
                            "topology_complexity",
                            "accepted_actions_per_shot",
                            "candidate_count_per_shot",
                            "combinations_evaluated_per_shot",
                            "model_us_per_shot",
                            "gate_us_per_shot",
                            "decode_us_per_shot",
                            "end_to_end_us_per_shot",
                        )
                    },
                }
            )
    write_csv(aggregate_dir / "summary.csv", aggregate_rows)

    comparisons = (
        ("bce_pointwise", "raw_pymatching"),
        ("bce_whole_cluster_v1", "raw_pymatching"),
        ("bce_combination_v2", "raw_pymatching"),
        ("bce_whole_cluster_v1", "bce_pointwise"),
        ("bce_combination_v2", "bce_pointwise"),
        ("bce_combination_v2", "bce_whole_cluster_v1"),
    )
    paired_rows: list[dict[str, Any]] = []
    harmful_total = 0
    repeats = int(config["statistics"]["bootstrap_repeats"])
    for basis_index, basis in enumerate(("X", "Z")):
        vectors_by_seed = []
        for seed in config["seeds"]:
            path = root / "evaluate" / f"seed_{int(seed)}" / "vectors" / f"t0_{basis}.npz"
            with np.load(path) as vectors:
                vectors_by_seed.append({key: vectors[key].copy() for key in vectors.files})
                harmful_total += int(vectors["harmful_whole_avoided"].sum())
        for comparison_index, (left, right) in enumerate(comparisons):
            left_error = np.concatenate(
                [vectors[f"{left}__errors"] for vectors in vectors_by_seed]
            )
            right_error = np.concatenate(
                [vectors[f"{right}__errors"] for vectors in vectors_by_seed]
            )
            low, high = paired_bootstrap_interval(
                left_error,
                right_error,
                seed=int(config["statistics"]["bootstrap_seed"])
                + basis_index * 100
                + comparison_index,
                repeats=repeats,
            )
            seed_deltas = [
                float(vectors[f"{left}__errors"].mean())
                - float(vectors[f"{right}__errors"].mean())
                for vectors in vectors_by_seed
            ]
            paired_rows.append(
                {
                    "basis": basis,
                    "left": left,
                    "right": right,
                    "shots": int(left_error.size),
                    "left_errors": int(left_error.sum()),
                    "right_errors": int(right_error.sum()),
                    "delta_ler": float(left_error.mean() - right_error.mean()),
                    "delta_ci_low": low,
                    "delta_ci_high": high,
                    "seed_deltas": json.dumps(seed_deltas),
                    "noninferior_at_0p005": high
                    <= float(config["statistics"]["ler_noninferiority_margin"]),
                }
            )
    write_csv(aggregate_dir / "paired_deltas.csv", paired_rows)

    index = {(row["basis"], row["method"]): row for row in aggregate_rows}
    core_noninferior = all(
        row["noninferior_at_0p005"]
        for row in paired_rows
        if row["left"] == "bce_combination_v2"
        and row["right"] in ("bce_pointwise", "raw_pymatching")
    )
    workload_reductions = {}
    for basis in ("X", "Z"):
        pointwise = float(index[(basis, "bce_pointwise")]["topology_complexity"])
        combination = float(index[(basis, "bce_combination_v2")]["topology_complexity"])
        workload_reductions[basis] = (pointwise - combination) / pointwise
    decisions = {
        "status": "COMPLETE",
        "formal_patent_evidence": False,
        "reason_not_formal": "single existing model at one distance (d=9)",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "core_noninferior": core_noninferior,
        "workload_reductions_vs_pointwise": workload_reductions,
        "harmful_whole_avoided": harmful_total,
    }
    atomic_json(aggregate_dir / "decisions.json", decisions)

    labels = {
        "raw_pymatching": "Raw PyMatching",
        "bce_pointwise": "BCE 逐位置提交",
        "bce_whole_cluster_v1": "BCE 整簇门控",
        "bce_combination_v2": "BCE 组合门控",
    }
    lines = [
        "# ising_domestic_fast BCE 门控验证",
        "",
        f"- checkpoint：`{checkpoint}`",
        f"- SHA-256：`{sha256_file(checkpoint)}`",
        "- 几何：d=9、r=9、XV（公共配置 O1）",
        f"- 避免整簇有害提交事件：{harmful_total}",
        "",
        "| 逻辑基 | 方法 | shots | errors | LER | 残余复杂度 | 残余密度 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for basis in ("X", "Z"):
        for method in METHODS:
            row = index[(basis, method)]
            lines.append(
                f"| {basis} | {labels[method]} | {row['shots']} | {row['errors']} | "
                f"{row['ler']:.6f} | {row['topology_complexity']:.6f} | "
                f"{row['residual_density']:.6f} |"
            )
    lines.extend(
        [
            "",
            "配对差异和 95% CI 见 `paired_deltas.csv`；本结果不是正式专利证据。",
        ]
    )
    (aggregate_dir / "results.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("COMPLETE")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--mode", choices=("full", "smoke"), default="full")
    parser.add_argument("--resume", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--seed", required=True, type=int)
    evaluate_parser.add_argument("--device", default="cuda:0")
    evaluate_parser.add_argument("--cpu-workers", default=30, type=int)
    subparsers.add_parser("aggregate")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = resolved_config(Path(args.config).resolve(), args.mode)
    checkpoint = Path(args.checkpoint or config["external_checkpoint"]).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    output = Path(args.output).resolve()
    if args.command == "evaluate":
        evaluate(config, args, output, checkpoint)
    elif args.command == "aggregate":
        aggregate(config, args, output, checkpoint)
    else:  # pragma: no cover
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
