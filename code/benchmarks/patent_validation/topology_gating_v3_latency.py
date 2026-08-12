# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pure-inference S07-v3 screening, test, and CPU-pinned throughput workflow."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import os
import random
import subprocess
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from benchmarks.patent_validation.common import (
    REPO_ROOT, atomic_json, load_yaml, sha256_file, wilson_interval, write_csv,
)
from benchmarks.patent_validation.topology_gating_v2_external_bce import (
    _ParallelGatePool, _array_sha256, _deep_merge, _detector_adjacency,
    _device, _geometry, _load_external_checkpoint, _predict_probabilities,
    _sample_stim, _seed_everything,
)
from evaluation.surface_topology_adapter import build_surface_action_adapter
from evaluation.topology_gating_v2 import GateConfig, config_from_mapping

DEFAULT_CONFIG = REPO_ROOT / "conf/experiments/patent/topology_gating_v3_latency.yaml"
DEFAULT_OUTPUT = REPO_ROOT / "outputs/patent_validation/topology_gating_v3_latency"
EVIDENCE_SCOPE = "single_distance_d9_train_free_latency_aligned_implementation_example"
BASE_METHODS = (
    "raw_pymatching", "bce_pointwise", "bce_whole_cluster_v1",
    "bce_combination_v2", "bce_combination_v3",
)
ABLATIONS = (
    "v3_no_pointwise_guard", "v3_legacy_workload",
    "v3_no_radius_pair", "v3_no_largest_component",
)


def _hash(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, check=False, capture_output=True, text=True
    ).stdout.strip()


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def resolved_config(path: Path, mode: str) -> dict[str, Any]:
    raw = load_yaml(path)
    modes = raw.pop("modes", {})
    result = raw if mode == "full" else _deep_merge(raw, modes[mode])
    if result.get("evidence_scope") != EVIDENCE_SCOPE:
        raise ValueError("unexpected evidence_scope")
    actual = len(_candidates(result))
    expected = int(result["gate"]["expected_candidate_count"])
    if actual != expected:
        raise ValueError(f"candidate matrix {actual} != {expected}")
    return result


def _identity(config: dict[str, Any], checkpoint: Path, mode: str, **extra: Any) -> dict[str, Any]:
    source_paths = [
        Path(__file__).resolve(),
        REPO_ROOT / "code/evaluation/topology_gating_v2.py",
        REPO_ROOT / "code/benchmarks/patent_validation/topology_gating_v2_small.py",
        REPO_ROOT / "code/benchmarks/patent_validation/topology_gating_v2_external_bce.py",
        REPO_ROOT / "code/scripts/patent/run_topology_gating_v3_latency.sh",
    ]
    return {
        "schema_version": 1, "evidence_scope": EVIDENCE_SCOPE,
        "config_sha256": _hash(config), "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "git_commit": _git("rev-parse", "HEAD"),
        "source_sha256": {str(path): sha256_file(path) for path in source_paths},
        "mode": mode, **extra,
    }


def _start(path: Path, identity: dict[str, Any]) -> dict[str, Any]:
    value = {
        **identity, "status": "running", "started_unix": time.time(),
        "git_status": _git("status", "--short"), "python": os.sys.version,
        "executable": os.sys.executable, "prefix": os.sys.prefix,
        "torch": torch.__version__, "numpy": np.__version__, "failures": [],
    }
    atomic_json(path, value)
    return value


def _resume(path: Path, identity: dict[str, Any], artifacts: list[Path], enabled: bool) -> bool:
    if not enabled or not path.exists():
        return False
    old = json.loads(path.read_text(encoding="utf-8"))
    if {key: old.get(key) for key in identity} != identity:
        raise RuntimeError(f"resume identity mismatch: {path}")
    if old.get("status") != "completed":
        return False
    hashes = old.get("artifact_sha256", {})
    for artifact in artifacts:
        if not artifact.exists() or hashes.get(str(artifact)) != sha256_file(artifact):
            raise RuntimeError(f"resume artifact mismatch: {artifact}")
    return True


def _finish(path: Path, manifest: dict[str, Any], artifacts: list[Path], **extra: Any) -> None:
    manifest.update(
        status="completed", completed_unix=time.time(),
        artifact_sha256={str(item): sha256_file(item) for item in artifacts}, **extra,
    )
    atomic_json(path, manifest)


def _failed(path: Path, manifest: dict[str, Any], exc: Exception) -> None:
    manifest.update(status="failed", completed_unix=time.time())
    manifest["failures"].append({"type": type(exc).__name__, "message": str(exc)})
    atomic_json(path, manifest)


def _matcher(circuit: Any) -> Any:
    import pymatching
    return pymatching.Matching.from_detector_error_model(
        circuit.detector_error_model(
            decompose_errors=True, approximate_disjoint_errors=True
        )
    )




def _dem_sha256(circuit: Any) -> str:
    dem = circuit.detector_error_model(
        decompose_errors=True, approximate_disjoint_errors=True
    )
    return hashlib.sha256(str(dem).encode("utf-8")).hexdigest()

def _candidates(config: dict[str, Any]) -> list[tuple[str, GateConfig]]:
    gate = config["gate"]
    matrix = itertools.product(
        gate["data_thresholds"], gate["component_weights"],
        gate["largest_weights"], gate["pair_weights"],
        gate["pair_radii"], gate["guard_margins"],
    )
    output = []
    for index, values in enumerate(matrix):
        data, component, largest, pair, radius, margin = values
        mapping = dict(gate["defaults"])
        mapping.update(
            data_threshold=float(data), workload_mode="topology_v3",
            workload_component_weight=float(component),
            workload_largest_weight=float(largest),
            workload_pair_weight=float(pair), workload_pair_radius=int(radius),
            pointwise_guard_enabled=True,
            pointwise_guard_relative_margin=float(margin),
        )
        output.append((f"v3_{index:03d}", config_from_mapping(mapping)))
    return output


def _v2(candidate: GateConfig) -> GateConfig:
    return replace(
        candidate, workload_mode="legacy", workload_component_weight=0.25,
        workload_pair_weight=0.1, workload_largest_weight=0.0,
        workload_pair_radius=1, pointwise_guard_enabled=False,
        pointwise_guard_relative_margin=0.0,
    )


def _phase_seed(config: dict[str, Any], seed: int, phase: str, basis_index: int) -> int:
    return seed + int(config["seed_offsets"][phase]) + 10000 * basis_index


def _cell(
    config: dict[str, Any], checkpoint: Path, seed: int, phase: str,
    task: str, basis: str, basis_index: int, shots: int, device_name: str,
) -> tuple[Any, ...]:
    data_seed = _phase_seed(config, seed, phase, basis_index)
    detectors, observable, circuit = _sample_stim(
        config, task=task, basis=basis, shots=shots, seed=data_seed
    )
    device = _device(device_name)
    model = _load_external_checkpoint(checkpoint, config, device)
    distance, rounds, rotation = _geometry(config)
    adapter = build_surface_action_adapter(distance, rounds, basis, rotation)
    probabilities, model_seconds = _predict_probabilities(
        model, detectors, adapter, device=device,
        batch_size=int(config["evaluation"]["batch_size"]),
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    adjacency = _detector_adjacency(circuit, detectors.shape[1])
    return (
        detectors, observable, circuit, adapter, probabilities,
        model_seconds, adjacency, data_seed,
    )


def _gate(
    pool: _ParallelGatePool, method: str, candidate: GateConfig, name: str,
    observable: np.ndarray, matcher: Any, model_seconds: float,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    started = time.perf_counter()
    values = pool.run(method, candidate)
    gate_seconds = time.perf_counter() - started
    started = time.perf_counter()
    global_prediction = np.asarray(
        matcher.decode_batch(values["residual"]), dtype=np.uint8
    ).reshape(-1)
    decode_seconds = time.perf_counter() - started
    errors = (values["frame"].astype(np.uint8) ^ global_prediction != observable).astype(np.uint8)
    low, high = wilson_interval(int(errors.sum()), len(errors))
    row = {
        "method": name, "shots": len(errors), "errors": int(errors.sum()),
        "ler": float(errors.mean()), "ler_ci_low": low, "ler_ci_high": high,
        "residual_density": float(values["density"].mean()),
        "topology_complexity": float(values["complexity"].mean()),
        "accepted_actions_per_shot": float(values["action_counts"].mean()),
        "candidate_count_per_shot": float(values["candidate_counts"].mean()),
        "combinations_evaluated_per_shot": float(values["combinations"].mean()),
        "pointwise_fallback_rate": float(values["pointwise_fallback"].mean()),
        "model_us_per_shot": model_seconds * 1e6 / len(errors),
        "gate_us_per_shot": gate_seconds * 1e6 / len(errors),
        "decode_batch_us_per_shot": decode_seconds * 1e6 / len(errors),
    }
    vectors = {
        f"{name}__errors": errors, f"{name}__residual": values["residual"],
        f"{name}__frame": values["frame"],
        f"{name}__action_counts": values["action_counts"],
        f"{name}__complexity": values["complexity"],
        f"{name}__density": values["density"],
        f"{name}__pointwise_fallback": values["pointwise_fallback"],
        f"{name}__combination_workload_score": values["combination_workload_score"],
        f"{name}__pointwise_workload_score": values["pointwise_workload_score"],
    }
    return row, vectors


def _raw(detectors: np.ndarray, observable: np.ndarray, matcher: Any) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    started = time.perf_counter()
    prediction = np.asarray(matcher.decode_batch(detectors), dtype=np.uint8).reshape(-1)
    seconds = time.perf_counter() - started
    errors = (prediction != observable).astype(np.uint8)
    low, high = wilson_interval(int(errors.sum()), len(errors))
    row = {
        "method": "raw_pymatching", "shots": len(errors), "errors": int(errors.sum()),
        "ler": float(errors.mean()), "ler_ci_low": low, "ler_ci_high": high,
        "residual_density": float(detectors.mean()), "topology_complexity": float("nan"),
        "accepted_actions_per_shot": 0.0, "candidate_count_per_shot": 0.0,
        "combinations_evaluated_per_shot": 0.0, "pointwise_fallback_rate": 0.0,
        "model_us_per_shot": 0.0, "gate_us_per_shot": 0.0,
        "decode_batch_us_per_shot": seconds * 1e6 / len(errors),
    }
    vectors = {
        "raw_pymatching__errors": errors, "raw_pymatching__residual": detectors,
        "raw_pymatching__frame": np.zeros(len(errors), dtype=np.uint8),
        "raw_pymatching__action_counts": np.zeros(len(errors), dtype=np.int16),
    }
    return row, vectors


def paired_ci(left: np.ndarray, right: np.ndarray, repeats: int, seed: int) -> tuple[float, float, float]:
    """Paired bootstrap CI over fixed 100-shot blocks."""

    delta = left.astype(np.float64) - right.astype(np.float64)
    blocks = np.asarray(
        [delta[start : start + 100].mean() for start in range(0, len(delta), 100)]
    )
    rng = np.random.default_rng(seed)
    samples = np.empty(repeats)
    for index in range(repeats):
        samples[index] = blocks[rng.integers(0, len(blocks), len(blocks))].mean()
    low, high = np.quantile(samples, (0.025, 0.975))
    return float(delta.mean()), float(low), float(high)


def run_screen(config: dict[str, Any], args: argparse.Namespace, root: Path, checkpoint: Path) -> None:
    seed = int(args.seed)
    job = root / "screen" / f"seed_{seed}"
    summary = job / "screening.csv"
    vector_paths = [job / f"{basis}_errors.npz" for basis in ("X", "Z")]
    manifest_path = job / "manifest.json"
    identity = _identity(config, checkpoint, args.mode, stage="screen", seed=seed)
    if _resume(manifest_path, identity, [summary, *vector_paths], args.resume):
        print(f"[resume] screen seed={seed}")
        return
    manifest = _start(manifest_path, identity)
    rows = []
    try:
        _seed_everything(seed)
        for basis_index, basis in enumerate(("X", "Z")):
            values = _cell(
                config, checkpoint, seed, "score_screen", "t0", basis, basis_index,
                int(config["shots"]["score_screen"]), args.device,
            )
            detectors, observable, circuit, adapter, probabilities, model_seconds, adjacency, data_seed = values
            matcher = _matcher(circuit)
            data_sha256 = _array_sha256(detectors, observable)
            dem_sha256 = _dem_sha256(circuit)
            vectors: dict[str, np.ndarray] = {}
            pool = _ParallelGatePool(
                detectors, probabilities, adapter, adjacency,
                adapter.valid_actions.numpy(), int(args.cpu_workers),
            )
            try:
                reference = config_from_mapping(config["gate"]["defaults"])
                methods = [
                    ("bce_pointwise", "pointwise", reference),
                    ("bce_combination_v2", "combination_v2", _v2(reference)),
                ] + [
                    (config_id, "combination_v3_latency_guarded", candidate)
                    for config_id, candidate in _candidates(config)
                ]
                for name, method, candidate in methods:
                    row, scientific = _gate(
                        pool, method, candidate, name, observable, matcher, model_seconds
                    )
                    row.update(
                        seed=seed, basis=basis, config_id=name,
                        data_seed=data_seed, data_sha256=data_sha256,
                        dem_sha256=dem_sha256,
                    )
                    if name.startswith("v3_"):
                        row["config_json"] = json.dumps(asdict(candidate), sort_keys=True)
                    rows.append(row)
                    vectors[f"{name}__errors"] = scientific[f"{name}__errors"]
            finally:
                pool.close()
            path = job / f"{basis}_errors.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, **vectors)
        write_csv(summary, rows)
        _finish(manifest_path, manifest, [summary, *vector_paths])
    except Exception as exc:
        _failed(manifest_path, manifest, exc)
        raise


def freeze_screen(config: dict[str, Any], args: argparse.Namespace, root: Path, checkpoint: Path) -> None:
    screening = root / "screening.csv"
    shortlist_path = root / "screen_shortlist.json"
    manifest_path = root / "screen_freeze_manifest.json"
    screen_manifests = [
        root / "screen" / f"seed_{int(seed)}" / "manifest.json"
        for seed in config["seeds"]
    ]
    identity = _identity(
        config, checkpoint, args.mode, stage="freeze_screen",
        screen_manifest_sha256={
            str(path): sha256_file(path) for path in screen_manifests
        },
    )
    if _resume(manifest_path, identity, [screening, shortlist_path], args.resume):
        print("[resume] freeze-screen")
        return
    manifest = _start(manifest_path, identity)
    try:
        all_rows = []
        for seed in config["seeds"]:
            all_rows += _rows(root / "screen" / f"seed_{int(seed)}" / "screening.csv")
        write_csv(screening, all_rows)
        result: dict[str, Any] = {}
        for basis_index, basis in enumerate(("X", "Z")):
            pooled: dict[str, list[np.ndarray]] = {}
            configs = {}
            basis_rows = [row for row in all_rows if row["basis"] == basis]
            for row in basis_rows:
                if row.get("config_json"):
                    configs[row["config_id"]] = json.loads(row["config_json"])
            for seed in config["seeds"]:
                with np.load(root / "screen" / f"seed_{int(seed)}" / f"{basis}_errors.npz") as values:
                    for key in values.files:
                        pooled.setdefault(key.removesuffix("__errors"), []).append(values[key])
            point = np.concatenate(pooled["bce_pointwise"])
            v2_errors = np.concatenate(pooled["bce_combination_v2"])
            ranked = []
            for config_id in configs:
                errors = np.concatenate(pooled[config_id])
                delta = float(errors.mean() - point.mean())
                _, _, v2_high = paired_ci(
                    errors, v2_errors, int(config["statistics"]["bootstrap_repeats"]),
                    int(config["statistics"]["bootstrap_seed"]) + basis_index,
                )
                selected = [row for row in basis_rows if row["config_id"] == config_id]
                complexity = float(np.mean([float(row["topology_complexity"]) for row in selected]))
                density = float(np.mean([float(row["residual_density"]) for row in selected]))
                rejected = delta > 0.002 or v2_high > 0.002
                ranked.append((rejected, delta, v2_high, complexity, density, config_id))
            viable = [item for item in ranked if not item[0]] or ranked
            viable.sort(key=lambda item: item[1:])
            result[basis] = [
                {"config_id": item[-1], "config": configs[item[-1]],
                 "screen_pointwise_delta": item[1], "screen_v2_ci_high": item[2]}
                for item in viable[: int(config["gate"]["screen_pareto_limit"])]
            ]
        atomic_json(shortlist_path, result)
        _finish(manifest_path, manifest, [screening, shortlist_path])
    except Exception as exc:
        _failed(manifest_path, manifest, exc)
        raise



def run_validation(config: dict[str, Any], args: argparse.Namespace, root: Path, checkpoint: Path) -> None:
    seed = int(args.seed)
    job = root / "validate" / f"seed_{seed}"
    summary = job / "validation.csv"
    vector_paths = [job / f"{basis}_vectors.npz" for basis in ("X", "Z")]
    shortlist_path = root / "screen_shortlist.json"
    manifest_path = job / "manifest.json"
    identity = _identity(
        config, checkpoint, args.mode, stage="validate", seed=seed,
        shortlist_sha256=sha256_file(shortlist_path),
    )
    if _resume(manifest_path, identity, [summary, *vector_paths], args.resume):
        print(f"[resume] validate seed={seed}")
        return
    manifest = _start(manifest_path, identity)
    shortlist = json.loads(shortlist_path.read_text(encoding="utf-8"))
    rows = []
    try:
        for basis_index, basis in enumerate(("X", "Z")):
            values = _cell(
                config, checkpoint, seed, "gate_validation", "t0", basis, basis_index,
                int(config["shots"]["gate_validation"]), args.device,
            )
            detectors, observable, circuit, adapter, probabilities, model_seconds, adjacency, data_seed = values
            matcher = _matcher(circuit)
            data_sha256 = _array_sha256(detectors, observable)
            dem_sha256 = _dem_sha256(circuit)
            candidates = [
                (item["config_id"], config_from_mapping(item["config"]))
                for item in shortlist[basis]
            ]
            reference = candidates[0][1]
            vectors: dict[str, np.ndarray] = {}
            pool = _ParallelGatePool(
                detectors, probabilities, adapter, adjacency,
                adapter.valid_actions.numpy(), int(args.cpu_workers),
            )
            try:
                methods = [
                    ("bce_pointwise", "pointwise", reference),
                    ("bce_combination_v2", "combination_v2", _v2(reference)),
                ] + [
                    (config_id, "combination_v3_latency_guarded", candidate)
                    for config_id, candidate in candidates
                ]
                for name, method, candidate in methods:
                    row, scientific = _gate(
                        pool, method, candidate, name, observable, matcher, model_seconds
                    )
                    row.update(
                        seed=seed, basis=basis, config_id=name,
                        data_seed=data_seed, data_sha256=data_sha256,
                        dem_sha256=dem_sha256,
                    )
                    rows.append(row)
                    vectors.update(scientific)
            finally:
                pool.close()
            latency_shots = min(
                len(detectors), int(config["latency"]["validation_shots_per_seed"])
            )
            compact = {
                key: value[:latency_shots] if key.endswith("__residual") else value
                for key, value in vectors.items()
                if key.endswith("__errors") or key.endswith("__residual")
            }
            compact["observable"] = observable
            path = job / f"{basis}_vectors.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, **compact)
        write_csv(summary, rows)
        _finish(manifest_path, manifest, [summary, *vector_paths])
    except Exception as exc:
        _failed(manifest_path, manifest, exc)
        raise


def _cpu_utilisation(cpu: int, interval: float = 0.25) -> float:
    def snapshot() -> tuple[int, int]:
        lines = Path("/proc/stat").read_text(encoding="utf-8").splitlines()
        fields = [int(value) for value in lines[cpu + 1].split()[1:]]
        idle = fields[3] + (fields[4] if len(fields) > 4 else 0)
        return sum(fields), idle
    total0, idle0 = snapshot()
    time.sleep(interval)
    total1, idle1 = snapshot()
    return 1.0 - (idle1 - idle0) / max(1, total1 - total0)


def verify_latency_isolation(cpu: int, strict: bool) -> dict[str, Any]:
    affinity = sorted(os.sched_getaffinity(0))
    utilisation = _cpu_utilisation(cpu)
    result = {
        "requested_cpu": cpu, "affinity": affinity,
        "cpu_utilisation_before": utilisation,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
    }
    if strict and affinity != [cpu]:
        raise RuntimeError(f"latency affinity {affinity} != [{cpu}]")
    if strict and utilisation > 0.20:
        raise RuntimeError(f"latency CPU {cpu} utilisation {utilisation:.3f} > 0.20")
    return result


def measure_batch_throughput(
    matcher: Any, residuals: dict[str, np.ndarray], seed_labels: np.ndarray,
    *, warmup: int, repeats: int, block_size: int, random_seed: int,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Measure decode_batch() throughput using balanced orders and paired blocks."""

    if not residuals:
        raise ValueError("no latency inputs")
    length = len(next(iter(residuals.values())))
    if any(len(value) != length for value in residuals.values()):
        raise ValueError("latency input lengths differ")
    if len(seed_labels) != length:
        raise ValueError("seed labels do not match latency inputs")
    methods = list(residuals)
    random.Random(random_seed).shuffle(methods)
    warmup = min(warmup, length)
    for method in methods:
        matcher.decode_batch(np.ascontiguousarray(residuals[method][:warmup]))

    sentinel_name = "raw_pymatching" if "raw_pymatching" in residuals else methods[0]
    sentinel_values = residuals[sentinel_name][:warmup]

    def sentinel() -> float:
        started = time.perf_counter_ns()
        matcher.decode_batch(np.ascontiguousarray(sentinel_values))
        return (time.perf_counter_ns() - started) / max(1, len(sentinel_values))

    before = sentinel()
    rows = []
    for repeat_index in range(repeats):
        offset = repeat_index % len(methods)
        order = methods[offset:] + methods[:offset]
        for seed in sorted(set(int(value) for value in seed_labels)):
            indices = np.flatnonzero(seed_labels == seed)
            for block_index, start in enumerate(range(0, len(indices), block_size)):
                block = indices[start : start + block_size]
                for order_index, method in enumerate(order):
                    batch = np.ascontiguousarray(residuals[method][block])
                    started = time.perf_counter_ns()
                    matcher.decode_batch(batch)
                    elapsed_ns = time.perf_counter_ns() - started
                    per_shot_us = elapsed_ns / max(1, len(block)) / 1000.0
                    rows.append(
                        {
                            "method": method, "repeat": repeat_index,
                            "order": order_index, "seed": seed, "block": block_index,
                            "shots": len(block), "measurement_mode": "decode_batch",
                            "batch_us": float(elapsed_ns / 1000.0),
                            "throughput_us_per_shot": float(per_shot_us),
                            # Compatibility fields: these hold batch time divided
                            # by shots, not a distribution of single-shot latency.
                            "mean_us": float(per_shot_us),
                            "median_us": float(per_shot_us),
                            "p95_us": float(per_shot_us),
                        }
                    )
    after = sentinel()
    return rows, {
        "raw_sentinel_before_ns": before, "raw_sentinel_after_ns": after,
        "raw_sentinel_drift": abs(after / before - 1.0) if before else float("inf"),
    }


def latency_delta(
    rows: list[dict[str, Any]], left: str, right: str, repeats: int, seed: int,
) -> dict[str, Any]:
    def indexed(method: str) -> dict[tuple[int, int, int], float]:
        return {
            (int(row["repeat"]), int(row["seed"]), int(row["block"])): float(row["mean_us"])
            for row in rows if row["method"] == method
        }
    left_rows, right_rows = indexed(left), indexed(right)
    keys = sorted(left_rows.keys() & right_rows.keys())
    delta = np.asarray([left_rows[key] - right_rows[key] for key in keys])
    strata: dict[int, list[int]] = {}
    for index, key in enumerate(keys):
        strata.setdefault(key[1], []).append(index)
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(repeats)
    for repeat_index in range(repeats):
        selected = []
        sampled_seeds = rng.choice(list(strata), size=len(strata), replace=True)
        for sampled_seed in sampled_seeds:
            indices = np.asarray(strata[int(sampled_seed)])
            selected += rng.choice(indices, size=len(indices), replace=True).tolist()
        bootstrap[repeat_index] = delta[selected].mean()
    low, high = np.quantile(bootstrap, (0.025, 0.975))
    left_values, right_values = np.asarray(list(left_rows.values())), np.asarray(list(right_rows.values()))
    return {
        "left": left, "right": right, "blocks": len(keys),
        "left_mean_us": float(left_values.mean()),
        "right_mean_us": float(right_values.mean()),
        "delta_us": float(delta.mean()), "delta_ci_low_us": float(low),
        "delta_ci_high_us": float(high),
        "relative_delta": float(delta.mean() / right_values.mean()),
    }


def freeze_gate(config: dict[str, Any], args: argparse.Namespace, root: Path, checkpoint: Path) -> None:
    frozen_path = root / "frozen_gate.json"
    latency_path = root / "validation_latency_blocks.csv"
    shortlist_path = root / "screen_shortlist.json"
    manifest_path = root / "freeze_manifest.json"
    validation_manifests = [
        root / "validate" / f"seed_{int(seed)}" / "manifest.json"
        for seed in config["seeds"]
    ]
    identity = _identity(
        config, checkpoint, args.mode, stage="freeze",
        shortlist_sha256=sha256_file(shortlist_path),
        validation_manifest_sha256={
            str(path): sha256_file(path) for path in validation_manifests
        },
    )
    if _resume(manifest_path, identity, [frozen_path, latency_path], args.resume):
        print("[resume] freeze")
        return
    manifest = _start(manifest_path, identity)
    shortlist = json.loads(shortlist_path.read_text(encoding="utf-8"))
    block_rows, frozen = [], {}
    try:
        isolation = verify_latency_isolation(int(args.latency_cpu), args.require_isolation)
        for basis_index, basis in enumerate(("X", "Z")):
            residuals: dict[str, list[np.ndarray]] = {}
            errors: dict[str, list[np.ndarray]] = {}
            labels = []
            for seed in config["seeds"]:
                path = root / "validate" / f"seed_{int(seed)}" / f"{basis}_vectors.npz"
                with np.load(path) as values:
                    count = len(values["bce_pointwise__residual"])
                    labels.append(np.full(count, int(seed)))
                    for key in values.files:
                        if key.endswith("__residual"):
                            residuals.setdefault(key.removesuffix("__residual"), []).append(values[key])
                        if key.endswith("__errors"):
                            errors.setdefault(key.removesuffix("__errors"), []).append(values[key])
            residual_arrays = {key: np.concatenate(value) for key, value in residuals.items()}
            error_arrays = {key: np.concatenate(value) for key, value in errors.items()}
            _, _, circuit = _sample_stim(
                config, task="t0", basis=basis, shots=1,
                seed=_phase_seed(config, int(config["seeds"][0]), "gate_validation", basis_index),
            )
            measured, sentinel = measure_batch_throughput(
                _matcher(circuit), residual_arrays, np.concatenate(labels),
                warmup=int(config["latency"]["warmup"]),
                repeats=int(config["latency"]["validation_repeats"]),
                block_size=int(config["latency"]["block_size"]),
                random_seed=int(config["statistics"]["bootstrap_seed"]) + basis_index,
            )
            for row in measured:
                row.update(basis=basis, phase="gate_validation")
                block_rows.append(row)
            ranked = []
            point, current_v2 = error_arrays["bce_pointwise"], error_arrays["bce_combination_v2"]
            validation_rows = [
                row for seed in config["seeds"]
                for row in _rows(root / "validate" / f"seed_{int(seed)}" / "validation.csv")
                if row["basis"] == basis
            ]
            for item in shortlist[basis]:
                config_id = item["config_id"]
                candidate_errors = error_arrays[config_id]
                point_delta = float(candidate_errors.mean() - point.mean())
                _, _, v2_high = paired_ci(
                    candidate_errors, current_v2,
                    int(config["statistics"]["bootstrap_repeats"]),
                    int(config["statistics"]["bootstrap_seed"]) + 10 + basis_index,
                )
                latency = latency_delta(
                    measured, config_id, "bce_pointwise",
                    int(config["statistics"]["latency_bootstrap_repeats"]),
                    int(config["statistics"]["bootstrap_seed"]) + 20 + basis_index,
                )
                selected = [row for row in validation_rows if row["config_id"] == config_id]
                complexity = float(np.mean([float(row["topology_complexity"]) for row in selected]))
                eligible = (
                    point_delta <= 0.0
                    and v2_high <= float(config["statistics"]["selection_v2_margin"])
                    and latency["delta_ci_high_us"] < 0.0
                )
                ranked.append((
                    not eligible, latency["relative_delta"], point_delta, complexity,
                    config_id, item["config"], v2_high, latency,
                ))
            ranked.sort(key=lambda item: item[:4])
            winner = ranked[0]
            frozen[basis] = {
                "config_id": winner[4], "config": winner[5],
                "selection_status": "PASS" if not winner[0] else "INCONCLUSIVE",
                "validation_pointwise_ler_delta": winner[2],
                "validation_v2_ci_high": winner[6],
                "validation_latency": winner[7], "sentinel": sentinel,
            }
        write_csv(latency_path, block_rows)
        atomic_json(frozen_path, frozen)
        _finish(manifest_path, manifest, [frozen_path, latency_path], isolation=isolation)
    except Exception as exc:
        _failed(manifest_path, manifest, exc)
        raise



def _main_configs(candidate: GateConfig) -> list[tuple[str, str, GateConfig]]:
    return [
        ("bce_pointwise", "pointwise", candidate),
        ("bce_whole_cluster_v1", "whole_cluster_v1", candidate),
        ("bce_combination_v2", "combination_v2", _v2(candidate)),
        ("bce_combination_v3", "combination_v3_latency_guarded", candidate),
    ]


def _ablation_configs(candidate: GateConfig) -> list[tuple[str, GateConfig]]:
    return [
        ("v3_no_pointwise_guard", replace(candidate, pointwise_guard_enabled=False)),
        (
            "v3_legacy_workload",
            replace(
                candidate, workload_mode="legacy", workload_component_weight=0.25,
                workload_pair_weight=0.1, workload_largest_weight=0.0,
                workload_pair_radius=1,
            ),
        ),
        ("v3_no_radius_pair", replace(candidate, workload_pair_weight=0.0)),
        ("v3_no_largest_component", replace(candidate, workload_largest_weight=0.0)),
    ]


def run_test(config: dict[str, Any], args: argparse.Namespace, root: Path, checkpoint: Path) -> None:
    seed = int(args.seed)
    job = root / "test" / f"seed_{seed}"
    summary, ablation = job / "summary.csv", job / "ablation.csv"
    vector_paths = [
        job / "vectors" / f"{task}_{basis}.npz"
        for task in ("t0", "measurement_drift_1p5") for basis in ("X", "Z")
    ]
    latency_paths = [job / "latency_inputs" / f"{basis}.npz" for basis in ("X", "Z")]
    frozen_path, manifest_path = root / "frozen_gate.json", job / "manifest.json"
    identity = _identity(
        config, checkpoint, args.mode, stage="test", seed=seed,
        frozen_gate_sha256=sha256_file(frozen_path),
    )
    artifacts = [summary, ablation, *vector_paths, *latency_paths]
    if _resume(manifest_path, identity, artifacts, args.resume):
        print(f"[resume] test seed={seed}")
        return
    manifest = _start(manifest_path, identity)
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    rows, ablation_rows, data_hashes = [], [], {}
    try:
        for task in ("t0", "measurement_drift_1p5"):
            shots = int(config["shots"]["test_t0" if task == "t0" else "test_drift"])
            phase = "test_t0" if task == "t0" else "test_drift"
            for basis_index, basis in enumerate(("X", "Z")):
                values = _cell(
                    config, checkpoint, seed, phase, task, basis, basis_index,
                    shots, args.device,
                )
                detectors, observable, circuit, adapter, probabilities, model_seconds, adjacency, data_seed = values
                data_hashes[f"{task}_{basis}"] = _array_sha256(detectors, observable)
                data_hashes[f"{task}_{basis}_dem"] = _dem_sha256(circuit)
                matcher = _matcher(circuit)
                candidate = config_from_mapping(frozen[basis]["config"])
                cell_rows, vectors = [], {}
                raw_row, raw_vectors = _raw(detectors, observable, matcher)
                cell_rows.append(raw_row)
                vectors.update(raw_vectors)
                pool = _ParallelGatePool(
                    detectors, probabilities, adapter, adjacency,
                    adapter.valid_actions.numpy(), int(args.cpu_workers),
                )
                try:
                    for name, method, method_config in _main_configs(candidate):
                        row, scientific = _gate(
                            pool, method, method_config, name,
                            observable, matcher, model_seconds,
                        )
                        cell_rows.append(row)
                        vectors.update(scientific)
                finally:
                    pool.close()
                for row in cell_rows:
                    row.update(
                        seed=seed, task=task, basis=basis,
                        scope="main", data_seed=data_seed,
                    )
                    rows.append(row)

                ablation_shots = min(shots, int(config["shots"]["ablation"]))
                pool = _ParallelGatePool(
                    detectors[:ablation_shots], probabilities[:ablation_shots],
                    adapter, adjacency, adapter.valid_actions.numpy(),
                    int(args.cpu_workers),
                )
                try:
                    for name, method_config in _ablation_configs(candidate):
                        row, scientific = _gate(
                            pool, "combination_v3_latency_guarded", method_config,
                            name, observable[:ablation_shots], matcher,
                            model_seconds * ablation_shots / shots,
                        )
                        row.update(
                            seed=seed, task=task, basis=basis,
                            scope="ablation", data_seed=data_seed,
                        )
                        ablation_rows.append(row)
                        for key, item in scientific.items():
                            vectors[f"ablation__{key}"] = item
                finally:
                    pool.close()

                vector_path = job / "vectors" / f"{task}_{basis}.npz"
                vector_path.parent.mkdir(parents=True, exist_ok=True)
                scientific = {
                    key: item for key, item in vectors.items()
                    if not key.endswith("__residual")
                }
                scientific["observable"] = observable
                np.savez_compressed(vector_path, **scientific)
                if task == "t0":
                    count = min(shots, int(config["latency"]["test_shots_per_seed"]))
                    latency_path = job / "latency_inputs" / f"{basis}.npz"
                    latency_path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(
                        latency_path,
                        **{
                            f"{method}__residual": vectors[f"{method}__residual"][:count]
                            for method in BASE_METHODS
                        },
                    )
        write_csv(summary, rows)
        write_csv(ablation, ablation_rows)
        _finish(manifest_path, manifest, artifacts, data_sha256=data_hashes)
    except Exception as exc:
        _failed(manifest_path, manifest, exc)
        raise


def run_latency(config: dict[str, Any], args: argparse.Namespace, root: Path, checkpoint: Path) -> None:
    block_path, delta_path = root / "latency_blocks.csv", root / "latency_deltas.csv"
    manifest_path, frozen_path = root / "latency_manifest.json", root / "frozen_gate.json"
    test_manifests = [
        root / "test" / f"seed_{int(seed)}" / "manifest.json"
        for seed in config["seeds"]
    ]
    identity = _identity(
        config, checkpoint, args.mode, stage="latency",
        frozen_gate_sha256=sha256_file(frozen_path),
        test_manifest_sha256={str(path): sha256_file(path) for path in test_manifests},
    )
    if _resume(manifest_path, identity, [block_path, delta_path], args.resume):
        print("[resume] latency")
        return
    manifest = _start(manifest_path, identity)
    block_rows, deltas = [], []
    try:
        isolation = verify_latency_isolation(int(args.latency_cpu), args.require_isolation)
        for basis_index, basis in enumerate(("X", "Z")):
            residuals: dict[str, list[np.ndarray]] = {}
            labels = []
            for seed in config["seeds"]:
                path = root / "test" / f"seed_{int(seed)}" / "latency_inputs" / f"{basis}.npz"
                with np.load(path) as values:
                    count = len(values[f"{BASE_METHODS[0]}__residual"])
                    labels.append(np.full(count, int(seed)))
                    for method in BASE_METHODS:
                        residuals.setdefault(method, []).append(values[f"{method}__residual"])
            residual_arrays = {key: np.concatenate(item) for key, item in residuals.items()}
            _, _, circuit = _sample_stim(
                config, task="t0", basis=basis, shots=1,
                seed=_phase_seed(config, int(config["seeds"][0]), "test_t0", basis_index),
            )
            measured, sentinel = measure_batch_throughput(
                _matcher(circuit), residual_arrays, np.concatenate(labels),
                warmup=int(config["latency"]["warmup"]),
                repeats=int(config["latency"]["repeats"]),
                block_size=int(config["latency"]["block_size"]),
                random_seed=int(config["statistics"]["bootstrap_seed"]) + 100 + basis_index,
            )
            if sentinel["raw_sentinel_drift"] > float(config["latency"]["max_sentinel_drift"]):
                raise RuntimeError(
                    f"{basis} sentinel drift {sentinel['raw_sentinel_drift']:.4f} exceeds limit"
                )
            for row in measured:
                row.update(basis=basis, phase="test")
                block_rows.append(row)
            for left, right in (
                ("bce_combination_v3", "bce_pointwise"),
                ("bce_combination_v3", "bce_combination_v2"),
            ):
                item = latency_delta(
                    measured, left, right,
                    int(config["statistics"]["latency_bootstrap_repeats"]),
                    int(config["statistics"]["bootstrap_seed"]) + 200 + basis_index,
                )
                item.update(basis=basis, **sentinel)
                deltas.append(item)
        write_csv(block_path, block_rows)
        write_csv(delta_path, deltas)
        _finish(manifest_path, manifest, [block_path, delta_path], isolation=isolation)
    except Exception as exc:
        _failed(manifest_path, manifest, exc)
        raise


def aggregate(config: dict[str, Any], args: argparse.Namespace, root: Path, checkpoint: Path) -> None:
    output = root / "aggregate"
    output.mkdir(parents=True, exist_ok=True)
    summary_path, delta_path = output / "summary.csv", output / "paired_deltas.csv"
    ablation_path, results_path = output / "ablation.csv", output / "results.md"
    required = [root / "frozen_gate.json", root / "latency_deltas.csv"]
    manifests = [root / "freeze_manifest.json", root / "latency_manifest.json"]
    for seed in config["seeds"]:
        required += [
            root / "test" / f"seed_{int(seed)}" / "summary.csv",
            root / "test" / f"seed_{int(seed)}" / "ablation.csv",
        ]
        manifests.append(root / "test" / f"seed_{int(seed)}" / "manifest.json")
    problems = [f"missing artifact: {path}" for path in required if not path.exists()]
    for manifest_path in manifests:
        if not manifest_path.exists():
            problems.append(f"missing manifest: {manifest_path}")
            continue
        stage = json.loads(manifest_path.read_text(encoding="utf-8"))
        if stage.get("status") != "completed":
            problems.append(f"incomplete manifest: {manifest_path}")
            continue
        for artifact_name, expected_hash in stage.get("artifact_sha256", {}).items():
            artifact = Path(artifact_name)
            if not artifact.exists() or sha256_file(artifact) != expected_hash:
                problems.append(f"artifact hash mismatch: {artifact}")
    if problems:
        results_path.write_text(
            "# S07 v3 batch-throughput-aligned validation\n\nStatus: **INCOMPLETE**\n\n"
            + "\n".join(f"- {problem}" for problem in problems) + "\n",
            encoding="utf-8",
        )
        print("INCOMPLETE")
        return

    rows = [
        row for seed in config["seeds"]
        for row in _rows(root / "test" / f"seed_{int(seed)}" / "summary.csv")
    ]
    ablations = [
        row for seed in config["seeds"]
        for row in _rows(root / "test" / f"seed_{int(seed)}" / "ablation.csv")
    ]
    write_csv(ablation_path, ablations)
    pooled = []
    for task in ("t0", "measurement_drift_1p5"):
        for basis in ("X", "Z"):
            for method in BASE_METHODS:
                selected = [
                    row for row in rows
                    if row["task"] == task and row["basis"] == basis
                    and row["method"] == method
                ]
                shots = sum(int(row["shots"]) for row in selected)
                errors = sum(int(row["errors"]) for row in selected)
                low, high = wilson_interval(errors, shots)
                pooled.append(
                    {
                        "task": task, "basis": basis, "method": method,
                        "shots": shots, "errors": errors, "ler": errors / shots,
                        "ler_ci_low": low, "ler_ci_high": high,
                        "residual_density": float(np.mean([float(row["residual_density"]) for row in selected])),
                        "topology_complexity": float(np.mean([float(row["topology_complexity"]) for row in selected])),
                        "accepted_actions_per_shot": float(np.mean([float(row["accepted_actions_per_shot"]) for row in selected])),
                        "pointwise_fallback_rate": float(np.mean([float(row["pointwise_fallback_rate"]) for row in selected])),
                    }
                )
    write_csv(summary_path, pooled)

    cache: dict[tuple[str, str, str], list[np.ndarray]] = {}
    for seed in config["seeds"]:
        for task in ("t0", "measurement_drift_1p5"):
            for basis in ("X", "Z"):
                path = root / "test" / f"seed_{int(seed)}" / "vectors" / f"{task}_{basis}.npz"
                with np.load(path) as values:
                    for method in BASE_METHODS:
                        cache.setdefault((task, basis, method), []).append(values[f"{method}__errors"])
    comparisons = [
        ("bce_combination_v3", "bce_pointwise"),
        ("bce_combination_v3", "bce_combination_v2"),
        ("bce_combination_v3", "raw_pymatching"),
        ("bce_combination_v2", "bce_pointwise"),
        ("bce_whole_cluster_v1", "bce_pointwise"),
    ]
    paired = []
    for task in ("t0", "measurement_drift_1p5"):
        for basis_index, basis in enumerate(("X", "Z")):
            for compare_index, (left, right) in enumerate(comparisons):
                left_by_seed, right_by_seed = cache[(task, basis, left)], cache[(task, basis, right)]
                left_values, right_values = np.concatenate(left_by_seed), np.concatenate(right_by_seed)
                estimate, low, high = paired_ci(
                    left_values, right_values,
                    int(config["statistics"]["bootstrap_repeats"]),
                    int(config["statistics"]["bootstrap_seed"]) + basis_index * 100 + compare_index,
                )
                paired.append(
                    {
                        "task": task, "basis": basis, "left": left, "right": right,
                        "delta": estimate, "delta_ci_low": low, "delta_ci_high": high,
                        "paired_shots": len(left_values),
                        "logical_error_events": int(np.logical_or(left_values, right_values).sum()),
                        "seed_deltas": json.dumps([
                            float(a.mean() - b.mean())
                            for a, b in zip(left_by_seed, right_by_seed)
                        ]),
                    }
                )
    write_csv(delta_path, paired)

    index = {(row["task"], row["basis"], row["left"], row["right"]): row for row in paired}
    latency = {
        (row["basis"], row["left"], row["right"]): row
        for row in _rows(root / "latency_deltas.csv")
    }
    latency_blocks = _rows(root / "latency_blocks.csv")
    reasons, no_go, inconclusive = [], False, False
    for basis in ("X", "Z"):
        point = index[("t0", basis, "bce_combination_v3", "bce_pointwise")]
        current = index[("t0", basis, "bce_combination_v3", "bce_combination_v2")]
        backend = latency[(basis, "bce_combination_v3", "bce_pointwise")]
        if min(int(point["logical_error_events"]), int(current["logical_error_events"])) < 200:
            inconclusive = True
            reasons.append(f"{basis}: fewer than 200 paired logical-error events")
        if float(point["delta_ci_low"]) > 0:
            no_go = True
            reasons.append(f"{basis}: v3 LER is significantly worse than pointwise")
        elif float(point["delta_ci_high"]) >= 0:
            inconclusive = True
            reasons.append(f"{basis}: v3 vs pointwise LER CI crosses zero")
        margin = float(config["statistics"]["v2_noninferiority_margin"])
        if float(current["delta_ci_low"]) > margin:
            no_go = True
            reasons.append(f"{basis}: v3 significantly violates +0.0002 v2 noninferiority")
        elif float(current["delta_ci_high"]) > margin:
            inconclusive = True
            reasons.append(f"{basis}: v3 vs v2 noninferiority CI crosses +0.0002")
        if float(backend["delta_ci_low_us"]) > 0:
            no_go = True
            reasons.append(f"{basis}: backend batch throughput is significantly worse")
        elif float(backend["delta_ci_high_us"]) >= 0:
            inconclusive = True
            reasons.append(f"{basis}: backend batch-throughput CI crosses zero")
        if not all(value < 0 for value in json.loads(point["seed_deltas"])):
            inconclusive = True
            reasons.append(f"{basis}: LER direction not consistent across seeds")
        latency_seed_deltas = []
        for seed in config["seeds"]:
            def latency_mean(method: str) -> float:
                selected = [
                    float(row["mean_us"]) for row in latency_blocks
                    if row["basis"] == basis and int(row["seed"]) == int(seed)
                    and row["method"] == method
                ]
                return float(np.mean(selected))
            latency_seed_deltas.append(
                latency_mean("bce_combination_v3") - latency_mean("bce_pointwise")
            )
        if not all(value < 0 for value in latency_seed_deltas):
            inconclusive = True
            reasons.append(f"{basis}: batch-throughput direction not consistent across seeds")
        for seed, drift, registered in zip(
            config["seeds"],
            cache[("measurement_drift_1p5", basis, "bce_combination_v3")],
            cache[("t0", basis, "bce_combination_v3")],
        ):
            if float(drift.mean() - registered.mean()) > 0.01:
                reasons.append(f"{basis} seed {seed}: BYPASS REQUIRED under measurement drift")
    status = "NO-GO" if no_go else ("INCONCLUSIVE" if inconclusive else "GO")
    frozen = json.loads((root / "frozen_gate.json").read_text(encoding="utf-8"))
    text = [
        "# S07 v3 batch-throughput-aligned validation", "", f"Status: **{status}**", "",
        "Fixed checkpoint, d=9/r=9, train-free single-distance implementation example.",
        "Backend timing is CPU-pinned decode_batch() time per shot, not single-shot latency.",
        "This is not general formal patent evidence.", "", "## Frozen gate", "",
        json.dumps(frozen, indent=2, ensure_ascii=False), "", "## Decision notes", "",
    ]
    text += [f"- {reason}" for reason in reasons] or ["- All registered GO criteria passed."]
    text += [
        "", "See summary.csv, paired_deltas.csv, ablation.csv, and the parent "
        "latency_blocks.csv / latency_deltas.csv for complete results.",
    ]
    results_path.write_text("\n".join(text) + "\n", encoding="utf-8")
    aggregate_artifacts = [summary_path, delta_path, ablation_path, results_path]
    atomic_json(
        output / "manifest.json",
        {
            **_identity(config, checkpoint, args.mode, stage="aggregate"),
            "status": "completed",
            "completed_unix": time.time(),
            "artifact_sha256": {
                str(path): sha256_file(path) for path in aggregate_artifacts
            },
        },
    )

    print(status)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--checkpoint")
    parser.add_argument("--mode", choices=("full", "smoke"), default="full")
    parser.add_argument("--resume", action="store_true")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("screen", "validate", "test"):
        item = commands.add_parser(name)
        item.add_argument("--seed", required=True, type=int)
        item.add_argument("--device", default="cuda:0")
        item.add_argument("--cpu-workers", default=30, type=int)
    commands.add_parser("freeze-screen")
    for name in ("freeze", "latency"):
        item = commands.add_parser(name)
        item.add_argument("--latency-cpu", default=27, type=int)
        item.add_argument("--require-isolation", action="store_true")
    commands.add_parser("aggregate")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = resolved_config(Path(args.config).resolve(), args.mode)
    checkpoint = Path(args.checkpoint or config["external_checkpoint"]).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if getattr(args, "cpu_workers", 1) < 1:
        raise ValueError("cpu-workers must be positive")
    root = Path(args.output).resolve() / args.mode
    root.mkdir(parents=True, exist_ok=True)
    dispatch = {
        "screen": run_screen, "freeze-screen": freeze_screen,
        "validate": run_validation, "freeze": freeze_gate,
        "test": run_test, "latency": run_latency, "aggregate": aggregate,
    }
    dispatch[args.command](config, args, root, checkpoint)


if __name__ == "__main__":
    main()
