#!/usr/bin/env python3
"""Strict online, seeded, paired evaluation of the patent gate for X and Z.

No stored corpus or correction label is read.  For each configured basis this
script builds the production surface-code Stim circuit, gives its detector
sampler an independent deterministic seed, and streams fresh detector and
observable rows.  Model input is reconstructed only from detector rows; the
observable remains evaluation-only.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
import pymatching
import torch
from omegaconf import OmegaConf

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
CODE_ROOT = REPO_ROOT / "code"
L_LOGICAL_ROOT = REPO_ROOT / "my_file" / "end_to_end" / "L_logical"
for path in (HERE, CODE_ROOT, L_LOGICAL_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from compare_three_paths import build_model_cfg
from data.predecoder_transform import dets_to_predecoder_inputs
from qec.noise_model import NoiseModel
from qec.surface_code.memory_circuit import MemoryCircuit
from qec.surface_code.stim_sample_io import normalize_code_rotation
from training.precision import match_input_to_model_memory_format
from workflows.run import _load_model

try:
    from .classical_gate import GateConfig, TopologyResidualGate
    from .evaluate import _decode_failures, _repo_path, _settings, _summary
    from .surface_code import apply_dense_actions, build_surface_action_space, dense_probabilities
except ImportError:  # ``python my_file/patent/strict_evaluate.py``
    from classical_gate import GateConfig, TopologyResidualGate
    from evaluate import _decode_failures, _repo_path, _settings, _summary
    from surface_code import apply_dense_actions, build_surface_action_space, dense_probabilities


DEFAULT_SETTINGS = HERE / "settings.yaml"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strict seeded online patent-gate evaluation.")
    parser.add_argument("--settings", type=Path, default=DEFAULT_SETTINGS)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--samples-per-basis", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _strict_settings(settings_path: Path) -> dict[str, Any]:
    raw = OmegaConf.load(settings_path.expanduser().resolve())
    values = OmegaConf.to_container(raw.get("patent_strict_evaluation", {}), resolve=True)
    if not isinstance(values, dict):
        raise ValueError("settings patent_strict_evaluation section must be a mapping")
    return values


def _parse_bases(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        normalized = value.strip().upper()
        if normalized in ("BOTH", "MIXED"):
            return ("X", "Z")
        values = [item.strip() for item in normalized.split(",")]
    elif isinstance(value, (list, tuple)):
        values = [str(item).strip().upper() for item in value]
    else:
        raise ValueError("bases must be X, Z, both, or a list containing X/Z")
    result = tuple(dict.fromkeys(values))
    if not result or any(basis not in ("X", "Z") for basis in result):
        raise ValueError("bases must contain only X and/or Z")
    return result


def _basis_seed(master_seed: int, basis: str) -> int:
    """Derive a stable independent 63-bit Stim seed for one basis."""

    if basis not in ("X", "Z"):
        raise ValueError("basis must be X or Z")
    basis_index = 0 if basis == "X" else 1
    state = np.random.SeedSequence([int(master_seed), basis_index]).generate_state(
        1, dtype=np.uint64
    )
    return int(state[0] % np.uint64(2**63 - 1))


def _noise_from_config(path: Path) -> tuple[NoiseModel | None, float, dict[str, Any]]:
    cfg = OmegaConf.load(path)
    raw = OmegaConf.select(cfg, "data.noise_model")
    if raw is not None:
        values = OmegaConf.to_container(raw, resolve=True)
        if not isinstance(values, dict):
            raise ValueError("data.noise_model must be a mapping")
        model = NoiseModel.from_config_dict(values)
        p_placeholder = float(model.get_max_probability())
        return model, p_placeholder, {
            "kind": "noise_model",
            "parameters": model.canonical_parameters(),
            "sha256": model.sha256(),
            "source": str(path),
        }
    p_value = OmegaConf.select(cfg, "data.p_error")
    if p_value is None:
        p_value = OmegaConf.select(cfg, "data.p_max")
    if p_value is None:
        raise ValueError("noise config must define data.noise_model, data.p_error, or data.p_max")
    p = float(p_value)
    return None, p, {"kind": "simple", "p_error": p, "source": str(path)}


def _new_stats() -> dict[str, Any]:
    return {
        "samples": 0,
        "raw_errors": 0,
        "original_errors": 0,
        "gate_errors": 0,
        "raw_residual_sum": 0,
        "original_residual_sum": 0,
        "gate_residual_sum": 0,
        "original_action_sum": 0,
        "gate_action_sum": 0,
        "candidate_sum": 0,
        "decision_sum": 0,
        "vetoed_action_sum": 0,
        "evaluated_sum": 0,
        "available_cluster_sum": 0,
        "processed_cluster_sum": 0,
        "paired_original": {
            "gate_helpful": 0, "gate_harmful": 0, "both_fail": 0, "both_succeed": 0,
        },
        "paired_raw": {
            "gate_helpful": 0, "gate_harmful": 0, "both_fail": 0, "both_succeed": 0,
        },
        "timing": {"sampling": 0.0, "input_transform": 0.0, "model": 0.0,
                   "classical_gate": 0.0, "global_decoding_all_three_paths": 0.0},
        "traces": [],
    }


def _merge_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    merged = _new_stats()
    scalar_keys = (
        "samples", "raw_errors", "original_errors", "gate_errors", "raw_residual_sum",
        "original_residual_sum", "gate_residual_sum", "original_action_sum",
        "gate_action_sum", "candidate_sum", "decision_sum", "vetoed_action_sum",
        "evaluated_sum", "available_cluster_sum", "processed_cluster_sum",
    )
    for row in rows:
        for key in scalar_keys:
            merged[key] += row[key]
        for group in ("paired_original", "paired_raw", "timing"):
            for key, value in row[group].items():
                merged[group][key] += value
    return merged


def _paired_update(target: dict[str, int], reference: np.ndarray, gate: np.ndarray) -> None:
    target["gate_helpful"] += int((reference & ~gate).sum())
    target["gate_harmful"] += int((~reference & gate).sum())
    target["both_fail"] += int((reference & gate).sum())
    target["both_succeed"] += int((~reference & ~gate).sum())


def _stats_report(stats: Mapping[str, Any], selection_mode: str) -> dict[str, Any]:
    total = int(stats["samples"])
    available = int(stats["available_cluster_sum"])
    if total <= 0:
        raise ValueError("cannot report empty strict statistics")
    timing = {key: float(value) for key, value in stats["timing"].items()}
    timing["gate_per_shot"] = timing["classical_gate"] / total
    return {
        "paths": {
            "raw_pymatching": _summary(
                stats["raw_errors"], total, stats["raw_residual_sum"], 0
            ),
            "ising_accept_all": _summary(
                stats["original_errors"], total,
                stats["original_residual_sum"], stats["original_action_sum"],
            ),
            "patent_gate": _summary(
                stats["gate_errors"], total,
                stats["gate_residual_sum"], stats["gate_action_sum"],
            ),
        },
        "paired_gate_vs_ising_accept_all": dict(stats["paired_original"]),
        "paired_gate_vs_raw_pymatching": dict(stats["paired_raw"]),
        "gate_search": {
            "selection_mode": selection_mode,
            "mean_candidates_per_shot": stats["candidate_sum"] / total,
            "mean_gate_decisions_per_shot": stats["decision_sum"] / total,
            "mean_vetoed_actions_per_shot": stats["vetoed_action_sum"] / total,
            "mean_evaluated_combinations_per_shot": stats["evaluated_sum"] / total,
            "mean_available_clusters_per_shot": stats["available_cluster_sum"] / total,
            "mean_processed_clusters_per_shot": stats["processed_cluster_sum"] / total,
            "processed_cluster_fraction": stats["processed_cluster_sum"] / available if available else 0.0,
        },
        "timing_seconds": timing,
    }


def run(
    settings_path: Path,
    *,
    seed_override: int | None = None,
    samples_override: int | None = None,
    batch_size_override: int | None = None,
    device_override: str | None = None,
    output_override: Path | None = None,
) -> dict[str, Any]:
    base = _settings(settings_path)
    strict = _strict_settings(settings_path)
    project_config = _repo_path(base["project_config"])
    checkpoint = _repo_path(base["checkpoint"])
    noise_config = _repo_path(strict.get("noise_config", base["project_config"]))
    output = _repo_path(output_override or strict["output"])
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Ising checkpoint not found: {checkpoint}")
    if not noise_config.is_file():
        raise FileNotFoundError(f"noise config not found: {noise_config}")

    project = OmegaConf.load(project_config)
    distance = int(strict.get("distance", OmegaConf.select(project, "distance")))
    n_rounds = int(strict.get("n_rounds", OmegaConf.select(project, "n_rounds")))
    rotation = normalize_code_rotation(
        strict.get("code_rotation", OmegaConf.select(project, "data.code_rotation") or "O1")
    )
    bases = _parse_bases(strict.get("bases", "both"))
    master_seed = int(seed_override if seed_override is not None else strict["seed"])
    samples_per_basis = int(
        samples_override if samples_override is not None else strict["samples_per_basis"]
    )
    batch_size = int(
        batch_size_override if batch_size_override is not None else strict.get("batch_size", 128)
    )
    if master_seed < 0:
        raise ValueError("strict evaluation seed must be non-negative")
    if min(distance, n_rounds, samples_per_basis, batch_size) <= 0:
        raise ValueError("distance, n_rounds, samples_per_basis, and batch_size must be positive")
    device = torch.device(
        device_override or strict.get("device") or base.get("device")
        or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    noise_model, p_placeholder, noise_metadata = _noise_from_config(noise_config)

    gate_mapping = dict(base.get("gate", {}))
    interaction_radius = int(gate_mapping.pop("interaction_radius", 1))
    gate_config = GateConfig.from_mapping(gate_mapping)
    probe_batch_size = int(base.get("mapping_probe_batch_size", 256))
    data_threshold = float(base.get("original_data_probability_threshold", 0.5))
    meas_threshold = float(base.get("original_measurement_probability_threshold", 0.5))
    if not (0.0 <= data_threshold <= 1.0 and 0.0 <= meas_threshold <= 1.0):
        raise ValueError("original probability thresholds must be in [0, 1]")
    log_every = max(1, int(strict.get("log_every_batches", 10)))
    trace_limit = max(0, int(strict.get("trace_first_n_per_basis", 0)))

    runtimes: dict[str, dict[str, Any]] = {}
    for basis in bases:
        memory = MemoryCircuit(
            distance=distance,
            idle_error=p_placeholder,
            sqgate_error=p_placeholder,
            tqgate_error=p_placeholder,
            spam_error=(2.0 / 3.0) * p_placeholder,
            n_rounds=n_rounds,
            basis=basis,
            code_rotation=rotation,
            noise_model=noise_model,
            add_boundary_detectors=True,
        )
        memory.set_error_rates()
        circuit = memory.stim_circuit
        dem = circuit.detector_error_model(
            decompose_errors=True, approximate_disjoint_errors=True
        )
        runtimes[basis] = {
            "circuit": circuit,
            "sampler_seed": _basis_seed(master_seed, basis),
            "sampler": circuit.compile_detector_sampler(seed=_basis_seed(master_seed, basis)),
            "matcher": pymatching.Matching.from_detector_error_model(dem),
            "num_detectors": int(circuit.num_detectors),
            "num_observables": int(circuit.num_observables),
        }
        if runtimes[basis]["num_observables"] <= 0:
            raise ValueError(f"strict {basis}-basis circuit has no logical observable")

    first_basis = bases[0]
    first_runtime = runtimes[first_basis]
    first_metadata = {
        "distance": distance,
        "n_rounds": n_rounds,
        "basis": first_basis,
        "code_rotation": rotation,
        "num_detectors": first_runtime["num_detectors"],
        "num_observables": first_runtime["num_observables"],
        "noise": noise_metadata,
    }
    model_args = SimpleNamespace(
        project_config=project_config, checkpoint=checkpoint, model_id=base.get("model_id")
    )
    model_cfg = build_model_cfg(model_args, first_metadata)
    model = _load_model(
        model_cfg, SimpleNamespace(rank=0, world_size=1, device=device)
    ).to(device).eval()

    mapping_seconds: dict[str, float] = {}
    for basis in bases:
        runtime = runtimes[basis]
        started = time.perf_counter()
        space = build_surface_action_space(
            model_cfg,
            distance=distance,
            n_rounds=n_rounds,
            basis=basis,
            rotation=rotation,
            device="cpu",
            probe_batch_size=probe_batch_size,
            interaction_radius=interaction_radius,
        )
        mapping_seconds[basis] = time.perf_counter() - started
        if space.num_detectors != runtime["num_detectors"]:
            raise ValueError(
                f"{basis}-basis online circuit has {runtime['num_detectors']} detectors, "
                f"but gate action space has {space.num_detectors}"
            )
        runtime["space"] = space
        runtime["gate"] = TopologyResidualGate(space, gate_config)

    basis_stats: dict[str, dict[str, Any]] = {}
    for basis in bases:
        runtime = runtimes[basis]
        space = runtime["space"]
        gate = runtime["gate"]
        matcher = runtime["matcher"]
        num_observables = runtime["num_observables"]
        stats = _new_stats()
        basis_stats[basis] = stats
        for batch_index, start in enumerate(range(0, samples_per_basis, batch_size)):
            count = min(batch_size, samples_per_basis - start)
            t0 = time.perf_counter()
            detector, observable = runtime["sampler"].sample(
                count, separate_observables=True
            )
            stats["timing"]["sampling"] += time.perf_counter() - t0
            detector = np.asarray(detector, dtype=np.uint8)
            observable = np.asarray(observable, dtype=np.uint8).reshape(count, num_observables)

            t0 = time.perf_counter()
            train_x, _, _ = dets_to_predecoder_inputs(
                torch.as_tensor(detector, dtype=torch.uint8, device=device),
                distance=distance,
                n_rounds=n_rounds,
                basis=basis,
                code_rotation=rotation,
            )
            train_x = match_input_to_model_memory_format(train_x.to(torch.float32), model)
            stats["timing"]["input_transform"] += time.perf_counter() - t0

            t0 = time.perf_counter()
            with torch.no_grad():
                logits = model(train_x).to(torch.float32).cpu().numpy()
            stats["timing"]["model"] += time.perf_counter() - t0
            probabilities = dense_probabilities(space, logits)

            dense_sigmoid = 1.0 / (1.0 + np.exp(-np.clip(logits, -80.0, 80.0)))
            dense_original = np.zeros_like(logits, dtype=np.uint8)
            dense_original[:, :2] = dense_sigmoid[:, :2] >= data_threshold
            dense_original[:, 2:] = dense_sigmoid[:, 2:] >= meas_threshold
            original_residual, original_frame, original_masks = apply_dense_actions(
                space, detector, dense_original
            )

            gated_residual = np.empty_like(detector)
            gated_frame = np.empty((count, space.num_logicals), dtype=np.uint8)
            gated_masks = np.zeros((count, space.num_actions), dtype=np.uint8)
            t0 = time.perf_counter()
            for local_index in range(count):
                result = gate.run(detector[local_index], probabilities[local_index])
                gated_residual[local_index] = result.residual_syndrome
                gated_frame[local_index] = result.local_logical_frame
                gated_masks[local_index] = result.accepted_mask
                stats["candidate_sum"] += result.candidate_count
                stats["decision_sum"] += len(result.decisions)
                stats["vetoed_action_sum"] += result.vetoed_count
                stats["evaluated_sum"] += result.evaluated_combinations
                stats["available_cluster_sum"] += result.available_cluster_evaluations
                stats["processed_cluster_sum"] += result.processed_cluster_evaluations
                if len(stats["traces"]) < trace_limit:
                    stats["traces"].append(
                        {"basis": basis, "shot": start + local_index, **result.summary()}
                    )
            stats["timing"]["classical_gate"] += time.perf_counter() - t0

            t0 = time.perf_counter()
            raw_prediction = np.asarray(
                matcher.decode_batch(np.ascontiguousarray(detector)), dtype=np.uint8
            ).reshape(observable.shape)
            raw_failure = np.any(raw_prediction != observable, axis=1)
            original_failure = _decode_failures(
                matcher, original_residual, original_frame, observable
            )
            gate_failure = _decode_failures(
                matcher, gated_residual, gated_frame, observable
            )
            stats["timing"]["global_decoding_all_three_paths"] += time.perf_counter() - t0

            stats["samples"] += count
            stats["raw_errors"] += int(raw_failure.sum())
            stats["original_errors"] += int(original_failure.sum())
            stats["gate_errors"] += int(gate_failure.sum())
            stats["raw_residual_sum"] += int(detector.sum())
            stats["original_residual_sum"] += int(original_residual.sum())
            stats["gate_residual_sum"] += int(gated_residual.sum())
            stats["original_action_sum"] += int(original_masks.sum())
            stats["gate_action_sum"] += int(gated_masks.sum())
            _paired_update(stats["paired_original"], original_failure, gate_failure)
            _paired_update(stats["paired_raw"], raw_failure, gate_failure)

            if (batch_index + 1) % log_every == 0 or start + count == samples_per_basis:
                seen = int(stats["samples"])
                print(
                    f"[strict patent evaluation] basis={basis} {seen}/{samples_per_basis}; "
                    f"raw={stats['raw_errors']/seen:.6g} "
                    f"original={stats['original_errors']/seen:.6g} "
                    f"gate={stats['gate_errors']/seen:.6g}",
                    flush=True,
                )

    combined_stats = _merge_stats(list(basis_stats.values()))
    report = {
        "artifact": "strict_online_seeded_patent_gate_evaluation_v1",
        "sampling": "online_stim_detector_sampler",
        "checkpoint": str(checkpoint),
        "project_config": str(project_config),
        "noise": noise_metadata,
        "distance": distance,
        "n_rounds": n_rounds,
        "code_rotation": rotation,
        "bases": list(bases),
        "samples_per_basis": samples_per_basis,
        "total_samples": samples_per_basis * len(bases),
        "batch_size": batch_size,
        "master_seed": master_seed,
        "basis_sampler_seeds": {
            basis: int(runtimes[basis]["sampler_seed"]) for basis in bases
        },
        "gate_config": {
            **{key: value for key, value in gate_config.__dict__.items() if key != "workload"},
            "workload": gate_config.workload.__dict__,
            "interaction_radius": interaction_radius,
        },
        "mapping_construction_seconds": mapping_seconds,
        "combined": _stats_report(combined_stats, gate_config.selection_mode),
        "per_basis": {
            basis: {
                "num_detectors": int(runtimes[basis]["num_detectors"]),
                "num_observables": int(runtimes[basis]["num_observables"]),
                "num_actions": int(runtimes[basis]["space"].num_actions),
                **_stats_report(basis_stats[basis], gate_config.selection_mode),
                "traces": basis_stats[basis]["traces"],
            }
            for basis in bases
        },
        "observable_usage": "evaluation_only_not_model_or_gate_input",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["combined"]["paths"], indent=2), flush=True)
    print(f"Wrote {output}", flush=True)
    return report


def main() -> None:
    args = _parse_args()
    run(
        args.settings,
        seed_override=args.seed,
        samples_override=args.samples_per_basis,
        batch_size_override=args.batch_size,
        device_override=args.device,
        output_override=args.output,
    )


if __name__ == "__main__":
    main()
