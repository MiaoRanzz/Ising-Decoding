# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Phase-1 benchmark for topology-residual utility gating v2.

This runner migrates the useful part of the earlier DEM experiment while
changing its gate from "commit an entire connected cluster" to the v2
candidate-domain semantics implemented in :mod:`evaluation.topology_gating_v2`.

The sampled DEM mechanisms are a reproducible proxy action set.  Consequently
this phase validates the core combination-search and state-update claims, but
does not yet validate the dependent four-spatial-channel Ising embodiment.
That limitation is recorded in every output manifest and result report.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional

from benchmarks.patent_validation.common import (
    REPO_ROOT,
    CandidateCorrectionNet,
    atomic_json,
    batch_indices,
    build_surface_bundle,
    detectors_to_grid,
    environment_manifest,
    load_yaml,
    paired_bootstrap_interval,
    render_results_markdown,
    sample_dem_bundle,
    seed_everything,
    tasks_from_config,
    wilson_interval,
    write_csv,
)
from evaluation.topology_gating_v2 import (
    DATA_X,
    DATA_Z,
    GateConfig,
    apply_actions,
    config_from_mapping,
    gate_actions,
    interaction_clusters,
    pointwise_actions,
    typed_candidates,
    workload_features,
    workload_score,
)


DEFAULT_CONFIG = REPO_ROOT / "conf/experiments/patent/topology_gating_v2.yaml"
DEFAULT_OUTPUT = REPO_ROOT / "outputs/patent_validation/topology_gating_v2"
EVIDENCE_SCOPE = "phase1_core_gate_dem_action_proxy"


def soft_odd_probability(probability: torch.Tensor, incidence: torch.Tensor) -> torch.Tensor:
    clipped = probability.clamp(1e-5, 0.499)
    log_product = torch.log1p(-2 * clipped) @ incidence.t()
    return 0.5 * (1 - torch.exp(log_product))


def train_model(
    model: torch.nn.Module,
    grid: torch.Tensor,
    faults: torch.Tensor,
    syndrome: torch.Tensor,
    observable: torch.Tensor,
    extended_h: torch.Tensor,
    extended_l: torch.Tensor,
    config: dict[str, Any],
    *,
    topology_loss: bool,
) -> torch.nn.Module:
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]))
    h_float = extended_h.to(device=device, dtype=torch.float32)
    logical_float = extended_l.reshape(1, -1).to(device=device, dtype=torch.float32)
    positive = faults.float().sum(dim=0)
    positive_weight = (
        (faults.shape[0] - positive) / positive.clamp_min(1)
    ).clamp(1, 100).to(device)
    model.train()
    for epoch in range(int(config["epochs"])):
        for index in batch_indices(
            grid.shape[0], int(config["batch_size"]), seed=int(config["seed"]) + epoch
        ):
            inputs = grid[index].to(device)
            targets = faults[index].float().to(device)
            target_syndrome = syndrome[index].float().to(device)
            target_observable = observable[index].float().to(device)
            logits = model(inputs)
            local_loss = functional.binary_cross_entropy_with_logits(
                logits, targets, pos_weight=positive_weight
            )
            loss = local_loss
            if topology_loss:
                probability = torch.sigmoid(logits)
                predicted_syndrome = soft_odd_probability(probability, h_float)
                residual_probability = (
                    target_syndrome * (1 - predicted_syndrome)
                    + (1 - target_syndrome) * predicted_syndrome
                )
                logical_probability = soft_odd_probability(
                    probability, logical_float
                ).squeeze(1)
                topology_term = residual_probability.mean()
                frame_term = functional.binary_cross_entropy(
                    logical_probability.clamp(1e-5, 1 - 1e-5), target_observable
                )
                calibration_term = (probability - targets).square().mean()
                loss = (
                    loss
                    + float(config["loss"]["topology"]) * topology_term
                    + float(config["loss"]["logical_frame"]) * frame_term
                    + float(config["loss"]["calibration"]) * calibration_term
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return model


def detector_adjacency(
    coordinates: dict[int, tuple[float, ...]], detector_count: int
) -> list[set[int]]:
    adjacency = [set() for _ in range(detector_count)]
    for left in range(detector_count):
        left_coordinate = coordinates.get(left, ())
        for right in range(left + 1, detector_count):
            right_coordinate = coordinates.get(right, ())
            if (
                left_coordinate
                and len(left_coordinate) == len(right_coordinate)
                and sum(
                    abs(first - second)
                    for first, second in zip(left_coordinate, right_coordinate)
                )
                <= 1.01
            ):
                adjacency[left].add(right)
                adjacency[right].add(left)
    return adjacency


def _whole_cluster_actions(
    syndrome: np.ndarray,
    probabilities: np.ndarray,
    action_types: list[str],
    extended_h: np.ndarray,
    extended_l: np.ndarray,
    adjacency: list[set[int]],
    config: GateConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    """Historical v1-style comparator that commits or rejects entire clusters."""

    candidates = typed_candidates(
        probabilities,
        action_types,
        data_threshold=config.data_threshold,
        measurement_threshold=config.measurement_threshold,
        max_candidates=config.max_candidates,
    )
    clusters = interaction_clusters(
        candidates,
        extended_h,
        adjacency,
        radius=config.interaction_radius,
    )
    state = syndrome.copy().astype(np.uint8)
    correction = np.zeros(extended_h.shape[1], dtype=np.uint8)
    local_frame = np.zeros(extended_l.shape[0], dtype=np.uint8)
    accepted_clusters = 0
    for cluster in clusters:
        proposed = np.zeros(extended_h.shape[1], dtype=np.uint8)
        proposed[list(cluster)] = 1
        trial, logical_delta = apply_actions(state, proposed, extended_h, extended_l)
        before = workload_features(state, adjacency)
        after = workload_features(trial, adjacency)
        if workload_score(after, len(state), config) <= workload_score(
            before, len(state), config
        ):
            correction[list(cluster)] = 1
            state = trial
            local_frame ^= logical_delta
            accepted_clusters += 1
    return correction, state, local_frame, {
        "candidate_count": float(candidates.size),
        "accepted_cluster_count": float(accepted_clusters),
        "combinations_evaluated": float(len(clusters)),
        "exact_search": 1.0,
    }


def _method_output(
    method: str,
    syndrome: np.ndarray,
    probabilities: np.ndarray,
    action_types: list[str],
    extended_h: np.ndarray,
    extended_l: np.ndarray,
    adjacency: list[set[int]],
    gate_config: GateConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    if method == "pointwise":
        correction = pointwise_actions(probabilities, action_types, gate_config)
        residual, local_frame = apply_actions(
            syndrome, correction, extended_h, extended_l
        )
        return correction, residual, local_frame, {
            "candidate_count": float(correction.sum()),
            "accepted_cluster_count": float("nan"),
            "combinations_evaluated": 0.0,
            "exact_search": 1.0,
        }
    if method == "whole_cluster_v1":
        return _whole_cluster_actions(
            syndrome,
            probabilities,
            action_types,
            extended_h,
            extended_l,
            adjacency,
            gate_config,
        )
    if method == "combination_v2":
        result = gate_actions(
            syndrome,
            probabilities,
            action_types,
            extended_h,
            extended_l,
            adjacency,
            gate_config,
        )
        return (
            result.accepted_actions,
            result.residual,
            result.local_logical_frame,
            {
                "candidate_count": float(result.candidate_count),
                "accepted_cluster_count": float(
                    sum(decision.accepted for decision in result.decisions)
                ),
                "combinations_evaluated": float(result.combinations_evaluated),
                "exact_search": float(result.exact_search),
            },
        )
    raise ValueError(f"unknown method {method!r}")


def decode_methods(
    model: torch.nn.Module,
    bundle,
    detectors: torch.Tensor,
    observable: torch.Tensor,
    config: dict[str, Any],
    *,
    gate_config: GateConfig,
    model_name: str,
    basis: str,
    include_raw: bool,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    import pymatching

    device = next(model.parameters()).device
    grid = detectors_to_grid(detectors, config["distance"], config["rounds"])
    probability_batches = []
    model.eval()
    model_started = time.perf_counter()
    with torch.no_grad():
        for start in range(0, len(grid), int(config["batch_size"])):
            probability_batches.append(
                torch.sigmoid(
                    model(grid[start : start + int(config["batch_size"])].to(device))
                ).cpu()
            )
    probabilities = torch.cat(probability_batches).numpy()
    model_seconds = time.perf_counter() - model_started
    h = bundle.h.numpy().astype(np.uint8)
    logical = bundle.logical.numpy().astype(np.uint8).reshape(1, -1)
    detector_bits = detectors.numpy().astype(np.uint8)
    observable_bits = observable.numpy().astype(np.uint8).reshape(-1)
    adjacency = detector_adjacency(bundle.detector_coordinates, h.shape[0])
    matcher = pymatching.Matching.from_detector_error_model(bundle.dem)
    action_type = DATA_Z if basis.upper() == "X" else DATA_X
    action_types = [action_type] * h.shape[1]
    rows: list[dict[str, Any]] = []
    error_vectors: dict[str, np.ndarray] = {}

    if include_raw:
        decode_started = time.perf_counter()
        raw_prediction = np.asarray(matcher.decode_batch(detector_bits), dtype=np.uint8).reshape(-1)
        decode_seconds = time.perf_counter() - decode_started
        raw_errors = raw_prediction != observable_bits
        low, high = wilson_interval(int(raw_errors.sum()), len(raw_errors))
        raw_complexity = np.mean(
            [workload_features(item, adjacency).component_square_sum for item in detector_bits]
        )
        rows.append(
            {
                "method": "raw_pymatching",
                "errors": int(raw_errors.sum()),
                "shots": len(raw_errors),
                "ler": float(raw_errors.mean()),
                "ler_ci_low": low,
                "ler_ci_high": high,
                "residual_density": float(detector_bits.mean()),
                "topology_complexity": float(raw_complexity),
                "acceptance_rate": 0.0,
                "candidate_count_per_shot": 0.0,
                "combinations_evaluated_per_shot": 0.0,
                "exact_search_rate": 1.0,
                "model_us_per_shot": 0.0,
                "gate_us_per_shot": 0.0,
                "decode_us_per_shot": decode_seconds * 1e6 / len(raw_errors),
                "end_to_end_us_per_shot": decode_seconds * 1e6 / len(raw_errors),
                "evidence_scope": EVIDENCE_SCOPE,
            }
        )
        error_vectors["raw_pymatching"] = raw_errors.astype(np.uint8)

    for method in ("pointwise", "whole_cluster_v1", "combination_v2"):
        gate_started = time.perf_counter()
        corrections = []
        residuals = []
        local_frames = []
        diagnostics = []
        for shot in range(len(detector_bits)):
            correction, residual, local_frame, diagnostic = _method_output(
                method,
                detector_bits[shot],
                probabilities[shot],
                action_types,
                h,
                logical,
                adjacency,
                gate_config,
            )
            corrections.append(correction)
            residuals.append(residual)
            local_frames.append(local_frame)
            diagnostics.append(diagnostic)
        gate_seconds = time.perf_counter() - gate_started
        correction_array = np.stack(corrections)
        residual_array = np.stack(residuals)
        local_array = np.stack(local_frames).reshape(-1)
        decode_started = time.perf_counter()
        global_prediction = np.asarray(
            matcher.decode_batch(residual_array), dtype=np.uint8
        ).reshape(-1)
        decode_seconds = time.perf_counter() - decode_started
        prediction = local_array ^ global_prediction
        errors = prediction != observable_bits
        low, high = wilson_interval(int(errors.sum()), len(errors))
        method_name = f"{model_name}_{method}"
        complexity = np.mean(
            [workload_features(item, adjacency).component_square_sum for item in residual_array]
        )
        rows.append(
            {
                "method": method_name,
                "errors": int(errors.sum()),
                "shots": len(errors),
                "ler": float(errors.mean()),
                "ler_ci_low": low,
                "ler_ci_high": high,
                "residual_density": float(residual_array.mean()),
                "topology_complexity": float(complexity),
                "acceptance_rate": float(correction_array.mean()),
                "candidate_count_per_shot": float(
                    np.mean([item["candidate_count"] for item in diagnostics])
                ),
                "combinations_evaluated_per_shot": float(
                    np.mean([item["combinations_evaluated"] for item in diagnostics])
                ),
                "exact_search_rate": float(
                    np.mean([item["exact_search"] for item in diagnostics])
                ),
                "model_us_per_shot": model_seconds * 1e6 / len(errors),
                "gate_us_per_shot": gate_seconds * 1e6 / len(errors),
                "decode_us_per_shot": decode_seconds * 1e6 / len(errors),
                "end_to_end_us_per_shot": (
                    model_seconds + gate_seconds + decode_seconds
                )
                * 1e6
                / len(errors),
                "evidence_scope": EVIDENCE_SCOPE,
            }
        )
        error_vectors[method_name] = errors.astype(np.uint8)
    return rows, error_vectors


def _gate_parameter_candidates(config: dict[str, Any]) -> list[GateConfig]:
    base = dict(config["gate"]["defaults"])
    base.update(config.get("gate_overrides", {}))
    candidates = []
    thresholds = config.get(
        "gate_candidate_thresholds", config["gate"]["candidate_thresholds"]
    )
    parameter_grid = config.get("gate_parameter_grid", config["gate"]["parameter_grid"])
    for threshold in thresholds:
        for override in parameter_grid:
            value = dict(base)
            value.update(override)
            value["data_threshold"] = float(threshold)
            value["measurement_threshold"] = float(threshold)
            candidates.append(config_from_mapping(value))
    return candidates


def _config_record(config: GateConfig) -> dict[str, Any]:
    return {
        "data_threshold": config.data_threshold,
        "measurement_threshold": config.measurement_threshold,
        "interaction_radius": config.interaction_radius,
        "max_combination_actions": config.max_combination_actions,
        "max_candidates": config.max_candidates,
        "combination_budget": config.combination_budget,
        "uncertainty_weight": config.uncertainty_weight,
        "logical_risk_weight": config.logical_risk_weight,
    }


def run(args: argparse.Namespace) -> None:
    config_path = Path(args.config).resolve()
    output_root = Path(args.output).resolve() / args.mode
    config = load_yaml(config_path)
    run_config = dict(config)
    run_config.update(config.get(args.mode, {}))
    seed = int(run_config["seed"])
    seed_everything(seed)
    manifest_path = output_root / "manifest.json"
    manifest = environment_manifest(
        config_path, args.mode, seed, evidence_scope=EVIDENCE_SCOPE
    )
    atomic_json(manifest_path, manifest)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_rows: list[dict[str, Any]] = []
    try:
        for task_index, task in enumerate(tasks_from_config(run_config)):
            for basis_index, basis in enumerate(("X", "Z")):
                raw_path = output_root / "raw" / f"{task.name}_{basis}.json"
                if args.resume and raw_path.exists():
                    existing = json.loads(raw_path.read_text(encoding="utf-8"))
                    expected_shots = int(run_config["test_shots"])
                    if existing and all(
                        row.get("shots") == expected_shots
                        and row.get("evidence_scope") == EVIDENCE_SCOPE
                        for row in existing
                    ):
                        all_rows.extend(existing)
                        print(f"[resume] {task.name}/{basis}")
                        continue
                print(f"[run] mode={args.mode} task={task.name} basis={basis} device={device}")
                bundle = build_surface_bundle(
                    run_config["distance"], run_config["rounds"], basis, task
                )
                train = sample_dem_bundle(
                    bundle,
                    run_config["train_shots"],
                    seed=seed + task_index * 100 + basis_index * 10,
                    device=device,
                )
                validation = sample_dem_bundle(
                    bundle,
                    run_config["validation_shots"],
                    seed=seed + 10000 + task_index * 100 + basis_index * 10,
                    device=device,
                )
                test = sample_dem_bundle(
                    bundle,
                    run_config["test_shots"],
                    seed=seed + 20000 + task_index * 100 + basis_index * 10,
                    device=device,
                )
                grid = detectors_to_grid(
                    train[0], run_config["distance"], run_config["rounds"]
                )
                models = {}
                for model_name, topology_loss in (("bce", False), ("topology", True)):
                    seed_everything(seed)
                    model = CandidateCorrectionNet(
                        bundle.h.shape[1], int(run_config["channels"])
                    ).to(device)
                    models[model_name] = train_model(
                        model,
                        grid,
                        train[2],
                        train[0],
                        train[1],
                        bundle.h,
                        bundle.logical,
                        run_config,
                        topology_loss=topology_loss,
                    )

                selected_gate = None
                selected_rank = None
                for gate_config in _gate_parameter_candidates(run_config):
                    rows, _ = decode_methods(
                        models["bce"],
                        bundle,
                        validation[0],
                        validation[1],
                        run_config,
                        gate_config=gate_config,
                        model_name="bce",
                        basis=basis,
                        include_raw=False,
                    )
                    row = next(
                        item for item in rows if item["method"] == "bce_combination_v2"
                    )
                    rank = (
                        row["ler"],
                        row["topology_complexity"],
                        row["end_to_end_us_per_shot"],
                    )
                    if selected_rank is None or rank < selected_rank:
                        selected_rank = rank
                        selected_gate = gate_config
                assert selected_gate is not None

                baseline_errors = None
                unit_rows = []
                for model_name, model in models.items():
                    rows, error_vectors = decode_methods(
                        model,
                        bundle,
                        test[0],
                        test[1],
                        run_config,
                        gate_config=selected_gate,
                        model_name=model_name,
                        basis=basis,
                        include_raw=model_name == "bce",
                    )
                    if model_name == "bce":
                        baseline_errors = error_vectors["bce_pointwise"]
                    for row in rows:
                        if model_name == "topology" and row["method"] != "topology_combination_v2":
                            continue
                        method_errors = error_vectors[row["method"]]
                        low, high = paired_bootstrap_interval(
                            method_errors,
                            baseline_errors,
                            seed=seed + 70000 + task_index * 10 + basis_index,
                            repeats=int(run_config["acceptance"]["bootstrap_repeats"]),
                        )
                        row.update(
                            {
                                "task": task.name,
                                "basis": basis,
                                **_config_record(selected_gate),
                                "ler_minus_pointwise": float(
                                    method_errors.mean() - baseline_errors.mean()
                                ),
                                "ler_minus_pointwise_ci_low": low,
                                "ler_minus_pointwise_ci_high": high,
                            }
                        )
                        unit_rows.append(row)
                        all_rows.append(row)
                atomic_json(raw_path, unit_rows)

        write_csv(output_root / "summary.csv", all_rows)
        pointwise = [item for item in all_rows if item["method"] == "bce_pointwise"]
        combination = [item for item in all_rows if item["method"] == "bce_combination_v2"]
        margin = float(run_config["acceptance"]["ler_noninferiority_margin"])
        supported = bool(pointwise and combination) and all(
            candidate["ler_minus_pointwise_ci_high"] <= margin
            for candidate in combination
        ) and any(
            candidate["topology_complexity"] < baseline["topology_complexity"]
            for baseline, candidate in zip(pointwise, combination)
        )
        if args.mode == "smoke":
            conclusion = (
                "结论：smoke 流水线完成；该结果只证明实现可运行，不构成专利效果证据。"
            )
        elif args.mode == "pilot":
            conclusion = (
                "结论：Phase 1 单 seed pilot 已完成；可用于冻结正式设计，不构成正式专利效果证据。"
            )
        elif supported:
            conclusion = "结论：Phase 1 的 DEM 动作代理支持 v2 核心组合门控假设。"
        else:
            conclusion = "结论：Phase 1 当前不支持或证据不足，不能作为正向专利效果结论。"
        render_results_markdown(
            output_root / "results.md",
            title="拓扑残差效用门控 v2 Phase 1 结果",
            rows=all_rows,
            conclusion=conclusion,
            limitations=[
                "DEM 误差机制是代理动作集合，尚未覆盖真实四通道空间动作映射。",
                "smoke 与 pilot 模式分别只用于链路检查和正式设计冻结，不进行正式效果判定。",
                "正式证据需要多随机种子、更多码距及独立冻结测试集。",
                "Python 参考门控耗时不能代表优化后的部署时延。",
            ],
        )
        manifest.update(
            status="completed",
            completed_unix=time.time(),
            result_rows=len(all_rows),
            conclusion=conclusion,
        )
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["completed_unix"] = time.time()
        manifest["failures"].append(
            {"type": type(exc).__name__, "message": str(exc)}
        )
        raise
    finally:
        atomic_json(manifest_path, manifest)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "pilot", "validation"), default="smoke")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
