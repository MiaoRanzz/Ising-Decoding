#!/usr/bin/env python3
"""Paired end-to-end evaluation of the patent's classical gate.

The evaluated paths use exactly the same stored shots:

1. raw syndrome -> PyMatching;
2. all thresholded Ising-fast actions -> residual -> PyMatching;
3. Ising-fast candidates -> topology-residual gate -> residual -> PyMatching.

Observable bits are never passed to the model or gate.  They are read only
after global decoding to calculate logical error rate.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from typing import Any

import numpy as np
from omegaconf import OmegaConf
import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
CODE_ROOT = REPO_ROOT / "code"
L_LOGICAL_ROOT = REPO_ROOT / "my_file" / "end_to_end" / "L_logical"
for path in (HERE, CODE_ROOT, L_LOGICAL_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from compare_three_paths import build_matcher, build_model_cfg, load_corpus
from training.precision import match_input_to_model_memory_format
from workflows.run import _load_model

try:
    from .classical_gate import GateConfig, TopologyResidualGate
    from .surface_code import apply_dense_actions, build_surface_action_space, dense_probabilities
except ImportError:  # ``python my_file/patent/evaluate.py``
    from classical_gate import GateConfig, TopologyResidualGate
    from surface_code import apply_dense_actions, build_surface_action_space, dense_probabilities


DEFAULT_SETTINGS = HERE / "settings.yaml"


def _repo_path(value: str | Path) -> Path:
    value = Path(value).expanduser()
    return value if value.is_absolute() else REPO_ROOT / value


def _settings(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"settings file not found: {path}")
    section = OmegaConf.to_container(OmegaConf.load(path).get("patent_evaluation", {}), resolve=True)
    if not isinstance(section, dict):
        raise ValueError("patent_evaluation must be a mapping")
    return section


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the topology-residual patent gate.")
    parser.add_argument("--settings", type=Path, default=DEFAULT_SETTINGS)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _summary(errors: int, samples: int, residual_sum: int = 0, action_sum: int = 0) -> dict[str, float | int]:
    ler = errors / samples if samples else float("nan")
    standard_error = math.sqrt(ler * (1.0 - ler) / samples) if samples else float("nan")
    result: dict[str, float | int] = {
        "logical_errors": int(errors),
        "samples": int(samples),
        "ler": float(ler),
        "ler_standard_error": float(standard_error),
    }
    if samples:
        result["mean_residual_weight"] = residual_sum / samples
        result["mean_action_count"] = action_sum / samples
    return result


def _decode_failures(matcher, residual: np.ndarray, local_frame: np.ndarray, observable: np.ndarray) -> np.ndarray:
    prediction = np.asarray(
        matcher.decode_batch(np.ascontiguousarray(residual, dtype=np.uint8)), dtype=np.uint8
    ).reshape(observable.shape)
    final_prediction = np.bitwise_xor(local_frame.astype(np.uint8), prediction)
    return np.any(final_prediction != observable, axis=1)


def run(settings_path: Path, *, max_samples: int | None = None, device_name: str | None = None,
        output_override: Path | None = None) -> dict[str, Any]:
    settings = _settings(settings_path)
    dataset_dir = _repo_path(settings["dataset_dir"])
    project_config = _repo_path(settings["project_config"])
    checkpoint = _repo_path(settings["checkpoint"])
    output = _repo_path(output_override or settings["output"])
    batch_size = int(settings.get("batch_size", 128))
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    device = torch.device(device_name or settings.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))

    if not dataset_dir.is_dir():
        raise FileNotFoundError(
            f"paired evaluation corpus not found: {dataset_dir}. "
            "Generate it with my_file/end_to_end/L_logical/generate_labeled_dataset.py "
            "or update patent_evaluation.dataset_dir."
        )
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Ising-fast checkpoint not found: {checkpoint}")

    metadata, dets_and_obs_all, train_x_all, _ = load_corpus(dataset_dir)
    available = int(metadata["num_samples"])
    configured_max = settings.get("max_samples")
    requested = max_samples if max_samples is not None else configured_max
    total = available if requested is None else min(available, int(requested))
    if total <= 0:
        raise ValueError("evaluation requires at least one sample")

    matcher, num_observables = build_matcher(metadata)
    if num_observables <= 0:
        raise ValueError("the evaluation circuit has no observable")
    model_args = SimpleNamespace(
        project_config=project_config,
        checkpoint=checkpoint,
        model_id=settings.get("model_id"),
    )
    cfg = build_model_cfg(model_args, metadata)
    model = _load_model(cfg, SimpleNamespace(rank=0, world_size=1, device=device)).to(device).eval()
    gate_mapping = dict(settings.get("gate", {}))
    interaction_radius = int(gate_mapping.pop("interaction_radius", 1))
    probe_batch_size = int(settings.get("mapping_probe_batch_size", 256))
    space = build_surface_action_space(
        cfg,
        distance=int(metadata["distance"]),
        n_rounds=int(metadata["n_rounds"]),
        basis=str(metadata["basis"]),
        rotation=str(metadata["code_rotation"]),
        device="cpu",
        probe_batch_size=probe_batch_size,
        interaction_radius=interaction_radius,
    )
    gate_config = GateConfig.from_mapping(gate_mapping)
    gate = TopologyResidualGate(space, gate_config)

    original_data_threshold = float(settings.get("original_data_probability_threshold", 0.5))
    original_meas_threshold = float(settings.get("original_measurement_probability_threshold", 0.5))
    if not (0.0 <= original_data_threshold <= 1.0 and 0.0 <= original_meas_threshold <= 1.0):
        raise ValueError("original probability thresholds must be in [0, 1]")

    raw_errors = original_errors = gated_errors = 0
    original_residual_sum = gated_residual_sum = 0
    original_action_sum = gated_action_sum = 0
    candidate_sum = evaluated_sum = decision_sum = 0
    gate_time = model_time = decoder_time = 0.0
    paired = {"gate_helpful": 0, "gate_harmful": 0, "both_fail": 0, "both_succeed": 0}
    traces: list[dict[str, Any]] = []
    trace_limit = int(settings.get("trace_first_n", 0))
    log_every = max(1, int(settings.get("log_every_batches", 10)))

    for batch_index, start in enumerate(range(0, total, batch_size)):
        end = min(start + batch_size, total)
        dets_and_obs = np.asarray(dets_and_obs_all[start:end], dtype=np.uint8)
        detector = np.array(dets_and_obs[:, :-num_observables], dtype=np.uint8, copy=True)
        observable = np.asarray(dets_and_obs[:, -num_observables:], dtype=np.uint8)
        train_x = torch.as_tensor(
            np.array(train_x_all[start:end], dtype=np.float32, copy=True), device=device
        )
        train_x = match_input_to_model_memory_format(train_x, model)
        t0 = time.perf_counter()
        with torch.no_grad():
            logits = model(train_x).to(torch.float32).cpu().numpy()
        model_time += time.perf_counter() - t0
        probabilities = dense_probabilities(space, logits)

        dense_original = np.zeros_like(logits, dtype=np.uint8)
        # Threshold in the model's native dense layout, then let ActionSpace
        # discard invalid syndrome-grid positions.
        dense_sigmoid = 1.0 / (1.0 + np.exp(-np.clip(logits, -80.0, 80.0)))
        dense_original[:, :2] = dense_sigmoid[:, :2] >= original_data_threshold
        dense_original[:, 2:] = dense_sigmoid[:, 2:] >= original_meas_threshold
        original_residual, original_frame, original_masks = apply_dense_actions(
            space, detector, dense_original
        )

        gated_residual = np.empty_like(detector)
        gated_frame = np.empty((end - start, space.num_logicals), dtype=np.uint8)
        gated_masks = np.zeros((end - start, space.num_actions), dtype=np.uint8)
        t0 = time.perf_counter()
        for local_index in range(end - start):
            result = gate.run(detector[local_index], probabilities[local_index])
            gated_residual[local_index] = result.residual_syndrome
            gated_frame[local_index] = result.local_logical_frame
            gated_masks[local_index] = result.accepted_mask
            candidate_sum += result.candidate_count
            evaluated_sum += result.evaluated_combinations
            decision_sum += len(result.decisions)
            if len(traces) < trace_limit:
                traces.append({"shot": start + local_index, **result.summary()})
        gate_time += time.perf_counter() - t0

        t0 = time.perf_counter()
        raw_prediction = np.asarray(
            matcher.decode_batch(np.ascontiguousarray(detector)), dtype=np.uint8
        ).reshape(observable.shape)
        raw_failure = np.any(raw_prediction != observable, axis=1)
        original_failure = _decode_failures(matcher, original_residual, original_frame, observable)
        gated_failure = _decode_failures(matcher, gated_residual, gated_frame, observable)
        decoder_time += time.perf_counter() - t0

        raw_errors += int(raw_failure.sum())
        original_errors += int(original_failure.sum())
        gated_errors += int(gated_failure.sum())
        original_residual_sum += int(original_residual.sum())
        gated_residual_sum += int(gated_residual.sum())
        original_action_sum += int(original_masks.sum())
        gated_action_sum += int(gated_masks.sum())
        paired["gate_helpful"] += int((original_failure & ~gated_failure).sum())
        paired["gate_harmful"] += int((~original_failure & gated_failure).sum())
        paired["both_fail"] += int((original_failure & gated_failure).sum())
        paired["both_succeed"] += int((~original_failure & ~gated_failure).sum())
        if (batch_index + 1) % log_every == 0 or end == total:
            print(
                f"[patent evaluation] {end}/{total} shots; "
                f"raw={raw_errors/end:.6g} original={original_errors/end:.6g} gate={gated_errors/end:.6g}",
                flush=True,
            )

    report = {
        "artifact": "topology_residual_patent_gate_evaluation_v1",
        "dataset_dir": str(dataset_dir),
        "checkpoint": str(checkpoint),
        "basis": str(metadata["basis"]),
        "distance": int(metadata["distance"]),
        "n_rounds": int(metadata["n_rounds"]),
        "num_detectors": space.num_detectors,
        "num_actions": space.num_actions,
        "num_logicals": space.num_logicals,
        "gate_config": {
            **{key: value for key, value in gate_config.__dict__.items() if key != "workload"},
            "workload": gate_config.workload.__dict__,
            "interaction_radius": interaction_radius,
        },
        "paths": {
            "raw_pymatching": _summary(raw_errors, total),
            "ising_accept_all": _summary(
                original_errors, total, original_residual_sum, original_action_sum
            ),
            "patent_gate": _summary(
                gated_errors, total, gated_residual_sum, gated_action_sum
            ),
        },
        "paired_gate_vs_ising_accept_all": paired,
        "gate_search": {
            "mean_candidates_per_shot": candidate_sum / total,
            "mean_accepted_decisions_per_shot": decision_sum / total,
            "mean_evaluated_combinations_per_shot": evaluated_sum / total,
        },
        "timing_seconds": {
            "model": model_time,
            "classical_gate": gate_time,
            "global_decoding_all_three_paths": decoder_time,
            "gate_per_shot": gate_time / total,
        },
        "observable_usage": "evaluation_only_not_model_or_gate_input",
        "traces": traces,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["paths"], indent=2), flush=True)
    print(f"Wrote {output}", flush=True)
    return report


def main() -> None:
    args = _parse_args()
    run(
        args.settings,
        max_samples=args.max_samples,
        device_name=args.device,
        output_override=args.output,
    )


if __name__ == "__main__":
    main()
