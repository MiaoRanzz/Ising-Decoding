#!/usr/bin/env python3
"""Select a structured checkpoint no-op bias on validation and report held-out LER."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from common import (DEFAULT_SETTINGS, build_base_model, build_structured_model, endpoint_outcomes, endpoint_pipeline,
                    repo_path, section, summarize, write_json)
from compare_three_paths import baseline_failures, final_failures, load_corpus
from evaluation.logical_error_rate import PreDecoderMemoryEvalModule, _build_stab_maps
from structured_actions import actions_from_logits
from train_structured_ising import split_rows


def actions_for_rows(model, train_x: np.ndarray, rows: np.ndarray, device: torch.device, batch_size: int, bias: float):
    result = []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            part = rows[start:start + batch_size]
            logits = model(torch.as_tensor(np.array(train_x[part], dtype=np.float32, copy=True), device=device))
            result.append(actions_from_logits(logits, bias).cpu().numpy().astype(np.uint8))
    return np.concatenate(result, axis=0)


def action_report(failure: np.ndarray, residual: np.ndarray, actions: np.ndarray) -> dict[str, int | float]:
    """Summarize one fixed-action endpoint path in the common report format."""
    return {
        **summarize(failure),
        "mean_residual_weight": float(residual.mean()),
        "mean_action_count": float(actions.sum(axis=(1, 2, 3, 4)).mean()),
    }


def main() -> None:
    settings = DEFAULT_SETTINGS.resolve()
    model_cfg, cfg = section(settings, "structured_model"), section(settings, "structured_evaluation")
    dataset_dir = repo_path(model_cfg["dataset_dir"])
    base_checkpoint, project_config = repo_path(model_cfg["base_checkpoint"]), repo_path(model_cfg["project_config"])
    checkpoint, output = repo_path(cfg["checkpoint"]), repo_path(cfg["output"])
    device = torch.device(cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    metadata, dets_and_obs, train_x, _ = load_corpus(dataset_dir)
    model, endpoint_cfg = build_structured_model(metadata, project_config, base_checkpoint, model_cfg.get("model_id"), device, checkpoint)
    model.eval()
    matcher, action_model, pipeline = endpoint_pipeline(endpoint_cfg, metadata, device)
    _, validation_rows, test_rows = split_rows(
        len(train_x), float(cfg.get("train_fraction", .70)), float(cfg.get("validation_fraction", .15)), int(cfg.get("split_seed", 12345))
    )
    batch_size = int(cfg.get("batch_size", 256))
    candidates = [float(value) for value in cfg.get("no_op_biases", [0.0])]
    validation = []
    for bias in candidates:
        actions = actions_for_rows(model, train_x, validation_rows, device, batch_size, bias)
        failed, residual = endpoint_outcomes(pipeline, action_model, matcher,
                                             np.asarray(dets_and_obs[validation_rows, :-1], dtype=np.uint8),
                                             np.asarray(dets_and_obs[validation_rows, -1:], dtype=np.uint8), actions,
                                             device, batch_size)
        validation.append({"no_op_bias": bias, **summarize(failed), "mean_residual_weight": float(residual.mean()),
                           "mean_action_count": float(actions.sum(axis=(1, 2, 3, 4)).mean())})
    selected = min(validation, key=lambda item: (item["logical_errors"], item["mean_residual_weight"], item["mean_action_count"]))
    bias = float(selected["no_op_bias"])
    test_dets_and_obs = np.asarray(dets_and_obs[test_rows], dtype=np.uint8)
    test_dets, test_obs = test_dets_and_obs[:, :-1], test_dets_and_obs[:, -1:]
    test_actions = actions_for_rows(model, train_x, test_rows, device, batch_size, bias)
    test_failure, test_residual = endpoint_outcomes(pipeline, action_model, matcher,
                                                    test_dets, test_obs, test_actions,
                                                    device, batch_size)
    teacher_report = action_report(test_failure, test_residual, test_actions)

    # Run all reference paths on the exact same held-out shots.  The oracle
    # structured model uses bias zero: it represents the learned head itself,
    # while teacher uses the validation-selected deployment bias above.
    oracle_checkpoint = repo_path(cfg["oracle_checkpoint"])
    oracle_model, _ = build_structured_model(
        metadata, project_config, base_checkpoint, model_cfg.get("model_id"), device, oracle_checkpoint
    )
    oracle_model.eval()
    oracle_actions = actions_for_rows(oracle_model, train_x, test_rows, device, batch_size, bias=0.0)
    oracle_failure, oracle_residual = endpoint_outcomes(
        pipeline, action_model, matcher, test_dets, test_obs, oracle_actions, device, batch_size
    )

    original_model, original_cfg = build_base_model(
        metadata, project_config, base_checkpoint, model_cfg.get("model_id"), device
    )
    original_maps = _build_stab_maps(int(metadata["distance"]), str(metadata["code_rotation"]))
    original_pipeline = PreDecoderMemoryEvalModule(original_model, original_cfg, original_maps, device).to(device).eval()
    original_failure, original_residual = final_failures(
        original_pipeline, matcher, test_dets_and_obs, 1, batch_size, device
    )
    baseline_failure = baseline_failures(matcher, test_dets_and_obs, 1, batch_size)

    report = {
        "teacher_checkpoint": str(checkpoint),
        "oracle_checkpoint": str(oracle_checkpoint),
        "base_ising_fast_checkpoint": str(base_checkpoint),
        "selected_on_validation": selected,
        "validation_candidates": validation,
        "held_out_test": {
            "pymatching": summarize(baseline_failure),
            "original_ising_fast": {**summarize(original_failure), "mean_residual_weight": float(original_residual.mean())},
            "structured_oracle": action_report(oracle_failure, oracle_residual, oracle_actions),
            "structured_teacher": teacher_report,
            "paired_vs_original": {
                "teacher_helpful": int((original_failure & ~test_failure).sum()),
                "teacher_harmful": int((~original_failure & test_failure).sum()),
                "oracle_helpful": int((original_failure & ~oracle_failure).sum()),
                "oracle_harmful": int((~original_failure & oracle_failure).sum()),
            },
        },
    }
    write_json(output, report)
    print(report)


if __name__ == "__main__":
    main()
