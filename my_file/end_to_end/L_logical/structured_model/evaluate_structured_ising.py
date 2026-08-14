#!/usr/bin/env python3
"""Select no-op bias on validation and report offline or fresh-shot held-out LER."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from common import (DEFAULT_SETTINGS, build_base_model, build_structured_model, endpoint_outcomes, endpoint_pipeline,
                    repo_path, section, summarize, write_json)
from compare_three_paths import baseline_failures, final_failures, load_corpus
from evaluation.logical_error_rate import PreDecoderMemoryEvalModule, _build_stab_maps
from structured_actions import actions_from_logits
from strict_evaluation import run_strict_evaluation
from train_structured_ising import split_rows


def actions_for_rows(model, train_x: np.ndarray, rows: np.ndarray, device: torch.device, batch_size: int, bias: float):
    result = []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            part = rows[start:start + batch_size]
            logits = model(torch.as_tensor(np.array(train_x[part], dtype=np.float32, copy=True), device=device))
            result.append(actions_from_logits(logits, bias).cpu().numpy().astype(np.uint8))
    return np.concatenate(result, axis=0)


def legacy_actions_for_rows(model, train_x: np.ndarray, rows: np.ndarray, device: torch.device, batch_size: int):
    """Return the original model's four ``logit >= 0`` correction actions."""
    result = []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            part = rows[start:start + batch_size]
            logits = model(torch.as_tensor(np.array(train_x[part], dtype=np.float32, copy=True), device=device))
            result.append((logits >= 0).cpu().numpy().astype(np.uint8))
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
    mode = str(cfg.get("mode", "full")).lower()
    if mode not in {"warm_start_audit", "full"}:
        raise ValueError("structured_evaluation.mode must be 'warm_start_audit' or 'full'")
    data_mode = str(model_cfg.get("data_mode", "offline")).lower()
    if data_mode not in {"offline", "strict"}:
        raise ValueError("structured_model.data_mode must be 'offline' or 'strict'")
    if data_mode == "strict":
        run_strict_evaluation(settings, model_cfg, cfg)
        return
    dataset_dir = repo_path(model_cfg["dataset_dir"])
    base_checkpoint, project_config = repo_path(model_cfg["base_checkpoint"]), repo_path(model_cfg["project_config"])
    output = repo_path(cfg["audit_output"] if mode == "warm_start_audit" else cfg["output"])
    device = torch.device(cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    metadata, dets_and_obs, train_x, _ = load_corpus(dataset_dir)
    _, validation_rows, test_rows = split_rows(
        len(train_x), float(cfg.get("train_fraction", .70)), float(cfg.get("validation_fraction", .15)), int(cfg.get("split_seed", 12345))
    )
    batch_size = int(cfg.get("batch_size", 256))
    test_dets_and_obs = np.asarray(dets_and_obs[test_rows], dtype=np.uint8)
    test_dets, test_obs = test_dets_and_obs[:, :-1], test_dets_and_obs[:, -1:]

    if mode == "warm_start_audit":
        # Do not load oracle/teacher checkpoints here. Checkpoints produced
        # before the trunk-conversion fix have a different and invalid layout.
        original_model, original_cfg = build_base_model(
            metadata, project_config, base_checkpoint, model_cfg.get("model_id"), device
        )
        matcher, action_model, pipeline = endpoint_pipeline(original_cfg, metadata, device)
        original_maps = _build_stab_maps(int(metadata["distance"]), str(metadata["code_rotation"]))
        original_pipeline = PreDecoderMemoryEvalModule(
            original_model, original_cfg, original_maps, device
        ).to(device).eval()
        original_pipeline_failure, original_pipeline_residual = final_failures(
            original_pipeline, matcher, test_dets_and_obs, 1, batch_size, device
        )
        original_actions = legacy_actions_for_rows(original_model, train_x, test_rows, device, batch_size)
        # Feed both action tensors through this same fixed-action pipeline.
        # These are the only outcomes suitable for an action-equivalence audit;
        # the complete original pipeline may reconstruct trainX and use a
        # different autocast path, so it is retained as a reference only.
        original_failure, original_residual = endpoint_outcomes(
            pipeline, action_model, matcher, test_dets, test_obs, original_actions, device, batch_size
        )

        warm_model, _ = build_structured_model(
            metadata, project_config, base_checkpoint, model_cfg.get("model_id"), device
        )
        warm_model.eval()
        warm_actions = actions_for_rows(warm_model, train_x, test_rows, device, batch_size, bias=0.0)
        warm_failure, warm_residual = endpoint_outcomes(
            pipeline, action_model, matcher, test_dets, test_obs, warm_actions, device, batch_size
        )
        action_mismatch = warm_actions != original_actions
        report = {
            "mode": mode,
            "data_mode": "offline",
            "comparison_path": "train_x_to_fixed_action_endpoint",
            "base_ising_fast_checkpoint": str(base_checkpoint),
            "held_out_test": {
                "original_ising_fast_fixed_actions": action_report(
                    original_failure, original_residual, original_actions
                ),
                "original_ising_fast_complete_pipeline_reference": {
                    **summarize(original_pipeline_failure),
                    "mean_residual_weight": float(original_pipeline_residual.mean()),
                },
                "structured_warm_start": action_report(warm_failure, warm_residual, warm_actions),
                "action_mismatch_bits": int(action_mismatch.sum()),
                "action_mismatch_shots": int(np.any(action_mismatch, axis=(1, 2, 3, 4)).sum()),
                "fixed_action_endpoint_failure_mismatch_shots": int((warm_failure != original_failure).sum()),
                "fixed_action_residual_weight_mismatch_shots": int((warm_residual != original_residual).sum()),
                "complete_vs_fixed_original_failure_mismatch_shots": int(
                    (original_pipeline_failure != original_failure).sum()
                ),
            },
        }
        write_json(output, report)
        print(report)
        return

    checkpoint = repo_path(cfg["checkpoint"])
    model, endpoint_cfg = build_structured_model(
        metadata, project_config, base_checkpoint, model_cfg.get("model_id"), device, checkpoint
    )
    model.eval()
    matcher, action_model, pipeline = endpoint_pipeline(endpoint_cfg, metadata, device)
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
    original_complete_failure, original_complete_residual = final_failures(
        original_pipeline, matcher, test_dets_and_obs, 1, batch_size, device
    )
    original_actions = legacy_actions_for_rows(original_model, train_x, test_rows, device, batch_size)
    original_failure, original_residual = endpoint_outcomes(
        pipeline, action_model, matcher, test_dets, test_obs, original_actions, device, batch_size
    )
    baseline_failure = baseline_failures(matcher, test_dets_and_obs, 1, batch_size)

    # This is the decisive heat-start audit.  It uses a freshly constructed
    # structured model with *no* structured checkpoint loaded.  If it differs
    # from original actions (beyond exact old-logit == 0 ties), the conversion
    # itself is wrong; if it matches, any later regression is from training.
    warm_model, _ = build_structured_model(
        metadata, project_config, base_checkpoint, model_cfg.get("model_id"), device
    )
    warm_model.eval()
    warm_actions = actions_for_rows(warm_model, train_x, test_rows, device, batch_size, bias=0.0)
    warm_failure, warm_residual = endpoint_outcomes(
        pipeline, action_model, matcher, test_dets, test_obs, warm_actions, device, batch_size
    )
    action_mismatch = warm_actions != original_actions

    report = {
        "data_mode": "offline",
        "comparison_path": "train_x_to_fixed_action_endpoint",
        "teacher_checkpoint": str(checkpoint),
        "oracle_checkpoint": str(oracle_checkpoint),
        "base_ising_fast_checkpoint": str(base_checkpoint),
        "selected_on_validation": selected,
        "validation_candidates": validation,
        "held_out_test": {
            "pymatching": summarize(baseline_failure),
            "original_ising_fast": action_report(original_failure, original_residual, original_actions),
            "original_ising_fast_complete_pipeline_reference": {
                **summarize(original_complete_failure),
                "mean_residual_weight": float(original_complete_residual.mean()),
                "failure_mismatch_vs_fixed_actions": int(
                    (original_complete_failure != original_failure).sum()
                ),
            },
            "warm_start_audit": {
                "structured_warm_start": action_report(warm_failure, warm_residual, warm_actions),
                "original_actions_from_train_x": action_report(original_failure, original_residual, original_actions),
                "action_mismatch_bits": int(action_mismatch.sum()),
                "action_mismatch_shots": int(np.any(action_mismatch, axis=(1, 2, 3, 4)).sum()),
                "endpoint_failure_mismatch_shots": int((warm_failure != original_failure).sum()),
            },
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
