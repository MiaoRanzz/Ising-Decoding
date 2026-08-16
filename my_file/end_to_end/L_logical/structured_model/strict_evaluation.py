"""Streaming evaluation on fresh shots for the strict structured workflow."""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from common import (build_base_model, build_structured_model, endpoint_outcomes,
                    endpoint_pipeline, evaluation_no_op_bias_config, repo_path,
                    section, write_json)
from compare_three_paths import baseline_failures, final_failures
from evaluation.logical_error_rate import PreDecoderMemoryEvalModule, _build_stab_maps
from strict_data import StrictSurfaceSampler, strict_batch_plan
from structured_actions import actions_from_logits


def _empty_action_stats() -> dict[str, float | int]:
    return {"errors": 0, "samples": 0, "residual_sum": 0, "action_sum": 0}


def _add_action_stats(stats: dict, failures: np.ndarray, residual: np.ndarray, actions: np.ndarray) -> None:
    stats["errors"] += int(failures.sum())
    stats["samples"] += int(failures.size)
    stats["residual_sum"] += int(residual.sum())
    stats["action_sum"] += int(actions.sum())


def _action_report(stats: dict) -> dict[str, int | float]:
    n = int(stats["samples"])
    return {
        "logical_errors": int(stats["errors"]), "samples": n,
        "ler": stats["errors"] / n,
        "mean_residual_weight": stats["residual_sum"] / n,
        "mean_action_count": stats["action_sum"] / n,
    }


def _actions(model, train_x: torch.Tensor, *, structured: bool, bias: float = 0.0) -> np.ndarray:
    with torch.no_grad():
        logits = model(train_x.to(dtype=torch.float32))
    if structured:
        return actions_from_logits(logits, bias).cpu().numpy().astype(np.uint8)
    return (logits >= 0).cpu().numpy().astype(np.uint8)


def run_strict_evaluation(settings, model_cfg: dict[str, Any], cfg: dict[str, Any]) -> None:
    if str(cfg.get("mode", "full")).lower() != "full":
        raise ValueError("strict data mode supports structured_evaluation.mode=full; use offline for warm_start_audit")
    strict_cfg = section(settings, "structured_strict_data")
    device = torch.device(cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    sampler = StrictSurfaceSampler(strict_cfg, device)
    validation_plan = strict_batch_plan(strict_cfg, "validation")
    test_plan = strict_batch_plan(strict_cfg, "test")
    validation_samples, test_samples = validation_plan.num_samples, test_plan.num_samples
    base_checkpoint = repo_path(model_cfg["base_checkpoint"])
    project_config = repo_path(model_cfg["project_config"])
    model_id = model_cfg.get("model_id")
    metadata = sampler.metadata()
    teacher_checkpoint = repo_path(cfg.get("strict_checkpoint", cfg["checkpoint"]))
    oracle_checkpoint = repo_path(cfg.get("strict_oracle_checkpoint", cfg["oracle_checkpoint"]))
    teacher, _ = build_structured_model(
        metadata, project_config, base_checkpoint, model_id, device, teacher_checkpoint
    )
    teacher.eval()

    endpoint_contexts = {}
    for basis in sampler.bases:
        basis_metadata = sampler.metadata(basis)
        unused, endpoint_cfg = build_base_model(
            basis_metadata, project_config, base_checkpoint, model_id, device
        )
        del unused
        endpoint_contexts[basis] = endpoint_pipeline(endpoint_cfg, basis_metadata, device)

    scan_biases, biases, fixed_bias = evaluation_no_op_bias_config(cfg)
    validation_stats = {bias: _empty_action_stats() for bias in biases}
    validation = []
    selected = None
    if scan_biases:
        total_batches = validation_plan.num_batches
        batch_size = validation_plan.batch_size
        for batch_index in range(total_batches):
            count = batch_size
            fresh = sampler.generate(
                stream="validation", step=batch_index, batch_size=count, with_endpoint=True
            )
            with torch.no_grad():
                logits = teacher(fresh.train_x.to(device=device, dtype=torch.float32))
            matcher, action_model, pipeline = endpoint_contexts[fresh.basis]
            num_obs = int(sampler.metadata(fresh.basis)["num_observables"])
            dets, obs = fresh.dets_and_obs[:, :-num_obs], fresh.dets_and_obs[:, -num_obs:]
            for bias in biases:
                actions = actions_from_logits(logits, bias).detach().cpu().numpy().astype(np.uint8)
                failure, residual = endpoint_outcomes(
                    pipeline, action_model, matcher, dets, obs, actions, device, batch_size
                )
                _add_action_stats(validation_stats[bias], failure, residual, actions)
            if ((batch_index + 1) % max(1, int(cfg.get("log_every_batches", 25))) == 0
                    or batch_index + 1 == total_batches):
                print(
                    f"[strict evaluation validation] {min((batch_index+1)*batch_size, validation_samples)}/"
                    f"{validation_samples} shots basis={fresh.basis}", flush=True
                )
        validation = [{"no_op_bias": bias, **_action_report(validation_stats[bias])} for bias in biases]
        selected = min(
            validation,
            key=lambda item: (item["logical_errors"], item["mean_residual_weight"], item["mean_action_count"]),
        )
        selected_bias = float(selected["no_op_bias"])
    else:
        selected_bias = float(fixed_bias)
        print(f"[strict evaluation] fixed no_op_bias={selected_bias}; validation scan skipped", flush=True)

    oracle, _ = build_structured_model(
        metadata, project_config, base_checkpoint, model_id, device, oracle_checkpoint
    )
    warm, _ = build_structured_model(metadata, project_config, base_checkpoint, model_id, device)
    oracle.eval(); warm.eval()
    original_contexts = {}
    for basis in sampler.bases:
        basis_metadata = sampler.metadata(basis)
        original_model, original_cfg = build_base_model(
            basis_metadata, project_config, base_checkpoint, model_id, device
        )
        maps = _build_stab_maps(int(basis_metadata["distance"]), str(basis_metadata["code_rotation"]))
        original_pipeline = PreDecoderMemoryEvalModule(
            original_model, original_cfg, maps, device
        ).to(device).eval()
        original_contexts[basis] = (original_model, original_pipeline)

    teacher_stats = _empty_action_stats()
    oracle_stats = _empty_action_stats()
    warm_stats = _empty_action_stats()
    original_fixed_stats = _empty_action_stats()
    original_errors = original_samples = original_residual_sum = 0
    baseline_errors = 0
    mismatch_bits = mismatch_shots = endpoint_mismatch = complete_fixed_mismatch = 0
    paired = {"teacher_helpful": 0, "teacher_harmful": 0, "oracle_helpful": 0, "oracle_harmful": 0}
    total_batches = test_plan.num_batches
    batch_size = test_plan.batch_size
    for batch_index in range(total_batches):
        count = batch_size
        fresh = sampler.generate(stream="test", step=batch_index, batch_size=count, with_endpoint=True)
        basis_metadata = sampler.metadata(fresh.basis)
        num_obs = int(basis_metadata["num_observables"])
        detsobs = fresh.dets_and_obs
        dets, obs = detsobs[:, :-num_obs], detsobs[:, -num_obs:]
        matcher, action_model, pipeline = endpoint_contexts[fresh.basis]
        train_x = fresh.train_x.to(device=device, dtype=torch.float32)
        teacher_actions = _actions(teacher, train_x, structured=True, bias=selected_bias)
        oracle_actions = _actions(oracle, train_x, structured=True)
        warm_actions = _actions(warm, train_x, structured=True)
        original_model, original_pipeline = original_contexts[fresh.basis]
        original_actions = _actions(original_model, train_x, structured=False)
        outcomes = []
        for actions in (teacher_actions, oracle_actions, warm_actions, original_actions):
            outcomes.append(endpoint_outcomes(
                pipeline, action_model, matcher, dets, obs, actions, device, batch_size
            ))
        (teacher_failure, teacher_residual), (oracle_failure, oracle_residual), \
            (warm_failure, warm_residual), (fixed_failure, fixed_residual) = outcomes
        original_failure, original_residual = final_failures(
            original_pipeline, matcher, detsobs, num_obs, batch_size, device
        )
        baseline_failure = baseline_failures(matcher, detsobs, num_obs, batch_size)
        _add_action_stats(teacher_stats, teacher_failure, teacher_residual, teacher_actions)
        _add_action_stats(oracle_stats, oracle_failure, oracle_residual, oracle_actions)
        _add_action_stats(warm_stats, warm_failure, warm_residual, warm_actions)
        _add_action_stats(original_fixed_stats, fixed_failure, fixed_residual, original_actions)
        original_errors += int(original_failure.sum())
        original_samples += int(original_failure.size)
        original_residual_sum += int(original_residual.sum())
        complete_fixed_mismatch += int((original_failure != fixed_failure).sum())
        baseline_errors += int(baseline_failure.sum())
        mismatch = warm_actions != original_actions
        mismatch_bits += int(mismatch.sum())
        mismatch_shots += int(np.any(mismatch, axis=(1, 2, 3, 4)).sum())
        endpoint_mismatch += int((warm_failure != fixed_failure).sum())
        paired["teacher_helpful"] += int((fixed_failure & ~teacher_failure).sum())
        paired["teacher_harmful"] += int((~fixed_failure & teacher_failure).sum())
        paired["oracle_helpful"] += int((fixed_failure & ~oracle_failure).sum())
        paired["oracle_harmful"] += int((~fixed_failure & oracle_failure).sum())
        if (batch_index + 1) % max(1, int(cfg.get("log_every_batches", 25))) == 0 or batch_index + 1 == total_batches:
            print(
                f"[strict evaluation test] {min((batch_index+1)*batch_size, test_samples)}/"
                f"{test_samples} shots basis={fresh.basis}", flush=True
            )

    report = {
        "data_mode": "strict",
        "comparison_path": "train_x_to_fixed_action_endpoint",
        "strict_session_seed": sampler.session_seed,
        "strict_reference_config": str(sampler.reference_path),
        "strict_noise_config": str(sampler.noise_config_path),
        "strict_validation_num_batches": validation_plan.num_batches,
        "strict_validation_batch_size": validation_plan.batch_size,
        "strict_test_num_batches": test_plan.num_batches,
        "strict_test_batch_size": test_plan.batch_size,
        "teacher_checkpoint": str(teacher_checkpoint),
        "oracle_checkpoint": str(oracle_checkpoint),
        "base_ising_fast_checkpoint": str(base_checkpoint),
        "no_op_bias_selection": {
            "mode": "scan" if scan_biases else "fixed",
            "selected_no_op_bias": selected_bias,
        },
        "selected_on_validation": selected,
        "validation_candidates": validation,
        "held_out_test": {
            "pymatching": {
                "logical_errors": baseline_errors, "samples": test_samples,
                "ler": baseline_errors / test_samples,
            },
            "original_ising_fast": _action_report(original_fixed_stats),
            "original_ising_fast_complete_pipeline_reference": {
                "logical_errors": original_errors, "samples": original_samples,
                "ler": original_errors / original_samples,
                "mean_residual_weight": original_residual_sum / original_samples,
                "failure_mismatch_vs_fixed_actions": complete_fixed_mismatch,
            },
            "warm_start_audit": {
                "structured_warm_start": _action_report(warm_stats),
                "original_actions_from_train_x": _action_report(original_fixed_stats),
                "action_mismatch_bits": mismatch_bits,
                "action_mismatch_shots": mismatch_shots,
                "endpoint_failure_mismatch_shots": endpoint_mismatch,
            },
            "structured_oracle": _action_report(oracle_stats),
            "structured_teacher": _action_report(teacher_stats),
            "paired_vs_original": paired,
        },
    }
    write_json(repo_path(cfg.get("strict_output", cfg["output"])), report)
    print(report)
