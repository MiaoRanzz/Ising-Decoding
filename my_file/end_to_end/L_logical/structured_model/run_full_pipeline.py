#!/usr/bin/env python3
"""Run the complete structured oracle -> teacher -> evaluation workflow."""
from __future__ import annotations

import gc

import torch

from common import DEFAULT_SETTINGS, repo_path, section
from evaluate_structured_ising import main as evaluate_main
from generate_endpoint_teacher import main as generate_teacher_main
from qec.dem_sampling import _reset_sampler_cache
from train_structured_ising import main as train_main


def release_memory() -> None:
    """Release models from the preceding stage before constructing the next."""
    # The multi-stream DEM cache intentionally owns strong references during a
    # stage so train/validation RNG states survive alternation. At a pipeline
    # boundary those streams are finished and must be released with the models.
    _reset_sampler_cache()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def validate_pipeline_paths(settings, data_mode: str) -> None:
    """Fail early if a full pipeline would train one artifact and test another."""
    oracle_cfg = section(settings, "structured_oracle_training")
    teacher_cfg = section(settings, "structured_teacher_training")
    evaluation_cfg = section(settings, "structured_evaluation")
    strict = data_mode == "strict"
    oracle_dir_key = "strict_output_dir" if strict else "output_dir"
    teacher_dir_key = "strict_output_dir" if strict else "output_dir"
    teacher_resume_key = "strict_resume_checkpoint" if strict else "resume_checkpoint"
    eval_teacher_key = "strict_checkpoint" if strict else "checkpoint"
    eval_oracle_key = "strict_oracle_checkpoint" if strict else "oracle_checkpoint"

    expected_oracle = (repo_path(oracle_cfg[oracle_dir_key]) / "best.pt").resolve()
    expected_teacher = (repo_path(teacher_cfg[teacher_dir_key]) / "best.pt").resolve()
    configured = {
        f"structured_teacher_training.{teacher_resume_key}": repo_path(
            teacher_cfg[teacher_resume_key]
        ).resolve(),
        f"structured_evaluation.{eval_teacher_key}": repo_path(
            evaluation_cfg[eval_teacher_key]
        ).resolve(),
        f"structured_evaluation.{eval_oracle_key}": repo_path(
            evaluation_cfg[eval_oracle_key]
        ).resolve(),
    }
    expected = {
        f"structured_teacher_training.{teacher_resume_key}": expected_oracle,
        f"structured_evaluation.{eval_teacher_key}": expected_teacher,
        f"structured_evaluation.{eval_oracle_key}": expected_oracle,
    }
    mismatches = [
        f"{key}: configured={configured[key]} expected={expected[key]}"
        for key in configured
        if configured[key] != expected[key]
    ]
    if mismatches:
        raise ValueError(
            f"{data_mode} pipeline checkpoint paths are inconsistent:\n  "
            + "\n  ".join(mismatches)
        )


def main() -> None:
    settings = DEFAULT_SETTINGS.resolve()
    model_cfg = section(settings, "structured_model")
    data_mode = str(model_cfg.get("data_mode", "offline")).lower()
    if data_mode not in {"offline", "strict"}:
        raise ValueError("structured_model.data_mode must be 'offline' or 'strict'")
    validate_pipeline_paths(settings, data_mode)

    total_stages = 4 if data_mode == "offline" else 3
    print(f"[pipeline 1/{total_stages}] training oracle (data_mode={data_mode})", flush=True)
    train_main("oracle")
    release_memory()

    if data_mode == "offline":
        print("[pipeline 2a/4] generating offline endpoint teacher corpus", flush=True)
        generate_teacher_main()
        release_memory()
        teacher_label = "3/4"
        evaluation_label = "4/4"
    else:
        # Strict training generates endpoint teachers inside every fresh batch.
        teacher_label = "2/3"
        evaluation_label = "3/3"

    print(f"[pipeline {teacher_label}] training teacher", flush=True)
    train_main("teacher")
    release_memory()

    print(f"[pipeline {evaluation_label}] evaluating", flush=True)
    evaluate_main()
    print("[pipeline done] oracle, teacher, and evaluation completed", flush=True)


if __name__ == "__main__":
    main()
