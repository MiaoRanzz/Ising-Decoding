#!/usr/bin/env python3
"""Run the complete structured oracle -> teacher -> evaluation workflow."""
from __future__ import annotations

import gc

import torch

from common import DEFAULT_SETTINGS, section
from evaluate_structured_ising import main as evaluate_main
from generate_endpoint_teacher import main as generate_teacher_main
from train_structured_ising import main as train_main


def release_memory() -> None:
    """Release models from the preceding stage before constructing the next."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    settings = DEFAULT_SETTINGS.resolve()
    model_cfg = section(settings, "structured_model")
    data_mode = str(model_cfg.get("data_mode", "offline")).lower()
    if data_mode not in {"offline", "strict"}:
        raise ValueError("structured_model.data_mode must be 'offline' or 'strict'")

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
