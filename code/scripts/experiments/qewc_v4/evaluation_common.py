#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared frozen paths and checkpoint helpers for QEWC-v4 evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
FORMAL_ROOT = REPO_ROOT / "outputs" / "qewc_v4" / "formal"
FINAL_STREAM_ROOT = (
    REPO_ROOT / "outputs" / "qewc_v3" / "evaluation_streams" / "formal" / "stress"
)
TASKS = ("T0", "T1", "T2", "T3", "T4")
HIGH_ORDER = ("T1", "T3", "T2", "T4")
REVERSE_ORDER = tuple(reversed(HIGH_ORDER))


@dataclass(frozen=True)
class EvaluationArchitecture:
    key: str
    short: str
    model_id: int
    base_config: str
    task_config_dir: str
    checkpoint_prefix: str


ARCHITECTURES = {
    "htnet_a101": EvaluationArchitecture(
        "htnet_a101",
        "H",
        111,
        "experiments/qewc_v4/htnet/T0",
        "experiments/qewc_v4/htnet",
        "HTnet",
    ),
    "ising_fast": EvaluationArchitecture(
        "ising_fast",
        "I",
        1,
        "experiments/qewc_v4/ising/t0_base",
        "experiments/qewc_v4/ising",
        "PreDecoderModelMemory_v1",
    ),
}


def run_ids(architecture: str) -> tuple[str, ...]:
    prefix = ARCHITECTURES[architecture].short
    return (
        f"{prefix}_T0",
        f"{prefix}_SEQ",
        f"{prefix}_EWC",
        f"{prefix}_REPLAY",
        f"{prefix}_MIX_TOTAL",
        f"{prefix}_MIX_EXPOSURE",
        f"{prefix}_REV_SEQ",
        f"{prefix}_REV_EWC",
    )


def config_path(architecture: str, task: str) -> Path:
    spec = ARCHITECTURES[architecture]
    relative = spec.base_config if task == "T0" else f"{spec.task_config_dir}/{task}"
    return REPO_ROOT / "conf" / f"{relative}.yaml"


def training_manifest_path(run_id: str) -> Path:
    return FORMAL_ROOT / run_id / "run_manifest.json"


def load_training(run_id: str) -> dict:
    path = training_manifest_path(run_id)
    return json.loads(path.read_text(encoding="utf-8"))


def checkpoint_items(training: dict) -> list[tuple[str, Path]]:
    field = "checkpoints" if training["method"] in {"mix_total", "mix_exposure"} else "endpoints"
    return [(name, Path(item["path"])) for name, item in training[field].items()]


def final_checkpoint(training: dict) -> Path:
    return checkpoint_items(training)[-1][1]
