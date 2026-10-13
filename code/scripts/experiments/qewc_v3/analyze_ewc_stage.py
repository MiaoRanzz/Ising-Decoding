#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Write per-layer Fisher and parameter-drift diagnostics for one EWC stage."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[4]
CODE_ROOT = REPO_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from scripts.compute_ewc_fisher import (  # noqa: E402
    _load_checkpoint_state_dict,
    select_model_checkpoint,
)
from training.ewc import canonical_param_name, load_ewc_state  # noqa: E402


def layer_name(parameter_name: str) -> str:
    parameter_name = canonical_param_name(parameter_name)
    return parameter_name.rsplit(".", 1)[0] if "." in parameter_name else parameter_name


def summarize_stage(
    checkpoint_state: dict[str, torch.Tensor],
    reference_state,
    current_fisher_state,
) -> dict:
    current = {
        canonical_param_name(name): value.detach().cpu().float()
        for name, value in checkpoint_state.items()
        if torch.is_tensor(value)
    }
    layers: dict[str, dict[str, float | int]] = {}
    missing = []
    for name, reference_mean in reference_state.mean.items():
        if name not in current:
            missing.append(name)
            continue
        delta = current[name] - reference_mean.detach().cpu().float()
        reference_fisher = reference_state.fisher[name].detach().cpu().float()
        current_fisher = current_fisher_state.fisher.get(name)
        current_fisher = (
            torch.zeros_like(reference_fisher)
            if current_fisher is None
            else current_fisher.detach().cpu().float()
        )
        key = layer_name(name)
        row = layers.setdefault(
            key,
            {
                "parameter_count": 0,
                "drift_squared_sum": 0.0,
                "drift_max_abs": 0.0,
                "fisher_weighted_drift": 0.0,
                "reference_fisher_sum": 0.0,
                "reference_fisher_max": 0.0,
                "current_fisher_sum": 0.0,
                "current_fisher_max": 0.0,
            },
        )
        row["parameter_count"] += int(delta.numel())
        row["drift_squared_sum"] += float(delta.square().sum())
        row["drift_max_abs"] = max(row["drift_max_abs"], float(delta.abs().max()))
        row["fisher_weighted_drift"] += float(
            0.5 * (reference_fisher * delta.square()).sum()
        )
        row["reference_fisher_sum"] += float(reference_fisher.sum())
        row["reference_fisher_max"] = max(
            row["reference_fisher_max"], float(reference_fisher.max())
        )
        row["current_fisher_sum"] += float(current_fisher.sum())
        row["current_fisher_max"] = max(
            row["current_fisher_max"], float(current_fisher.max())
        )

    if missing:
        raise ValueError(f"Checkpoint is missing reference parameters: {missing}")

    for row in layers.values():
        count = int(row["parameter_count"])
        row["drift_l2"] = math.sqrt(float(row.pop("drift_squared_sum")))
        row["drift_rms"] = row["drift_l2"] / math.sqrt(count)
        row["reference_fisher_mean"] = row["reference_fisher_sum"] / count
        row["current_fisher_mean"] = row["current_fisher_sum"] / count

    totals = {
        "parameter_count": sum(int(row["parameter_count"]) for row in layers.values()),
        "fisher_weighted_drift": sum(
            float(row["fisher_weighted_drift"]) for row in layers.values()
        ),
        "drift_l2": math.sqrt(
            sum(float(row["drift_l2"]) ** 2 for row in layers.values())
        ),
        "reference_fisher_sum": sum(
            float(row["reference_fisher_sum"]) for row in layers.values()
        ),
        "current_fisher_sum": sum(
            float(row["current_fisher_sum"]) for row in layers.values()
        ),
    }
    return {
        "reference_task": reference_state.task_name,
        "current_fisher_task": current_fisher_state.task_name,
        "totals": totals,
        "layers": dict(sorted(layers.items())),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--current-fisher", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = select_model_checkpoint(args.checkpoint)
    checkpoint_state = _load_checkpoint_state_dict(checkpoint, torch.device("cpu"))
    reference = load_ewc_state(args.reference)
    current_fisher = load_ewc_state(args.current_fisher)
    summary = summarize_stage(checkpoint_state, reference, current_fisher)
    summary.update(
        checkpoint=str(checkpoint),
        reference_state=str(args.reference),
        current_fisher_state=str(args.current_fisher),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"[QEWC v3] EWC stage diagnostics: {args.output}")


if __name__ == "__main__":
    main()
