#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Derive EWC lambda candidates from a fixed first-switch probe checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[4]
CODE_ROOT = REPO_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from model.factory import ModelFactory  # noqa: E402
from scripts.compute_ewc_fisher import (  # noqa: E402
    _build_training_generator,
    _load_checkpoint_state_dict,
    _load_public_config,
)
from training.ewc import diagonal_ewc_penalty, load_ewc_state  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def lambda_candidates(
    data_loss: float, ewc_penalty: float, target_ratios: list[float]
) -> dict[str, float]:
    if data_loss <= 0 or ewc_penalty <= 0:
        raise ValueError("data loss and EWC penalty must both be positive")
    return {
        f"{ratio:g}": float(ratio) * float(data_loss) / float(ewc_penalty)
        for ratio in target_ratios
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference-fisher", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--positive-weight", type=float, default=1.0)
    parser.add_argument(
        "--target-ratio",
        action="append",
        type=float,
        default=None,
        help="Target lambda*L_EWC/L_data; defaults to .001,.01,.1,1.",
    )
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint if args.checkpoint.is_absolute() else REPO_ROOT / args.checkpoint
    reference_path = (
        args.reference_fisher
        if args.reference_fisher.is_absolute()
        else REPO_ROOT / args.reference_fisher
    )
    output = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    target_ratios = args.target_ratio or [0.001, 0.01, 0.1, 1.0]
    if any(ratio <= 0 for ratio in target_ratios):
        raise ValueError("target ratios must be positive")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cfg = _load_public_config(args.config_name)
    model = ModelFactory.create_model(cfg).to(device).eval()
    model.load_state_dict(_load_checkpoint_state_dict(checkpoint, device), strict=True)
    reference = load_ewc_state(reference_path)
    generator = _build_training_generator(cfg, device=device, seed=args.seed, verbose=False)
    train_x, train_y = generator.generate_batch(step=0, batch_size=args.batch_size)
    train_x = train_x.to(device)
    train_y = train_y.to(device)
    positive_weight = torch.tensor(float(args.positive_weight), device=device)
    data_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        model(train_x), train_y, reduction="sum", pos_weight=positive_weight
    ) / args.batch_size
    penalty = diagonal_ewc_penalty(model, [reference])
    candidates = lambda_candidates(float(data_loss), float(penalty), target_ratios)
    payload = {
        "protocol": "fixed_T1_batch_at_SEQ_1k_probe_endpoint",
        "config": args.config_name,
        "checkpoint": {"path": str(checkpoint), "sha256": sha256(checkpoint)},
        "reference_fisher": {
            "path": str(reference_path),
            "sha256": sha256(reference_path),
            "task": reference.task_name,
        },
        "seed": args.seed,
        "batch_size": args.batch_size,
        "positive_weight": args.positive_weight,
        "data_loss_per_sample_sum_outputs": float(data_loss),
        "ewc_penalty_unscaled": float(penalty),
        "ratio_at_lambda_one": float(penalty / data_loss),
        "lambda_by_target_ratio": candidates,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"[QEWC v3] EWC lambda candidates: {output}")


if __name__ == "__main__":
    main()
