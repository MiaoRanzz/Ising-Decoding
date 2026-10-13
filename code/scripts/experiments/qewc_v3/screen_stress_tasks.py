#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Measure batch-level difficulty and conflict for QEWC v3 Stress candidates."""

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

from model.factory import ModelFactory  # noqa: E402
from scripts.compute_ewc_fisher import (  # noqa: E402
    _build_training_generator,
    _load_checkpoint_state_dict,
    _load_public_config,
)


STRESS_DIR = REPO_ROOT / "conf" / "experiments" / "qewc_v3" / "stress"
BASE_CONFIG = "experiments/qewc_v3/realistic/t0_base"


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    denominator = float(left.norm() * right.norm())
    return float(torch.dot(left, right) / denominator) if denominator else float("nan")


def measure_task(
    model: torch.nn.Module,
    config_name: str,
    *,
    device: torch.device,
    seed: int,
    batch_size: int,
    batches: int,
) -> tuple[dict, torch.Tensor, torch.Tensor]:
    cfg = _load_public_config(config_name)
    generator = _build_training_generator(cfg, device=device, seed=seed, verbose=False)
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction="mean")
    gradient_sum = [torch.zeros_like(param, device="cpu", dtype=torch.float32)
                    for param in model.parameters() if param.requires_grad]
    representation_sum = None
    loss_sum = 0.0
    syndrome_ones = 0
    syndrome_elements = 0
    sample_count = 0
    captured = []

    def capture_representation(_module, _inputs, output):
        captured.append(output.detach().float().mean(dim=(0, 2, 3, 4)).cpu())

    handle = model.head_activation.register_forward_hook(capture_representation)
    try:
        for step in range(batches):
            train_x, train_y = generator.generate_batch(step=step, batch_size=batch_size)
            train_x = train_x.to(device)
            train_y = train_y.to(device)
            model.zero_grad(set_to_none=True)
            output = model(train_x)
            loss = loss_fn(output, train_y)
            loss.backward()
            loss_sum += float(loss.detach())
            syndrome_ones += int(torch.count_nonzero(train_x).item())
            syndrome_elements += int(train_x.numel())
            sample_count += int(train_x.shape[0])
            for accumulator, parameter in zip(
                gradient_sum, (p for p in model.parameters() if p.requires_grad)
            ):
                if parameter.grad is not None:
                    accumulator.add_(parameter.grad.detach().float().cpu())
            current_representation = captured.pop()
            representation_sum = (
                current_representation
                if representation_sum is None
                else representation_sum + current_representation
            )
    finally:
        handle.remove()
        model.zero_grad(set_to_none=True)

    gradient = torch.cat([value.reshape(-1) for value in gradient_sum]) / batches
    representation = representation_sum / batches
    metrics = {
        "config": config_name,
        "samples": sample_count,
        "batches": batches,
        "batch_size": batch_size,
        "bce_nll": loss_sum / batches,
        "syndrome_density": syndrome_ones / syndrome_elements,
        "gradient_norm": float(gradient.norm()),
        "representation_norm": float(representation.norm()),
    }
    return metrics, gradient, representation


def pairwise(names: list[str], vectors: dict[str, torch.Tensor], metric: str) -> dict:
    rows = {}
    for left in names:
        rows[left] = {}
        for right in names:
            if metric == "cosine":
                value = cosine(vectors[left], vectors[right])
            elif metric == "distance":
                value = float(torch.linalg.vector_norm(vectors[left] - vectors[right]))
            else:
                raise ValueError(metric)
            rows[left][right] = value
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.batches <= 0:
        raise ValueError("batch size and batches must be positive")
    checkpoint = args.checkpoint if args.checkpoint.is_absolute() else REPO_ROOT / args.checkpoint
    output = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cfg = _load_public_config(BASE_CONFIG)
    model = ModelFactory.create_model(cfg).to(device).eval()
    model.load_state_dict(_load_checkpoint_state_dict(checkpoint, device), strict=True)

    candidates = [
        ("T0", BASE_CONFIG),
        *(
            (path.stem, f"experiments/qewc_v3/stress/{path.stem}")
            for path in sorted(STRESS_DIR.glob("*.yaml"))
        ),
    ]
    metrics = {}
    gradients = {}
    representations = {}
    for name, config_name in candidates:
        print(f"[Stress screen] task={name} config={config_name}", flush=True)
        task_metrics, gradient, representation = measure_task(
            model,
            config_name,
            device=device,
            seed=args.seed,
            batch_size=args.batch_size,
            batches=args.batches,
        )
        metrics[name] = task_metrics
        gradients[name] = gradient
        representations[name] = representation

    names = [name for name, _ in candidates]
    payload = {
        "checkpoint": str(checkpoint),
        "seed": args.seed,
        "estimator": "mean_gradient_over_fixed_minibatches",
        "representation": "mean_head_activation_channels",
        "metrics": metrics,
        "gradient_cosine": pairwise(names, gradients, "cosine"),
        "representation_distance": pairwise(names, representations, "distance"),
    }
    if any(not math.isfinite(value["bce_nll"]) for value in metrics.values()):
        raise RuntimeError("non-finite stress-screen metric")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"[Stress screen] wrote {output}")


if __name__ == "__main__":
    main()
