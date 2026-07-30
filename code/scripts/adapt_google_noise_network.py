#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adapt the paper noise-learning network to Google hardware syndromes."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from noise_learning.google_qec import GoogleQECDataset, PARAMETER_NAMES
from noise_learning.paper_input import google_experiment_to_paper_tensor
from noise_learning.paper_model import NoiseLearningNetwork
from qec.noise_model import NoiseModel


def _aggregate_logits(
    model: NoiseLearningNetwork,
    values: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    total = torch.zeros(25, device=device)
    count = 0
    model.eval()
    with torch.no_grad():
        for offset in range(0, values.shape[0], batch_size):
            batch = values[offset : offset + batch_size].to(device)
            logits = model.per_sample_logits(batch)
            total += logits.sum(dim=0)
            count += logits.shape[0]
    return total / count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--fit-result", type=Path, required=True)
    parser.add_argument("--experiment-key", action="append", required=True)
    parser.add_argument("--max-shots", type=int, default=20_000)
    parser.add_argument("--validation-shots", type=int, default=2_000)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=3e-2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prediction-output", type=Path, required=True)
    parser.add_argument(
        "--training-config",
        type=Path,
        help="Also emit a complete decoder-training YAML from this base config.",
    )
    parser.add_argument("--training-config-output", type=Path)
    parser.add_argument("--target-distance", type=int)
    parser.add_argument("--target-rounds", type=int)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    fit_payload = yaml.safe_load(args.fit_result.read_text())
    target_model = NoiseModel.from_config_dict(fit_payload["noise_model"])
    target_values = torch.tensor(
        [target_model.canonical_parameters()[name] for name in PARAMETER_NAMES],
        dtype=torch.float32,
        device=device,
    )

    dataset = GoogleQECDataset(args.source)
    train_sets: list[torch.Tensor] = []
    validation_sets: list[torch.Tensor] = []
    for key in args.experiment_key:
        experiment = dataset.load(key, max_shots=args.max_shots)
        values = google_experiment_to_paper_tensor(experiment)
        validation_count = min(args.validation_shots, max(1, values.shape[0] // 5))
        train_sets.append(values[:-validation_count])
        validation_sets.append(values[-validation_count:])
        print(
            f"Mapped {key}: train={values.shape[0] - validation_count}, "
            f"validation={validation_count}, shape={tuple(values.shape[1:])}"
        )

    model = NoiseLearningNetwork().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    generator = torch.Generator().manual_seed(args.seed)
    final_loss = float("nan")
    for step in range(args.steps):
        source_index = step % len(train_sets)
        source = train_sets[source_index]
        indices = torch.randint(
            source.shape[0], (args.batch_size,), generator=generator
        )
        batch = source[indices].to(device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        predicted = model(batch)
        loss = torch.mean((torch.log(predicted) - torch.log(target_values)) ** 2)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach())
        if step < 3 or (step + 1) % 25 == 0:
            print(f"step={step + 1}/{args.steps} log_mse={final_loss:.8f}")

    logits_sum = torch.zeros(25, device=device)
    sample_count = 0
    for values in validation_sets:
        logits = _aggregate_logits(
            model, values, batch_size=args.batch_size, device=device
        )
        logits_sum += logits * values.shape[0]
        sample_count += values.shape[0]
    inferred = model.bounded_log_space(logits_sum / sample_count).detach().cpu()
    target_cpu = target_values.detach().cpu()
    validation_log_rmse = float(
        torch.sqrt(torch.mean((torch.log(inferred) - torch.log(target_cpu)) ** 2))
    )
    inferred_model = NoiseModel.from_config_dict(
        dict(zip(PARAMETER_NAMES, inferred.tolist(), strict=True))
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "architecture": "Chamberland2026NoiseLearning_Eqs58_61",
            "parameter_names": PARAMETER_NAMES,
            "target_noise_model": target_model.canonical_parameters(),
            "target_noise_model_sha256": target_model.sha256(),
            "inferred_noise_model": inferred_model.canonical_parameters(),
            "inferred_noise_model_sha256": inferred_model.sha256(),
            "validation_log_rmse": validation_log_rmse,
            "steps": args.steps,
            "final_training_log_mse": final_loss,
            "experiment_keys": args.experiment_key,
        },
        args.output,
    )
    prediction = {
        "noise_model": inferred_model.canonical_parameters(),
        "noise_model_sha256": inferred_model.sha256(),
        "adaptation": {
            "architecture": "Chamberland2026NoiseLearning_Eqs58_61",
            "target_noise_model_sha256": target_model.sha256(),
            "validation_log_rmse": validation_log_rmse,
            "steps": args.steps,
            "final_training_log_mse": final_loss,
            "experiment_keys": args.experiment_key,
            "checkpoint": str(args.output),
        },
    }
    args.prediction_output.parent.mkdir(parents=True, exist_ok=True)
    args.prediction_output.write_text(yaml.safe_dump(prediction, sort_keys=False))
    if args.training_config:
        if args.training_config_output is None:
            raise SystemExit(
                "--training-config-output is required with --training-config"
            )
        config = yaml.safe_load(args.training_config.read_text())
        config.setdefault("data", {})["noise_model"] = (
            inferred_model.canonical_parameters()
        )
        config["data"]["skip_noise_upscaling"] = True
        if args.target_distance is not None:
            config["distance"] = int(args.target_distance)
        if args.target_rounds is not None:
            config["n_rounds"] = int(args.target_rounds)
        args.training_config_output.parent.mkdir(parents=True, exist_ok=True)
        args.training_config_output.write_text(
            yaml.safe_dump(config, sort_keys=False)
        )
        print(f"Wrote decoder training config: {args.training_config_output}")
    print(
        "NOISE_NETWORK_ADAPT_OK",
        f"target_sha256={target_model.sha256()}",
        f"inferred_sha256={inferred_model.sha256()}",
        f"validation_log_rmse={validation_log_rmse:.8f}",
        f"checkpoint={args.output}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
