#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run one real GPU optimizer step with a learned 25-parameter config."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from omegaconf import OmegaConf

from data.generator_torch import QCDataGeneratorTorch
from model.factory import ModelFactory
from qec.noise_model import NoiseModel, get_training_upscaled_noise_model
from workflows.config_validator import apply_public_defaults_and_model, validate_public_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output", type=Path,
        default=Path("outputs/google_noise_learning/smoke_train_checkpoint.pt"),
    )
    args = parser.parse_args()

    public_cfg = OmegaConf.load(args.config)
    spec = validate_public_config(public_cfg)
    cfg = apply_public_defaults_and_model(public_cfg, spec)
    raw_model = OmegaConf.to_container(cfg.data.noise_model, resolve=True)
    learned = NoiseModel.from_config_dict(dict(raw_model))
    active, upscale = get_training_upscaled_noise_model(
        learned,
        code_type=str(cfg.code),
        skip_upscale=bool(cfg.data.skip_noise_upscaling),
    )
    if active.sha256() != learned.sha256():
        raise RuntimeError("training noise differs from learned hardware noise")

    device = torch.device(args.device)
    generator = QCDataGeneratorTorch(
        distance=int(cfg.distance),
        n_rounds=int(cfg.n_rounds),
        p_error=1.25 * active.get_max_probability(),
        p_min=1.25 * active.get_max_probability(),
        p_max=1.25 * active.get_max_probability(),
        measure_basis=str(cfg.meas_basis),
        rank=device.index or 0,
        global_rank=0,
        mode="train",
        verbose=True,
        timelike_he=bool(cfg.data.timelike_he),
        num_he_cycles=int(cfg.data.num_he_cycles),
        use_weight2=bool(cfg.data.use_weight2_timelike),
        max_passes_w1=int(cfg.data.max_passes_w1),
        max_passes_w2=int(cfg.data.max_passes_w2),
        decompose_y=False,
        precomputed_frames_dir=None,
        code_rotation=str(cfg.data.code_rotation),
        noise_model=active,
        use_compile=False,
        device=device,
    )
    model = ModelFactory.create_model(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.optimizer.lr))
    x, y = generator.generate_batch(step=0, batch_size=args.batch_size)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits = model(x)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, y.to(dtype=torch.float32), reduction="mean"
    )
    loss.backward()
    optimizer.step()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "noise_model_sha256": active.sha256(),
            "noise_model": active.canonical_parameters(),
            "loss": float(loss.detach()),
            "input_shape": tuple(x.shape),
            "target_shape": tuple(y.shape),
        },
        args.output,
    )
    print(
        "SMOKE_TRAIN_OK",
        f"noise_sha256={active.sha256()}",
        f"upscale={upscale['message']}",
        f"x_shape={tuple(x.shape)}",
        f"y_shape={tuple(y.shape)}",
        f"loss={float(loss.detach()):.8f}",
        f"device={device}",
        f"checkpoint={args.output}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
