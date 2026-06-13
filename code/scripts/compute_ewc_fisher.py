#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Estimate and save a diagonal EWC Fisher state for one predecoder task."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Mapping

import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from data.generator_torch import QCDataGeneratorTorch
from model.factory import ModelFactory
from qec.noise_model import NoiseModel, get_training_upscaled_noise_model
from training.ewc import estimate_diagonal_fisher, save_ewc_state
from training.train import resolve_precomputed_frames_dir
from workflows.config_validator import apply_public_defaults_and_model, validate_public_config


def _strip_known_prefixes(name: str) -> str:
    prefixes = ("module.", "_orig_mod.")
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if name.startswith(prefix):
                name = name[len(prefix):]
                changed = True
    return name


def normalize_state_dict_keys(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Normalize DDP / torch.compile checkpoint keys to plain module keys."""
    return {_strip_known_prefixes(str(key)): value for key, value in state_dict.items()}


def select_model_checkpoint(path: str | Path) -> Path:
    """Select the latest model state-dict checkpoint from a file or model directory."""
    candidate = Path(path)
    if candidate.is_file():
        return candidate
    if not candidate.is_dir():
        raise FileNotFoundError(f"Checkpoint path does not exist: {candidate}")

    files = [
        item for item in candidate.glob("*.pt")
        if item.is_file() and not item.name.startswith("checkpoint")
    ]
    if not files:
        raise FileNotFoundError(f"No model .pt files found under {candidate}")

    def score(item: Path):
        numbers = [int(x) for x in re.findall(r"\d+", item.stem)]
        epoch = numbers[-1] if numbers else -1
        return (epoch, item.stat().st_mtime, item.name)

    return sorted(files, key=score)[-1]


def _load_checkpoint_state_dict(path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    raw = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(raw, dict):
        raise ValueError(f"Unsupported checkpoint format in {path}: {type(raw).__name__}")
    if "model_state_dict" in raw:
        state_dict = raw["model_state_dict"]
    elif "state_dict" in raw:
        state_dict = raw["state_dict"]
    else:
        state_dict = raw
    return normalize_state_dict_keys(state_dict)


def _load_public_config(config_name: str):
    config_path = REPO_ROOT / "conf" / f"{config_name}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    public_cfg = OmegaConf.load(config_path)
    if "workflow" not in public_cfg:
        public_cfg.workflow = {}
    public_cfg.workflow.task = "train"
    model_spec = validate_public_config(public_cfg)
    return apply_public_defaults_and_model(public_cfg, model_spec)


def _build_training_generator(cfg, *, device: torch.device, seed: int, verbose: bool):
    noise_model_cfg = getattr(cfg.data, "noise_model", None)
    if noise_model_cfg is None:
        raise ValueError("EWC Fisher estimation requires cfg.data.noise_model for a single task")
    nm_dict = OmegaConf.to_container(noise_model_cfg, resolve=True)
    user_noise_model = NoiseModel.from_config_dict(dict(nm_dict))
    skip_upscale = bool(getattr(cfg.data, "skip_noise_upscaling", False))
    train_noise_model, upscale_info = get_training_upscaled_noise_model(
        user_noise_model,
        code_type=getattr(cfg.data, "code_type", "surface_code"),
        skip_upscale=skip_upscale,
    )
    p_error = float(1.25 * train_noise_model.get_max_probability())

    precomputed_frames_dir = resolve_precomputed_frames_dir(
        getattr(cfg.data, "precomputed_frames_dir", None),
        cfg.distance,
        cfg.n_rounds,
        cfg.meas_basis,
        rank=0,
    )

    if verbose:
        print(
            "[EWC Fisher] noise_model "
            f"sha256={train_noise_model.sha256()} "
            f"p_placeholder={p_error:.6g} "
            f"max_group={float(upscale_info['max_group']):.6g}"
        )

    compute_dtype = None
    dtype_raw = getattr(cfg.data, "compute_dtype", None)
    if dtype_raw is not None:
        compute_dtype = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }.get(str(dtype_raw), None)

    return QCDataGeneratorTorch(
        distance=cfg.distance,
        n_rounds=cfg.n_rounds,
        p_error=p_error,
        p_min=p_error,
        p_max=p_error,
        measure_basis=cfg.meas_basis,
        rank=0,
        global_rank=0,
        mode="train",
        verbose=verbose,
        base_seed=seed,
        timelike_he=bool(getattr(cfg.data, "timelike_he", True)),
        num_he_cycles=int(getattr(cfg.data, "num_he_cycles", 1)),
        max_passes_w1=int(getattr(cfg.data, "max_passes_w1", 32)),
        max_passes_w2=int(getattr(cfg.data, "max_passes_w2", 32)),
        decompose_y=False,
        precomputed_frames_dir=precomputed_frames_dir,
        code_rotation=getattr(cfg.data, "code_rotation", "XV"),
        noise_model=train_noise_model,
        device=device,
        use_compile=bool(getattr(cfg.data, "use_compile", False)),
        compile_chunk_size=int(getattr(cfg.data, "compile_chunk_size", 2)),
        compute_dtype=compute_dtype,
        use_weight2=bool(getattr(cfg.data, "use_weight2", False)),
        use_coset_search=bool(getattr(cfg.data, "use_coset_search", False)),
        coset_max_generators=int(getattr(cfg.data, "coset_max_generators", 20)),
        use_dense_overlap=bool(getattr(cfg.data, "use_dense_overlap", False)),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", required=True, help="conf/<name>.yaml without extension")
    parser.add_argument("--checkpoint", required=True, help="model .pt file or models directory")
    parser.add_argument("--output", required=True, help="output EWC .pt path")
    parser.add_argument("--task-name", required=True, help="task label stored in the EWC state")
    parser.add_argument("--num-samples", type=int, default=65536)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cfg = _load_public_config(args.config_name)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = bool(getattr(cfg, "enable_matmul_tf32", True))
        torch.backends.cudnn.allow_tf32 = bool(getattr(cfg, "enable_cudnn_tf32", True))

    checkpoint = select_model_checkpoint(args.checkpoint)
    model = ModelFactory.create_model(cfg).to(device)
    state_dict = _load_checkpoint_state_dict(checkpoint, device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint load mismatch for {checkpoint}: missing={missing}, unexpected={unexpected}"
        )

    generator = _build_training_generator(cfg, device=device, seed=args.seed, verbose=True)
    params = sum(p.numel() for p in model.parameters())
    print(
        f"[EWC Fisher] task={args.task_name} config={args.config_name} "
        f"checkpoint={checkpoint} params={params:,} samples={args.num_samples} batch={args.batch_size}"
    )
    ewc_state = estimate_diagonal_fisher(
        model,
        generator,
        task_name=args.task_name,
        num_samples=int(args.num_samples),
        batch_size=int(args.batch_size),
        device=device,
        enable_fp16=bool(args.fp16),
        enable_bf16=bool(args.bf16),
    )
    save_ewc_state(ewc_state, args.output)
    print(f"[EWC Fisher] saved {args.output}")


if __name__ == "__main__":
    main()
