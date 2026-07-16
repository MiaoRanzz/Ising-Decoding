#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate fixed unknown T0-random noise configs for generalization tests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from omegaconf import OmegaConf

CODE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_ROOT.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from qec.noise_model import NoiseModel  # noqa: E402
from scripts.config_paths import config_name_from_path  # noqa: E402


DEFAULT_BASE_CONFIG = "conf/config_domestic_fast_opt_stfusion_r9_x_seq_ewc_t0_base.yaml"
DEFAULT_PREFIX = "config_unknown_t0_random"
DEFAULT_OUTPUT_DIR = "conf/experiments/unknown_t0_random"
DEFAULT_SEED = 20260714
DEFAULT_NUM_ENVS = 5
DEFAULT_FRAC = 0.25


def rel(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def _plain_mapping(value: Any) -> dict[str, float]:
    raw = OmegaConf.to_container(value, resolve=True) if hasattr(value, "items") else value
    if raw is None:
        raise ValueError("base config does not contain data.noise_model")
    return {str(key): float(item) for key, item in dict(raw).items()}


def load_base_noise_model(base_config: str | Path) -> dict[str, float]:
    cfg = OmegaConf.load(rel(base_config))
    noise_cfg = getattr(getattr(cfg, "data", None), "noise_model", None)
    noise = _plain_mapping(noise_cfg)
    return NoiseModel.from_config_dict(noise).to_config_dict()


def generate_perturbed_noise_models(
    base_noise: Mapping[str, float],
    *,
    num_envs: int = DEFAULT_NUM_ENVS,
    frac: float = DEFAULT_FRAC,
    seed: int = DEFAULT_SEED,
) -> list[dict[str, Any]]:
    if int(num_envs) <= 0:
        raise ValueError(f"num_envs must be positive, got {num_envs}")
    if float(frac) < 0:
        raise ValueError(f"frac must be non-negative, got {frac}")

    base = NoiseModel.from_config_dict(dict(base_noise)).to_config_dict()
    rng = np.random.default_rng(int(seed))
    generated = []
    for env_index in range(int(num_envs)):
        nm = NoiseModel.from_config_dict(base)
        nm.randomize_around_reference(frac=float(frac), rng=rng)
        noise = {key: float(value) for key, value in nm.to_config_dict().items()}
        multipliers = {
            key: (float(noise[key]) / float(base[key]) if float(base[key]) else None)
            for key in noise
        }
        generated.append(
            {
                "env_index": env_index,
                "env_key": f"e{env_index:02d}",
                "noise_model": noise,
                "multipliers": multipliers,
                "noise_model_sha256": NoiseModel.from_config_dict(noise).sha256(),
            }
        )
    return generated


def _config_name(prefix: str, seed: int, env_index: int) -> str:
    return f"{prefix}_s{int(seed)}_e{env_index:02d}"


def _render_config(base_cfg: Any, noise_model: Mapping[str, float], *, header: str) -> str:
    cfg = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=True))
    cfg.data.noise_model = dict(noise_model)
    return header + OmegaConf.to_yaml(cfg, resolve=True)


def write_unknown_configs(
    *,
    base_config: str | Path = DEFAULT_BASE_CONFIG,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    num_envs: int = DEFAULT_NUM_ENVS,
    frac: float = DEFAULT_FRAC,
    seed: int = DEFAULT_SEED,
    prefix: str = DEFAULT_PREFIX,
    manifest: str | Path | None = None,
) -> tuple[list[Path], dict[str, Any]]:
    base_path = rel(base_config)
    if not base_path.exists():
        raise FileNotFoundError(base_path)
    out_dir = rel(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = OmegaConf.load(base_path)
    base_noise = load_base_noise_model(base_path)
    generated = generate_perturbed_noise_models(
        base_noise,
        num_envs=int(num_envs),
        frac=float(frac),
        seed=int(seed),
    )

    paths = []
    environments = []
    for item in generated:
        env_index = int(item["env_index"])
        config_name = _config_name(prefix, int(seed), env_index)
        filename = f"{config_name}.yaml"
        path = out_dir / filename
        header = (
            "# Auto-generated unknown T0 random-noise environment.\n"
            f"# base_config: {base_path.name}\n"
            f"# seed: {int(seed)}\n"
            f"# frac: {float(frac):.6g}\n"
            f"# env_key: {item['env_key']}\n"
            f"# noise_model_sha256: {item['noise_model_sha256']}\n\n"
        )
        path.write_text(
            _render_config(base_cfg, item["noise_model"], header=header),
            encoding="utf-8",
        )
        paths.append(path)
        environments.append(
            {
                "env_index": env_index,
                "env_key": item["env_key"],
                "config_name": config_name_from_path(path),
                "config_filename": filename,
                "noise_model_sha256": item["noise_model_sha256"],
                "multipliers": item["multipliers"],
            }
        )

    manifest_payload = {
        "base_config": str(base_config),
        "num_envs": int(num_envs),
        "frac": float(frac),
        "seed": int(seed),
        "prefix": prefix,
        "environments": environments,
    }
    if manifest is None:
        manifest_path = REPO_ROOT / "outputs" / "analysis" / f"unknown_t0_random_manifest_s{int(seed)}.json"
    else:
        manifest_path = rel(manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest_payload, indent=2, sort_keys=True), encoding="utf-8")
    manifest_payload["manifest_path"] = str(manifest_path)
    return paths, manifest_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-envs", type=int, default=DEFAULT_NUM_ENVS)
    parser.add_argument("--frac", type=float, default=DEFAULT_FRAC)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--manifest", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths, manifest = write_unknown_configs(
        base_config=args.base_config,
        output_dir=args.output_dir,
        num_envs=args.num_envs,
        frac=args.frac,
        seed=args.seed,
        prefix=args.prefix,
        manifest=args.manifest,
    )
    for path in paths:
        print(f"[write] {path}")
    print(f"[write] {manifest['manifest_path']}")


if __name__ == "__main__":
    main()
