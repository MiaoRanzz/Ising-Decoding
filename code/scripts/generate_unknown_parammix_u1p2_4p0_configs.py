#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate parameter-level random training-axis composite noise configs."""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from omegaconf import OmegaConf

CODE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_ROOT.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from qec.noise_model import NoiseModel  # noqa: E402
from scripts.config_paths import config_name_from_path  # noqa: E402
from scripts.generate_unknown_seqfav_composite_configs import (  # noqa: E402
    AXES,
    DEFAULT_BASE_CONFIG,
)


DESIGN_LABEL = "parameter-level random training-axis mixed OOD test"
DEFAULT_PREFIX = "config_unknown_parammix_u1p2_4p0"
DEFAULT_MANIFEST = "outputs/analysis/unknown_parammix_u1p2_4p0_manifest.json"
DEFAULT_OUTPUT_DIR = "conf/experiments/unknown_parammix_u1p2_4p0"
AXIS_ORDER = ("meas_all", "cnot_all", "idle_all", "z_bias")
MIN_MULTIPLIER = 1.2
MAX_MULTIPLIER = 4.0
NUM_REPLICATES = 5
DEFAULT_SEED = 20260714


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


def _axis_signature(active_axes: Sequence[str]) -> str:
    return "+".join(active_axes)


def _default_env_specs() -> list[dict[str, Any]]:
    specs = []
    for size in (2, 3, 4):
        for active_axes in combinations(AXIS_ORDER, size):
            env_index = len(specs)
            specs.append(
                {
                    "env_index": env_index,
                    "env_key": f"e{env_index:02d}",
                    "active_axes": tuple(active_axes),
                    "axis_signature": _axis_signature(active_axes),
                    "combination_size": size,
                    "contains_z_bias": "z_bias" in active_axes,
                    "contains_cnot_z_bias": (
                        "cnot_all" in active_axes and "z_bias" in active_axes
                    ),
                    "purpose": f"{size}-axis parameter-level random composite",
                }
            )
    return specs


DEFAULT_ENV_SPECS: list[dict[str, Any]] = _default_env_specs()


def _normalize_spec(raw_spec: Mapping[str, Any]) -> dict[str, Any]:
    env_index = int(raw_spec["env_index"])
    active_axes = tuple(str(axis) for axis in raw_spec["active_axes"])
    if not 2 <= len(active_axes) <= 4:
        raise ValueError(f"parameter-mix env must activate 2, 3, or 4 axes, got {active_axes}")
    unknown = [axis for axis in active_axes if axis not in AXIS_ORDER]
    if unknown:
        raise ValueError(f"unknown parameter-mix axes: {unknown}")
    if len(set(active_axes)) != len(active_axes):
        raise ValueError(f"duplicate active axes: {active_axes}")
    return {
        "env_index": env_index,
        "env_key": str(raw_spec.get("env_key", f"e{env_index:02d}")),
        "active_axes": active_axes,
        "axis_signature": str(raw_spec.get("axis_signature", _axis_signature(active_axes))),
        "combination_size": len(active_axes),
        "contains_z_bias": "z_bias" in active_axes,
        "contains_cnot_z_bias": "cnot_all" in active_axes and "z_bias" in active_axes,
        "purpose": str(
            raw_spec.get("purpose", f"{len(active_axes)}-axis parameter-level random composite")
        ),
    }


def _active_parameters(
    base_noise: Mapping[str, float],
    active_axes: Sequence[str],
) -> tuple[str, ...]:
    active_set = {
        parameter
        for axis_name in active_axes
        for parameter in AXES[axis_name]
    }
    missing = sorted(active_set - set(base_noise))
    if missing:
        raise ValueError(f"active axes reference missing noise parameters: {missing}")
    return tuple(key for key in base_noise if key in active_set)


def _probability_totals(noise: Mapping[str, float]) -> dict[str, float]:
    return {
        "cnot_total": sum(value for key, value in noise.items() if key.startswith("p_cnot_")),
        "idle_cnot_total": sum(
            value for key, value in noise.items() if key.startswith("p_idle_cnot_")
        ),
        "idle_spam_total": sum(
            value for key, value in noise.items() if key.startswith("p_idle_spam_")
        ),
    }


def generate_parammix_noise_models(
    base_noise: Mapping[str, float],
    env_specs: Sequence[Mapping[str, Any]] = DEFAULT_ENV_SPECS,
    *,
    num_replicates: int = NUM_REPLICATES,
    seed: int = DEFAULT_SEED,
    min_multiplier: float = MIN_MULTIPLIER,
    max_multiplier: float = MAX_MULTIPLIER,
) -> list[dict[str, Any]]:
    if num_replicates <= 0:
        raise ValueError(f"num_replicates must be positive, got {num_replicates}")
    if not 0 <= min_multiplier < max_multiplier:
        raise ValueError(
            f"multiplier range must satisfy 0 <= min < max, got "
            f"[{min_multiplier}, {max_multiplier})"
        )

    base = NoiseModel.from_config_dict(dict(base_noise)).to_config_dict()
    generated = []
    for raw_spec in env_specs:
        spec = _normalize_spec(raw_spec)
        active_parameters = _active_parameters(base, spec["active_axes"])
        for replicate_index in range(num_replicates):
            rng = np.random.default_rng(
                np.random.SeedSequence([int(seed), spec["env_index"], replicate_index])
            )
            sampled = rng.uniform(min_multiplier, max_multiplier, size=len(active_parameters))
            parameter_multipliers = {key: 1.0 for key in base}
            for key, multiplier in zip(active_parameters, sampled):
                parameter_multipliers[key] = float(multiplier)

            noise = {
                key: float(base_value) * parameter_multipliers[key]
                for key, base_value in base.items()
            }
            validated = NoiseModel.from_config_dict(noise)
            noise = validated.to_config_dict()
            generated.append(
                {
                    **spec,
                    "replicate_index": replicate_index,
                    "replicate_key": f"r{replicate_index:02d}",
                    "generation_seed": int(seed),
                    "seed_sequence": [int(seed), spec["env_index"], replicate_index],
                    "active_parameters": list(active_parameters),
                    "parameter_multipliers": parameter_multipliers,
                    "noise_model": {key: float(value) for key, value in noise.items()},
                    "probability_totals": _probability_totals(noise),
                    "noise_model_sha256": validated.sha256(),
                }
            )
    return generated


def _render_config(base_cfg: Any, noise_model: Mapping[str, float], *, header: str) -> str:
    cfg = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=True))
    cfg.data.noise_model = dict(noise_model)
    return header + OmegaConf.to_yaml(cfg, resolve=True)


def _config_name(prefix: str, env_index: int, replicate_index: int) -> str:
    return f"{prefix}_e{env_index:02d}_r{replicate_index:02d}"


def write_parammix_configs(
    *,
    base_config: str | Path = DEFAULT_BASE_CONFIG,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    prefix: str = DEFAULT_PREFIX,
    manifest: str | Path = DEFAULT_MANIFEST,
    env_specs: Sequence[Mapping[str, Any]] = DEFAULT_ENV_SPECS,
    num_replicates: int = NUM_REPLICATES,
    seed: int = DEFAULT_SEED,
) -> tuple[list[Path], dict[str, Any]]:
    base_path = rel(base_config)
    if not base_path.exists():
        raise FileNotFoundError(base_path)
    out_dir = rel(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = OmegaConf.load(base_path)
    base_noise = load_base_noise_model(base_path)
    generated = generate_parammix_noise_models(
        base_noise,
        env_specs,
        num_replicates=num_replicates,
        seed=seed,
    )

    paths = []
    environments = []
    for item in generated:
        config_name = _config_name(
            prefix,
            int(item["env_index"]),
            int(item["replicate_index"]),
        )
        filename = f"{config_name}.yaml"
        path = out_dir / filename
        header = (
            "# Auto-generated parameter-level random training-axis composite noise.\n"
            f"# design: {DESIGN_LABEL}\n"
            f"# base_config: {base_path.name}\n"
            f"# env_key: {item['env_key']}\n"
            f"# replicate_key: {item['replicate_key']}\n"
            f"# active_axes: {item['axis_signature']}\n"
            f"# distribution: Uniform({MIN_MULTIPLIER}, {MAX_MULTIPLIER})\n"
            f"# seed_sequence: {json.dumps(item['seed_sequence'])}\n"
            f"# noise_model_sha256: {item['noise_model_sha256']}\n\n"
        )
        path.write_text(
            _render_config(base_cfg, item["noise_model"], header=header),
            encoding="utf-8",
        )
        paths.append(path)
        environments.append(
            {
                "env_index": int(item["env_index"]),
                "env_key": item["env_key"],
                "replicate_index": int(item["replicate_index"]),
                "replicate_key": item["replicate_key"],
                "config_name": config_name_from_path(path),
                "config_filename": filename,
                "active_axes": list(item["active_axes"]),
                "axis_signature": item["axis_signature"],
                "combination_size": int(item["combination_size"]),
                "contains_z_bias": bool(item["contains_z_bias"]),
                "contains_cnot_z_bias": bool(item["contains_cnot_z_bias"]),
                "generation_seed": int(item["generation_seed"]),
                "seed_sequence": item["seed_sequence"],
                "active_parameters": item["active_parameters"],
                "parameter_multipliers": item["parameter_multipliers"],
                "noise_model": item["noise_model"],
                "probability_totals": item["probability_totals"],
                "noise_model_sha256": item["noise_model_sha256"],
            }
        )

    env_count = len({int(item["env_index"]) for item in generated})
    manifest_payload = {
        "design": DESIGN_LABEL,
        "base_config": str(base_config),
        "prefix": prefix,
        "distribution": {
            "name": "Uniform",
            "minimum": MIN_MULTIPLIER,
            "maximum_exclusive": MAX_MULTIPLIER,
            "sampling_granularity": "physical_parameter",
        },
        "generation_seed": int(seed),
        "num_envs": env_count,
        "num_replicates": int(num_replicates),
        "num_realizations": len(generated),
        "axis_order": list(AXIS_ORDER),
        "axes": {name: list(keys) for name, keys in AXES.items()},
        "environments": environments,
    }
    manifest_path = rel(manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    manifest_payload["manifest_path"] = str(manifest_path)
    return paths, manifest_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--num-replicates", type=int, default=NUM_REPLICATES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths, manifest = write_parammix_configs(
        base_config=args.base_config,
        output_dir=args.output_dir,
        prefix=args.prefix,
        manifest=args.manifest,
        num_replicates=args.num_replicates,
        seed=args.seed,
    )
    print(f"[write] {manifest['manifest_path']}")
    print(f"[write] {len(paths)} configs")


if __name__ == "__main__":
    main()
