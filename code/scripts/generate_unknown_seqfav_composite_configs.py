#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate fixed seq-favoring composite OOD noise configs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from omegaconf import OmegaConf

CODE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_ROOT.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from qec.noise_model import NoiseModel  # noqa: E402
from scripts.config_paths import config_name_from_path  # noqa: E402


DEFAULT_BASE_CONFIG = "conf/config_domestic_fast_opt_stfusion_r9_x_seq_ewc_t0_base.yaml"
DEFAULT_PREFIX = "config_unknown_seqfav_composite_v1"
DEFAULT_MANIFEST = "outputs/analysis/unknown_seqfav_composite_v1_manifest.json"
DEFAULT_OUTPUT_DIR = "conf/experiments/unknown_seqfav_composite_v1"
DESIGN_LABEL = "seq-favoring OOD stress test"

CNOT_KEYS = (
    "p_cnot_IX",
    "p_cnot_IY",
    "p_cnot_IZ",
    "p_cnot_XI",
    "p_cnot_XX",
    "p_cnot_XY",
    "p_cnot_XZ",
    "p_cnot_YI",
    "p_cnot_YX",
    "p_cnot_YY",
    "p_cnot_YZ",
    "p_cnot_ZI",
    "p_cnot_ZX",
    "p_cnot_ZY",
    "p_cnot_ZZ",
)

AXES: dict[str, tuple[str, ...]] = {
    "meas_all": ("p_meas_X", "p_meas_Z"),
    "cnot_all": CNOT_KEYS,
    "idle_all": (
        "p_idle_cnot_X",
        "p_idle_cnot_Y",
        "p_idle_cnot_Z",
        "p_idle_spam_X",
        "p_idle_spam_Y",
        "p_idle_spam_Z",
    ),
    "z_bias": (
        "p_prep_X",
        "p_meas_X",
        "p_idle_cnot_Z",
        "p_idle_spam_Z",
        "p_cnot_IZ",
        "p_cnot_XZ",
        "p_cnot_YZ",
        "p_cnot_ZI",
        "p_cnot_ZX",
        "p_cnot_ZY",
        "p_cnot_ZZ",
    ),
}

DEFAULT_ENV_SPECS: list[dict[str, Any]] = [
    {
        "env_index": 0,
        "env_key": "e00",
        "axis_multipliers": {"meas_all": 1.35, "cnot_all": 1.45, "idle_all": 1.00, "z_bias": 1.00},
        "purpose": "measurement + CNOT composite",
    },
    {
        "env_index": 1,
        "env_key": "e01",
        "axis_multipliers": {"meas_all": 1.00, "cnot_all": 1.55, "idle_all": 1.35, "z_bias": 1.00},
        "purpose": "CNOT + idle composite",
    },
    {
        "env_index": 2,
        "env_key": "e02",
        "axis_multipliers": {"meas_all": 1.25, "cnot_all": 1.00, "idle_all": 1.00, "z_bias": 1.60},
        "purpose": "measurement + Z-bias composite",
    },
    {
        "env_index": 3,
        "env_key": "e03",
        "axis_multipliers": {"meas_all": 1.00, "cnot_all": 1.35, "idle_all": 1.00, "z_bias": 1.60},
        "purpose": "CNOT + Z-bias composite",
    },
    {
        "env_index": 4,
        "env_key": "e04",
        "axis_multipliers": {"meas_all": 1.35, "cnot_all": 1.45, "idle_all": 1.35, "z_bias": 1.60},
        "purpose": "all-axis hard composite",
    },
]


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


def _config_name(prefix: str, env_index: int) -> str:
    return f"{prefix}_e{int(env_index):02d}"


def _render_config(base_cfg: Any, noise_model: Mapping[str, float], *, header: str) -> str:
    cfg = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=True))
    cfg.data.noise_model = dict(noise_model)
    return header + OmegaConf.to_yaml(cfg, resolve=True)


def _parameter_multipliers(
    base_noise: Mapping[str, float],
    axis_multipliers: Mapping[str, float],
) -> dict[str, float]:
    multipliers = {key: 1.0 for key in base_noise}
    for axis_name, axis_multiplier in axis_multipliers.items():
        if axis_name not in AXES:
            raise ValueError(f"unknown seq-favoring axis: {axis_name}")
        value = float(axis_multiplier)
        if value < 0:
            raise ValueError(f"axis multiplier must be non-negative, got {axis_name}={value}")
        for key in AXES[axis_name]:
            if key not in base_noise:
                raise ValueError(f"axis {axis_name} references missing noise parameter {key}")
            multipliers[key] = max(multipliers[key], value)
    return multipliers


def generate_seqfav_noise_models(
    base_noise: Mapping[str, float],
    env_specs: Sequence[Mapping[str, Any]] = DEFAULT_ENV_SPECS,
) -> list[dict[str, Any]]:
    base = NoiseModel.from_config_dict(dict(base_noise)).to_config_dict()
    generated = []
    for raw_spec in env_specs:
        env_index = int(raw_spec["env_index"])
        axis_multipliers = {
            str(key): float(value)
            for key, value in dict(raw_spec["axis_multipliers"]).items()
        }
        param_multipliers = _parameter_multipliers(base, axis_multipliers)
        noise = {
            key: float(base_value) * float(param_multipliers[key])
            for key, base_value in base.items()
        }
        noise = NoiseModel.from_config_dict(noise).to_config_dict()
        generated.append(
            {
                "env_index": env_index,
                "env_key": str(raw_spec.get("env_key", f"e{env_index:02d}")),
                "purpose": str(raw_spec.get("purpose", "")),
                "axis_multipliers": axis_multipliers,
                "parameter_multipliers": param_multipliers,
                "noise_model": {key: float(value) for key, value in noise.items()},
                "noise_model_sha256": NoiseModel.from_config_dict(noise).sha256(),
            }
        )
    return generated


def write_seqfav_configs(
    *,
    base_config: str | Path = DEFAULT_BASE_CONFIG,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    prefix: str = DEFAULT_PREFIX,
    manifest: str | Path | None = DEFAULT_MANIFEST,
) -> tuple[list[Path], dict[str, Any]]:
    base_path = rel(base_config)
    if not base_path.exists():
        raise FileNotFoundError(base_path)
    out_dir = rel(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = OmegaConf.load(base_path)
    base_noise = load_base_noise_model(base_path)
    generated = generate_seqfav_noise_models(base_noise)

    paths = []
    environments = []
    for item in generated:
        env_index = int(item["env_index"])
        config_name = _config_name(prefix, env_index)
        filename = f"{config_name}.yaml"
        path = out_dir / filename
        axis_json = json.dumps(item["axis_multipliers"], sort_keys=True)
        header = (
            "# Auto-generated seq-favoring composite OOD noise environment.\n"
            f"# design: {DESIGN_LABEL}\n"
            f"# base_config: {base_path.name}\n"
            f"# env_key: {item['env_key']}\n"
            f"# purpose: {item['purpose']}\n"
            f"# axis_multipliers: {axis_json}\n"
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
                "purpose": item["purpose"],
                "axis_multipliers": item["axis_multipliers"],
                "parameter_multipliers": item["parameter_multipliers"],
                "noise_model_sha256": item["noise_model_sha256"],
            }
        )

    manifest_payload = {
        "design": DESIGN_LABEL,
        "base_config": str(base_config),
        "prefix": prefix,
        "num_envs": len(generated),
        "axes": {name: list(keys) for name, keys in AXES.items()},
        "environments": environments,
    }
    manifest_path = rel(manifest) if manifest is not None else rel(DEFAULT_MANIFEST)
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths, manifest = write_seqfav_configs(
        base_config=args.base_config,
        output_dir=args.output_dir,
        prefix=args.prefix,
        manifest=args.manifest,
    )
    for path in paths:
        print(f"[write] {path}")
    print(f"[write] {manifest['manifest_path']}")


if __name__ == "__main__":
    main()
