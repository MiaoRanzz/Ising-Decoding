#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate Z-basis seq-favoring stress-test candidate and fixed configs."""

from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path
from typing import Any, Mapping, Sequence

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


DESIGN_LABEL = "Z-basis seq-favoring stress test"
DEFAULT_CANDIDATE_PREFIX = "config_unknown_seqfav_zstress_v2_candidate"
DEFAULT_FIXED_PREFIX = "config_unknown_seqfav_zstress_v2"
DEFAULT_CANDIDATE_MANIFEST = "outputs/analysis/unknown_seqfav_zstress_v2_candidate_manifest.json"
DEFAULT_FIXED_MANIFEST = "outputs/analysis/unknown_seqfav_zstress_v2_manifest.json"
DEFAULT_OUTPUT_DIR = "conf/experiments/unknown_seqfav_zstress_v2"

AXIS_ORDER = ("meas_all", "cnot_all", "idle_all", "z_bias")
DEFAULT_GRID: dict[str, list[float]] = {
    "meas_all": [1.00, 1.10, 1.20, 1.30],
    "cnot_all": [1.35, 1.45, 1.55, 1.65],
    "idle_all": [1.00, 1.20, 1.35, 1.50],
    "z_bias": [1.00, 1.35, 1.50, 1.65],
}


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


def _render_config(base_cfg: Any, noise_model: Mapping[str, float], *, header: str) -> str:
    cfg = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=True))
    cfg.data.noise_model = dict(noise_model)
    return header + OmegaConf.to_yaml(cfg, resolve=True)


def _axis_signature(axis_multipliers: Mapping[str, float]) -> str:
    active = [axis for axis in AXIS_ORDER if float(axis_multipliers[axis]) != 1.0]
    return "+".join(active)


def _parameter_multipliers(
    base_noise: Mapping[str, float],
    axis_multipliers: Mapping[str, float],
) -> dict[str, float]:
    multipliers = {key: 1.0 for key in base_noise}
    for axis_name in AXIS_ORDER:
        axis_multiplier = float(axis_multipliers.get(axis_name, 1.0))
        if axis_name not in AXES:
            raise ValueError(f"unknown zstress axis: {axis_name}")
        if axis_multiplier < 0:
            raise ValueError(f"axis multiplier must be non-negative, got {axis_name}={axis_multiplier}")
        for key in AXES[axis_name]:
            if key not in base_noise:
                raise ValueError(f"axis {axis_name} references missing noise parameter {key}")
            multipliers[key] = max(multipliers[key], axis_multiplier)
    return multipliers


def _is_combination_ood(axis_multipliers: Mapping[str, float]) -> bool:
    active_axes = [axis for axis in AXIS_ORDER if float(axis_multipliers[axis]) != 1.0]
    return 2 <= len(active_axes) < len(AXIS_ORDER)


def generate_candidate_specs(grid: Mapping[str, Sequence[float]] = DEFAULT_GRID) -> list[dict[str, Any]]:
    missing = [axis for axis in AXIS_ORDER if axis not in grid]
    if missing:
        raise ValueError(f"candidate grid missing axes: {missing}")

    specs = []
    for values in product(*(grid[axis] for axis in AXIS_ORDER)):
        axis_multipliers = {axis: float(value) for axis, value in zip(AXIS_ORDER, values)}
        if not _is_combination_ood(axis_multipliers):
            continue
        candidate_index = len(specs)
        specs.append(
            {
                "candidate_index": candidate_index,
                "candidate_key": f"candidate_{candidate_index:03d}",
                "axis_multipliers": axis_multipliers,
                "axis_signature": _axis_signature(axis_multipliers),
                "purpose": "candidate Z-basis seq-favoring OOD composite",
            }
        )
    return specs


def generate_zstress_noise_models(
    base_noise: Mapping[str, float],
    specs: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    base = NoiseModel.from_config_dict(dict(base_noise)).to_config_dict()
    generated = []
    for raw_spec in specs:
        candidate_index = int(raw_spec.get("candidate_index", len(generated)))
        axis_multipliers = {
            str(axis): float(value)
            for axis, value in dict(raw_spec["axis_multipliers"]).items()
        }
        param_multipliers = _parameter_multipliers(base, axis_multipliers)
        noise = {
            key: float(base_value) * float(param_multipliers[key])
            for key, base_value in base.items()
        }
        noise = NoiseModel.from_config_dict(noise).to_config_dict()
        generated.append(
            {
                "candidate_index": candidate_index,
                "candidate_key": str(raw_spec.get("candidate_key", f"candidate_{candidate_index:03d}")),
                "env_index": int(raw_spec.get("env_index", candidate_index)),
                "env_key": str(raw_spec.get("env_key", f"e{candidate_index:02d}")),
                "axis_multipliers": axis_multipliers,
                "axis_signature": str(raw_spec.get("axis_signature", _axis_signature(axis_multipliers))),
                "purpose": str(raw_spec.get("purpose", "")),
                "parameter_multipliers": param_multipliers,
                "noise_model": {key: float(value) for key, value in noise.items()},
                "noise_model_sha256": NoiseModel.from_config_dict(noise).sha256(),
            }
        )
    return generated


def _write_configs(
    *,
    base_config: str | Path,
    output_dir: str | Path,
    manifest: str | Path,
    specs: Sequence[Mapping[str, Any]],
    prefix: str,
    fixed: bool,
) -> tuple[list[Path], dict[str, Any]]:
    base_path = rel(base_config)
    if not base_path.exists():
        raise FileNotFoundError(base_path)
    out_dir = rel(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = OmegaConf.load(base_path)
    base_noise = load_base_noise_model(base_path)
    generated = generate_zstress_noise_models(base_noise, specs)

    paths = []
    environments = []
    for out_index, item in enumerate(generated):
        if fixed:
            config_name = f"{prefix}_e{out_index:02d}"
            env_key = f"e{out_index:02d}"
        else:
            config_name = f"{prefix}_{int(item['candidate_index']):03d}"
            env_key = item["candidate_key"]
        filename = f"{config_name}.yaml"
        path = out_dir / filename
        axis_json = json.dumps(item["axis_multipliers"], sort_keys=True)
        header = (
            "# Auto-generated Z-basis seq-favoring stress-test noise environment.\n"
            f"# design: {DESIGN_LABEL}\n"
            f"# base_config: {base_path.name}\n"
            f"# env_key: {env_key}\n"
            f"# candidate_key: {item['candidate_key']}\n"
            f"# axis_signature: {item['axis_signature']}\n"
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
                "env_index": out_index,
                "env_key": env_key,
                "candidate_index": int(item["candidate_index"]),
                "candidate_key": item["candidate_key"],
                "config_name": config_name_from_path(path),
                "config_filename": filename,
                "axis_signature": item["axis_signature"],
                "axis_multipliers": item["axis_multipliers"],
                "parameter_multipliers": item["parameter_multipliers"],
                "noise_model_sha256": item["noise_model_sha256"],
            }
        )

    manifest_payload = {
        "design": DESIGN_LABEL,
        "base_config": str(base_config),
        "prefix": prefix,
        "fixed": bool(fixed),
        "num_envs": len(generated),
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


def write_candidate_configs(
    *,
    base_config: str | Path = DEFAULT_BASE_CONFIG,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    manifest: str | Path = DEFAULT_CANDIDATE_MANIFEST,
    candidate_specs: Sequence[Mapping[str, Any]] | None = None,
    prefix: str = DEFAULT_CANDIDATE_PREFIX,
) -> tuple[list[Path], dict[str, Any]]:
    specs = list(candidate_specs) if candidate_specs is not None else generate_candidate_specs()
    return _write_configs(
        base_config=base_config,
        output_dir=output_dir,
        manifest=manifest,
        specs=specs,
        prefix=prefix,
        fixed=False,
    )


def write_fixed_configs(
    *,
    base_config: str | Path = DEFAULT_BASE_CONFIG,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    manifest: str | Path = DEFAULT_FIXED_MANIFEST,
    selected_specs: Sequence[Mapping[str, Any]],
    prefix: str = DEFAULT_FIXED_PREFIX,
) -> tuple[list[Path], dict[str, Any]]:
    fixed_specs = []
    for env_index, raw_spec in enumerate(selected_specs):
        spec = dict(raw_spec)
        spec["env_index"] = env_index
        spec["env_key"] = f"e{env_index:02d}"
        fixed_specs.append(spec)
    return _write_configs(
        base_config=base_config,
        output_dir=output_dir,
        manifest=manifest,
        specs=fixed_specs,
        prefix=prefix,
        fixed=True,
    )


def _parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--manifest", default=DEFAULT_CANDIDATE_MANIFEST)
    parser.add_argument("--prefix", default=DEFAULT_CANDIDATE_PREFIX)
    parser.add_argument("--meas-all", default=",".join(str(v) for v in DEFAULT_GRID["meas_all"]))
    parser.add_argument("--cnot-all", default=",".join(str(v) for v in DEFAULT_GRID["cnot_all"]))
    parser.add_argument("--idle-all", default=",".join(str(v) for v in DEFAULT_GRID["idle_all"]))
    parser.add_argument("--z-bias", default=",".join(str(v) for v in DEFAULT_GRID["z_bias"]))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    grid = {
        "meas_all": _parse_float_list(args.meas_all),
        "cnot_all": _parse_float_list(args.cnot_all),
        "idle_all": _parse_float_list(args.idle_all),
        "z_bias": _parse_float_list(args.z_bias),
    }
    specs = generate_candidate_specs(grid)
    paths, manifest = write_candidate_configs(
        base_config=args.base_config,
        output_dir=args.output_dir,
        manifest=args.manifest,
        candidate_specs=specs,
        prefix=args.prefix,
    )
    for path in paths:
        print(f"[write] {path}")
    print(f"[write] {manifest['manifest_path']}")
    print(f"[count] candidates={len(paths)}")


if __name__ == "__main__":
    main()
