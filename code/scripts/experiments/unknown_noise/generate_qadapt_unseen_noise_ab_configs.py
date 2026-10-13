#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate frozen A/B unseen-noise evaluation configs for QAdapt.

Family A assigns different multipliers to the active training axes.  Parameters
shared by more than one axis receive the maximum applicable multiplier, matching
the non-compounding convention used by the fixed multiplier-grid experiment.

Family B independently perturbs all 25 physical noise parameters in log space.
Both families are generated from a fixed seed and written to one immutable-style
manifest before inference is run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from itertools import product
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from omegaconf import OmegaConf

CODE_ROOT = Path(__file__).resolve().parents[3]
REPO_ROOT = CODE_ROOT.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from qec.noise_model import NoiseModel  # noqa: E402
from scripts.config_paths import config_name_from_path  # noqa: E402
from scripts.experiments.unknown_noise.generate_unknown_axismix_grid_u1p2_5p0_configs import (  # noqa: E402
    AXES,
    AXIS_ORDER,
    DEFAULT_ENV_SPECS,
    load_base_noise_model,
)


DEFAULT_BASE_CONFIG = "conf/examples/qadapt/config_qadapt_t0_base.yaml"
DEFAULT_OUTPUT_DIR = "outputs/generated_configs/qadapt_unseen_noise_ab"
DEFAULT_MANIFEST = "outputs/analysis/qadapt_unseen_noise_ab_manifest.json"
DEFAULT_SEED = 20260806
DEFAULT_REPLICATES_PER_AXIS_SET = 5
DEFAULT_INDEPENDENT_CONFIGS = 55
ASYMMETRIC_MULTIPLIERS = (1.2, 1.5, 2.0, 2.5, 3.0)
INDEPENDENT_MIN_MULTIPLIER = 0.5
INDEPENDENT_MAX_MULTIPLIER = 3.0
DESIGN_LABEL = "QAdapt frozen-checkpoint unseen-noise A/B evaluation"


def rel(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


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


def _sha256_json(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validated_noise(
    base_noise: Mapping[str, float],
    parameter_multipliers: Mapping[str, float],
) -> tuple[dict[str, float], str]:
    noise = {
        key: float(base_value) * float(parameter_multipliers[key])
        for key, base_value in base_noise.items()
    }
    validated = NoiseModel.from_config_dict(noise)
    return validated.to_config_dict(), validated.sha256()


def _asymmetric_parameter_multipliers(
    base_noise: Mapping[str, float],
    axis_multipliers: Mapping[str, float],
) -> dict[str, float]:
    parameter_multipliers = {key: 1.0 for key in base_noise}
    for axis in AXIS_ORDER:
        multiplier = float(axis_multipliers.get(axis, 1.0))
        if multiplier == 1.0:
            continue
        for key in AXES[axis]:
            if key not in base_noise:
                raise ValueError(f"axis {axis} references missing parameter {key}")
            parameter_multipliers[key] = max(parameter_multipliers[key], multiplier)
    return parameter_multipliers


def generate_asymmetric_axis_family(
    base_noise: Mapping[str, float],
    *,
    seed: int = DEFAULT_SEED,
    replicates_per_axis_set: int = DEFAULT_REPLICATES_PER_AXIS_SET,
    multiplier_choices: Sequence[float] = ASYMMETRIC_MULTIPLIERS,
) -> list[dict[str, Any]]:
    """Generate family A with unique non-uniform active-axis multiplier tuples."""

    if replicates_per_axis_set <= 0:
        raise ValueError("replicates_per_axis_set must be positive")
    choices = tuple(float(value) for value in multiplier_choices)
    if len(set(choices)) < 2 or any(value <= 0 for value in choices):
        raise ValueError("multiplier_choices need at least two distinct positive values")

    generated: list[dict[str, Any]] = []
    for raw_spec in DEFAULT_ENV_SPECS:
        env_index = int(raw_spec["env_index"])
        active_axes = tuple(str(axis) for axis in raw_spec["active_axes"])
        candidates = [
            values
            for values in product(choices, repeat=len(active_axes))
            if len(set(values)) > 1
        ]
        if replicates_per_axis_set > len(candidates):
            raise ValueError(
                f"requested {replicates_per_axis_set} unique tuples for {active_axes}, "
                f"but only {len(candidates)} are available"
            )
        rng = np.random.default_rng(np.random.SeedSequence([seed, 0, env_index]))
        selected = rng.choice(
            len(candidates), size=replicates_per_axis_set, replace=False
        )
        for replicate_index, candidate_index in enumerate(selected.tolist()):
            active_values = candidates[int(candidate_index)]
            axis_multipliers = {axis: 1.0 for axis in AXIS_ORDER}
            axis_multipliers.update(dict(zip(active_axes, active_values, strict=True)))
            parameter_multipliers = _asymmetric_parameter_multipliers(
                base_noise, axis_multipliers
            )
            noise, noise_sha256 = _validated_noise(base_noise, parameter_multipliers)
            generated.append(
                {
                    "family": "A_asymmetric_axes",
                    "family_key": "A",
                    "env_index": env_index,
                    "env_key": str(raw_spec["env_key"]),
                    "replicate_index": replicate_index,
                    "replicate_key": f"r{replicate_index:02d}",
                    "active_axes": list(active_axes),
                    "axis_signature": str(raw_spec["axis_signature"]),
                    "combination_size": len(active_axes),
                    "axis_multipliers": axis_multipliers,
                    "parameter_multipliers": parameter_multipliers,
                    "overlap_rule": "maximum_applicable_axis_multiplier",
                    "generation_seed": int(seed),
                    "seed_sequence": [int(seed), 0, env_index],
                    "noise_model": noise,
                    "probability_totals": _probability_totals(noise),
                    "noise_model_sha256": noise_sha256,
                }
            )
    return generated


def generate_independent_parameter_family(
    base_noise: Mapping[str, float],
    *,
    seed: int = DEFAULT_SEED,
    num_configs: int = DEFAULT_INDEPENDENT_CONFIGS,
    minimum_multiplier: float = INDEPENDENT_MIN_MULTIPLIER,
    maximum_multiplier: float = INDEPENDENT_MAX_MULTIPLIER,
) -> list[dict[str, Any]]:
    """Generate family B by independently perturbing all 25 parameters."""

    if len(base_noise) != 25:
        raise ValueError(f"family B requires exactly 25 parameters, got {len(base_noise)}")
    if num_configs <= 0:
        raise ValueError("num_configs must be positive")
    if not 0 < minimum_multiplier < maximum_multiplier:
        raise ValueError("independent multiplier bounds must satisfy 0 < min < max")

    keys = tuple(base_noise)
    log_low = math.log(float(minimum_multiplier))
    log_high = math.log(float(maximum_multiplier))
    generated: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    for config_index in range(num_configs):
        rng = np.random.default_rng(
            np.random.SeedSequence([seed, 1, config_index])
        )
        values = np.exp(rng.uniform(log_low, log_high, size=len(keys)))
        parameter_multipliers = {
            key: float(value) for key, value in zip(keys, values, strict=True)
        }
        noise, noise_sha256 = _validated_noise(base_noise, parameter_multipliers)
        if noise_sha256 in seen_hashes:
            raise RuntimeError(f"duplicate family-B noise model at index {config_index}")
        seen_hashes.add(noise_sha256)
        generated.append(
            {
                "family": "B_independent_25p",
                "family_key": "B",
                "env_index": config_index,
                "env_key": f"e{config_index:02d}",
                "replicate_index": 0,
                "replicate_key": "r00",
                "active_axes": list(AXIS_ORDER),
                "axis_signature": "independent_all_25_parameters",
                "combination_size": len(AXIS_ORDER),
                "parameter_multipliers": parameter_multipliers,
                "multiplier_min_observed": float(values.min()),
                "multiplier_geometric_mean": float(np.exp(np.log(values).mean())),
                "multiplier_max_observed": float(values.max()),
                "generation_seed": int(seed),
                "seed_sequence": [int(seed), 1, config_index],
                "noise_model": noise,
                "probability_totals": _probability_totals(noise),
                "noise_model_sha256": noise_sha256,
            }
        )
    return generated


def _render_config(base_cfg: Any, noise_model: Mapping[str, float], header: str) -> str:
    cfg = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=True))
    cfg.data.noise_model = dict(noise_model)
    return header + OmegaConf.to_yaml(cfg, resolve=True)


def write_unseen_noise_ab_configs(
    *,
    base_config: str | Path = DEFAULT_BASE_CONFIG,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    manifest: str | Path = DEFAULT_MANIFEST,
    seed: int = DEFAULT_SEED,
    replicates_per_axis_set: int = DEFAULT_REPLICATES_PER_AXIS_SET,
    independent_configs: int = DEFAULT_INDEPENDENT_CONFIGS,
    independent_min_multiplier: float = INDEPENDENT_MIN_MULTIPLIER,
    independent_max_multiplier: float = INDEPENDENT_MAX_MULTIPLIER,
) -> tuple[list[Path], dict[str, Any]]:
    base_path = rel(base_config)
    if not base_path.exists():
        raise FileNotFoundError(base_path)
    out_dir = rel(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = OmegaConf.load(base_path)
    base_noise = load_base_noise_model(base_path)
    family_a = generate_asymmetric_axis_family(
        base_noise,
        seed=seed,
        replicates_per_axis_set=replicates_per_axis_set,
    )
    family_b = generate_independent_parameter_family(
        base_noise,
        seed=seed,
        num_configs=independent_configs,
        minimum_multiplier=independent_min_multiplier,
        maximum_multiplier=independent_max_multiplier,
    )

    paths: list[Path] = []
    environments: list[dict[str, Any]] = []
    for item in [*family_a, *family_b]:
        family_dir = out_dir / item["family"]
        family_dir.mkdir(parents=True, exist_ok=True)
        if item["family_key"] == "A":
            stem = (
                f"config_qadapt_unseen_A_{item['env_key']}_"
                f"{item['replicate_key']}"
            )
        else:
            stem = f"config_qadapt_unseen_B_{item['env_key']}"
        path = family_dir / f"{stem}.yaml"
        header = (
            "# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.\n"
            "# SPDX-License-Identifier: Apache-2.0\n\n"
            "# Auto-generated frozen-checkpoint unseen-noise evaluation config.\n"
            f"# design: {DESIGN_LABEL}\n"
            f"# family: {item['family']}\n"
            f"# generation_seed: {seed}\n"
            f"# noise_model_sha256: {item['noise_model_sha256']}\n\n"
        )
        path.write_text(
            _render_config(base_cfg, item["noise_model"], header),
            encoding="utf-8",
        )
        paths.append(path)
        environment = {
            key: value for key, value in item.items() if key != "noise_model"
        }
        environment.update(
            {
                "config_filename": path.name,
                "config_path": str(path.relative_to(REPO_ROOT)),
                "config_name": config_name_from_path(path),
                "noise_model": item["noise_model"],
            }
        )
        environments.append(environment)

    payload: dict[str, Any] = {
        "design": DESIGN_LABEL,
        "design_version": 1,
        "base_config": str(Path(base_config)),
        "seed": int(seed),
        "frozen_evaluation": {
            "distances": [7, 9],
            "n_rounds": 9,
            "bases": ["X", "Z"],
            "primary_comparison": "r9x_seq_ewc_e100:r9x_seq_noewc_e100",
            "checkpoint_selection_after_generation": False,
        },
        "family_A": {
            "description": "different multipliers across active training axes",
            "axis_order": list(AXIS_ORDER),
            "axes": {name: list(keys) for name, keys in AXES.items()},
            "multiplier_choices": list(ASYMMETRIC_MULTIPLIERS),
            "replicates_per_axis_set": int(replicates_per_axis_set),
            "overlap_rule": "maximum_applicable_axis_multiplier",
            "num_configs": len(family_a),
        },
        "family_B": {
            "description": "independent log-uniform perturbation of all 25 parameters",
            "distribution": "LogUniform",
            "minimum_multiplier": float(independent_min_multiplier),
            "maximum_multiplier": float(independent_max_multiplier),
            "num_parameters": len(base_noise),
            "num_configs": len(family_b),
        },
        "num_configs": len(environments),
        "environments": environments,
    }
    payload["design_sha256"] = _sha256_json(payload)
    manifest_path = rel(manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    payload["manifest_path"] = str(manifest_path)
    return paths, payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--replicates-per-axis-set",
        type=int,
        default=DEFAULT_REPLICATES_PER_AXIS_SET,
    )
    parser.add_argument(
        "--independent-configs", type=int, default=DEFAULT_INDEPENDENT_CONFIGS
    )
    parser.add_argument(
        "--independent-min-multiplier",
        type=float,
        default=INDEPENDENT_MIN_MULTIPLIER,
    )
    parser.add_argument(
        "--independent-max-multiplier",
        type=float,
        default=INDEPENDENT_MAX_MULTIPLIER,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths, payload = write_unseen_noise_ab_configs(
        base_config=args.base_config,
        output_dir=args.output_dir,
        manifest=args.manifest,
        seed=args.seed,
        replicates_per_axis_set=args.replicates_per_axis_set,
        independent_configs=args.independent_configs,
        independent_min_multiplier=args.independent_min_multiplier,
        independent_max_multiplier=args.independent_max_multiplier,
    )
    print(f"[write] {payload['manifest_path']}")
    print(f"[design_sha256] {payload['design_sha256']}")
    print(f"[write] {len(paths)} configs")


if __name__ == "__main__":
    main()
