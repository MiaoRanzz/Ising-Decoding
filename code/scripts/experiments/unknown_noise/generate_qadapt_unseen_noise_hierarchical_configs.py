#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate frozen absolute-domain hierarchical unseen-noise configs.

The four preparation/measurement probabilities are sampled directly. For
idle and CNOT Pauli channels, total error probabilities are sampled first and
then split over channel components with Dirichlet draws. This keeps every
sample inside a valid probability simplex without anchoring it to T0.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from omegaconf import OmegaConf

CODE_ROOT = Path(__file__).resolve().parents[3]
REPO_ROOT = CODE_ROOT.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from qec.noise_model import CNOT_ERROR_TYPES, NoiseModel  # noqa: E402
from scripts.config_paths import config_name_from_path  # noqa: E402


DEFAULT_BASE_CONFIG = "conf/examples/qadapt/config_qadapt_t0_base.yaml"
DEFAULT_OUTPUT_DIR = "outputs/generated_configs/qadapt_unseen_noise_hierarchical"
DEFAULT_MANIFEST = "outputs/analysis/qadapt_unseen_noise_hierarchical_manifest.json"
DEFAULT_SEED = 20260807
DEFAULT_NUM_CONFIGS = 55
DESIGN_LABEL = "QAdapt absolute-domain hierarchical unseen-noise evaluation"
FAMILY = "B2_hierarchical_absolute"
FAMILY_KEY = "B2"

# Operational physically plausible domain for this study. These are absolute
# probabilities, not multipliers of T0.
SCALAR_RANGES: dict[str, tuple[float, float]] = {
    "p_prep_X": (1.0e-4, 5.0e-3),
    "p_prep_Z": (1.0e-4, 5.0e-3),
    "p_meas_X": (3.0e-3, 3.0e-2),
    "p_meas_Z": (3.0e-3, 3.0e-2),
}
TOTAL_RANGES: dict[str, tuple[float, float]] = {
    "idle_cnot_total": (3.0e-4, 5.0e-3),
    "idle_spam_total": (6.0e-4, 1.0e-2),
    "cnot_total": (3.0e-3, 3.0e-2),
}
DIRICHLET_ALPHA: dict[str, float] = {
    "idle_cnot": 0.75,
    "idle_spam": 0.75,
    "cnot": 0.35,
}
TRAINING_CONFIGS = (
    "conf/examples/qadapt/config_qadapt_t0_base.yaml",
    "conf/examples/qadapt/config_qadapt_t1_meas_1p5.yaml",
    "conf/examples/qadapt/config_qadapt_t2_cnot_1p5.yaml",
    "conf/examples/qadapt/config_qadapt_t3_idle_1p5.yaml",
    "conf/examples/qadapt/config_qadapt_t4_z_bias_1p5.yaml",
)
MIN_OUTSIDE_TRAINING_ENVELOPE = 8


def rel(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def _sha256_json(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _load_noise(path: str | Path) -> dict[str, float]:
    cfg = OmegaConf.load(rel(path))
    raw = OmegaConf.to_container(cfg.data.noise_model, resolve=True)
    return NoiseModel.from_config_dict(dict(raw)).to_config_dict()


def load_training_noises(
    paths: Sequence[str | Path] = TRAINING_CONFIGS,
) -> list[dict[str, float]]:
    return [_load_noise(path) for path in paths]


def _latin_hypercube_log_values(
    rng: np.random.Generator,
    num_configs: int,
    ranges: Mapping[str, tuple[float, float]],
) -> dict[str, np.ndarray]:
    values: dict[str, np.ndarray] = {}
    for key, (low, high) in ranges.items():
        if not 0 < low < high:
            raise ValueError(f"invalid positive range for {key}: {(low, high)}")
        quantiles = (
            rng.permutation(num_configs) + rng.random(num_configs)
        ) / num_configs
        values[key] = np.exp(
            math.log(low) + quantiles * (math.log(high) - math.log(low))
        )
    return values


def _training_envelope(
    training_noises: Sequence[Mapping[str, float]],
) -> dict[str, tuple[float, float]]:
    keys = tuple(training_noises[0])
    return {
        key: (
            min(float(noise[key]) for noise in training_noises),
            max(float(noise[key]) for noise in training_noises),
        )
        for key in keys
    }


def _outside_envelope(
    noise: Mapping[str, float],
    envelope: Mapping[str, tuple[float, float]],
) -> list[str]:
    return [
        key
        for key, value in noise.items()
        if float(value) < envelope[key][0] or float(value) > envelope[key][1]
    ]


def _minimum_log_rms_distance(
    noise: Mapping[str, float],
    training_noises: Sequence[Mapping[str, float]],
) -> float:
    keys = tuple(noise)
    candidate = np.log10([float(noise[key]) for key in keys])
    distances = []
    for training_noise in training_noises:
        training = np.log10([float(training_noise[key]) for key in keys])
        distances.append(float(np.sqrt(np.mean((candidate - training) ** 2))))
    return min(distances)


def generate_hierarchical_family(
    *,
    seed: int = DEFAULT_SEED,
    num_configs: int = DEFAULT_NUM_CONFIGS,
    training_noises: Sequence[Mapping[str, float]] | None = None,
    minimum_outside_training_envelope: int = MIN_OUTSIDE_TRAINING_ENVELOPE,
) -> list[dict[str, Any]]:
    """Generate deterministic, valid, absolute-domain hierarchical samples."""

    if num_configs <= 0:
        raise ValueError("num_configs must be positive")
    if training_noises is None:
        training_noises = load_training_noises()
    envelope = _training_envelope(training_noises)
    rng = np.random.default_rng(seed)
    scalar_values = _latin_hypercube_log_values(rng, num_configs, SCALAR_RANGES)
    total_values = _latin_hypercube_log_values(rng, num_configs, TOTAL_RANGES)
    generated: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()

    for config_index in range(num_configs):
        config_rng = np.random.default_rng(
            np.random.SeedSequence([seed, 2, config_index])
        )
        idle_cnot_weights = config_rng.dirichlet(
            np.full(3, DIRICHLET_ALPHA["idle_cnot"])
        )
        idle_spam_weights = config_rng.dirichlet(
            np.full(3, DIRICHLET_ALPHA["idle_spam"])
        )
        cnot_weights = config_rng.dirichlet(
            np.full(15, DIRICHLET_ALPHA["cnot"])
        )
        noise: dict[str, float] = {
            key: float(values[config_index])
            for key, values in scalar_values.items()
        }
        noise.update(
            {
                f"p_idle_cnot_{pauli}": float(
                    total_values["idle_cnot_total"][config_index] * weight
                )
                for pauli, weight in zip(
                    ("X", "Y", "Z"), idle_cnot_weights, strict=True
                )
            }
        )
        noise.update(
            {
                f"p_idle_spam_{pauli}": float(
                    total_values["idle_spam_total"][config_index] * weight
                )
                for pauli, weight in zip(
                    ("X", "Y", "Z"), idle_spam_weights, strict=True
                )
            }
        )
        noise.update(
            {
                f"p_cnot_{pauli}": float(
                    total_values["cnot_total"][config_index] * weight
                )
                for pauli, weight in zip(
                    CNOT_ERROR_TYPES, cnot_weights, strict=True
                )
            }
        )
        validated = NoiseModel.from_config_dict(noise)
        noise = validated.to_config_dict()
        noise_sha256 = validated.sha256()
        if noise_sha256 in seen_hashes:
            raise RuntimeError(f"duplicate hierarchical noise at index {config_index}")
        seen_hashes.add(noise_sha256)
        outside = _outside_envelope(noise, envelope)
        if len(outside) < minimum_outside_training_envelope:
            raise RuntimeError(
                f"config {config_index} has only {len(outside)} parameters outside "
                f"the training envelope; require {minimum_outside_training_envelope}"
            )
        generated.append(
            {
                "family": FAMILY,
                "family_key": FAMILY_KEY,
                "env_index": config_index,
                "env_key": f"e{config_index:02d}",
                "replicate_index": 0,
                "replicate_key": "r00",
                "active_axes": [
                    "prep",
                    "measurement",
                    "idle_cnot",
                    "idle_spam",
                    "cnot",
                ],
                "axis_signature": "hierarchical_absolute_25p",
                "combination_size": 5,
                "generation_seed": int(seed),
                "seed_sequence": [int(seed), 2, config_index],
                "sampled_totals": {
                    key: float(values[config_index])
                    for key, values in total_values.items()
                },
                "outside_training_envelope_parameters": outside,
                "outside_training_envelope_count": len(outside),
                "min_log10_rms_distance_to_training":
                    _minimum_log_rms_distance(noise, training_noises),
                "noise_model": noise,
                "noise_model_sha256": noise_sha256,
            }
        )
    return generated


def _render_config(
    base_cfg: Any, noise_model: Mapping[str, float], header: str
) -> str:
    cfg = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=True))
    cfg.data.noise_model = dict(noise_model)
    return header + OmegaConf.to_yaml(cfg, resolve=True)


def write_hierarchical_configs(
    *,
    base_config: str | Path = DEFAULT_BASE_CONFIG,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    manifest: str | Path = DEFAULT_MANIFEST,
    seed: int = DEFAULT_SEED,
    num_configs: int = DEFAULT_NUM_CONFIGS,
) -> tuple[list[Path], dict[str, Any]]:
    base_path = rel(base_config)
    base_cfg = OmegaConf.load(base_path)
    training_noises = load_training_noises()
    training_envelope = _training_envelope(training_noises)
    family = generate_hierarchical_family(
        seed=seed,
        num_configs=num_configs,
        training_noises=training_noises,
    )
    out_dir = rel(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    environments: list[dict[str, Any]] = []
    for item in family:
        path = out_dir / f"config_qadapt_unseen_B2_{item['env_key']}.yaml"
        header = (
            "# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.\n"
            "# SPDX-License-Identifier: Apache-2.0\n\n"
            "# Auto-generated absolute-domain hierarchical unseen-noise config.\n"
            f"# design: {DESIGN_LABEL}\n"
            f"# generation_seed: {seed}\n"
            f"# noise_model_sha256: {item['noise_model_sha256']}\n\n"
        )
        path.write_text(
            _render_config(base_cfg, item["noise_model"], header),
            encoding="utf-8",
        )
        paths.append(path)
        environment = dict(item)
        environment.update(
            {
                "config_filename": path.name,
                "config_path": str(path.relative_to(REPO_ROOT)),
                "config_name": config_name_from_path(path),
            }
        )
        environments.append(environment)

    payload: dict[str, Any] = {
        "design": DESIGN_LABEL,
        "design_version": 1,
        "base_config_used_for_non_noise_settings_only": str(Path(base_config)),
        "t0_anchored_noise_sampling": False,
        "seed": int(seed),
        "training_configs_for_ood_audit": list(TRAINING_CONFIGS),
        "training_envelope": {
            key: list(bounds) for key, bounds in training_envelope.items()
        },
        "minimum_parameters_outside_training_envelope":
            MIN_OUTSIDE_TRAINING_ENVELOPE,
        "sampling": {
            "scalar_method": "log-space Latin hypercube",
            "scalar_absolute_ranges": {
                key: list(value) for key, value in SCALAR_RANGES.items()
            },
            "total_method": "log-space Latin hypercube",
            "channel_total_absolute_ranges": {
                key: list(value) for key, value in TOTAL_RANGES.items()
            },
            "composition_method": "symmetric Dirichlet",
            "dirichlet_alpha": DIRICHLET_ALPHA,
            "num_parameters": 25,
        },
        "frozen_evaluation": {
            "distances": [7, 9],
            "n_rounds": 9,
            "bases": ["X", "Z"],
            "num_samples_per_basis": 262144,
            "primary_comparison":
                "r9x_seq_ewc_e100:r9x_seq_noewc_e100",
            "checkpoint_selection_after_generation": False,
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
    parser.add_argument("--num-configs", type=int, default=DEFAULT_NUM_CONFIGS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths, payload = write_hierarchical_configs(
        base_config=args.base_config,
        output_dir=args.output_dir,
        manifest=args.manifest,
        seed=args.seed,
        num_configs=args.num_configs,
    )
    print(f"[write] {payload['manifest_path']}")
    print(f"[design_sha256] {payload['design_sha256']}")
    print(f"[write] {len(paths)} configs")


if __name__ == "__main__":
    main()
