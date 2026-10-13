#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate the 16 physical-noise configs used by QEWC v3 Stress screening."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[4]
BASE_CONFIG = REPO_ROOT / "conf" / "experiments" / "qewc_v3" / "realistic" / "t0_base.yaml"
OUTPUT_DIR = REPO_ROOT / "conf" / "experiments" / "qewc_v3" / "stress"
MULTIPLIERS = (1.5, 2.0, 3.0, 4.0)
FAMILIES = {
    "measurement": ("p_meas_X", "p_meas_Z"),
    "cnot": (
        "p_cnot_IX", "p_cnot_IY", "p_cnot_IZ", "p_cnot_XI", "p_cnot_XX",
        "p_cnot_XY", "p_cnot_XZ", "p_cnot_YI", "p_cnot_YX", "p_cnot_YY",
        "p_cnot_YZ", "p_cnot_ZI", "p_cnot_ZX", "p_cnot_ZY", "p_cnot_ZZ",
    ),
    "idle": (
        "p_idle_cnot_X", "p_idle_cnot_Y", "p_idle_cnot_Z",
        "p_idle_spam_X", "p_idle_spam_Y", "p_idle_spam_Z",
    ),
    "z_bias": (
        "p_prep_X", "p_meas_X", "p_idle_cnot_Z", "p_idle_spam_Z",
        "p_cnot_IZ", "p_cnot_XZ", "p_cnot_YZ", "p_cnot_ZI",
        "p_cnot_ZX", "p_cnot_ZY", "p_cnot_ZZ",
    ),
}


def label(multiplier: float) -> str:
    return str(multiplier).replace(".", "p")


def build_configs(base: dict) -> dict[str, dict]:
    configs = {}
    for family, keys in FAMILIES.items():
        for multiplier in MULTIPLIERS:
            config = deepcopy(base)
            for key in keys:
                config["data"]["noise_model"][key] = round(
                    float(base["data"]["noise_model"][key]) * multiplier, 10
                )
            configs[f"{family}_{label(multiplier)}x"] = config
    return configs


def main() -> None:
    base = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    header = (
        "# @package _global_\n"
        "# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.\n"
        "# SPDX-License-Identifier: Apache-2.0\n\n"
    )
    for name, config in build_configs(base).items():
        path = OUTPUT_DIR / f"{name}.yaml"
        path.write_text(
            header + yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )
        print(path.relative_to(REPO_ROOT))


if __name__ == "__main__":
    main()
