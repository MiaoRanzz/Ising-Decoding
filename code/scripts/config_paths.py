#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for Hydra config names stored below ``conf/``."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


CODE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_ROOT.parent
CONF_ROOT = REPO_ROOT / "conf"
EXPERIMENTS_ROOT = CONF_ROOT / "experiments"


def rel(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def config_path(config_name: str | Path) -> Path:
    """Return the YAML path for a Hydra config name below ``conf/``.

    Historical generated experiment configs lived directly under ``conf/``. If a
    basename no longer exists there, look for a unique match under
    ``conf/experiments/*/`` so old ad-hoc commands keep working.
    """
    raw = str(config_name)
    if raw.endswith(".yaml"):
        raw = raw[:-5]
    direct = CONF_ROOT / f"{raw}.yaml"
    if direct.exists() or "/" in raw or "\\" in raw:
        return direct
    matches = sorted(EXPERIMENTS_ROOT.glob(f"*/{raw}.yaml"))
    if len(matches) == 1:
        return matches[0]
    return direct


def config_name_from_path(path: str | Path) -> str:
    """Return the Hydra config name for a YAML path when it is below a ``conf/`` dir."""
    path = rel(path)
    try:
        relative = path.relative_to(CONF_ROOT)
    except ValueError:
        parts = path.parts
        if "conf" not in parts:
            return path.stem
        conf_index = len(parts) - 1 - list(reversed(parts)).index("conf")
        relative = Path(*parts[conf_index + 1 :])
    return relative.with_suffix("").as_posix()


def config_basename(config_name: str | Path) -> str:
    """Return the final component of a Hydra config name."""
    raw = str(config_name)
    if raw.endswith(".yaml"):
        raw = raw[:-5]
    return Path(raw).name


def config_lookup_with_basename(
    environments: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...],
) -> dict[str, dict[str, Any]]:
    """Map both full config names and historical basenames to manifest rows."""
    lookup: dict[str, dict[str, Any]] = {}
    for env in environments:
        item = dict(env)
        full = str(item["config_name"])
        lookup[full] = item
        lookup.setdefault(config_basename(full), item)
    return lookup
