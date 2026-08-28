# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed configuration boundary for the standalone decoder."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sysconfig
from typing import Any

from omegaconf import DictConfig


def deployment_root() -> Path:
    """Return the source checkout or installed shared-data directory."""
    source_root = Path(__file__).resolve().parents[2]
    if (source_root / "conf").is_dir():
        return source_root
    installed_root = Path(sysconfig.get_path("data")) / "share/ising_decoder"
    return installed_root


def _resolve(root: Path, value: Any) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else (root / path).resolve()


@dataclass(frozen=True)
class DecoderConfig:
    """Runtime configuration for one basis-specific decoder instance."""

    checkpoint_path: Path
    geometry_path: Path
    dem_path: Path
    basis: str
    device: str = "cuda"
    distance: int = 9
    n_rounds: int = 9
    rotation: str = "XV"
    return_residual: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "checkpoint_path", Path(self.checkpoint_path))
        object.__setattr__(self, "geometry_path", Path(self.geometry_path))
        object.__setattr__(self, "dem_path", Path(self.dem_path))
        object.__setattr__(self, "basis", str(self.basis).upper())
        object.__setattr__(self, "rotation", str(self.rotation).upper())
        if self.basis not in ("X", "Z"):
            raise ValueError(f"basis must be 'X' or 'Z', got {self.basis!r}")
        if self.distance != 9 or self.n_rounds != 9 or self.rotation != "XV":
            raise ValueError(
                "This deployment artifact supports only distance=9, "
                "n_rounds=9, rotation='XV'"
            )

    @classmethod
    def from_omegaconf(
        cls,
        cfg: DictConfig,
        *,
        root: Path | None = None,
    ) -> "DecoderConfig":
        """Convert a Hydra-composed config into the core's plain dataclass."""
        base = deployment_root() if root is None else Path(root).resolve()
        return cls(
            checkpoint_path=_resolve(base, cfg.checkpoint_path),
            geometry_path=_resolve(base, cfg.geometry_path),
            dem_path=_resolve(base, cfg.basis.dem_path),
            basis=str(cfg.basis.name).upper(),
            device=str(cfg.device),
            distance=int(cfg.distance),
            n_rounds=int(cfg.n_rounds),
            rotation=str(cfg.rotation).upper(),
            return_residual=bool(cfg.return_residual),
        )

    @property
    def num_detectors(self) -> int:
        half = (self.distance * self.distance - 1) // 2
        return 2 * self.n_rounds * half
