# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load fixed d=9/XV surface-code geometry artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass(frozen=True)
class SurfaceGeometry:
    """Arrays needed by detector preprocessing and residual construction."""

    hx: torch.Tensor
    hz: torch.Tensor
    stab_x: torch.Tensor
    stab_z: torch.Tensor
    weight_x: torch.Tensor
    weight_z: torch.Tensor


def load_geometry(path: Path) -> SurfaceGeometry:
    """Load a geometry NPZ created from the authoritative training code."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Geometry artifact not found: {path}")
    with np.load(path, allow_pickle=False) as arrays:
        return SurfaceGeometry(
            hx=torch.as_tensor(arrays["Hx_i32"], dtype=torch.float32),
            hz=torch.as_tensor(arrays["Hz_i32"], dtype=torch.float32),
            stab_x=torch.as_tensor(arrays["stab_x"], dtype=torch.long),
            stab_z=torch.as_tensor(arrays["stab_z"], dtype=torch.long),
            weight_x=torch.as_tensor(arrays["w_mapXgrid"], dtype=torch.float32),
            weight_z=torch.as_tensor(arrays["w_mapZgrid"], dtype=torch.float32),
        )
