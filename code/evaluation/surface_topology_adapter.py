# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Map the surface pre-decoder's four spatial channels to topology actions.

The action order is exactly the contiguous PyTorch order of a tensor with
shape ``(4, rounds, distance, distance)``.  The detector row order matches
``SurfaceDetectorInputTransform`` and ``PreDecoderMemoryEvalModule``:

* the first basis-dependent stabilizer block;
* interleaved X/Z stabilizer blocks for subsequent rounds;
* the final boundary-detector block, which the current pre-decoder leaves
  unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from evaluation.topology_gating_v2 import (
    DATA_X,
    DATA_Z,
    MEASUREMENT_X,
    MEASUREMENT_Z,
)
from qec.surface_code.data_mapping import (
    compute_stabX_to_data_index_map,
    compute_stabZ_to_data_index_map,
    construct_X_stab_Parity_check_Mat,
    construct_Z_stab_Parity_check_Mat,
)
from qec.surface_code.memory_circuit import SurfaceCode


CHANNEL_ACTION_TYPES = (DATA_Z, DATA_X, MEASUREMENT_X, MEASUREMENT_Z)


def _parity_matrices(distance: int, rotation: str) -> tuple[torch.Tensor, torch.Tensor]:
    rotation = str(rotation).upper()
    if rotation == "XV":
        return (
            construct_X_stab_Parity_check_Mat(distance).to(torch.uint8),
            construct_Z_stab_Parity_check_Mat(distance).to(torch.uint8),
        )
    if rotation not in ("XH", "ZV", "ZH"):
        raise ValueError(f"unsupported surface-code rotation {rotation!r}")
    code = SurfaceCode(
        distance,
        first_bulk_syndrome_type=rotation[0],
        rotated_type=rotation[1],
    )
    return torch.as_tensor(code.hx, dtype=torch.uint8), torch.as_tensor(
        code.hz, dtype=torch.uint8
    )


@dataclass(frozen=True)
class SurfaceActionAdapter:
    """Frozen four-channel action maps for one basis and geometry."""

    distance: int
    rounds: int
    basis: str
    rotation: str
    extended_h: torch.Tensor
    extended_l: torch.Tensor
    valid_actions: torch.Tensor
    detector_training_mask: torch.Tensor
    action_types: tuple[str, ...]
    x_stabilizer_grid_indices: torch.Tensor
    z_stabilizer_grid_indices: torch.Tensor

    @property
    def action_count(self) -> int:
        return int(self.extended_h.shape[1])

    @property
    def detector_count(self) -> int:
        return int(self.extended_h.shape[0])

    @property
    def num_stabilizers(self) -> int:
        return (self.distance * self.distance - 1) // 2

    def flatten_spatial(self, value: torch.Tensor) -> torch.Tensor:
        """Flatten ``(B,4,T,D,D)`` values into the registered action order."""

        expected = (4, self.rounds, self.distance, self.distance)
        if value.ndim != 5 or tuple(value.shape[1:]) != expected:
            raise ValueError(
                f"expected spatial tensor (B,{','.join(map(str, expected))}), "
                f"got {tuple(value.shape)}"
            )
        return value.contiguous().reshape(value.shape[0], -1)

    def detector_state_from_train_x(self, train_x: torch.Tensor) -> torch.Tensor:
        """Recover the flattened detector state represented by a model input."""

        expected = (4, self.rounds, self.distance, self.distance)
        if train_x.ndim != 5 or tuple(train_x.shape[1:]) != expected:
            raise ValueError(f"unexpected trainX shape {tuple(train_x.shape)}")
        batch = train_x.shape[0]
        x_grid = train_x[:, 0].reshape(batch, self.rounds, -1)
        z_grid = train_x[:, 1].reshape(batch, self.rounds, -1)
        x_syn = x_grid.index_select(2, self.x_stabilizer_grid_indices.to(train_x.device))
        z_syn = z_grid.index_select(2, self.z_stabilizer_grid_indices.to(train_x.device))
        first = x_syn[:, 0] if self.basis == "X" else z_syn[:, 0]
        rest = torch.stack((x_syn[:, 1:], z_syn[:, 1:]), dim=2).reshape(batch, -1)
        boundary = torch.zeros(
            (batch, self.num_stabilizers), dtype=train_x.dtype, device=train_x.device
        )
        return torch.cat((first, rest, boundary), dim=1)

    def action_type_mask(self, allowed: Sequence[str]) -> np.ndarray:
        allowed_set = set(allowed)
        return np.asarray(
            [valid and kind in allowed_set for valid, kind in zip(
                self.valid_actions.tolist(), self.action_types, strict=True
            )],
            dtype=bool,
        )

    def numpy_maps(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
        return (
            self.extended_h.cpu().numpy().astype(np.uint8, copy=False),
            self.extended_l.cpu().numpy().astype(np.uint8, copy=False),
            self.valid_actions.cpu().numpy().astype(bool, copy=False),
            list(self.action_types),
        )


def build_surface_action_adapter(
    distance: int,
    rounds: int,
    basis: str,
    rotation: str = "XV",
) -> SurfaceActionAdapter:
    """Build explicit detector/logical maps for all four spatial channels."""

    distance = int(distance)
    rounds = int(rounds)
    basis = str(basis).upper()
    rotation = str(rotation).upper()
    if distance < 3 or distance % 2 == 0:
        raise ValueError("distance must be an odd integer >= 3")
    if rounds < 2:
        raise ValueError("rounds must be >= 2")
    if basis not in ("X", "Z"):
        raise ValueError("basis must be X or Z")

    d2 = distance * distance
    half = (d2 - 1) // 2
    action_count = 4 * rounds * d2
    main_count = half * (2 * rounds - 1)
    detector_count = main_count + half
    h = torch.zeros((detector_count, action_count), dtype=torch.uint8)
    logical = torch.zeros((1, action_count), dtype=torch.uint8)
    valid = torch.zeros(action_count, dtype=torch.bool)
    train_mask = torch.zeros(detector_count, dtype=torch.bool)
    train_mask[:main_count] = True
    hx, hz = _parity_matrices(distance, rotation)
    x_grid = compute_stabX_to_data_index_map(distance, rotation).to(torch.long)
    z_grid = compute_stabZ_to_data_index_map(distance, rotation).to(torch.long)

    def action_index(channel: int, time_index: int, data_index: int) -> int:
        return ((channel * rounds + time_index) * d2) + data_index

    def detector_row(kind: str, time_index: int, stabilizer: int) -> int | None:
        if time_index == 0:
            if kind != basis:
                return None
            return stabilizer
        offset = 0 if kind == "X" else half
        return half + (time_index - 1) * 2 * half + offset + stabilizer

    for time_index in range(rounds):
        for data_index in range(d2):
            z_action = action_index(0, time_index, data_index)
            x_action = action_index(1, time_index, data_index)
            valid[z_action] = True
            valid[x_action] = True
            for stabilizer in torch.nonzero(hx[:, data_index], as_tuple=False).reshape(-1).tolist():
                row = detector_row("X", time_index, int(stabilizer))
                if row is not None:
                    h[row, z_action] ^= 1
            for stabilizer in torch.nonzero(hz[:, data_index], as_tuple=False).reshape(-1).tolist():
                row = detector_row("Z", time_index, int(stabilizer))
                if row is not None:
                    h[row, x_action] ^= 1

        for kind, channel, grid_indices in (
            ("X", 2, x_grid),
            ("Z", 3, z_grid),
        ):
            boundary_invalid = (basis == "X" and kind == "Z") or (
                basis == "Z" and kind == "X"
            )
            if boundary_invalid and time_index in (0, rounds - 1):
                continue
            for stabilizer, grid_index in enumerate(grid_indices.tolist()):
                action = action_index(channel, time_index, int(grid_index))
                valid[action] = True
                for affected_time in (time_index, time_index + 1):
                    if affected_time >= rounds:
                        continue
                    row = detector_row(kind, affected_time, stabilizer)
                    if row is not None:
                        h[row, action] ^= 1

    if rotation in ("XV", "ZH"):
        logical_x = set(range(distance))
        logical_z = set(range(0, d2, distance))
    else:
        logical_x = set(range(0, d2, distance))
        logical_z = set(range(distance))
    logical_channel = 0 if basis == "X" else 1
    logical_support = logical_x if basis == "X" else logical_z
    for time_index in range(rounds):
        for data_index in logical_support:
            logical[0, action_index(logical_channel, time_index, data_index)] = 1

    action_types = tuple(
        CHANNEL_ACTION_TYPES[channel]
        for channel in range(4)
        for _ in range(rounds * d2)
    )
    if len(action_types) != action_count:
        raise AssertionError("action type construction mismatch")
    if torch.any(h[:, ~valid]) or torch.any(logical[:, ~valid]):
        raise AssertionError("invalid spatial actions must have zero detector/logical support")
    return SurfaceActionAdapter(
        distance=distance,
        rounds=rounds,
        basis=basis,
        rotation=rotation,
        extended_h=h,
        extended_l=logical,
        valid_actions=valid,
        detector_training_mask=train_mask,
        action_types=action_types,
        x_stabilizer_grid_indices=x_grid,
        z_stabilizer_grid_indices=z_grid,
    )


__all__ = ["SurfaceActionAdapter", "build_surface_action_adapter"]
