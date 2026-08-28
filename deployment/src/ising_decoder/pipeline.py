# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standalone detector-input pre-decoder pipeline.

This file is adapted from ``code/data/predecoder_transform.py`` and
``code/evaluation/logical_error_rate.py``. It keeps the d=9/T=9/XV threshold
path and omits training, distributed evaluation, and export machinery.
"""

from __future__ import annotations

import torch
from torch import nn

from .geometry import SurfaceGeometry


class PreDecoderPipeline(nn.Module):
    """Map detector bits to ``pre_L`` and residual detector bits."""

    def __init__(
        self,
        model: nn.Module,
        geometry: SurfaceGeometry,
        *,
        basis: str,
        distance: int = 9,
        n_rounds: int = 9,
    ):
        super().__init__()
        self.model = model
        self.distance = int(distance)
        self.n_rounds = int(n_rounds)
        self.distance_sq = self.distance * self.distance
        self.half = (self.distance_sq - 1) // 2
        self.basis = str(basis).upper()
        if self.basis not in ("X", "Z"):
            raise ValueError(f"basis must be 'X' or 'Z', got {basis!r}")

        stab_x = geometry.stab_x.to(dtype=torch.long)
        stab_z = geometry.stab_z.to(dtype=torch.long)
        col_idx = torch.arange(self.half)

        scatter_x = torch.zeros(self.distance_sq, self.half, dtype=torch.float32)
        scatter_x[stab_x, col_idx] = 1.0
        scatter_z = torch.zeros(self.distance_sq, self.half, dtype=torch.float32)
        scatter_z[stab_z, col_idx] = 1.0
        gather_x = scatter_x.t().contiguous()
        gather_z = scatter_z.t().contiguous()

        scatter_perm_x = torch.full(
            (self.distance_sq,), self.half, dtype=torch.long
        )
        scatter_perm_x[stab_x] = col_idx
        scatter_perm_z = torch.full(
            (self.distance_sq,), self.half, dtype=torch.long
        )
        scatter_perm_z[stab_z] = col_idx
        self.register_buffer("scatter_perm_x", scatter_perm_x)
        self.register_buffer("scatter_perm_z", scatter_perm_z)
        self.register_buffer(
            "zero_pad_row",
            torch.zeros(1, 1, 1, dtype=torch.float32),
        )

        logical_x = torch.zeros((1, self.distance_sq), dtype=torch.float32)
        logical_z = torch.zeros((1, self.distance_sq), dtype=torch.float32)
        logical_x[0, : self.distance] = 1
        logical_z[0, :: self.distance] = 1
        logical_padded = torch.zeros(
            self.half,
            self.distance_sq,
            dtype=torch.float32,
        )
        logical_padded[0] = logical_x[0] if self.basis == "X" else logical_z[0]

        self.register_buffer(
            "post_matrices",
            torch.stack(
                [
                    geometry.hx.to(dtype=torch.float32),
                    geometry.hz.to(dtype=torch.float32),
                    gather_x,
                    gather_z,
                    logical_padded,
                ],
                dim=0,
            ),
        )
        self.register_buffer(
            "w_mapXgrid",
            geometry.weight_x.to(dtype=torch.float32),
        )
        self.register_buffer(
            "w_mapZgrid",
            geometry.weight_z.to(dtype=torch.float32),
        )

    def _transform(
        self,
        dets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = dets.shape[0]
        timeline_len = 2 * self.n_rounds
        device = dets.device
        dets_timeline = (
            dets.to(torch.float32)
            .view(batch_size, timeline_len, self.half)
            .permute(0, 2, 1)
            .contiguous()
        )
        zero_col = self.zero_pad_row.expand(batch_size, self.half, 1)
        padded = torch.cat([dets_timeline, zero_col], dim=2)
        sentinel_idx = timeline_len

        x_bulk_idx = torch.arange(
            1,
            timeline_len - 1,
            2,
            dtype=torch.long,
            device=device,
        )
        z_bulk_idx = torch.arange(
            2,
            timeline_len,
            2,
            dtype=torch.long,
            device=device,
        )
        zero_idx = torch.zeros(1, dtype=torch.long, device=device)
        sentinel = torch.full(
            (1,),
            sentinel_idx,
            dtype=torch.long,
            device=device,
        )
        if self.basis == "X":
            idx_x = torch.cat([zero_idx, x_bulk_idx])
            idx_z = torch.cat([sentinel, z_bulk_idx[:-1], sentinel])
        else:
            idx_z = torch.cat([zero_idx, z_bulk_idx])
            idx_x = torch.cat([sentinel, x_bulk_idx[:-1], sentinel])

        x_syn_diff = torch.index_select(padded, 2, idx_x)
        z_syn_diff = torch.index_select(padded, 2, idx_z)

        boundary_mask = torch.cat(
            [
                torch.zeros(1, device=device, dtype=torch.float32),
                torch.ones(
                    self.n_rounds - 2,
                    device=device,
                    dtype=torch.float32,
                ),
                torch.zeros(1, device=device, dtype=torch.float32),
            ]
        ).view(1, self.n_rounds, 1, 1)
        if self.basis == "X":
            x_present = self.w_mapXgrid.expand(
                batch_size, self.n_rounds, self.distance, self.distance
            )
            z_present = (self.w_mapZgrid * boundary_mask).expand(
                batch_size, self.n_rounds, self.distance, self.distance
            )
        else:
            x_present = (self.w_mapXgrid * boundary_mask).expand(
                batch_size, self.n_rounds, self.distance, self.distance
            )
            z_present = self.w_mapZgrid.expand(
                batch_size, self.n_rounds, self.distance, self.distance
            )

        zero_pad = self.zero_pad_row.expand(batch_size, 1, self.n_rounds)
        x_grid = torch.index_select(
            torch.cat([x_syn_diff, zero_pad], dim=1),
            1,
            self.scatter_perm_x,
        )
        z_grid = torch.index_select(
            torch.cat([z_syn_diff, zero_pad], dim=1),
            1,
            self.scatter_perm_z,
        )
        x_type = (
            x_grid.reshape(
                batch_size,
                self.distance,
                self.distance,
                self.n_rounds,
            )
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        z_type = (
            z_grid.reshape(
                batch_size,
                self.distance,
                self.distance,
                self.n_rounds,
            )
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        train_x = torch.cat(
            [
                x_type.unsqueeze(1),
                z_type.unsqueeze(1),
                x_present.unsqueeze(1),
                z_present.unsqueeze(1),
            ],
            dim=1,
        ).contiguous()
        baseline = (
            dets_timeline.permute(0, 2, 1)
            .contiguous()
            .view(batch_size, -1)
        )
        return train_x, x_syn_diff, z_syn_diff, baseline

    def forward(self, dets: torch.Tensor) -> torch.Tensor:
        batch_size = dets.shape[0]
        train_x, x_syn_diff, z_syn_diff, baseline = self._transform(dets)
        logits = self.model(train_x)

        z_data_corr = (logits[:, 0] >= 0.0).to(torch.int32)
        x_data_corr = (logits[:, 1] >= 0.0).to(torch.int32)
        syn_x_grid = (logits[:, 2] >= 0.0).to(torch.int32)
        syn_z_grid = (logits[:, 3] >= 0.0).to(torch.int32)

        z_corr_flat = (
            z_data_corr.permute(0, 2, 3, 1)
            .contiguous()
            .view(batch_size, self.distance_sq, self.n_rounds)
        )
        x_corr_flat = (
            x_data_corr.permute(0, 2, 3, 1)
            .contiguous()
            .view(batch_size, self.distance_sq, self.n_rounds)
        )
        syn_x_flat = (
            syn_x_grid.permute(0, 2, 3, 1)
            .contiguous()
            .view(batch_size, self.distance_sq, self.n_rounds)
        )
        syn_z_flat = (
            syn_z_grid.permute(0, 2, 3, 1)
            .contiguous()
            .view(batch_size, self.distance_sq, self.n_rounds)
        )
        pre_l_input = z_corr_flat if self.basis == "X" else x_corr_flat
        all_flat = torch.stack(
            [
                z_corr_flat,
                x_corr_flat,
                syn_x_flat,
                syn_z_flat,
                pre_l_input,
            ],
            dim=1,
        ).to(torch.float32)
        all_results = torch.matmul(self.post_matrices, all_flat)
        pre_l = (
            all_results[:, 4, 0:1, :]
            .remainder(2)
            .sum(dim=-1)
            .remainder(2)
            .view(-1)
            .to(torch.int32)
        )

        induced = all_results[:, :2].remainder(2)
        predicted_syndrome = all_results[:, 2:4]
        input_syndrome = torch.stack([x_syn_diff, z_syn_diff], dim=1)
        residual_first = (
            input_syndrome[..., 0:1]
            + predicted_syndrome[..., 0:1]
            + induced[..., 0:1]
        ).remainder(2)
        residual_rest = (
            input_syndrome[..., 1:]
            + predicted_syndrome[..., 1:]
            + predicted_syndrome[..., :-1]
            + induced[..., 1:]
        ).remainder(2)
        residual_xz = torch.cat([residual_first, residual_rest], dim=-1)

        basis_index = 0 if self.basis == "X" else 1
        initial = residual_xz[:, basis_index, :, 0:1].reshape(batch_size, -1)
        rest = (
            residual_xz[:, :, :, 1:]
            .permute(0, 3, 1, 2)
            .contiguous()
            .reshape(batch_size, -1)
        )
        boundary = baseline[:, -self.half :]
        residual = torch.cat([initial, rest, boundary], dim=1).to(torch.int32)
        residual = residual.reshape(batch_size, 2 * self.n_rounds * self.half)
        return torch.cat([pre_l.view(batch_size, 1), residual], dim=1).to(
            torch.uint8
        )
