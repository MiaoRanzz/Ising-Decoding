# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Union

import torch
import torch.nn as nn
from torch.amp import autocast


TensorDict = Dict[str, torch.Tensor]


@dataclass
class EWCState:
    task_name: str
    mean: TensorDict
    fisher: TensorDict


def canonical_param_name(name: str) -> str:
    """Normalize parameter names across plain, DDP and torch.compile wrappers."""
    prefixes = ("module.", "_orig_mod.")
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if name.startswith(prefix):
                name = name[len(prefix):]
                changed = True
    return name


def iter_trainable_named_parameters(model: nn.Module):
    for name, param in model.named_parameters():
        if param.requires_grad and param.dtype.is_floating_point:
            yield canonical_param_name(name), param


@torch.no_grad()
def capture_parameter_snapshot(model: nn.Module, *, device: Union[str, torch.device] = "cpu") -> TensorDict:
    return {
        name: param.detach().to(device=device, dtype=torch.float32).clone()
        for name, param in iter_trainable_named_parameters(model)
    }


def diagonal_ewc_penalty(model: nn.Module, states: Iterable[EWCState]) -> torch.Tensor:
    params = dict(iter_trainable_named_parameters(model))
    device = next(model.parameters()).device
    penalty = torch.zeros((), device=device, dtype=torch.float32)
    for state in states:
        for name, mean in state.mean.items():
            param = params.get(name)
            fisher = state.fisher.get(name)
            if param is None or fisher is None:
                continue
            mean_t = mean.to(device=param.device, dtype=param.dtype)
            fisher_t = fisher.to(device=param.device, dtype=param.dtype)
            penalty = penalty + 0.5 * (fisher_t * (param - mean_t).pow(2)).sum().float()
    return penalty


def add_ewc_penalty_to_loss(
    base_loss: torch.Tensor,
    ewc_penalty: torch.Tensor,
    *,
    ewc_lambda: float,
    batch_size: int,
) -> torch.Tensor:
    if ewc_lambda <= 0 or batch_size <= 0:
        return base_loss
    return base_loss + float(batch_size) * float(ewc_lambda) * ewc_penalty.to(
        device=base_loss.device, dtype=base_loss.dtype
    )


def save_ewc_state(state: EWCState, path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "task_name": state.task_name,
            "mean": {k: v.detach().cpu() for k, v in state.mean.items()},
            "fisher": {k: v.detach().cpu() for k, v in state.fisher.items()},
        },
        path,
    )


def load_ewc_state(path: Union[str, Path], *, device: Union[str, torch.device] = "cpu") -> EWCState:
    data = torch.load(Path(path), map_location=device, weights_only=False)
    return EWCState(
        task_name=str(data["task_name"]),
        mean={canonical_param_name(k): v.detach().to(device=device, dtype=torch.float32) for k, v in data["mean"].items()},
        fisher={
            canonical_param_name(k): v.detach().to(device=device, dtype=torch.float32)
            for k, v in data["fisher"].items()
        },
    )


def load_ewc_states(path_or_paths: Union[str, Path, Iterable[Union[str, Path]]],
                    *,
                    device: Union[str, torch.device] = "cpu") -> List[EWCState]:
    if isinstance(path_or_paths, (str, Path)):
        path = Path(path_or_paths)
        if not path.exists():
            return []
        paths = sorted(path.glob("*.pt")) if path.is_dir() else [path]
    else:
        paths = sorted(Path(p) for p in path_or_paths)
    return [load_ewc_state(path, device=device) for path in paths]


def merge_ewc_states(states: Iterable[EWCState], task_name: str = "merged") -> EWCState:
    states = list(states)
    if not states:
        return EWCState(task_name=task_name, mean={}, fisher={})
    fisher_sum: TensorDict = {}
    mean_weighted_sum: TensorDict = {}
    for state in states:
        for name, fisher in state.fisher.items():
            mean = state.mean[name]
            f = fisher.detach().cpu().float()
            m = mean.detach().cpu().float()
            fisher_sum[name] = fisher_sum.get(name, torch.zeros_like(f)) + f
            mean_weighted_sum[name] = mean_weighted_sum.get(name, torch.zeros_like(m)) + f * m
    merged_mean = {
        name: torch.where(fisher_sum[name] > 0, mean_weighted_sum[name] / fisher_sum[name], mean_weighted_sum[name])
        for name in fisher_sum
    }
    return EWCState(task_name=task_name, mean=merged_mean, fisher=fisher_sum)


def estimate_diagonal_fisher(
    model: nn.Module,
    generator,
    *,
    task_name: str,
    num_samples: int,
    batch_size: int,
    device: Union[str, torch.device],
    enable_fp16: bool = False,
    enable_bf16: bool = False,
) -> EWCState:
    model.eval()
    device = torch.device(device)
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction="sum")
    fisher = {
        name: torch.zeros_like(param.detach(), device=device, dtype=torch.float32)
        for name, param in iter_trainable_named_parameters(model)
    }
    total_samples = 0
    steps = int(math.ceil(float(num_samples) / float(batch_size)))
    if enable_fp16:
        autocast_dtype = torch.float16
    elif enable_bf16:
        autocast_dtype = torch.bfloat16
    else:
        autocast_dtype = torch.float32

    for step in range(steps):
        current_batch = min(batch_size, num_samples - total_samples)
        if current_batch <= 0:
            break
        train_x, train_y = generator.generate_batch(step=step, batch_size=current_batch)
        train_x = train_x.to(device, non_blocking=True)
        train_y = train_y.to(device, non_blocking=True)
        model.zero_grad(set_to_none=True)
        use_autocast = device.type == "cuda" and (enable_fp16 or enable_bf16)
        with autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_autocast):
            outputs = model(train_x)
            loss = loss_fn(outputs, train_y) / max(1, current_batch)
        loss.backward()
        for name, param in iter_trainable_named_parameters(model):
            if param.grad is not None:
                fisher[name] += param.grad.detach().float().pow(2) * current_batch
        total_samples += current_batch

    scale = max(1, total_samples)
    fisher = {name: value.detach().cpu() / scale for name, value in fisher.items()}
    mean = capture_parameter_snapshot(model)
    model.zero_grad(set_to_none=True)
    return EWCState(task_name=task_name, mean=mean, fisher=fisher)
