"""Fixed local-group representation and group-level safe-no-op risk gate.

Groups are *not learned*: they are connected components of active proposal
packets in a small space-time neighbourhood, split deterministically to bound
their extent.  The learned gate makes exactly one accept/no-op decision per
group, so a correlated group is removed atomically.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from collections import deque
import numpy as np
import torch
from torch import nn

GROUP_HELPFUL, GROUP_NEUTRAL, GROUP_HARMFUL = 1, 0, -1
RISK_CLASS_HARMFUL, RISK_CLASS_NEUTRAL, RISK_CLASS_HELPFUL = 0, 1, 2
RISK_CLASS_NAMES = ("harmful", "neutral", "helpful")


def split_rows(total: int, train_fraction: float, validation_fraction: float, seed: int):
    if total < 3 or train_fraction <= 0 or validation_fraction <= 0 or train_fraction + validation_fraction >= 1:
        raise ValueError("need non-empty train/validation/test shot splits")
    order = np.random.default_rng(seed).permutation(total)
    a, b = int(total * train_fraction), int(total * (train_fraction + validation_fraction))
    if a == 0 or a == b or b == total:
        raise ValueError("split fractions produced an empty split")
    return np.sort(order[:a]), np.sort(order[a:b]), np.sort(order[b:])


@dataclass(frozen=True)
class GroupingConfig:
    time_radius: int = 1
    spatial_radius: int = 1
    max_group_members: int = 8
    max_time_span: int = 3
    max_spatial_span: int = 3


@dataclass(frozen=True)
class GroupGateArchitecture:
    input_channels: int = 12
    hidden_channels: int = 64
    num_hidden_layers: int = 3
    kernel_size: int = 3
    dropout: float = 0.05
    group_hidden_channels: int = 128
    num_risk_classes: int = 3
    def to_dict(self): return asdict(self)


def proposal_actions_from_logits(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 5 or logits.shape[1] != 4:
        raise ValueError("proposal logits must be (batch, 4, rounds, distance, distance)")
    return logits >= 0


def packet_activity(actions: np.ndarray) -> np.ndarray:
    return np.stack((actions[0] | actions[1], actions[2], actions[3]), axis=0)


def _neighbors(node, present, cfg: GroupingConfig):
    p, t, y, x = node
    for dt in range(-cfg.time_radius, cfg.time_radius + 1):
        for dy in range(-cfg.spatial_radius, cfg.spatial_radius + 1):
            for dx in range(-cfg.spatial_radius, cfg.spatial_radius + 1):
                if abs(dy) + abs(dx) > cfg.spatial_radius:
                    continue
                for q in range(3):
                    other = (q, t + dt, y + dy, x + dx)
                    if other != node and other in present:
                        yield other


def _fits(group, candidate, cfg: GroupingConfig) -> bool:
    if len(group) >= cfg.max_group_members:
        return False
    points = group + [candidate]
    ts, ys, xs = [z[1] for z in points], [z[2] for z in points], [z[3] for z in points]
    return max(ts) - min(ts) + 1 <= cfg.max_time_span and max(ys) - min(ys) + 1 <= cfg.max_spatial_span and max(xs) - min(xs) + 1 <= cfg.max_spatial_span


def build_groups(actions: np.ndarray, cfg: GroupingConfig) -> list[np.ndarray]:
    """Return a deterministic partition of active packets as (members, 4)."""
    active = packet_activity(actions)
    nodes = {tuple(int(v) for v in row) for row in np.argwhere(active)}
    unassigned, groups = set(nodes), []
    while unassigned:
        seed = min(unassigned)
        group, queue = [], deque([seed])
        queued = {seed}
        while queue:
            node = queue.popleft()
            if node not in unassigned or not _fits(group, node, cfg):
                continue
            unassigned.remove(node); group.append(node)
            for other in _neighbors(node, nodes, cfg):
                if other in unassigned and other not in queued:
                    queued.add(other); queue.append(other)
        groups.append(np.asarray(group, dtype=np.int16))
    return groups


def gate_features(train_x: torch.Tensor, proposal_logits: torch.Tensor, proposal_actions: torch.Tensor) -> torch.Tensor:
    return torch.cat((train_x.float(), proposal_logits.float(), proposal_actions.float()), dim=1)


class LocalGroupSafeNoOpGate(nn.Module):
    """3-D context trunk followed by a three-class group-risk head."""
    def __init__(self, architecture: GroupGateArchitecture = GroupGateArchitecture()):
        super().__init__()
        if architecture.input_channels != 12 or architecture.num_hidden_layers < 1 or architecture.num_risk_classes != 3:
            raise ValueError("invalid group-gate architecture")
        self.architecture = architecture
        layers, c = [], architecture.input_channels
        for _ in range(architecture.num_hidden_layers):
            layers += [nn.Conv3d(c, architecture.hidden_channels, architecture.kernel_size, padding=architecture.kernel_size // 2), nn.GELU(approximate="tanh"), nn.Dropout3d(architecture.dropout)]
            c = architecture.hidden_channels
        self.trunk = nn.Sequential(*layers)
        # mean member context, max member context, whole-shot context, size/type statistics
        self.head = nn.Sequential(nn.Linear(3*c + 4, architecture.group_hidden_channels), nn.GELU(approximate="tanh"), nn.Dropout(architecture.dropout), nn.Linear(architecture.group_hidden_channels, architecture.num_risk_classes))

    def forward(self, features: torch.Tensor, group_members: torch.Tensor, group_ptr: torch.Tensor) -> torch.Tensor:
        """Score groups. members rows are [batch_index, packet_type, t, y, x]."""
        context = self.trunk(features)
        groups = int(group_ptr.numel() - 1)
        if groups == 0:
            return context.new_empty((0,))
        counts = group_ptr[1:] - group_ptr[:-1]
        gidx = torch.repeat_interleave(torch.arange(groups, device=context.device), counts)
        b, p, t, y, x = group_members.T.long()
        emb = context[b, :, t, y, x]
        total = context.new_zeros((groups, context.shape[1])); total.index_add_(0, gidx, emb)
        mean = total / counts[:, None].clamp_min(1)
        maximum = torch.full_like(total, -torch.inf)
        maximum.scatter_reduce_(0, gidx[:, None].expand_as(emb), emb, reduce="amax", include_self=True)
        shot_context = context.mean(dim=(2, 3, 4))[b[group_ptr[:-1]]]
        type_counts = context.new_zeros((groups, 3)); type_counts.index_add_(0, gidx, torch.nn.functional.one_hot(p, 3).to(context.dtype))
        stats = torch.cat((counts[:, None].to(context.dtype) / 8.0, type_counts / counts[:, None].clamp_min(1)), dim=1)
        return self.head(torch.cat((mean, maximum, shot_context, stats), dim=1))
