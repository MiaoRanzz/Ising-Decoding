"""Structured action representation and warm-startable Ising-fast model.

The original model independently thresholds four correction logits.  This
module preserves its trunk but replaces that output with three mutually
exclusive action distributions:

* data: ``[no_op, z, x, y]``;
* X-syndrome: ``[no_op, apply]``;
* Z-syndrome: ``[no_op, apply]``.

The warm start is action-equivalent to the old four-logit model away from
exact threshold ties: ``[0, z, x, z + x, 0, sx, 0, sz]``. Therefore changing
representation does not materially change the initial decoder policy.
"""
from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F


DATA_CLASSES = ("no_op", "z", "x", "y")
SYNDROME_CLASSES = ("no_op", "apply")
STRUCTURED_CHANNELS = 8


@dataclass(frozen=True)
class StructuredLoss:
    total: torch.Tensor
    data: torch.Tensor
    sx: torch.Tensor
    sz: torch.Tensor


class StructuredIsingFast(nn.Module):
    """An Ising-fast trunk with a single structured 3-D action head."""

    def __init__(self, trunk: nn.Module, action_head: nn.Conv3d):
        super().__init__()
        if action_head.out_channels != STRUCTURED_CHANNELS:
            raise ValueError(f"structured head must emit {STRUCTURED_CHANNELS} logits")
        self.trunk = trunk
        self.action_head = action_head

    def forward(self, train_x: torch.Tensor) -> torch.Tensor:
        return self.action_head(self.trunk(train_x))


def _last_conv(layers: list[nn.Module]) -> tuple[int, nn.Conv3d]:
    if not layers or not isinstance(layers[-1], nn.Conv3d):
        raise ValueError("expected the source Ising-fast model to end in Conv3d")
    return len(layers) - 1, layers[-1]


def from_ising_fast(source_model: nn.Module) -> StructuredIsingFast:
    """Split a ``PreDecoderModelMemory_v1`` into trunk/head and warm-start it.

    The source model must expose its sequential convolution stack as ``net``.
    This intentionally targets model_id=1, whose final layer has four useful
    correction logits.  A widened source head is accepted; only its first four
    channels are semantically meaningful to the existing evaluator.
    """
    source_net = getattr(source_model, "net", None)
    if not isinstance(source_net, nn.Sequential):
        raise TypeError("source checkpoint is not a sequential Ising-fast model")
    # IMPORTANT: index the Sequential directly. ``Module.children()`` uses
    # ``named_children()``, which de-duplicates repeated module objects.  The
    # original Ising-fast construction reuses one activation instance after
    # several convolutions, so ``list(source_net.children())`` silently drops
    # activation occurrences and changes the trunk computation completely.
    # Sequential iteration preserves every registered position and therefore
    # reconstructs the exact original forward path.
    source_layers = list(source_net)
    index, old_head = _last_conv(source_layers)
    if old_head.out_channels < 4:
        raise ValueError("source head needs at least four correction channels")
    trunk = nn.Sequential(*source_layers[:index])
    head = nn.Conv3d(
        old_head.in_channels,
        STRUCTURED_CHANNELS,
        kernel_size=old_head.kernel_size,
        stride=old_head.stride,
        padding=old_head.padding,
        dilation=old_head.dilation,
        groups=old_head.groups,
        bias=old_head.bias is not None,
        padding_mode=old_head.padding_mode,
    ).to(device=old_head.weight.device, dtype=old_head.weight.dtype)
    with torch.no_grad():
        head.weight.zero_()
        if head.bias is not None:
            head.bias.zero_()
        # data: no-op, Z, X, Y.  Softmax(argmax) reproduces independent
        # thresholding of old logits z and x exactly.
        head.weight[1].copy_(old_head.weight[0])
        head.weight[2].copy_(old_head.weight[1])
        head.weight[3].copy_(old_head.weight[0] + old_head.weight[1])
        head.weight[5].copy_(old_head.weight[2])
        head.weight[7].copy_(old_head.weight[3])
        if head.bias is not None and old_head.bias is not None:
            head.bias[1].copy_(old_head.bias[0])
            head.bias[2].copy_(old_head.bias[1])
            head.bias[3].copy_(old_head.bias[0] + old_head.bias[1])
            head.bias[5].copy_(old_head.bias[2])
            head.bias[7].copy_(old_head.bias[3])
    return StructuredIsingFast(trunk, head)


def split_logits(logits: torch.Tensor, no_op_bias: float = 0.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return data/sx/sz logits, optionally adding a common no-op bias."""
    if logits.ndim != 5 or logits.shape[1] != STRUCTURED_CHANNELS:
        raise ValueError("structured logits must have shape (B, 8, T, D, D)")
    data, sx, sz = logits[:, :4].clone(), logits[:, 4:6].clone(), logits[:, 6:8].clone()
    if no_op_bias:
        data[:, 0].add_(no_op_bias)
        sx[:, 0].add_(no_op_bias)
        sz[:, 0].add_(no_op_bias)
    return data, sx, sz


def actions_from_logits(logits: torch.Tensor, no_op_bias: float = 0.0) -> torch.Tensor:
    """Decode structured logits into the evaluator's four binary actions."""
    data, sx, sz = split_logits(logits, no_op_bias)
    data_choice = data.argmax(dim=1)
    output = torch.zeros((logits.shape[0], 4, *logits.shape[2:]), dtype=torch.bool, device=logits.device)
    output[:, 0] = (data_choice == 1) | (data_choice == 3)  # Z part of Y included.
    output[:, 1] = (data_choice == 2) | (data_choice == 3)  # X part of Y included.
    output[:, 2] = sx.argmax(dim=1) == 1
    output[:, 3] = sz.argmax(dim=1) == 1
    return output


def targets_from_actions(actions: torch.Tensor) -> torch.Tensor:
    """Map four binary action planes to [data, sx, sz] categorical targets."""
    if actions.ndim != 5 or actions.shape[1] != 4:
        raise ValueError("actions must have shape (B, 4, T, D, D)")
    actions = actions.to(torch.long)
    data = actions[:, 0] + 2 * actions[:, 1]
    return torch.stack((data, actions[:, 2], actions[:, 3]), dim=1)


def packet_mask_from_action_difference(before: torch.Tensor, after: torch.Tensor) -> torch.Tensor:
    """Return the three structured positions modified between two action tensors."""
    if before.shape != after.shape or before.ndim != 5 or before.shape[1] != 4:
        raise ValueError("before/after must be same-shaped four-channel action tensors")
    changed = before.to(torch.bool) ^ after.to(torch.bool)
    return torch.stack((changed[:, 0] | changed[:, 1], changed[:, 2], changed[:, 3]), dim=1)


def structured_cross_entropy(
    logits: torch.Tensor,
    actions: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> StructuredLoss:
    """Mean categorical CE for each action family; optional mask is Bx3xTxDxD."""
    target = targets_from_actions(actions)
    data, sx, sz = split_logits(logits)
    parts = (data, sx, sz)
    losses: list[torch.Tensor] = []
    for family, family_logits in enumerate(parts):
        value = F.cross_entropy(family_logits, target[:, family], reduction="none")
        if mask is None:
            losses.append(value.mean())
        else:
            family_mask = mask[:, family].to(value.dtype)
            denom = family_mask.sum().clamp_min(1.0)
            losses.append((value * family_mask).sum() / denom)
    total = sum(losses) / len(losses)
    return StructuredLoss(total=total, data=losses[0], sx=losses[1], sz=losses[2])
