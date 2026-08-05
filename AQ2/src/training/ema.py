"""Exponential Moving Average (EMA) for model parameters.

This module provides a reusable EMA implementation for training neural networks.
The implementation supports a *warm-up* (bias-correction) schedule where the
effective decay starts close to zero and anneals to the target value over the
first ~1e4 updates.  This avoids the common pitfall of using a cold EMA shadow
still dominated by the random initialization during short runs.
"""

from typing import Dict

import torch


class ExponentialMovingAverage:
    """Tracks exponential moving average of model parameters.

    Args:
        model: The model whose parameters to track.
        decay: The asymptotic decay factor for the moving average (default: 0.9999).
        warmup_steps: Number of updates over which to anneal the effective decay
            from a small value to ``decay``.  Default 10000 matches the standard
            ``min(decay, (1+t)/(10+t))`` schedule.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        decay: float = 0.9999,
        warmup_steps: int = 10000,
    ) -> None:
        self.decay = decay
        self.warmup_steps = warmup_steps
        self.num_updates = 0
        self.shadow: Dict[str, torch.Tensor] = {}
        self.backup: Dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def _effective_decay(self) -> float:
        """Return bias-corrected decay for the current update count."""
        t = self.num_updates
        # Standard warmup schedule: starts ~0.09 at t=0, reaches 0.999 at t≈9k,
        # and asymptotes to ``self.decay``.
        warmed = (1.0 + t) / (10.0 + t)
        return min(self.decay, warmed)

    def update(self, model: torch.nn.Module) -> None:
        """Update the shadow parameters with current model parameters."""
        d = self._effective_decay()
        self.num_updates += 1
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = d * self.shadow[name] + (1.0 - d) * param.data

    def apply_shadow(self, model: torch.nn.Module) -> None:
        """Apply shadow parameters to the model (backup current params first)."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self, model: torch.nn.Module) -> None:
        """Restore original parameters from backup."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}

    def state_dict(self) -> dict:
        """Return state dict for checkpointing."""
        return {
            "shadow": self.shadow,
            "decay": self.decay,
            "warmup_steps": self.warmup_steps,
            "num_updates": self.num_updates,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Load state dict from checkpoint."""
        self.shadow = state_dict.get("shadow", {})
        self.decay = state_dict.get("decay", self.decay)
        self.warmup_steps = state_dict.get("warmup_steps", self.warmup_steps)
        self.num_updates = state_dict.get("num_updates", self.num_updates)

    def to_device(self, device: torch.device) -> None:
        """Move shadow and backup tensors to the specified device."""
        for name, tensor in list(self.shadow.items()):
            if torch.is_tensor(tensor):
                self.shadow[name] = tensor.to(device)
        for name, tensor in list(self.backup.items()):
            if torch.is_tensor(tensor):
                self.backup[name] = tensor.to(device)
