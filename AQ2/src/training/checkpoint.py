"""Checkpoint loading and saving utilities.

This module provides common utilities for loading and saving model checkpoints.
"""

import os
from typing import Dict, Optional, Tuple

import torch


def strip_module_prefix(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Remove 'module.' prefix from state dict keys (added by DDP).
    
    Args:
        state: Model state dict, potentially with 'module.' prefixed keys.
        
    Returns:
        State dict with 'module.' prefix removed from keys.
    """
    if not state:
        return state
    sample_key = next(iter(state.keys()))
    if isinstance(sample_key, str) and sample_key.startswith("module."):
        return {k.replace("module.", "", 1): v for k, v in state.items()}
    return state


def extract_ema_shadow(checkpoint: dict) -> Optional[Dict[str, torch.Tensor]]:
    """Extract EMA shadow parameters from a checkpoint.
    
    Args:
        checkpoint: Loaded checkpoint dict.
        
    Returns:
        EMA shadow state dict if present, None otherwise.
    """
    ema_state = checkpoint.get("ema_state_dict")
    if not isinstance(ema_state, dict) or not ema_state:
        return None
    if "shadow" in ema_state and isinstance(ema_state["shadow"], dict):
        shadow = ema_state["shadow"]
        if shadow:
            return shadow
    # Some older checkpoints may store EMA params directly.
    if all(isinstance(k, str) and torch.is_tensor(v) for k, v in ema_state.items()):
        return ema_state
    return None


def extract_model_state_dict(
    checkpoint: dict, use_ema: bool = False
) -> Dict[str, torch.Tensor]:
    """Extract model state dict from checkpoint, optionally using EMA weights.
    
    If use_ema=True and an EMA shadow is present, this overlays EMA parameter
    tensors on top of the saved model_state_dict so that any non-EMA entries
    (e.g., non-parameter tensors) are preserved.
    
    Args:
        checkpoint: Loaded checkpoint dict.
        use_ema: If True, use EMA weights when available.
        
    Returns:
        Model state dict ready for loading.
        
    Raises:
        ValueError: If unable to extract a model state dict.
    """
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must be a dict.")

    base = checkpoint.get("model_state_dict")
    if not isinstance(base, dict) or not base:
        # Allow raw state dict checkpoints.
        base = checkpoint
    if not isinstance(base, dict) or not base:
        raise ValueError("Could not extract a model state dict from checkpoint.")
    base = strip_module_prefix(base)

    if not use_ema:
        return base

    shadow = extract_ema_shadow(checkpoint)
    if shadow is None:
        return base
    shadow = strip_module_prefix(shadow)

    merged: Dict[str, torch.Tensor] = dict(base)
    for k, v in shadow.items():
        if isinstance(k, str) and torch.is_tensor(v):
            merged[k] = v
    return merged


def adapt_cycle_embedding(
    state: Dict[str, torch.Tensor], model: torch.nn.Module, verbose: bool = True
) -> Dict[str, torch.Tensor]:
    """Resize cycle embedding if checkpoint and model have different sizes.
    
    Args:
        state: Model state dict from checkpoint.
        model: Target model to load into.
        verbose: If True, print a message when resizing.
        
    Returns:
        Updated state dict with resized cycle embedding if needed.
    """
    key = "readout.cycle_embed.weight"
    if key not in state:
        return state
    ckpt_weight = state[key]
    model_weight = model.readout.cycle_embed.weight
    if ckpt_weight.shape == model_weight.shape:
        return state
    new_weight = model_weight.data.clone()
    rows = min(ckpt_weight.shape[0], model_weight.shape[0])
    new_weight[:rows] = ckpt_weight[:rows]
    if verbose:
        print(
            f"[LOAD] Resized cycle embedding from {tuple(ckpt_weight.shape)} to {tuple(model_weight.shape)}",
            flush=True,
        )
    updated = dict(state)
    updated[key] = new_weight
    return updated


def save_checkpoint(
    path: str,
    model_state: dict,
    ema_state: dict,
    optimizer_state: dict,
    meta: dict,
) -> None:
    """Save a training checkpoint.
    
    Args:
        path: Output file path.
        model_state: Model state dict.
        ema_state: EMA state dict.
        optimizer_state: Optimizer state dict.
        meta: Additional metadata dict.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {
            "model_state_dict": model_state,
            "ema_state_dict": ema_state,
            "optimizer_state_dict": optimizer_state,
            **meta,
        },
        path,
    )
