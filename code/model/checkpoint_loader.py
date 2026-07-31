# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load one explicitly identified pre-decoder checkpoint."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def load_model_checkpoint(
    cfg: Any,
    *,
    checkpoint: Path,
    model_id: int,
    distributed: Any,
) -> torch.nn.Module:
    """Load a ``.pt`` or ``.safetensors`` checkpoint for one public model ID."""

    path = Path(checkpoint).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    if path.suffix.lower() != ".safetensors":
        from workflows.run import _load_model

        cfg.model_checkpoint_file = str(path)
        return _load_model(cfg, distributed)

    from export.safetensors_utils import load_safetensors

    model, metadata = load_safetensors(
        str(path),
        model_id=None,
        device=str(distributed.device),
    )
    embedded_model_id = metadata.get("model_id")
    if embedded_model_id is not None and str(embedded_model_id) != str(model_id):
        raise ValueError(
            f"SafeTensors model_id mismatch for {path}: "
            f"CLI requested {model_id}, file metadata contains {embedded_model_id}"
        )
    cfg.enable_fp16 = metadata.get("quant_format") == "fp16"
    cfg.model_checkpoint_file = str(path)
    return model
