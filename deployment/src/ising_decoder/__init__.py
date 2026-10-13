# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standalone surface-code pre-decoder package."""

from .config import DecoderConfig
from .decoder import DecodeResult, StageLatencies, SurfaceDecoderCore

__all__ = [
    "DecodeResult",
    "DecoderConfig",
    "StageLatencies",
    "SurfaceDecoderCore",
]
