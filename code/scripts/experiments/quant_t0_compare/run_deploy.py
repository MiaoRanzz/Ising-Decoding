#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deployment entry point with SafeTensors output-directory preparation."""

from __future__ import annotations

import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[3]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from scripts.experiments.quant_t0_compare import run as workflow  # noqa: E402


_ensure_reference_checkpoint = workflow.ensure_reference_checkpoint


def ensure_reference_checkpoint(job, output_dir, *, dry_run):
    if job.reference_checkpoint is None and not dry_run:
        output = (
            output_dir
            / "artifacts"
            / job.model
            / f"e{job.epoch}"
            / "model_fp32.safetensors"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
    return _ensure_reference_checkpoint(job, output_dir, dry_run=dry_run)


workflow.ensure_reference_checkpoint = ensure_reference_checkpoint


if __name__ == "__main__":
    workflow.main()
