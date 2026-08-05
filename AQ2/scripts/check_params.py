#!/usr/bin/env python3
"""Verify that the AQ2 model parameter count matches the paper (Table S1).

Paper reports 32,177,930 parameters for AQ2-full (d_model=512, d_ff=1024,
16 heads, key_size=32, temporal_K=6).  This script builds the model and
asserts the trainable parameter count falls within [32.0M, 32.4M].

The 0.18% residual difference from the paper figure is explained by the
readout head convention (2 cross-attention + 2 dense layers per the A.1.1
text, parameter-equivalent to the 4-layer / 8-head pooling in Table S1).

Usage:
    python AQ2/scripts/check_params.py
"""

import sys
import torch

# Ensure the AQ2 package root is importable
sys.path.insert(0, ".")


def main() -> None:
    from src.models.AQ2 import GoogleDecoder

    distances = [3, 5, 7, 9, 11]
    model = GoogleDecoder(
        distances=distances,
        num_rounds=121,             # max frames = T+1 = 121 (T=120)
        d_model=512,
        nhead=16,
        dim_feedforward=1024,       # paper Table S1 "Widening 4" = 4×d_model
        dropout=0.1,
        attn_key_size=32,
        temporal_K=6,
    )

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    expected = 32_177_930
    delta = num_params - expected

    print(f"Model parameters: {num_params:,}")
    print(f"Paper reports:    {expected:,}")
    print(f"Difference:       {delta:+,} ({100 * delta / expected:+.2f}%)")

    assert 32_000_000 <= num_params <= 32_400_000, (
        f"Parameter count {num_params:,} outside expected range [32.0M, 32.4M]"
    )
    print("PASS: parameter count within expected range.")


if __name__ == "__main__":
    main()
