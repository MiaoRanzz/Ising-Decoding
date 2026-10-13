#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Consolidate a newly estimated Fisher into one Online EWC state."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from training.ewc import load_ewc_state, save_ewc_state  # noqa: E402
from training.online_ewc import update_online_ewc_state  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", required=True)
    parser.add_argument("--current", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gamma", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    previous = load_ewc_state(args.previous)
    current = load_ewc_state(args.current)
    consolidated = update_online_ewc_state(previous, current, gamma=args.gamma)
    save_ewc_state(consolidated, args.output)
    print(
        f"[Online EWC] previous={args.previous} current={args.current} "
        f"gamma={args.gamma:g} output={args.output}"
    )


if __name__ == "__main__":
    main()
