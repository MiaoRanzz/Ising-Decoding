#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reproduce the paper's d=5/d=7, ten-round Willow evaluation."""

from __future__ import annotations

import sys

from infer_google_benchmark import main as run_google_benchmark


PAPER_WILLOW_DEFAULT_ARGS = (
    "--distances",
    "5",
    "7",
    "--rounds",
    "10",
)


def main(argv: list[str] | None = None) -> int:
    user_args = list(sys.argv[1:] if argv is None else argv)
    return run_google_benchmark([*PAPER_WILLOW_DEFAULT_ARGS, *user_args])


if __name__ == "__main__":
    raise SystemExit(main())
