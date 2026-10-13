# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Retain two complete checkpoint pairs within explicitly supplied run roots."""
import json
import re
from pathlib import Path


def prune(root: Path, audit: Path, protected_epochs=()):
    import torch
    removed = 0
    if not root.exists():
        return removed
    for folder in [p for p in root.rglob('*') if p.is_dir() and p.name in ('models', 'best_model')]:
        states = {}
        for path in folder.glob('checkpoint.0.*.pt'):
            match = re.fullmatch(r'checkpoint\.0\.(\d+)\.pt', path.name)
            if match:
                states[int(match[1])] = path
        pairs = {}
        for epoch, state in states.items():
            weights = [p for p in folder.glob(f'*.0.{epoch}.pt')
                       if p.name.startswith(('HTnet.', 'PreDecoderModelMemory_v1.'))]
            if len(weights) == 1:
                pairs[epoch] = (weights[0], state)
        if len(pairs) <= 2:
            continue
        newest = sorted(pairs)[-2:]
        try:
            for epoch in newest:
                model, state = pairs[epoch]
                weights = torch.load(model, map_location='cpu', weights_only=True)
                saved = torch.load(state, map_location='cpu', weights_only=False)
                if saved.get('epoch') != epoch or not weights:
                    raise ValueError('Incomplete checkpoint')
        except (OSError, RuntimeError, EOFError, ValueError):
            # Saving can be in progress; retain everything until a later poll.
            continue
        for epoch in sorted(pairs):
            if epoch >= newest[0] or epoch in protected_epochs:
                continue
            for path in pairs[epoch]:
                if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
                    raise RuntimeError(f'Unsafe cleanup target: {path}')
                with audit.open('a') as stream:
                    stream.write(json.dumps({'path': str(path), 'bytes': path.stat().st_size,
                                             'retained_epochs': newest}) + '\n')
                path.unlink()
                removed += 1
    return removed
