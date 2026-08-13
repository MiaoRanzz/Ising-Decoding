"""Unit tests for the structured action representation.

Run with an environment that has the project's PyTorch dependency installed:
``python -m unittest my_file.end_to_end.L_logical.structured_model.test_structured_actions``.
"""
from __future__ import annotations

import unittest

import torch
from torch import nn

try:  # Supports both ``python file.py`` and ``python -m ...``.
    from .structured_actions import (actions_from_logits, from_ising_fast,
                                     packet_mask_from_action_difference,
                                     targets_from_actions)
except ImportError:
    from structured_actions import (actions_from_logits, from_ising_fast,
                                    packet_mask_from_action_difference,
                                    targets_from_actions)


class _Original(nn.Module):
    def __init__(self):
        super().__init__()
        # Match the real Ising-fast construction: the same activation module
        # is registered at more than one Sequential position. ``children()``
        # de-duplicates it, while Sequential iteration must preserve it.
        activation = nn.GELU()
        self.net = nn.Sequential(
            nn.Conv3d(4, 6, 3, padding=1), activation,
            nn.Conv3d(6, 6, 3, padding=1), activation,
            nn.Conv3d(6, 4, 3, padding=1),
        )


class TestStructuredActions(unittest.TestCase):
    def test_warm_start_matches_independent_threshold_actions(self):
        torch.manual_seed(7)
        source = _Original().eval()
        structured = from_ising_fast(source).eval()
        x = torch.randn(3, 4, 4, 5, 5)
        old = source(x) >= 0
        new = actions_from_logits(structured(x))
        self.assertTrue(torch.equal(old, new))

    def test_conversion_preserves_every_sequential_position(self):
        source = _Original().eval()
        structured = from_ising_fast(source).eval()
        self.assertEqual(len(structured.trunk), len(source.net) - 1)

    def test_round_trip_and_packet_difference(self):
        actions = torch.zeros((1, 4, 2, 3, 3), dtype=torch.bool)
        actions[:, 0, 0, 1, 1] = True
        actions[:, 1, 0, 1, 1] = True
        actions[:, 2, 1, 0, 0] = True
        target = targets_from_actions(actions)
        self.assertEqual(int(target[0, 0, 0, 1, 1]), 3)  # Y
        self.assertEqual(int(target[0, 1, 1, 0, 0]), 1)
        changed = actions.clone()
        changed[:, :2, 0, 1, 1] = False
        mask = packet_mask_from_action_difference(actions, changed)
        self.assertTrue(bool(mask[0, 0, 0, 1, 1]))
        self.assertFalse(bool(mask[0, 1, 0, 1, 1]))


if __name__ == "__main__":
    unittest.main()
