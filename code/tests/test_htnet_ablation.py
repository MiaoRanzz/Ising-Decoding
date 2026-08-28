# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the three-factor HTNet architecture ablation."""

import itertools
import sys
import unittest
from pathlib import Path

import torch
from omegaconf import OmegaConf

_repo_code = Path(__file__).resolve().parent.parent
if str(_repo_code) not in sys.path:
    sys.path.insert(0, str(_repo_code))

from model.factory import ModelFactory
from model.predecoder import AxisChannelGate3D, ChannelGate3D
from workflows.config_validator import apply_public_defaults_and_model, validate_public_config


def _config(branch_mixer=True, axis_channel_gate=True, input_skip_head=True):
    public_cfg = OmegaConf.create(
        {
            "model_id": 111,
            "distance": 9,
            "n_rounds": 9,
            "htnet_ablation": {
                "branch_mixer": branch_mixer,
                "axis_channel_gate": axis_channel_gate,
                "input_skip_head": input_skip_head,
            },
        }
    )
    spec = validate_public_config(public_cfg)
    cfg = apply_public_defaults_and_model(public_cfg, spec)
    cfg.model.channels = 8
    cfg.model.expand_channels = 12
    cfg.model.num_blocks = 1
    cfg.model.joint_groups = 3
    cfg.model.norm_groups = 4
    return cfg


class TestHTNetAblation(unittest.TestCase):

    def test_all_eight_combinations_preserve_output_shape(self):
        x = torch.randn(2, 4, 3, 3, 3)
        for mixer, axis_gate, input_skip in itertools.product((False, True), repeat=3):
            with self.subTest(mixer=mixer, axis_gate=axis_gate, input_skip=input_skip):
                model = ModelFactory.create_model(_config(mixer, axis_gate, input_skip))
                self.assertEqual(tuple(model(x).shape), (2, 4, 3, 3, 3))

    def test_switches_select_expected_modules_and_head_width(self):
        enabled = ModelFactory.create_model(_config())
        disabled = ModelFactory.create_model(_config(False, False, False))

        self.assertNotIsInstance(enabled.blocks[0].branch_mixer, torch.nn.Identity)
        self.assertIsInstance(disabled.blocks[0].branch_mixer, torch.nn.Identity)
        self.assertIsInstance(enabled.blocks[0].gate, AxisChannelGate3D)
        self.assertIsInstance(disabled.blocks[0].gate, ChannelGate3D)
        self.assertEqual(enabled.head_hidden.in_channels, 12)
        self.assertEqual(disabled.head_hidden.in_channels, 8)

    def test_default_configuration_matches_all_enabled(self):
        cfg = _config()
        cfg.pop("htnet_ablation")
        model = ModelFactory.create_model(cfg)
        self.assertNotIsInstance(model.blocks[0].branch_mixer, torch.nn.Identity)
        self.assertIsInstance(model.blocks[0].gate, AxisChannelGate3D)
        self.assertTrue(model.use_input_skip_head)

    def test_ablation_rejected_for_non_htnet_model(self):
        cfg = OmegaConf.create(
            {
                "model_id": 1,
                "distance": 9,
                "n_rounds": 9,
                "htnet_ablation": {"branch_mixer": False},
            }
        )
        with self.assertRaisesRegex(ValueError, "requires model_id=111"):
            validate_public_config(cfg)


if __name__ == "__main__":
    unittest.main()
