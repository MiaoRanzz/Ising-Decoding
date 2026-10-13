# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import unittest
from unittest.mock import patch

from omegaconf import OmegaConf

from replay.config import resolve_replay_settings


class TestReplayStageTransition(unittest.TestCase):

    def test_existing_buffer_load_is_independent_of_training_resume(self):
        cfg = OmegaConf.create(
            {
                "output": "outputs/test",
                "load_checkpoint": False,
                "replay": {"enabled": False},
            }
        )
        env = {
            "PREDECODER_REPLAY_ENABLED": "1",
            "PREDECODER_REPLAY_TASK_ID": "T1",
            "PREDECODER_REPLAY_LOAD_EXISTING": "1",
        }
        with patch.dict(os.environ, env, clear=False):
            settings = resolve_replay_settings(cfg)
        self.assertTrue(settings.enabled)
        self.assertTrue(settings.load_existing)
        self.assertFalse(bool(cfg.load_checkpoint))


if __name__ == "__main__":
    unittest.main()
