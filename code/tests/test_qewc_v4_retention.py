# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import tempfile
from pathlib import Path
import unittest
import torch
from scripts.experiments.qewc_v4.retain_checkpoints import prune
from scripts.experiments.qewc_v4.run_b30_pipeline import task_command


class RetentionTest(unittest.TestCase):
    def test_protected_mix_evaluation_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            models = root / 'models'
            models.mkdir()
            for epoch in (29, 30, 31, 32):
                torch.save({'w': torch.ones(1)}, models / f'HTnet.0.{epoch}.pt')
                torch.save({'epoch': epoch}, models / f'checkpoint.0.{epoch}.pt')
            self.assertEqual(prune(root, root / 'audit.jsonl', protected_epochs=(30,)), 2)
            self.assertTrue((models / 'HTnet.0.30.pt').exists())
            self.assertTrue((models / 'checkpoint.0.30.pt').exists())
            self.assertFalse((models / 'HTnet.0.29.pt').exists())

    def test_keep_two_complete_pairs_and_pending_save(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            models = root / 'stage_1_T1/models'
            models.mkdir(parents=True)
            for epoch in (1, 2, 3):
                torch.save({'w': torch.ones(1)}, models / f'HTnet.0.{epoch}.pt')
                torch.save({'epoch': epoch}, models / f'checkpoint.0.{epoch}.pt')
            pending = models / 'HTnet.0.4.pt'
            pending.write_bytes(b'partial')
            fisher = root / 'fisher.pt'
            fisher.write_bytes(b'untouched')
            self.assertEqual(prune(root, root / 'audit.jsonl'), 2)
            self.assertFalse((models / 'HTnet.0.1.pt').exists())
            self.assertTrue((models / 'checkpoint.0.2.pt').exists())
            self.assertTrue(pending.exists())
            self.assertTrue(fisher.exists())

    def test_budget30_command(self):
        command = task_command('htnet_a101', 'T1', 'test', 30, 4194304, 0)
        self.assertEqual(command[command.index('--epochs-per-task') + 1], '30')
        self.assertEqual(command[command.index('--accumulate-steps') + 1], '2')


if __name__ == '__main__':
    unittest.main()
