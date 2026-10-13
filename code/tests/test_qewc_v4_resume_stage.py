# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import unittest
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from scripts.experiments.qewc_v4.resume_stage import combine, LOSS, replay_resume_stage, validate_replay, prepare, finish


class ResumeStageTest(unittest.TestCase):
    def test_mix_resume_preserves_logs_and_archives_early_stop(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / 'outputs/mix'
            models = output / 'models'
            models.mkdir(parents=True)
            (models / 'HTnet.0.2.pt').touch()
            (output / 'early_stopping.json').write_text('{}')
            old = root / 'logs/mix_20261001'
            old.mkdir(parents=True)
            (old / 'train.log').write_text('LOSS train 0.1 valid 0.2\n' * 3)
            rows = [{'epoch': i, 'optimizer_steps_global': (i + 1) * 512} for i in range(3)]
            (output / 'resource_metrics.json').write_text(json.dumps({'epochs': rows[:2]}))
            args = SimpleNamespace(run_name='mix', checkpoint_prefix='HTnet',
                                   epochs_per_task=3, steps_per_task=1536, method='mix_exposure')
            task = SimpleNamespace(name='mix_exposure')
            env = {'EXTRA_PARAMS': '++init_model_checkpoint_file=stage0.pt'}
            with patch('scripts.experiments.qewc_v4.resume_stage.validate_pair'):
                context = prepare(args, task, None, env, models / 'HTnet.0.3.pt', root)
                self.assertEqual(env['FRESH_START'], '0')
                self.assertEqual(env['EXTRA_PARAMS'], '++early_stopping.enabled=False')
                self.assertFalse((output / 'early_stopping.json').exists())
                self.assertTrue((context[0] / 'original_early_stopping.json').exists())
                new = root / 'logs/mix_99999999'
                new.mkdir()
                (new / 'train.log').write_text('LOSS train 0.3 valid 0.4\n')
                (output / 'resource_metrics.json').write_text(json.dumps({'epochs': rows[2:]}))
                finish(context, args, task, None, models / 'HTnet.0.3.pt')
            self.assertEqual(json.loads((output / 'resource_metrics.json').read_text())['epochs'], rows)
            self.assertEqual(json.loads((context[0] / 'recovery.json').read_text())['status'], 'completed')

    def test_partial_replay_preserves_snapshot_until_current_stage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = SimpleNamespace(method='replay', resume=True, dry_run=False,
                                   run_root=root, order_names=['T1', 'T3'],
                                   checkpoint_prefix='HTnet', epochs_per_task=30)
            for stage, task, epoch in ((1, 'T1', 30), (2, 'T3', 7)):
                folder = root / f'stage_{stage}_{task}' / 'models'
                folder.mkdir(parents=True)
                (folder / f'HTnet.0.{epoch}.pt').touch()
            self.assertEqual(replay_resume_stage(args), 2)
            args.resume = False
            self.assertEqual(replay_resume_stage(args), 0)

    def test_replay_requires_matching_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'balanced_manifest.json').write_text(json.dumps({'tasks': ['T0', 'T1']}))
            args = SimpleNamespace(replay_dir=root, replay_capacity=2,
                                   order_names=['T1', 'T3'], base_task=SimpleNamespace(name='T0'))
            snapshot = dict(task_id='T3', epoch=7, global_step=3584,
                            world_size=1, buffer_size_per_rank=2)
            buffer = dict(snapshot_epoch=7, snapshot_global_step=3584, world_size=1,
                          rank=0, capacity=2, size=2, train_x=torch.ones(2, 1),
                          train_y=torch.ones(2, 1), task_ids=['T0', 'T1'],
                          admission_rng_state=(), retrieval_rng_state=())
            for epoch, valid in ((7, True), (6, False)):
                buffer['snapshot_epoch'] = epoch
                with patch('torch.load', side_effect=[{'metadata': {'replay_snapshot': snapshot}}, buffer]):
                    if valid:
                        validate_replay(args, SimpleNamespace(name='T3'), root / 'HTnet.0.7.pt', 7, 512)
                    else:
                        with self.assertRaises(AssertionError):
                            validate_replay(args, SimpleNamespace(name='T3'), root / 'HTnet.0.7.pt', 7, 512)

    def test_valid_b30_resume(self):
        rows = [{'epoch': i, 'optimizer_steps_global': (i + 1) * 512} for i in range(30)]
        text, merged = combine('LOSS train 0.1 valid 0.2\n' * 14,
                               'LOSS train 0.3 valid 0.4\n' * 17,
                               rows[:14], rows[13:], 13, 30, 512)
        self.assertEqual(merged['epochs'], rows)
        self.assertEqual(len(LOSS.findall(text)), 30)

    def test_missing_epoch_rejected(self):
        with self.assertRaises(AssertionError):
            combine('LOSS train 0.1 valid 0.2\n', '', [], [], 1, 30, 512)


if __name__ == '__main__':
    unittest.main()
