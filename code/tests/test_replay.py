# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import tempfile
import unittest
from collections import Counter

import torch
import torch.nn as nn

from replay import ReplayBuffer, combine_current_and_replay_loss
from training.ewc import (
    EWCState,
    add_ewc_penalty_to_loss,
    capture_parameter_snapshot,
)


def _batch(start: int, size: int = 4):
    values = torch.arange(start, start + size, dtype=torch.float32)
    train_x = values.view(size, 1, 1, 1, 1).expand(size, 2, 1, 1, 1).clone()
    train_y = (values.remainder(2)).view(size, 1, 1, 1, 1)
    return train_x, train_y


class TestReplayLoss(unittest.TestCase):

    def test_no_replay_returns_original_sum(self):
        current = torch.tensor(12.0, requires_grad=True)
        combined = combine_current_and_replay_loss(
            current,
            None,
            current_batch_size=4,
            replay_batch_size=0,
            replay_lambda=1.0,
        )
        self.assertIs(combined, current)

    def test_equal_weight_sources_are_normalized(self):
        current = torch.tensor(8.0)
        replay = torch.tensor(12.0)
        combined = combine_current_and_replay_loss(
            current,
            replay,
            current_batch_size=2,
            replay_batch_size=4,
            replay_lambda=1.0,
        )
        # B * ((8/2 + 12/4) / 2) = 21.
        self.assertEqual(combined.item(), 21.0)

    def test_ewc_keeps_batch_independent_effective_scale(self):
        current = torch.tensor(8.0)
        replay = torch.tensor(12.0)
        data_loss = combine_current_and_replay_loss(
            current,
            replay,
            current_batch_size=2,
            replay_batch_size=4,
            replay_lambda=1.0,
        )
        total = add_ewc_penalty_to_loss(
            data_loss, torch.tensor(0.25), ewc_lambda=2.0, batch_size=6
        )
        # train_epoch later divides gradients by six samples; the EWC term is
        # 6 * 2 * .25 here, yielding 2 * .25 after that division.
        self.assertEqual(total.item(), 24.0)

    def test_negative_lambda_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            combine_current_and_replay_loss(
                torch.tensor(1.0),
                torch.tensor(1.0),
                current_batch_size=1,
                replay_batch_size=1,
                replay_lambda=-1.0,
            )


class TestReplayBuffer(unittest.TestCase):

    def test_equal_task_streams_remain_approximately_balanced(self):
        replay = ReplayBuffer(capacity=2000, seed=23)
        for task in range(5):
            train_x, train_y = _batch(task * 10000, 10000)
            replay.observe(
                train_x,
                train_y,
                task_id=f"T{task}",
                basis="X" if task % 2 == 0 else "Z",
                step=task,
            )
        counts = Counter(replay.state_dict()["task_ids"])
        self.assertEqual(sum(counts.values()), 2000)
        # Five equal-sized streams should each occupy roughly 20%, rather than
        # the exponential recency distribution caused by re-keying old rows.
        for task in range(5):
            self.assertGreaterEqual(counts[f"T{task}"], 320)
            self.assertLessEqual(counts[f"T{task}"], 480)

    def test_current_task_is_admitted_but_not_retrieved(self):
        replay = ReplayBuffer(capacity=8, seed=7)
        train_x, train_y = _batch(0)
        replay.observe(train_x, train_y, task_id="T0", basis="X", step=0)
        self.assertEqual(len(replay), 4)
        self.assertEqual(replay.eligible_count("T0"), 0)
        self.assertIsNone(replay.sample(2, current_task_id="T0", device="cpu"))
        sampled = replay.sample(2, current_task_id="T1", device="cpu")
        self.assertIsNotNone(sampled)
        self.assertEqual(len(sampled), 2)
        self.assertEqual(set(sampled.task_ids), {"T0"})

    def test_capacity_and_shape_validation(self):
        replay = ReplayBuffer(capacity=3, seed=11)
        train_x, train_y = _batch(0, 4)
        replay.observe(train_x, train_y, task_id="T0", basis="X", step=0)
        replay.observe(train_x, train_y, task_id="T1", basis="Z", step=1)
        self.assertEqual(len(replay), 3)
        self.assertEqual(replay.seen_count, 8)
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            replay.observe(
                torch.zeros(1, 3, 1, 1, 1),
                torch.zeros(1, 1, 1, 1, 1),
                task_id="T2",
                basis="X",
                step=2,
            )

    def test_save_load_restores_rng_and_next_actions(self):
        replay = ReplayBuffer(capacity=6, seed=19)
        train_x, train_y = _batch(0, 6)
        replay.observe(train_x, train_y, task_id="T0", basis="X", step=0)
        # Advance retrieval RNG before saving.
        replay.sample(2, current_task_id="T1", device="cpu")

        with tempfile.TemporaryDirectory() as tmpdir:
            replay.save(tmpdir, epoch=3, global_step=17)
            restored = ReplayBuffer(capacity=6, seed=19)
            self.assertTrue(restored.load(tmpdir))
            self.assertEqual(restored.snapshot_epoch, 3)
            self.assertEqual(restored.snapshot_global_step, 17)
            self.assertEqual(restored.seen_count, replay.seen_count)

            expected = replay.sample(3, current_task_id="T1", device="cpu")
            actual = restored.sample(3, current_task_id="T1", device="cpu")
            self.assertEqual(actual.sample_ids, expected.sample_ids)
            torch.testing.assert_close(actual.train_x, expected.train_x)
            torch.testing.assert_close(actual.train_y, expected.train_y)

            next_x, next_y = _batch(20, 4)
            expected_admitted = replay.observe(
                next_x, next_y, task_id="T1", basis="Z", step=1
            )
            actual_admitted = restored.observe(
                next_x, next_y, task_id="T1", basis="Z", step=1
            )
            self.assertEqual(actual_admitted, expected_admitted)
            self.assertEqual(restored.state_dict()["sample_ids"], replay.state_dict()["sample_ids"])

    def test_world_size_mismatch_is_rejected(self):
        replay = ReplayBuffer(capacity=4, seed=1, rank=0, world_size=1)
        train_x, train_y = _batch(0)
        replay.observe(train_x, train_y, task_id="T0", basis="X", step=0)
        state = replay.state_dict()
        other = ReplayBuffer(capacity=4, seed=1, rank=0, world_size=2)
        with self.assertRaisesRegex(ValueError, "world_size mismatch"):
            other.load_state_dict(state)


class _TrainGenerator:

    def generate_batch(self, step, batch_size, **_kwargs):
        x = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, -0.5]],
            dtype=torch.float32,
        )[:batch_size]
        y = torch.ones(batch_size, 1, dtype=torch.float32)
        return x, y

    def get_current_basis(self, _step):
        return "X"


class _Writer:

    def add_scalar(self, *_args, **_kwargs):
        return None


class TestReplayTrainEpochIntegration(unittest.TestCase):

    def test_replay_and_ewc_train_together(self):
        from training.train import train_epoch

        replay = ReplayBuffer(capacity=8, seed=5)
        history_x = torch.tensor(
            [[-1.0, 0.0], [0.0, -1.0], [-1.0, -1.0], [-0.5, 0.5]]
        )
        history_y = torch.zeros(4, 1)
        replay.observe(history_x, history_y, task_id="T0", basis="X", step=0)

        model = nn.Linear(2, 1)
        mean = capture_parameter_snapshot(model)
        state = EWCState(
            task_name="T0",
            mean=mean,
            fisher={name: torch.ones_like(value) for name, value in mean.items()},
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        loss, global_step = train_epoch(
            _TrainGenerator(), steps_per_epoch=1, batch_size=4,
            cumulative_steps_before_epoch=0, epoch_number=0, model=model,
            optimizer=optimizer, scaler=scaler, scheduler=scheduler,
            tb_writer=_Writer(), device=torch.device("cpu"), enable_fp16=False,
            ewc_states=[state], ewc_lambda=2.0, replay_buffer=replay,
            replay_task_id="T1", replay_ratio=0.5, replay_lambda=1.0,
        )
        self.assertTrue(torch.isfinite(torch.tensor(loss)))
        self.assertEqual(global_step, 1)
        self.assertEqual(replay.seen_count, 6)
        self.assertEqual(replay.eligible_count("T1"), 4)


if __name__ == "__main__":
    unittest.main()
