# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest
from scripts.experiments.qewc_v4.recover_budget_e50 import merge_history, LOSS


class RecoveryHistoryTest(unittest.TestCase):
    def test_unsaved_epoch_is_replaced_not_double_counted(self):
        old = 'LOSS train 0.1 valid 0.2\n' * 50
        new = 'LOSS train 0.3 valid 0.4\n'
        rows = [{'epoch': i, 'optimizer_steps_global': (i + 1) * 512} for i in range(50)]
        log, result = merge_history(old, new, rows, rows[49:], 49)
        self.assertEqual(len(LOSS.findall(log)), 50)
        self.assertEqual(LOSS.findall(log)[-1], ('0.3', '0.4'))
        self.assertEqual(result['epochs'], rows)

    def test_incomplete_recovery_rejected(self):
        rows = [{'epoch': i, 'optimizer_steps_global': (i + 1) * 512} for i in range(50)]
        with self.assertRaises(ValueError):
            merge_history('LOSS train 0.1 valid 0.2\n' * 46, '', rows[:46], [], 46)


if __name__ == '__main__':
    unittest.main()
