# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

from scripts.experiments.qewc_v3.extract_validation_bce import continual_row, mix_rows


class TestQewcV3ValidationBce(unittest.TestCase):

    def test_continual_row_labels_only_current_task(self):
        row = continual_row("[date] LOSS train 0.2 valid 0.3", "C2", "T3")
        self.assertEqual(row["validation_bce_T3"], 0.3)
        self.assertIsNone(row["validation_bce_T1"])

    def test_mix_rows_keep_all_task_validation_losses(self):
        chunks = []
        for epoch in range(4):
            chunks.append(f"LOSS train 0.{epoch + 1} valid 0.5")
            chunks.extend(f"[Mix Validation] T{task} loss=0.{task + 1}" for task in range(5))
        rows = mix_rows("\n".join(chunks))
        self.assertEqual([row["checkpoint"] for row in rows], ["M1", "M2", "M3", "M4"])
        self.assertEqual(rows[-1]["validation_bce_T4"], 0.5)


if __name__ == "__main__":
    unittest.main()
