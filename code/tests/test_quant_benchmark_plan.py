# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest
from pathlib import Path

from scripts.experiments.quant_t0_compare.run import expand_jobs, load_plan


class TestQuantBenchmarkPlan(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.plan_path = (
            Path(__file__).resolve().parents[1]
            / "scripts/experiments/quant_t0_compare/experiment.yaml"
        )
        cls.plan = load_plan(cls.plan_path)

    def test_smoke_matrix_has_one_int8_calibration(self):
        jobs = expand_jobs(self.plan, smoke=True)
        self.assertEqual(len(jobs), 12)
        self.assertEqual(sum(job.precision == "int8" for job in jobs), 4)

    def test_screen_matrix_has_three_int8_calibrations(self):
        jobs = expand_jobs(self.plan, smoke=False)
        self.assertEqual(len(jobs), 20)
        self.assertEqual(sum(job.precision == "int8" for job in jobs), 12)

    def test_model_filter_and_paths(self):
        jobs = expand_jobs(self.plan, smoke=True, only_model="htnet")
        self.assertEqual({job.model for job in jobs}, {"htnet"})
        self.assertTrue(all(job.checkpoint.is_absolute() for job in jobs))


if __name__ == "__main__":
    unittest.main()
