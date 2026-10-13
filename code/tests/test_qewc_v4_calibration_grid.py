# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from scripts.experiments.qewc_v4.run_calibration_grid import value_name, run_jobs


class TestQewcV4CalibrationGrid(unittest.TestCase):

    def test_parallel_jobs_preserve_first_gpu_and_execute_once(self):
        jobs = [('ewc', 'lambda_0p1', 0.1), ('ewc', 'lambda_1', 1), ('replay', 'ratio_0p1', 0.1)]
        seen = []
        with patch('scripts.experiments.qewc_v4.run_calibration_grid.run_candidate',
                   side_effect=lambda args, *job: seen.append((args.gpu, job))):
            run_jobs(SimpleNamespace(gpus=['0', '1'], gpu='0'), jobs)
        self.assertIn(('0', jobs[0]), seen)
        self.assertIn(('1', jobs[1]), seen)
        self.assertCountEqual([job for _, job in seen], jobs)

    def test_empty_refinement_jobs(self):
        with patch('scripts.experiments.qewc_v4.run_calibration_grid.run_candidate') as run:
            run_jobs(SimpleNamespace(gpus=['0', '1']), [])
        run.assert_not_called()
    def test_candidate_names_are_path_safe_and_stable(self):
        self.assertEqual(value_name("lambda", 0.1), "lambda_0p1")
        self.assertEqual(value_name("lambda", 1000), "lambda_1000")
        self.assertEqual(value_name("lambda", 10 ** 0.5), "lambda_3p16228")


if __name__ == "__main__":
    unittest.main()
