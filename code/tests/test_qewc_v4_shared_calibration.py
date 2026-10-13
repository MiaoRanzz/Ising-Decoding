# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from scripts.experiments.qewc_v4 import run_shared_calibration as shared


class SharedCalibrationTest(unittest.TestCase):
    def test_formal_passes_frozen_manifest_to_validation_and_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(shared, 'ROOT', root), \
                 patch.object(shared, 'run_ids', return_value=('H_T0',)), \
                 patch.object(shared.pipeline, 'execute') as execute:
                shared.formal('htnet_a101', 0)
            self.assertEqual(execute.call_count, 2)
            for call in execute.call_args_list:
                arguments = call.args[1]
                self.assertEqual(arguments[arguments.index('--manifest') + 1],
                                 str(root / 'outputs/qewc_v4/frozen_experiment_manifest.yaml'))

    def test_missing_summary_blocks_continuation(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(shared, 'CAL', Path(directory)):
                with self.assertRaisesRegex(RuntimeError, 'without summary'):
                    shared.wait_for_calibration(999999999)

    def test_transfer_parameters_but_not_fisher_or_capacity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cal = root / 'calibration'
            (cal / 'htnet_a101').mkdir(parents=True)
            source = {'architecture': 'htnet_a101', 'budget_B': 30, 'selected_lambda': 10,
                      'fisher_samples': 4096, 'selected_replay_ratio': 0.25,
                      'initial_fisher_path': 'htnet.pt', 'replay_capacity_samples': 123}
            (cal / 'htnet_a101/calibration_summary.json').write_text(json.dumps(source))
            fisher = root / 'ising.pt'
            fisher.write_bytes(b'architecture-specific state')
            capacity = {'capacity_samples': 456, 'target_bytes': 27, 'byte_tolerance': 0.05}
            with patch.object(shared, 'ROOT', root), patch.object(shared, 'CAL', cal):
                result = shared.transferred_summary(source, fisher, capacity)
            self.assertEqual(result['selected_lambda'], 10)
            self.assertEqual(result['selected_replay_ratio'], 0.25)
            self.assertEqual(result['fisher_samples'], 4096)
            self.assertEqual(result['initial_fisher_path'], 'ising.pt')
            self.assertEqual(result['replay_capacity_samples'], 456)
            self.assertFalse(result['independent_parameter_search'])
            self.assertEqual(result['parameter_source_architecture'], 'htnet_a101')


if __name__ == '__main__':
    unittest.main()
