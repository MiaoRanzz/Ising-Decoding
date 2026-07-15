# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


CODE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_ROOT.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from evaluation.lqcloud_inference import (
    build_model_detector_permutation,
    collect_hardware_measurement_memory,
    load_lqcloud_hardware_samples,
    normalize_measurement_memory,
    parse_measurement_log,
)


HAS_QEC_DEPS = all(
    importlib.util.find_spec(name) is not None
    for name in ("numpy", "stim", "pymatching")
)


class TestLQCloudMeasurementParser(unittest.TestCase):

    def _write_log(self, text: str) -> Path:
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False, encoding="utf-8")
        self.addCleanup(Path(tmp.name).unlink, missing_ok=True)
        with tmp:
            tmp.write(text)
        return Path(tmp.name)

    def test_parses_direct_memory_list(self):
        path = self._write_log("status line\n['0101', '1110']\n")
        self.assertEqual(
            parse_measurement_log(path, expected_width=4),
            [[0, 1, 0, 1], [1, 1, 1, 0]],
        )

    def test_normalizes_result_get_memory_output(self):
        self.assertEqual(
            normalize_measurement_memory(
                ["0101", "1110"],
                expected_width=4,
            ),
            [[0, 1, 0, 1], [1, 1, 1, 0]],
        )

    def test_hardware_collection_consumes_result_get_memory(self):
        calls = {}

        class FakeResult:

            def get_memory(self):
                calls["get_memory"] = calls.get("get_memory", 0) + 1
                return ["0101", "1110"]

        def fake_runner(**kwargs):
            calls["runner_kwargs"] = kwargs
            return FakeResult()

        cfg = SimpleNamespace(
            n_rounds=9,
            lqcloud=SimpleNamespace(
                shots=100,
                backend_name="QZ01-surface_code",
                circuit_type="memory_z",
            ),
        )
        memory = collect_hardware_measurement_memory(
            cfg,
            initial_state=[0] * 9,
            hardware_runner=fake_runner,
        )

        self.assertEqual(memory, ["0101", "1110"])
        self.assertEqual(calls["get_memory"], 1)
        self.assertEqual(calls["runner_kwargs"]["cycle"], 9)
        self.assertEqual(calls["runner_kwargs"]["shots"], 100)
        self.assertEqual(calls["runner_kwargs"]["backend_name"], "QZ01-surface_code")

    def test_parses_memory_from_result_dict_and_can_reverse(self):
        path = self._write_log("{'success': True, 'memory': ['0101', '1110'], 'shots': 2}\n")
        self.assertEqual(
            parse_measurement_log(path, expected_width=4, bit_order="reverse", max_shots=1),
            [[1, 0, 1, 0]],
        )

    def test_rejects_wrong_shot_width(self):
        path = self._write_log("['010', '111']\n")
        with self.assertRaisesRegex(ValueError, "expected 4"):
            parse_measurement_log(path, expected_width=4)

    def test_repository_hardware_log_contains_100_d3_round9_shots(self):
        path = REPO_ROOT / "my_file/lqcloud/lqcloud_d3_surface_code/measurement.log"
        shots = parse_measurement_log(path, expected_width=81)
        self.assertEqual(len(shots), 100)
        self.assertTrue(all(len(shot) == 81 for shot in shots))


class TestLQCloudDetectorPermutation(unittest.TestCase):

    class _LQCircuits:

        @staticmethod
        def generate_qubit_coords(distance, mirror=True):
            return {}

        @staticmethod
        def generate_z_x_measure_qubits(distance, mirror=True):
            return [10, 11, 14, 16], [9, 12, 13, 15]

        @staticmethod
        def generate_cz_pattern_and_stabilizer_qubits(**kwargs):
            supports = {
                9: [0, 1, 3, 4],
                10: [1, 2, 4, 5],
                11: [3, 4, 6, 7],
                12: [4, 5, 7, 8],
                13: [1, 2],
                14: [5, 8],
                15: [6, 7],
                16: [0, 3],
            }
            return [], supports

    class _XHSurfaceCode:
        hx = [
            [0, 1, 1, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 1, 1],
            [0, 0, 0, 0, 0, 0, 1, 1, 0],
        ]
        hz = [
            [1, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 1, 1, 0],
            [0, 1, 1, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 1],
        ]

    def test_d3_xh_permutation_is_complete_and_reorders_each_bulk_round(self):
        permutation = build_model_detector_permutation(
            distance=3,
            n_rounds=2,
            basis="Z",
            code_rotation="XH",
            lq_circuits=self._LQCircuits,
            surface_code=self._XHSurfaceCode,
        )
        self.assertEqual(
            permutation,
            [3, 1, 0, 2, 8, 4, 7, 10, 11, 6, 5, 9, 15, 13, 12, 14],
        )
        self.assertEqual(sorted(permutation), list(range(16)))


@unittest.skipUnless(HAS_QEC_DEPS, "requires numpy, stim, and pymatching")
class TestLQCloudHardwareRegression(unittest.TestCase):

    def test_checked_in_hardware_log_reproduces_existing_ler(self):
        import numpy as np
        import pymatching

        cfg = SimpleNamespace(
            distance=3,
            n_rounds=9,
            data=SimpleNamespace(code_rotation="XH"),
            lqcloud=SimpleNamespace(
                source="log",
                measurement_file=(
                    REPO_ROOT / "my_file/lqcloud/lqcloud_d3_surface_code/measurement.log"
                ),
                circuit_type="memory_z",
                initial_state=[0] * 9,
                reset=False,
                mirror=True,
                bit_order="as_returned",
                max_shots=0,
            ),
        )
        samples = load_lqcloud_hardware_samples(cfg)
        self.assertEqual(samples.measurements.shape, (100, 81))
        self.assertEqual(samples.lq_dets.shape, (100, 72))
        self.assertEqual(samples.model_dets.shape, (100, 72))
        self.assertEqual(samples.observables.shape, (100, 1))
        self.assertAlmostEqual(float(samples.observables.mean()), 0.28)

        matcher = pymatching.Matching.from_detector_error_model(
            samples.circuit.detector_error_model(
                decompose_errors=True,
                approximate_disjoint_errors=True,
            )
        )
        predictions = np.asarray(matcher.decode_batch(samples.lq_dets)).reshape(-1, 1)
        self.assertEqual(int(np.any(predictions != samples.observables, axis=1).sum()), 21)


if __name__ == "__main__":
    unittest.main()
