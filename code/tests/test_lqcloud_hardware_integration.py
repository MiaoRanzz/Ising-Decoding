# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
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
    _time_lqcloud_pymatching_latency,
    build_model_detector_permutation,
    collect_hardware_measurement_memory,
    load_measurement_file,
    load_lqcloud_hardware_samples,
    normalize_measurement_memory,
)


HAS_QEC_DEPS = all(
    importlib.util.find_spec(name) is not None
    for name in ("numpy", "stim", "pymatching")
)


class TestLQCloudMeasurementFile(unittest.TestCase):

    def test_latency_uses_configured_subset_and_original_timer_contract(self):
        calls = {}

        def fake_timer(**kwargs):
            calls.update(kwargs)
            return 12.5, 4.25

        baseline_us, predecoder_us, sample_count = _time_lqcloud_pymatching_latency(
            matcher="matcher",
            baseline_syndromes=list(range(5)),
            residual_syndromes=list(range(4)),
            n_rounds=9,
            num_samples=3,
            warmup_iterations=7,
            timer=fake_timer,
        )

        self.assertEqual((baseline_us, predecoder_us, sample_count), (12.5, 4.25, 3))
        self.assertEqual(calls["matcher"], "matcher")
        self.assertEqual(calls["baseline_syndromes"], [0, 1, 2])
        self.assertEqual(calls["residual_syndromes"], [0, 1, 2])
        self.assertEqual(calls["n_rounds"], 9)
        self.assertEqual(calls["warmup_iterations"], 7)

    def test_latency_can_be_disabled(self):
        baseline_us, predecoder_us, sample_count = _time_lqcloud_pymatching_latency(
            matcher=None,
            baseline_syndromes=[0],
            residual_syndromes=[0],
            n_rounds=9,
            num_samples=0,
            warmup_iterations=50,
            timer=lambda **_: self.fail("disabled latency must not call the timer"),
        )

        self.assertTrue(baseline_us != baseline_us)
        self.assertTrue(predecoder_us != predecoder_us)
        self.assertEqual(sample_count, 0)

    def _write_json(self, value) -> Path:
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8")
        self.addCleanup(Path(tmp.name).unlink, missing_ok=True)
        with tmp:
            json.dump(value, tmp)
        return Path(tmp.name)

    def test_loads_get_memory_json_list(self):
        path = self._write_json(["0101", "1110"])
        self.assertEqual(
            load_measurement_file(path, expected_width=4),
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
            distance=3,
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

    def test_can_reverse_and_limit_loaded_shots(self):
        path = self._write_json(["0101", "1110"])
        self.assertEqual(
            load_measurement_file(path, expected_width=4, bit_order="reverse", max_shots=1),
            [[1, 0, 1, 0]],
        )

    def test_rejects_wrong_shot_width(self):
        path = self._write_json(["010", "111"])
        with self.assertRaisesRegex(ValueError, "expected 4"):
            load_measurement_file(path, expected_width=4)

    def test_repository_measurement_file_contains_d3_round9_shots(self):
        path = REPO_ROOT / "lqcloud_measurements/measurement_5.json"
        shots = load_measurement_file(path, expected_width=81)
        self.assertEqual(len(shots), 10000)
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
                source="file",
                measurement_file=(
                    REPO_ROOT / "lqcloud_measurements/measurement_5.json"
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
        self.assertEqual(samples.measurements.shape, (10000, 81))
        self.assertEqual(samples.lq_dets.shape, (10000, 72))
        self.assertEqual(samples.model_dets.shape, (10000, 72))
        self.assertEqual(samples.observables.shape, (10000, 1))

        matcher = pymatching.Matching.from_detector_error_model(
            samples.circuit.detector_error_model(
                decompose_errors=True,
                approximate_disjoint_errors=True,
            )
        )
        predictions = np.asarray(matcher.decode_batch(samples.lq_dets)).reshape(-1, 1)
        self.assertEqual(predictions.shape, samples.observables.shape)


if __name__ == "__main__":
    unittest.main()
