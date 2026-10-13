# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import sys
import tempfile
import unittest
from pathlib import Path
import zipfile

import numpy as np
import stim
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from noise_learning.google_qec import GoogleQECDataset, inject_noise, unpack_b8
from noise_learning.paper_model import NoiseLearningNetwork
from qec.noise_model import NoiseModel


class TestPaperNoiseLearningNetwork(unittest.TestCase):
    def test_architecture_shape_bounds_and_parameter_count(self):
        model = NoiseLearningNetwork().eval()
        values = torch.zeros(3, 4, 2, 5, 5)
        with torch.no_grad():
            predicted = model(values)
        self.assertEqual(tuple(predicted.shape), (25,))
        self.assertTrue(bool(torch.all(predicted >= 1e-5)))
        self.assertTrue(bool(torch.all(predicted <= 3e-2)))
        count = sum(parameter.numel() for parameter in model.parameters())
        self.assertGreater(count, 1_200_000)
        self.assertLess(count, 1_300_000)

    def test_forward_uses_post_mlp_logit_averaging(self):
        torch.manual_seed(2)
        model = NoiseLearningNetwork().eval()
        values = torch.randn(2, 8, 5, 5)
        with torch.no_grad():
            logits = model.per_sample_logits(values)
            expected = model.bounded_log_space(logits.mean(dim=0))
            actual = model(values)
        torch.testing.assert_close(actual, expected)


class TestGoogleQECNoiseLearning(unittest.TestCase):
    def test_unpack_b8_handles_non_byte_aligned_rows(self):
        expected = np.array([[0, 1, 1, 0, 1], [1, 0, 0, 1, 0]], dtype=bool)
        packed = np.packbits(expected, axis=1, bitorder="little")
        actual = unpack_b8(packed.tobytes(), shots=2, bits_per_shot=5)
        np.testing.assert_array_equal(actual, expected)

    def test_dataset_reads_experiment_directly_from_zip(self):
        circuit = stim.Circuit(
            """
            QUBIT_COORDS(0, 0) 0
            QUBIT_COORDS(1, 0) 1
            R 1
            H 1
            CZ 1 0
            H 1
            M 1
            DETECTOR rec[-1]
            M 0
            OBSERVABLE_INCLUDE(0) rec[-1]
            """
        )
        metadata = {
            "basis": "Z",
            "rounds": 1,
            "shots": 2,
            "distance": 1,
            "data_qubit_coords": [[0, 0]],
            "meas_qubit_coords": [[1, 0]],
        }
        packed = np.packbits(np.array([[0], [1]], dtype=np.uint8), axis=1, bitorder="little")
        with tempfile.TemporaryDirectory() as tmp:
            archive_path = Path(tmp) / "data.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                root = "dataset/d1/Z/r1"
                archive.writestr(f"{root}/metadata.json", json.dumps(metadata))
                archive.writestr(f"{root}/circuit_ideal.stim", str(circuit))
                archive.writestr(f"{root}/detection_events.b8", packed.tobytes())
                archive.writestr(f"{root}/obs_flips_actual.b8", packed.tobytes())
            dataset = GoogleQECDataset(archive_path)
            experiment = dataset.load("dataset/d1/Z/r1")
        self.assertEqual(experiment.detection_events.shape, (2, 1))
        self.assertEqual(experiment.observable_flips.shape, (2, 1))
        self.assertEqual(experiment.metadata["basis"], "Z")

    def test_implicit_initial_z_preparation_uses_p_prep_z(self):
        circuit = stim.Circuit(
            """
            QUBIT_COORDS(0, 0) 0
            QUBIT_COORDS(1, 0) 1
            TICK
            H 1
            CZ 1 0
            H 1
            M 1
            DETECTOR rec[-1]
            """
        )
        noisy = inject_noise(
            circuit,
            NoiseModel(p_prep_Z=0.123),
            basis="Z",
            data_qubits=[0],
            measurement_qubits=[1],
        )
        lines = str(noisy).splitlines()
        self.assertIn("X_ERROR(0.123) 0", lines)
        self.assertLess(lines.index("X_ERROR(0.123) 0"), lines.index("TICK"))

    def test_x_basis_measurement_fault_flips_physical_mz(self):
        circuit = stim.Circuit(
            """
            QUBIT_COORDS(0, 0) 0
            QUBIT_COORDS(1, 0) 1
            R 1
            H 1
            CZ 1 0
            H 1
            M 1
            DETECTOR rec[-1]
            """
        )
        noisy = inject_noise(
            circuit,
            NoiseModel(p_meas_X=0.123),
            basis="X",
            data_qubits=[0],
            measurement_qubits=[1],
        )
        lines = str(noisy).splitlines()
        self.assertIn("X_ERROR(0.123) 1", lines)
        self.assertNotIn("Z_ERROR(0.123) 1", lines)
        self.assertLess(lines.index("X_ERROR(0.123) 1"), lines.index("M 1"))

    def test_injection_uses_all_channel_families(self):
        circuit = stim.Circuit(
            """
            QUBIT_COORDS(0, 0) 0
            QUBIT_COORDS(1, 0) 1
            R 1
            H 1
            CZ 1 0
            H 1
            M 1
            DETECTOR rec[-1]
            """
        )
        model = NoiseModel.from_single_p(0.003)
        noisy = inject_noise(
            circuit,
            model,
            basis="Z",
            data_qubits=[0],
            measurement_qubits=[1],
        )
        text = str(noisy)
        self.assertIn("PAULI_CHANNEL_2", text)
        self.assertIn("PAULI_CHANNEL_1", text)
        self.assertIn("X_ERROR", text)
        noisy.detector_error_model(approximate_disjoint_errors=True)


if __name__ == "__main__":
    unittest.main()
