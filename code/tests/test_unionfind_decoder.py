# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.decoder_backends import (  # noqa: E402
    LDPC_UNIONFIND,
    PYMATCHING,
    UnionFindDecoderAdapter,
    enabled_inference_backends,
    normalize_decoder_name,
    validation_decoder_name,
)


class _FakeUnionFind:

    def __init__(self, check_matrix=None, uf_method=None):
        self.check_matrix = check_matrix
        self.uf_method = uf_method

    def decode(self, syndrome):
        # A deterministic error-mechanism correction for projection tests.
        syndrome = np.asarray(syndrome, dtype=np.uint8)
        return np.array([syndrome[0], syndrome[1], syndrome[0] ^ syndrome[1]], dtype=np.uint8)


class TestUnionFindDecoderAdapter(unittest.TestCase):

    def test_decode_projects_correction_to_all_observables(self):
        observables = np.array([[1, 0, 1], [0, 1, 1]], dtype=np.uint8)
        adapter = UnionFindDecoderAdapter(_FakeUnionFind(), observables)

        self.assertEqual(adapter.decode([1, 0]).tolist(), [0, 1])
        np.testing.assert_array_equal(
            adapter.decode_batch([[1, 0], [0, 1], [1, 1]]),
            np.array([[0, 1], [1, 0], [1, 1]], dtype=np.uint8),
        )

    def test_single_observable_batch_matches_pymatching_shape(self):
        adapter = UnionFindDecoderAdapter(_FakeUnionFind(), np.array([[1, 0, 1]], dtype=np.uint8))
        prediction = adapter.decode_batch([[1, 0], [0, 1]])
        self.assertEqual(prediction.shape, (2,))
        np.testing.assert_array_equal(prediction, np.array([0, 1], dtype=np.uint8))

    def test_from_dem_constructs_peeling_decoder(self):
        matrices = SimpleNamespace(
            check_matrix="dense-H",
            observables_matrix=np.array([[1, 0, 1]], dtype=np.uint8),
        )
        adapter = UnionFindDecoderAdapter.from_detector_error_model(
            "dem",
            converter=lambda dem: matrices,
            decoder_cls=_FakeUnionFind,
            sparse_matrix_factory=lambda matrix: f"sparse({matrix})",
        )
        self.assertEqual(adapter._decoder.check_matrix, "sparse(dense-H)")
        self.assertEqual(adapter._decoder.uf_method, "peeling")

    def test_decoder_names_and_training_default(self):
        self.assertEqual(normalize_decoder_name("MWPM"), PYMATCHING)
        self.assertEqual(normalize_decoder_name("union-find"), LDPC_UNIONFIND)
        self.assertEqual(validation_decoder_name(SimpleNamespace()), PYMATCHING)
        self.assertEqual(
            validation_decoder_name(SimpleNamespace(validation_decoder="uf")), LDPC_UNIONFIND
        )
        self.assertEqual(
            enabled_inference_backends(
                SimpleNamespace(backend=SimpleNamespace(pymatching=True, ldpc_unionfind=False))
            ),
            (PYMATCHING,),
        )
        self.assertEqual(
            enabled_inference_backends(
                SimpleNamespace(backend=SimpleNamespace(pymatching=False, ldpc_unionfind=True))
            ),
            (LDPC_UNIONFIND,),
        )
        with self.assertRaises(ValueError):
            enabled_inference_backends(
                SimpleNamespace(backend=SimpleNamespace(pymatching=False, ldpc_unionfind=False))
            )
        with self.assertRaises(ValueError):
            normalize_decoder_name("unknown")


if __name__ == "__main__":
    unittest.main()
