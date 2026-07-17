# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small decoder adapters shared by training and inference workflows.

PyMatching consumes detector events and returns predicted observables directly.
``ldpc.UnionFindDecoder`` instead returns an error-mechanism correction.  The
adapter below projects that correction through the detector error model's
observable matrix, exposing the same ``decode``/``decode_batch`` contract used
by the existing evaluation pipeline.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np


PYMATCHING = "pymatching"
UNIONFIND = "unionfind"
SUPPORTED_DECODERS = (PYMATCHING, UNIONFIND)


def normalize_decoder_name(name: Any) -> str:
    """Return a canonical decoder name or raise a user-facing error."""
    value = str(name or PYMATCHING).strip().lower().replace("-", "").replace("_", "")
    aliases = {
        "pymatching": PYMATCHING,
        "matching": PYMATCHING,
        "mwpm": PYMATCHING,
        "unionfind": UNIONFIND,
        "uf": UNIONFIND,
    }
    if value not in aliases:
        raise ValueError(
            f"Unsupported decoder {name!r}. Choose one of: {', '.join(SUPPORTED_DECODERS)}"
        )
    return aliases[value]


class UnionFindDecoderAdapter:
    """Expose an LDPC Union-Find decoder with PyMatching-like methods."""

    def __init__(self, decoder: Any, observables_matrix: Any):
        self._decoder = decoder
        if hasattr(observables_matrix, "toarray"):
            observables_matrix = observables_matrix.toarray()
        self._observables = np.asarray(observables_matrix, dtype=np.uint8)
        if self._observables.ndim == 1:
            self._observables = self._observables.reshape(1, -1)
        if self._observables.ndim != 2:
            raise ValueError("observables_matrix must be a 2-D matrix")

    @classmethod
    def from_detector_error_model(
        cls,
        det_model: Any,
        *,
        converter: Optional[Callable[[Any], Any]] = None,
        decoder_cls: Optional[type] = None,
        sparse_matrix_factory: Optional[Callable[[Any], Any]] = None,
    ) -> "UnionFindDecoderAdapter":
        """Build Union-Find's check matrix and logical projection from a Stim DEM."""
        if converter is None:
            try:
                from beliefmatching.belief_matching import (
                    detector_error_model_to_check_matrices,
                )
            except ImportError as exc:
                raise RuntimeError(
                    "Union-Find decoding requires beliefmatching. Install the public "
                    "inference requirements before selecting validation_decoder=unionfind."
                ) from exc
            converter = detector_error_model_to_check_matrices

        if decoder_cls is None:
            try:
                from ldpc.union_find_decoder import UnionFindDecoder
            except ImportError as exc:
                raise RuntimeError(
                    "Union-Find decoding requires the ldpc package. Install the public "
                    "inference requirements before selecting validation_decoder=unionfind."
                ) from exc
            decoder_cls = UnionFindDecoder

        if sparse_matrix_factory is None:
            try:
                from scipy.sparse import csc_matrix
            except ImportError as exc:
                raise RuntimeError(
                    "Union-Find decoding requires scipy. Install the public inference "
                    "requirements before selecting validation_decoder=unionfind."
                ) from exc
            sparse_matrix_factory = csc_matrix

        matrices = converter(det_model)
        check_matrix = sparse_matrix_factory(matrices.check_matrix)
        decoder = decoder_cls(check_matrix, uf_method="peeling")
        return cls(decoder, matrices.observables_matrix)

    def decode(self, syndrome: Any, **_: Any):
        correction = np.asarray(
            self._decoder.decode(np.ascontiguousarray(syndrome, dtype=np.uint8)),
            dtype=np.uint8,
        ).reshape(-1)
        if correction.size != self._observables.shape[1]:
            raise ValueError(
                f"Union-Find correction width {correction.size} does not match DEM "
                f"mechanism count {self._observables.shape[1]}"
            )
        prediction = (self._observables.astype(np.int64) @ correction.astype(np.int64)) % 2
        prediction = prediction.astype(np.uint8, copy=False)
        return prediction[0] if prediction.size == 1 else prediction

    def decode_batch(self, syndromes: Any, **_: Any) -> np.ndarray:
        rows = np.asarray(syndromes, dtype=np.uint8)
        if rows.ndim != 2:
            raise ValueError(f"syndromes must have shape (shots, detectors), got {rows.shape}")
        predictions = [self.decode(row) for row in rows]
        result = np.asarray(predictions, dtype=np.uint8)
        if self._observables.shape[0] == 1:
            return result.reshape(-1)
        return result.reshape(rows.shape[0], self._observables.shape[0])


def build_decoder(det_model: Any, name: Any):
    """Construct one supported decoder from the same detector error model."""
    backend = normalize_decoder_name(name)
    if backend == UNIONFIND:
        return UnionFindDecoderAdapter.from_detector_error_model(det_model)

    try:
        import pymatching
    except ImportError as exc:
        raise RuntimeError(
            "PyMatching decoding requires pymatching. Install the public inference requirements."
        ) from exc
    return pymatching.Matching.from_detector_error_model(det_model)


def validation_decoder_name(cfg: Any) -> str:
    """Read the training-time validation backend (PyMatching by default)."""
    return normalize_decoder_name(getattr(cfg, "validation_decoder", PYMATCHING))
