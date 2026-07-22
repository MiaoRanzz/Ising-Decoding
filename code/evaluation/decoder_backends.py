# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small decoder adapters shared by training and inference workflows.

PyMatching consumes detector events and returns predicted observables directly.
The Union-Find libraries instead return error-mechanism corrections.  The
adapters below project those corrections through the detector error model's
observable matrix, exposing the same ``decode``/``decode_batch`` contract used
by the existing evaluation pipeline.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
import importlib
import importlib.util
import os
from pathlib import Path
import threading
from typing import Any, Callable, Optional

import numpy as np


PYMATCHING = "pymatching"
LDPC_UNIONFIND = "ldpc_unionfind"
NBI_HYQ_UNIONFIND = "nbi_hyq_unionfind"
SUPPORTED_DECODERS = (PYMATCHING, LDPC_UNIONFIND, NBI_HYQ_UNIONFIND)
BACKEND_LABELS = {
    PYMATCHING: "PyMatching",
    LDPC_UNIONFIND: "LDPC Union-Find",
    NBI_HYQ_UNIONFIND: "NBI-HYQ Union-Find",
}
_BACKEND_DEFAULTS = {
    PYMATCHING: True,
    LDPC_UNIONFIND: True,
    # This backend requires a separately cloned and compiled native library.
    NBI_HYQ_UNIONFIND: False,
}
_NBI_HYQ_CONSTRUCTION_LOCK = threading.Lock()


def normalize_decoder_name(name: Any) -> str:
    """Return a canonical decoder name or raise a user-facing error."""
    value = str(name or PYMATCHING).strip().lower().replace("-", "").replace("_", "")
    aliases = {
        "pymatching": PYMATCHING,
        "matching": PYMATCHING,
        "mwpm": PYMATCHING,
        "unionfind": LDPC_UNIONFIND,
        "ldpcunionfind": LDPC_UNIONFIND,
        "uf": LDPC_UNIONFIND,
        "nbihyqunionfind": NBI_HYQ_UNIONFIND,
        "nbihyqufdecoder": NBI_HYQ_UNIONFIND,
        "hyqunionfind": NBI_HYQ_UNIONFIND,
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
                    "inference requirements before selecting validation_decoder=ldpc_unionfind."
                ) from exc
            converter = detector_error_model_to_check_matrices

        if decoder_cls is None:
            try:
                from ldpc.union_find_decoder import UnionFindDecoder
            except ImportError as exc:
                raise RuntimeError(
                    "Union-Find decoding requires the ldpc package. Install the public "
                    "inference requirements before selecting validation_decoder=ldpc_unionfind."
                ) from exc
            decoder_cls = UnionFindDecoder

        if sparse_matrix_factory is None:
            try:
                from scipy.sparse import csc_matrix
            except ImportError as exc:
                raise RuntimeError(
                    "Union-Find decoding requires scipy. Install the public inference "
                    "requirements before selecting validation_decoder=ldpc_unionfind."
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


@contextmanager
def _temporary_working_directory(path: Path):
    """Temporarily enter the upstream wrapper directory while it loads its .so."""
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _load_nbi_hyq_decoder_class() -> tuple[type, Path]:
    """Load ``UFDecoder`` from a separately cloned nbi-hyq/uf_decoder tree."""
    repo_root = Path(__file__).resolve().parents[2]
    configured_root = os.environ.get("NBI_HYQ_UF_DECODER_ROOT")
    candidates = []
    if configured_root:
        candidates.append(Path(configured_root).expanduser())
    candidates.append(repo_root / "third_party" / "uf_decoder")

    for root in candidates:
        wrapper_path = root.resolve() / "py_wrapper" / "py_decoder.py"
        if not wrapper_path.is_file():
            continue
        module_name = "_nbi_hyq_uf_decoder_py_wrapper"
        spec = importlib.util.spec_from_file_location(module_name, wrapper_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Unable to load NBI-HYQ wrapper from {wrapper_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        decoder_cls = getattr(module, "UFDecoder", None)
        if decoder_cls is None:
            raise RuntimeError(f"{wrapper_path} does not define the expected UFDecoder class")
        return decoder_cls, wrapper_path.parent

    try:
        module = importlib.import_module("py_wrapper.py_decoder")
    except ImportError as exc:
        locations = ", ".join(str(path) for path in candidates)
        raise RuntimeError(
            "NBI-HYQ Union-Find requires a cloned and compiled nbi-hyq/uf_decoder. "
            "Set NBI_HYQ_UF_DECODER_ROOT to that repository, place it at "
            f"third_party/uf_decoder, or add it to PYTHONPATH. Checked: {locations}."
        ) from exc

    decoder_cls = getattr(module, "UFDecoder", None)
    module_file = getattr(module, "__file__", None)
    if decoder_cls is None or module_file is None:
        raise RuntimeError("py_wrapper.py_decoder must define UFDecoder in a filesystem module")
    return decoder_cls, Path(module_file).resolve().parent


class NbiHyqUnionFindDecoderAdapter(UnionFindDecoderAdapter):
    """Expose nbi-hyq/uf_decoder's general-LDPC path with PyMatching-like methods."""

    @classmethod
    def from_detector_error_model(
        cls,
        det_model: Any,
        *,
        converter: Optional[Callable[[Any], Any]] = None,
        decoder_cls: Optional[type] = None,
        sparse_matrix_factory: Optional[Callable[[Any], Any]] = None,
        wrapper_directory: Optional[Path] = None,
    ) -> "NbiHyqUnionFindDecoderAdapter":
        if converter is None:
            try:
                from beliefmatching.belief_matching import (
                    detector_error_model_to_check_matrices,
                )
            except ImportError as exc:
                raise RuntimeError(
                    "NBI-HYQ Union-Find decoding requires beliefmatching."
                ) from exc
            converter = detector_error_model_to_check_matrices

        if sparse_matrix_factory is None:
            try:
                from scipy.sparse import csr_matrix
            except ImportError as exc:
                raise RuntimeError("NBI-HYQ Union-Find decoding requires scipy.") from exc
            sparse_matrix_factory = csr_matrix

        if decoder_cls is None:
            decoder_cls, wrapper_directory = _load_nbi_hyq_decoder_class()

        matrices = converter(det_model)
        # The upstream wrapper accepts scipy COO/CSR (but not CSC) matrices.
        check_matrix = sparse_matrix_factory(matrices.check_matrix)
        try:
            if wrapper_directory is None:
                decoder = decoder_cls(check_matrix)
            else:
                # UFDecoder loads ../build/libSpeedDecoder.so relative to cwd.
                with _NBI_HYQ_CONSTRUCTION_LOCK:
                    with _temporary_working_directory(Path(wrapper_directory)):
                        decoder = decoder_cls(check_matrix)
        except OSError as exc:
            raise RuntimeError(
                "Unable to load nbi-hyq/uf_decoder's build/libSpeedDecoder.so. "
                "Build the upstream repository on Linux before enabling "
                "backend.nbi_hyq_unionfind."
            ) from exc

        for method_name in ("ldpc_decode", "ldpc_decode_batch"):
            if not callable(getattr(decoder, method_name, None)):
                raise RuntimeError(
                    f"NBI-HYQ UFDecoder is missing required method {method_name}()."
                )
        return cls(decoder, matrices.observables_matrix)

    def decode(self, syndrome: Any, **_: Any):
        syndrome_row = np.ascontiguousarray(syndrome, dtype=np.uint8).reshape(-1)
        mechanism_count = self._observables.shape[1]
        erasures = np.zeros(mechanism_count, dtype=np.uint8)
        self._decoder.correction = np.zeros(mechanism_count, dtype=np.uint8)
        self._decoder.ldpc_decode(syndrome_row, erasures)
        correction = np.asarray(self._decoder.correction, dtype=np.uint8).reshape(-1)
        if correction.size != mechanism_count:
            raise ValueError(
                f"NBI-HYQ correction width {correction.size} does not match DEM "
                f"mechanism count {mechanism_count}"
            )
        prediction = (self._observables.astype(np.int64) @ correction.astype(np.int64)) % 2
        prediction = prediction.astype(np.uint8, copy=False)
        return prediction[0] if prediction.size == 1 else prediction

    def decode_batch(self, syndromes: Any, **_: Any) -> np.ndarray:
        rows = np.ascontiguousarray(syndromes, dtype=np.uint8)
        if rows.ndim != 2:
            raise ValueError(f"syndromes must have shape (shots, detectors), got {rows.shape}")
        shots = rows.shape[0]
        mechanism_count = self._observables.shape[1]
        erasures = np.zeros(shots * mechanism_count, dtype=np.uint8)
        self._decoder.correction = np.zeros(shots * mechanism_count, dtype=np.uint8)
        self._decoder.ldpc_decode_batch(rows.reshape(-1), erasures, shots)
        corrections = np.asarray(self._decoder.correction, dtype=np.uint8)
        if corrections.size != shots * mechanism_count:
            raise ValueError(
                f"NBI-HYQ batch correction size {corrections.size} does not match "
                f"expected size {shots * mechanism_count}"
            )
        corrections = corrections.reshape(shots, mechanism_count)
        predictions = (
            corrections.astype(np.int64) @ self._observables.astype(np.int64).T
        ) % 2
        predictions = predictions.astype(np.uint8, copy=False)
        if self._observables.shape[0] == 1:
            return predictions.reshape(-1)
        return predictions


def build_decoder(det_model: Any, name: Any):
    """Construct one supported decoder from the same detector error model."""
    backend = normalize_decoder_name(name)
    if backend == LDPC_UNIONFIND:
        return UnionFindDecoderAdapter.from_detector_error_model(det_model)
    if backend == NBI_HYQ_UNIONFIND:
        return NbiHyqUnionFindDecoderAdapter.from_detector_error_model(det_model)

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


def _backend_flag(backend_cfg: Any, name: str, default: bool) -> bool:
    """Read one ``backend.<name>`` flag while accepting dict and OmegaConf configs."""
    if backend_cfg is None:
        value = default
    elif isinstance(backend_cfg, Mapping):
        value = backend_cfg.get(name, default)
    else:
        value = getattr(backend_cfg, name, default)
    if not isinstance(value, bool):
        raise ValueError(f"backend.{name} must be true or false, got {value!r}")
    return value


def enabled_inference_backends(cfg: Any) -> tuple[str, ...]:
    """Return the global decoder backends enabled for inference workflows."""
    backend_cfg = getattr(cfg, "backend", None)
    names = tuple(
        name
        for name in SUPPORTED_DECODERS
        if _backend_flag(backend_cfg, name, default=_BACKEND_DEFAULTS[name])
    )
    if not names:
        raise ValueError("At least one inference backend must be enabled.")
    return names
