# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
DEM sampling utilities for training data generation.

Sampling runs on the GPU via cuQuantum's cuStabilizer BitMatrixSampler with
optional CuPy zero-copy DLPack transfers.  cuquantum>=26.3.0 is required.

This module provides the sampling functions needed by MemoryCircuitTorch
to generate training batches from precomputed DEM matrices (H, p, A).
"""

from __future__ import annotations

import time
from collections import OrderedDict, deque
from dataclasses import dataclass
import threading

import torch
import numpy as np

try:
    from cuquantum.stabilizer.dem_sampling import BitMatrixSampler
    from cuquantum.stabilizer.simulator import Options
    _CUSTAB_AVAILABLE = True
except ImportError:
    # This should only happen if cuquantum is not installed. That is expected
    # for some test environments that don't need DEM sampling, so handle that
    # gracefully here.
    BitMatrixSampler = None  # type: ignore[misc, assignment]
    Options = None  # type: ignore[misc, assignment]
    _CUSTAB_AVAILABLE = False

try:
    import cupy as _cp  # noqa: F401
    _CUPY_AVAILABLE = True
except ImportError:
    _CUPY_AVAILABLE = False


def _custab_available() -> bool:
    """True if custabilizer (cuquantum.stabilizer) is present. For use by tests/skip logic."""
    return _CUSTAB_AVAILABLE


@dataclass
class _SamplerCacheEntry:
    """One persistent cuST RNG stream for one exact H/p tensor pair."""

    H: torch.Tensor
    p: torch.Tensor
    HT: torch.Tensor
    sampler: object
    max_shots: int
    device_id: int
    initial_seed: int | None
    rebuild_count: int = 0


# cuST keeps RNG state inside BitMatrixSampler.  A single global sampler is not
# sufficient: constructing validation/test generators replaces the training H
# tensor and used to discard the training RNG state.  Keep one sampler per
# exact H/p object pair so independent generators own independent, persistent
# streams while retaining the fast no-rebuild path within every stream.
_sampler_cache: "OrderedDict[tuple[int, int, int], _SamplerCacheEntry]" = OrderedDict()
_sampler_cache_lock = threading.RLock()

# Backward-compatible aliases for diagnostics/tests that inspect the most
# recently used sampler.  They are not used to choose a cached stream.
_cached_sampler = None
_cached_H: "torch.Tensor | None" = None
_cached_HT: "torch.Tensor | None" = None
_cached_max_shots: int = 0
_cached_device_id: int | None = None
_cached_seed: "int | None" = None

_DEM_TIMINGS_S: deque[float] = deque(maxlen=200)
_custab_path_logged: bool = False

_MIN_MAX_SHOTS = 1024


def get_dem_sampling_avg_ms() -> float:
    """Average duration of recent dem_sampling calls in milliseconds (for training log)."""
    if not _DEM_TIMINGS_S:
        return 0.0
    return (sum(_DEM_TIMINGS_S) / len(_DEM_TIMINGS_S)) * 1000.0


def _reset_sampler_cache() -> None:
    """Reset every module-level sampler stream and legacy diagnostic alias."""
    global _cached_sampler, _cached_H, _cached_HT, _cached_max_shots, _cached_device_id, _cached_seed
    with _sampler_cache_lock:
        _sampler_cache.clear()
        _cached_sampler = None
        _cached_H = None
        _cached_HT = None
        _cached_max_shots = 0
        _cached_device_id = None
        _cached_seed = None


def _resize_seed(initial_seed: int, rebuild_count: int) -> int:
    """Derive a deterministic non-overlapping stream when max_shots must grow."""
    return (int(initial_seed) + int(rebuild_count) * 1_000_000_007) & 0x7FFF_FFFF


def dem_sampling(
    H: torch.Tensor,
    p: torch.Tensor,
    batch_size: int,
    device_id: int | None = None,
    seed: int | None = None,
) -> torch.Tensor:
    """
    Sample errors from a detector error model (DEM) via cuST BitMatrixSampler.

    Args:
        H: (2*num_detectors, num_errors) uint8 - Detector-error incidence matrix
        p: (num_errors,) float32 - Per-error probabilities
        batch_size: int - Number of samples to generate
        device_id: Optional int - Device ID for cuST. If omitted, infer from
            H.device when H is on CUDA.
        seed: Optional int - RNG seed passed directly to BitMatrixSampler.
            When provided, a fresh sampler is created with that seed so repeated
            calls with the same seed produce identical outputs.
            When None (default), the cached sampler is reused across calls.

    Returns:
        frames_xz: (batch_size, 2*num_detectors) uint8 - Detector outcomes
    """
    global _cached_sampler, _cached_H, _cached_HT, _cached_max_shots
    global _cached_device_id, _cached_seed, _custab_path_logged

    if BitMatrixSampler is None or Options is None:
        raise RuntimeError("dem_sampling requires cuquantum>=26.3.0 (stabilizer)")

    if H.ndim != 2:
        raise ValueError(f"H must be 2-D, got ndim={H.ndim}")
    if p.ndim != 1:
        raise ValueError(f"p must be 1-D, got ndim={p.ndim}")
    if H.shape[1] != p.shape[0]:
        raise ValueError(f"H has {H.shape[1]} columns but p has {p.shape[0]} entries")

    if device_id is None:
        if H.is_cuda:
            device_index = H.device.index
            device_id = int(torch.cuda.current_device() if device_index is None else device_index)
        else:
            device_id = 0

    gpu_native = _CUPY_AVAILABLE and H.is_cuda
    cache_key = (id(H), id(p), int(device_id))

    # Sampling and cache mutation share a lock.  The normal trainer has one
    # producer thread, but this also prevents two prefetch users from advancing
    # the same cuST RNG stream concurrently.
    with _sampler_cache_lock:
        entry = _sampler_cache.get(cache_key)
        # Retaining strong tensor references in the entry prevents Python id
        # reuse from ever selecting a sampler belonging to a destroyed tensor.
        if entry is not None and (entry.H is not H or entry.p is not p):
            del _sampler_cache[cache_key]
            entry = None

        explicit_reset = seed is not None
        needs_resize = entry is not None and batch_size > entry.max_shots
        if entry is None or explicit_reset or needs_resize:
            max_shots = max(batch_size, _MIN_MAX_SHOTS)
            rebuild_count = 0 if entry is None or explicit_reset else entry.rebuild_count + 1
            initial_seed = int(seed) if explicit_reset else (entry.initial_seed if entry is not None else None)
            build_seed = int(seed) if explicit_reset else None
            if needs_resize and initial_seed is not None:
                # cuST cannot grow an existing sampler.  Starting a derived
                # deterministic stream avoids replaying the beginning of the
                # original seeded sequence.
                build_seed = _resize_seed(initial_seed, rebuild_count)

            HT = H.T
            if gpu_native:
                import cupy as cp
                with cp.cuda.Device(device_id):
                    H_in = cp.from_dlpack(HT.detach())
                    p_in = cp.from_dlpack(p.detach().to(torch.float64))
                pkg = "cupy"
            else:
                H_in = HT.detach().cpu().numpy().astype(np.uint8)
                p_in = p.detach().cpu().numpy().astype(np.float64)
                pkg = "numpy"
            bms_kwargs: dict = {"package": pkg, "options": Options(device_id=device_id)}
            if build_seed is not None:
                bms_kwargs["seed"] = int(build_seed)
            sampler = BitMatrixSampler(H_in, p_in, max_shots, **bms_kwargs)
            entry = _SamplerCacheEntry(
                H=H,
                p=p,
                HT=HT,
                sampler=sampler,
                max_shots=max_shots,
                device_id=int(device_id),
                initial_seed=initial_seed,
                rebuild_count=rebuild_count,
            )
            _sampler_cache[cache_key] = entry
        else:
            _sampler_cache.move_to_end(cache_key)

        _cached_sampler = entry.sampler
        _cached_H = entry.H
        _cached_HT = entry.HT
        _cached_max_shots = entry.max_shots
        _cached_device_id = entry.device_id
        _cached_seed = entry.initial_seed

        t0 = time.perf_counter()
        if gpu_native:
            import cupy as cp
            with cp.cuda.Device(device_id):
                entry.sampler.sample(batch_size)
                out = entry.sampler.get_outcomes(bit_packed=False)
        else:
            entry.sampler.sample(batch_size)
            out = entry.sampler.get_outcomes(bit_packed=False)
        # Detach from cuST's reusable outcome buffer before releasing the lock;
        # otherwise a subsequent sample could overwrite a tensor still being
        # formatted by MemoryCircuitTorch.
        if isinstance(out, np.ndarray):
            out = torch.as_tensor(out, device=H.device).to(dtype=torch.uint8).clone()
        else:
            out = torch.from_dlpack(out).to(dtype=torch.uint8).clone()
    _DEM_TIMINGS_S.append(time.perf_counter() - t0)

    if not _custab_path_logged:
        print(
            f"---- [dem_sampling] Using cuST BitMatrixSampler path "
            f"(max_shots={_cached_max_shots}, gpu_native={gpu_native}, device_id={device_id})"
        )
        _custab_path_logged = True

    return out


def measure_from_stacked_frames(
    frames_xz: torch.Tensor,
    meas_qubits: torch.Tensor,
    meas_bases: torch.Tensor,
    nq: int,
) -> torch.Tensor:
    """
    Extract measurement outcomes from stacked frame data.

    Convention: Z-basis measurement reads the X-component of the Pauli frame
    (anti-commutation), and X-basis measurement reads the Z-component.

    Args:
        frames_xz: (batch_size, 2*num_detectors) uint8 - Stacked [X|Z] detector frames
        meas_qubits: (num_meas,) long - Qubit indices for measurements
        meas_bases: (num_meas,) long - Basis for each measurement (0=X, 1=Z)
        nq: int - Total number of qubits

    Returns:
        meas_old: (batch_size, n_rounds, num_meas) uint8 - Measurement outcomes
    """
    meas_qubits = torch.as_tensor(meas_qubits, device=frames_xz.device,
                                  dtype=torch.long).reshape(-1)
    meas_bases = torch.as_tensor(meas_bases, device=frames_xz.device, dtype=torch.long).reshape(-1)
    D = frames_xz.shape[1] // 2
    R = D // int(nq)
    assert D == R * int(nq), f"Detector count {D} must be divisible by nq={nq}"

    idx = (torch.arange(R, device=frames_xz.device)[:, None] * int(nq) +
           meas_qubits[None, :]).reshape(-1)
    x = frames_xz[:, :D].index_select(1, idx).reshape(frames_xz.shape[0], R, -1)
    z = frames_xz[:, D:].index_select(1, idx).reshape(frames_xz.shape[0], R, -1)
    return torch.where(meas_bases[None, None, :] == 1, x, z).to(torch.uint8)


def timelike_syndromes(
    frames_xz: torch.Tensor,
    A: torch.Tensor,
    meas_old: torch.Tensor,
) -> torch.Tensor:
    """
    Apply timelike corrections to measurements using the A matrix.

    A is a linear map over GF(2) producing s2 from frames_xz;
    meas_new = s2 ^ meas_old.

    Args:
        frames_xz: (batch_size, 2*num_detectors) uint8
        A: (n_rounds*num_meas, 2*num_detectors) uint8
        meas_old: (batch_size, n_rounds, num_meas) uint8

    Returns:
        meas_new: (batch_size, n_rounds, num_meas) uint8
    """
    s2 = torch.remainder(frames_xz.float() @ A.t().float(), 2).to(torch.uint8)
    return (s2 ^ meas_old.reshape(meas_old.shape[0], -1)).reshape_as(meas_old)
