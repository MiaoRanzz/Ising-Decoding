"""
SI1000 Dataset for pretraining AlphaQubit 2 — circuit-backed, paper-aligned (v2).

Rewrite of 2026-08-05. Aligns the data pipeline with the AlphaQubit 2 paper
(arXiv:2512.07737v2) in three ways:

1. Detector layout (paper A.1.1/A.3.1)
   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   Detectors are mapped onto *true* (round, stabilizer-slot) positions by
   parsing the circuit itself (``_analyze_circuit``) instead of a flat
   ``reshape(T, N)``.  One experiment is represented as ``T+1`` frames of
   ``N = d²−1`` slots, in ``STABILIZER_LOCATIONS[d]`` order:

     frame 0      : e = round-1 detectors vs. initial state (Z slots); X slots = 0
     frame t      : e = detector comparing rounds t+1 vs. t (all slots)
     frame T      : e = final data-derived Z detectors; X slots = 0  (end marker)

   ``m`` is the cumulative XOR of ``e`` per slot (measurement reconstruction).
   Because the training cycle counts (24/48/72/120) are multiples of the
   temporal chunk K=6, the final frame always forms a chunk of its own
   (1 real + K−1 zero-padded frames) — the paper's "the final cycle ... is
   embedded by itself" (A.1.1).  The X-slot zeros in frame 0/T give the
   network an explicit, learnable start/end-of-experiment signal.

2. Noise model (paper A.3.2, *modern* SI1000 interpretation)
   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   The circuit is generated with ``after_clifford_depolarization=0`` and all
   gate noise is injected explicitly (``_apply_modern_si1000``):

     CX  → DEPOLARIZE2(p)          H → DEPOLARIZE1(p/10)   (SI1000 1q = p/10)
     MR  → M + [Idle(p/10) + ResonatorIdle(2p) on data qubits]
           R + [Idle(p/10) + ResonatorIdle(2p) on data qubits]

   Measurement (5p) and reset (2p) flips from the generator are kept.
   The same processed circuit feeds FlipSimulator (training) and
   ``detector_error_model`` (validation), so train/eval noise is identical.

3. Same-shot auxiliary labels (paper A.2.3/A.3.3)
   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   ``stim.FlipSimulator`` runs the full experiment and every chunk-boundary
   truncated experiment with one shared random seed per noise group.  The
   noise realisation of the shared circuit prefix is then byte-identical
   (verified in ``scripts/verify_pipeline_v2.py``), so the pseudo-terminated
   frames/observables are genuine same-shot fake endings.

   Noiseless observable labels (heads ②③④) are obtained by inserting
   ``MPP Z·Z·…`` (multi-Pauli measurement over the observable line parsed
   from ``OBSERVABLE_INCLUDE``) at every chunk boundary of the training
   circuits (``_insert_noiseless_mpps``).  Pauli-error states are eigenstates
   of the logical-Z line at round boundaries, so these measurements are
   deterministic and non-demolition; they do not perturb detectors,
   observables, or the RNG stream (verified).  Targets per chunk c:

     ① pseudo[c]          = fake-ending observable (same shot)
     ② noiseless[c]       = MPP value at the end of chunk c
     ③ noiseless_diff[c]  = ②[c] ⊕ ②[c−1]        (②[−1] := 0)
     ④ noiseless→inter[c] = ②[c−1] ⊕ ①[c]

torch is only required for the final tensor conversion; the circuit and
frame logic is pure stim+numpy so it can be verified without torch
(scripts/verify_pipeline_v2.py injects a numpy-backed torch stub).
"""

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import stim

try:  # torch is only needed to wrap outputs; see module docstring.
    import torch
except ModuleNotFoundError:  # pragma: no cover - stim-only verification env
    torch = None

from src.data.google_data_utils import STABILIZER_LOCATIONS


# =========================================================================
# Circuit construction — modern SI1000 noise (paper A.3.2)
# =========================================================================

_TWO_QUBIT_GATES = {
    "CX", "CZ", "CNOT", "SWAP", "ISWAP", "SQ_ISWAP",
    "XCX", "XCY", "XCZ", "YCX", "YCY", "YCZ", "ZCX", "ZCY", "ZCZ",
}
_ONE_QUBIT_GATES = {"H", "S", "X", "Y", "Z", "SQ_X", "SQ_Y", "SQ_Z", "SQ_XZ", "SQ_YZ"}
# Instructions we expect and pass through unchanged (after gate-noise-free
# generation the only noise ops present are flip channels + before-round idle).
_PASSTHROUGH = {
    "QUBIT_COORDS", "R", "MR", "M", "X_ERROR", "TICK", "DETECTOR",
    "OBSERVABLE_INCLUDE", "DEPOLARIZE1", "DEPOLARIZE2", "SHIFT_COORDS",
}


def _qubit_groups(flat: stim.Circuit, distance: int) -> tuple[list[int], dict[int, int]]:
    """Split qubits into (data qubits, ancilla qubit -> stabilizer slot).

    Ancillas are identified by their ``QUBIT_COORDS`` matching an entry of
    ``STABILIZER_LOCATIONS[distance]`` (verified to be an exact bijection for
    d = 3, 5, 7, 9, 11).
    """
    slot_of_xy = {
        (int(p[0]), int(p[1])): i for i, p in enumerate(STABILIZER_LOCATIONS[distance])
    }
    data: list[int] = []
    slot_of_qubit: dict[int, int] = {}
    for q, xy in flat.get_final_qubit_coordinates().items():
        key = (int(round(xy[0])), int(round(xy[1])))
        if key in slot_of_xy:
            slot_of_qubit[q] = slot_of_xy[key]
        else:
            data.append(q)
    n = distance * distance - 1
    assert len(slot_of_qubit) == n, (
        f"d={distance}: found {len(slot_of_qubit)} ancilla qubits, expected {n}"
    )
    assert len(data) == distance * distance, (
        f"d={distance}: found {len(data)} data qubits, expected {distance * distance}"
    )
    return sorted(data), slot_of_qubit


def _apply_modern_si1000(flat: stim.Circuit, p: float,
                         data_qubits: list[int]) -> stim.Circuit:
    """Inject the modern SI1000 noise channels (paper A.3.2).

    Input circuit must be generated with ``after_clifford_depolarization=0``
    so that gate noise is fully controlled here:

      - after each two-qubit gate block: DEPOLARIZE2(p) on the same targets
      - after each one-qubit gate block: DEPOLARIZE1(p/10)   (SI1000 1q rate)
      - each MR is split into M + R ("measurement and reset count as two
        separate operations with full sets of noise applied"):
            M   → DEPOLARIZE1(p/10) + DEPOLARIZE1(2p) on idling data qubits
            R   → DEPOLARIZE1(p/10) + DEPOLARIZE1(2p) on idling data qubits

    The generator's X_ERROR(5p) pre-measurement, X_ERROR(2p) post-reset and
    DEPOLARIZE1(p/10) before-round channels are kept as-is; the post-reset
    X_ERROR(2p) emitted after each MR lands right after the split R, matching
    the paper's example.  Splitting MR does not change the measurement record
    (R produces no record), so all DETECTOR rec offsets stay valid.
    """
    out = stim.Circuit()
    for ins in flat:
        name = ins.name
        if name in _TWO_QUBIT_GATES:
            out.append(ins)
            out.append("DEPOLARIZE2", ins.targets_copy(), p)
        elif name in _ONE_QUBIT_GATES:
            out.append(ins)
            out.append("DEPOLARIZE1", ins.targets_copy(), p / 10.0)
        elif name == "MR":
            targets = ins.targets_copy()
            out.append("M", targets)
            out.append("DEPOLARIZE1", list(data_qubits), p / 10.0)
            out.append("DEPOLARIZE1", list(data_qubits), 2 * p)
            out.append("R", targets)
            out.append("DEPOLARIZE1", list(data_qubits), p / 10.0)
            out.append("DEPOLARIZE1", list(data_qubits), 2 * p)
        else:
            assert name in _PASSTHROUGH, f"unexpected instruction {name!r} in generated circuit"
            out.append(ins)
    return out


# =========================================================================
# Circuit layout analysis — true (round, slot) detector mapping
# =========================================================================

@dataclass
class CircuitLayout:
    """True detector layout of one (distance, rounds) circuit.

    det_index[f, s]  = detector column carrying frame f, slot s (-1 if none)
    anc_meas_index[r, s] = measurement-record index of slot s in round r
    """
    distance: int
    rounds: int                     # T (stabilizer rounds); frames = T + 1
    n_spatial: int
    slot_of_qubit: dict[int, int]
    data_qubits: list[int]
    obs_line: list[int]
    is_z: np.ndarray                # (N,) bool
    det_index: np.ndarray           # (T+1, N) int64
    anc_meas_index: np.ndarray      # (T, N) int64

    @property
    def num_frames(self) -> int:
        return self.rounds + 1


def _analyze_circuit(flat: stim.Circuit, distance: int, rounds: int) -> CircuitLayout:
    """Parse a (processed, flattened) circuit into a CircuitLayout.

    Walks the instruction stream once, tracking the measurement record, then
    resolves every DETECTOR's rec targets to (round, slot) or data qubits:

      - 1 ancilla rec from round 1            -> frame 0 (marks the slot Z-type)
      - ancilla recs, same slot, rounds r,r+1 -> frame r
      - contains data recs (+ round-T rec)    -> frame T (final data-derived)
    """
    n_spatial = distance * distance - 1
    data_qubits, slot_of_qubit = _qubit_groups(flat, distance)

    m_count = 0
    anc_blocks: list[tuple[int, list[int]]] = []   # (base_rec, slots per target)
    final_block: Optional[tuple[int, list[int]]] = None
    pending_dets: list[tuple[int, list[int], int]] = []  # (m_at, offsets, column)
    pending_obs_offsets: Optional[list[int]] = None
    obs_line: Optional[list[int]] = None

    for ins in flat:
        name = ins.name
        if name in ("M", "MR"):
            qubits = [t.qubit_value for t in ins.targets_copy()]
            if qubits and all(q in slot_of_qubit for q in qubits):
                anc_blocks.append((m_count, [slot_of_qubit[q] for q in qubits]))
            else:
                final_block = (m_count, qubits)
            m_count += len(qubits)
        elif name == "DETECTOR":
            offsets = [int(t.value) for t in ins.targets_copy()]
            pending_dets.append((m_count, offsets, len(pending_dets)))
        elif name == "OBSERVABLE_INCLUDE":
            # Defer resolution until idx_to_data is populated (after the loop).
            pending_obs_offsets = [int(t.value) for t in ins.targets_copy()]

    assert len(anc_blocks) == rounds, (
        f"d={distance} T={rounds}: found {len(anc_blocks)} ancilla measurement blocks"
    )
    assert final_block is not None, "no data-qubit measurement block found"

    # measurement record index -> (round, slot) for ancillas, -> qubit for data
    anc_meas_index = np.full((rounds, n_spatial), -1, dtype=np.int64)
    idx_to_rs: dict[int, tuple[int, int]] = {}
    for bi, (base, slots) in enumerate(anc_blocks):
        for j, s in enumerate(slots):
            anc_meas_index[bi, s] = base + j
            idx_to_rs[base + j] = (bi + 1, s)
    fb_base, fb_qubits = final_block
    idx_to_data = {fb_base + j: q for j, q in enumerate(fb_qubits)}

    # Resolve OBSERVABLE_INCLUDE offsets → data-qubit line (deferred from loop)
    assert pending_obs_offsets is not None, "no OBSERVABLE_INCLUDE found in circuit"
    obs_line = []
    for off in pending_obs_offsets:
        rec = m_count + off                             # off is negative
        assert rec in idx_to_data, (
            f"OBSERVABLE_INCLUDE rec[{off}] not in final data block"
        )
        obs_line.append(idx_to_data[rec])

    det_index = np.full((rounds + 1, n_spatial), -1, dtype=np.int64)
    is_z = np.zeros(n_spatial, dtype=bool)

    for m_at, offsets, col in pending_dets:
        anc_refs: list[tuple[int, int]] = []
        data_refs: list[int] = []
        for off in offsets:
            rec = m_at + off
            if rec in idx_to_rs:
                anc_refs.append(idx_to_rs[rec])
            else:
                assert rec in idx_to_data, f"DETECTOR rec[{off}] resolves nowhere"
                data_refs.append(idx_to_data[rec])
        if data_refs:
            # final data-derived detector: keeps the round-T ancilla reference
            assert anc_refs, "final detector without ancilla reference"
            assert {r for r, _ in anc_refs} == {rounds}
            slot = anc_refs[0][1]
            frame = rounds
            is_z[slot] = True
        else:
            slots_seen = {s for _, s in anc_refs}
            rounds_seen = {r for r, _ in anc_refs}
            assert len(slots_seen) == 1, f"detector spans slots {slots_seen}"
            slot = slots_seen.pop()
            if rounds_seen == {1}:
                frame = 0
                is_z[slot] = True
            else:
                assert len(rounds_seen) == 2 and max(rounds_seen) == min(rounds_seen) + 1, (
                    f"detector spans rounds {rounds_seen}"
                )
                frame = min(rounds_seen)
        assert det_index[frame, slot] == -1, f"duplicate detector at frame {frame} slot {slot}"
        det_index[frame, slot] = col

    # structural assertions: Z slots measured every frame; X slots only in bulk
    z_slots = np.where(is_z)[0]
    x_slots = np.where(~is_z)[0]
    assert 2 * len(z_slots) == n_spatial, f"expected half Z slots, got {len(z_slots)}/{n_spatial}"
    assert (det_index[:, z_slots] >= 0).all(), "Z slot missing a detector"
    assert (det_index[[0, rounds]][:, x_slots] == -1).all(), "X slot unexpectedly measured at boundary"
    assert (det_index[1:rounds][:, x_slots] >= 0).all(), "X slot missing a bulk detector"
    assert len(pending_dets) == flat.num_detectors
    assert obs_line is not None and len(obs_line) == distance, (
        f"observable line {obs_line} should have {distance} qubits"
    )

    return CircuitLayout(
        distance=distance, rounds=rounds, n_spatial=n_spatial,
        slot_of_qubit=slot_of_qubit, data_qubits=data_qubits,
        obs_line=obs_line, is_z=is_z,
        det_index=det_index, anc_meas_index=anc_meas_index,
    )


def _detectors_to_frames(det_flips: np.ndarray,
                         layout: CircuitLayout) -> tuple[np.ndarray, np.ndarray]:
    """(B, num_detectors) detector flips -> (m, e) frames, each (B, T+1, N).

    Slots with no detector in a frame (X slots at frames 0 and T) stay 0 —
    the explicit start/end-of-experiment marker.  m = cumulative XOR of e.
    """
    det_flips = np.asarray(det_flips, dtype=bool)
    B = det_flips.shape[0]
    e = np.zeros((B, layout.num_frames, layout.n_spatial), dtype=bool)
    idx = layout.det_index
    valid = idx >= 0
    e[:, valid] = det_flips[:, idx[valid]]
    m = np.logical_xor.accumulate(e, axis=1)
    return m.astype(np.float32), e.astype(np.float32)


# =========================================================================
# Circuit builders (cached by the dataset)
# =========================================================================

def _build_processed_circuit(distance: int, rounds: int,
                             p: float) -> tuple[stim.Circuit, CircuitLayout]:
    """Generated circuit + modern SI1000 noise + parsed layout."""
    flat = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=0.0,      # gate noise injected manually
        before_round_data_depolarization=p / 10.0,
        before_measure_flip_probability=5 * p,
        after_reset_flip_probability=2 * p,
    ).flattened()
    data_qubits, _ = _qubit_groups(flat, distance)
    proc = _apply_modern_si1000(flat, p, data_qubits)
    layout = _analyze_circuit(proc, distance, rounds)
    return proc, layout


def _insert_noiseless_mpps(flat: stim.Circuit, obs_line: list[int],
                           temporal_K: int,
                           ancilla_qubits: set[int]) -> tuple[stim.Circuit, list[int]]:
    """Insert ``MPP Z·Z·…`` over the observable line at every chunk boundary.

    A boundary is the end of every temporal_K-th ancilla measurement round;
    the MPP is placed just before the TICK closing that round, i.e. after the
    round's reset+noise, where the data block is a logical-Z eigenstate
    (deterministic, non-demolition measurement).  The MPP right before the
    final data-measurement section is inserted *before* the X_ERROR(5p)
    pre-measurement channel of the data qubits, so the noiseless value is not
    contaminated by termination readout noise.

    Returns (circuit_with_mpps, mpp_record_indices) — the indices of the MPP
    results inside FlipSimulator.get_measurement_flips().

    Each inserted MPP adds one entry to the measurement record, so the rec
    targets of later DETECTOR / OBSERVABLE_INCLUDE instructions must be shifted
    by the number of MPPs inserted *between* the referenced measurement and the
    instruction itself.  The function performs a two-pass bookkeeping to apply
    that shift correctly.
    """
    # ------------------------------------------------------------------
    # Pass 1: discover measurement rounds and chunk-boundary rounds.
    # ------------------------------------------------------------------
    abs_round: list[int] = []          # original absolute index -> round
    boundary_rounds: set[int] = set()
    m_count = 0
    anc_block_count = 0
    total_rounds = 0
    for ins in flat:
        name = ins.name
        if name in ("M", "MR"):
            qubits = [t.qubit_value for t in ins.targets_copy()]
            is_ancilla = qubits and all(q in ancilla_qubits for q in qubits)
            if is_ancilla:
                anc_block_count += 1
                total_rounds = max(total_rounds, anc_block_count)
                if anc_block_count % temporal_K == 0:
                    boundary_rounds.add(anc_block_count)
                r = anc_block_count
            else:
                # Final data-measurement block sits in a virtual round T+1.
                r = total_rounds + 1
            n = len(qubits)
            abs_round.extend([r] * n)
            m_count += n

    # mpp_before_round[t] = # of MPPs inserted at boundaries of rounds < t
    mpp_before_round = [0] * (total_rounds + 3)
    for br in boundary_rounds:
        for t in range(br + 1, len(mpp_before_round)):
            mpp_before_round[t] += 1

    # ------------------------------------------------------------------
    # Pass 2: build the output circuit with MPPs and shifted rec targets.
    # ------------------------------------------------------------------
    out = stim.Circuit()
    mpp = stim.Circuit("MPP " + "*".join(f"Z{q}" for q in obs_line))
    m_count = 0
    anc_block_count = 0
    pending_mpp = False
    mpp_inserted_total = 0
    mpp_rec_indices: list[int] = []

    for ins in flat:
        name = ins.name
        if name in ("M", "MR"):
            qubits = [t.qubit_value for t in ins.targets_copy()]
            is_ancilla = qubits and all(q in ancilla_qubits for q in qubits)
            if is_ancilla:
                anc_block_count += 1
                if anc_block_count % temporal_K == 0:
                    pending_mpp = True
            elif pending_mpp:
                # Final chunk boundary coincides with the last stabilizer round.
                # There is no closing TICK; insert the MPP right before the data
                # measurement block (after any preceding pre-measurement noise).
                mpp_rec_indices.append(m_count)
                out += mpp
                m_count += 1
                mpp_inserted_total += 1
                pending_mpp = False
            m_count += len(qubits)
        elif name == "TICK" and pending_mpp:
            mpp_rec_indices.append(m_count)
            out += mpp
            m_count += 1
            mpp_inserted_total += 1
            pending_mpp = False
        elif name == "X_ERROR" and pending_mpp:
            qubits = [t.qubit_value for t in ins.targets_copy()]
            if qubits and all(q not in ancilla_qubits for q in qubits):
                # Insert before the 5p pre-measurement channel on data qubits so
                # the noiseless MPP value is not corrupted by readout noise.
                mpp_rec_indices.append(m_count)
                out += mpp
                m_count += 1
                mpp_inserted_total += 1
                pending_mpp = False
        elif name in ("DETECTOR", "OBSERVABLE_INCLUDE"):
            # Determine the "round" this instruction belongs to.
            # abs_round is indexed by the *original* measurement count, so we
            # subtract the MPPs already inserted from the current record count.
            orig_m_count = m_count - mpp_inserted_total
            if name == "OBSERVABLE_INCLUDE":
                detector_round = total_rounds + 1
            else:
                # DETECTOR: its rec targets reveal whether it is a bulk, initial,
                # or final data-derived detector.  We use the newest referenced
                # measurement round as a proxy for the detector's frame.
                ref_rounds = [
                    abs_round[orig_m_count + t.value]
                    for t in ins.targets_copy()
                    if t.is_measurement_record_target
                ]
                detector_round = max(ref_rounds) if ref_rounds else total_rounds + 1

            shifted = []
            for t in ins.targets_copy():
                if t.is_measurement_record_target:
                    ref_abs = orig_m_count + t.value
                    ref_round = abs_round[ref_abs]
                    n_mpp_between = (
                        mpp_before_round[detector_round]
                        - mpp_before_round[ref_round]
                    )
                    shifted.append(stim.target_rec(t.value - n_mpp_between))
                else:
                    shifted.append(t)
            out.append(name, shifted, ins.gate_args_copy())
            continue
        out.append(ins)
    assert not pending_mpp, (
        "last chunk boundary had no closing TICK or final data measurement"
    )
    return out, mpp_rec_indices


# =========================================================================
# Dataset
# =========================================================================

@dataclass
class _Bundle:
    """Circuits/layouts for one (noise, T, K) training configuration."""
    nq: int
    full: stim.Circuit                # with inline noiseless MPPs
    layout: CircuitLayout
    mpp_rec_indices: list[int]        # one per chunk boundary (C = T // K)
    truncs: list[stim.Circuit]        # with inline MPPs; truncs[C-1] is full
    trunc_layouts: list[CircuitLayout]


class SI1000DEMDataset:
    """Training / validation dataset with on-the-fly circuit generation."""

    def __init__(
        self,
        data_dir: str = "",                # kept for API compat — ignored
        distance: int = 3,
        basis: Optional[str] = None,
        rounds: Optional[int] = None,
        shuffle_sources: bool = True,
        uniform_rounds: bool = True,
        seed: Optional[int] = None,
    ):
        self.distance = distance
        self.basis = basis.upper() if basis else None
        self.rounds = rounds
        self.shuffle_sources = shuffle_sources
        self.uniform_rounds = uniform_rounds
        self.rng = np.random.default_rng(seed)

        self.n_spatial = distance * distance - 1
        self.stabilizer_indices = (
            torch.arange(self.n_spatial, dtype=torch.long) if torch is not None else None
        )

        # Training cycle counts — multiples of temporal_K=6 (paper Table S1)
        self.available_rounds = [24, 48, 72, 120]
        self.max_rounds = max(self.available_rounds)      # cycles
        self.max_frames = self.max_rounds + 1             # + final data frame

        self._proc_cache: dict[tuple, tuple[stim.Circuit, CircuitLayout]] = {}
        self._sampler_cache: dict[tuple, stim.CompiledDemSampler] = {}
        self._bundle_cache: dict[tuple, _Bundle] = {}

        if basis is None:
            print(
                f"[DATA] d={distance} circuit dataset (v2 aligned frames): "
                f"available_rounds={self.available_rounds}, "
                f"frames={self.available_rounds[0] + 1}..{self.max_frames}",
                flush=True,
            )

    # ------------------------------------------------------------------
    # Circuit caches
    # ------------------------------------------------------------------

    def _get_processed(self, noise: float, T: int) -> tuple[stim.Circuit, CircuitLayout]:
        key = (round(noise, 10), T)
        hit = self._proc_cache.get(key)
        if hit is None:
            hit = _build_processed_circuit(self.distance, T, noise)
            self._proc_cache[key] = hit
        return hit

    def _get_sampler(self, noise: float, T: int) -> stim.CompiledDemSampler:
        key = (round(noise, 10), T)
        cached = self._sampler_cache.get(key)
        if cached is None:
            circuit, _ = self._get_processed(noise, T)
            cached = circuit.detector_error_model().compile_sampler()
            self._sampler_cache[key] = cached
        return cached

    def _get_bundle(self, noise: float, T: int, temporal_K: int) -> _Bundle:
        key = (round(noise, 10), T, temporal_K)
        hit = self._bundle_cache.get(key)
        if hit is not None:
            return hit
        full_plain, layout = self._get_processed(noise, T)
        ancillas = set(layout.slot_of_qubit)
        full, mpp_idx = _insert_noiseless_mpps(full_plain, layout.obs_line,
                                               temporal_K, ancillas)
        C = T // temporal_K
        assert len(mpp_idx) == C, f"expected {C} MPP boundaries, got {len(mpp_idx)}"
        truncs: list[stim.Circuit] = []
        trunc_layouts: list[CircuitLayout] = []
        for c in range(C):
            r = (c + 1) * temporal_K
            if r == T:
                tc, tl = full, layout
            else:
                plain, tl = self._get_processed(noise, r)
                tc, _ = _insert_noiseless_mpps(plain, layout.obs_line,
                                               temporal_K, ancillas)
            truncs.append(tc)
            trunc_layouts.append(tl)
        hit = _Bundle(nq=full.num_qubits, full=full, layout=layout,
                      mpp_rec_indices=mpp_idx, truncs=truncs,
                      trunc_layouts=trunc_layouts)
        self._bundle_cache[key] = hit
        return hit

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _t(x: np.ndarray):
        """numpy -> torch float32 tensor (identity under the verify stub)."""
        return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))

    # ------------------------------------------------------------------
    # Validation batch (DEM — fast, correct marginals)
    # ------------------------------------------------------------------

    def generate_batch(
        self, batch_size: int, rounds: Optional[int] = None,
        scale_factor: float = 1.0,
    ) -> tuple:
        """Validation batch — no auxiliary labels. Returns T+1 frames."""
        T = rounds or self.max_rounds
        noise = 0.0015 * scale_factor
        sampler = self._get_sampler(noise, T)
        _, layout = self._get_processed(noise, T)
        dets_bool, obs_bool, _ = sampler.sample(batch_size)
        m, e = _detectors_to_frames(dets_bool, layout)
        return (
            self._t(m),
            self._t(e),
            self.stabilizer_indices,
            self._t(obs_bool[:, 0:1].astype(np.float32)),
            T + 1,                                    # frames (was: rounds)
        )

    def generate_batch_mixed(
        self, batch_size: int, noise_levels: list[float],
        noise_probs: list[float], rounds: Optional[int] = None,
    ) -> tuple:
        """Training batch without auxiliary labels (per-sample noise, A.2.1)."""
        if self.uniform_rounds and rounds is None:
            T = int(self.rng.choice(self.available_rounds))
        else:
            T = rounds or self.max_rounds

        idx = self.rng.choice(len(noise_levels), size=batch_size, p=noise_probs)
        scales = [float(noise_levels[i]) for i in idx]
        scale_groups: dict[float, list[int]] = {}
        for b, sc in enumerate(scales):
            scale_groups.setdefault(float(sc), []).append(b)

        N = self.n_spatial
        F = T + 1
        m_full = np.zeros((batch_size, F, N), dtype=np.float32)
        e_full = np.zeros((batch_size, F, N), dtype=np.float32)
        tgt_full = np.zeros((batch_size, 1), dtype=np.float32)

        for sc, indices in scale_groups.items():
            noise = 0.0015 * sc
            sampler = self._get_sampler(noise, T)
            _, layout = self._get_processed(noise, T)
            dets_bool, obs_bool, _ = sampler.sample(len(indices))
            m, e = _detectors_to_frames(dets_bool, layout)
            ii = np.asarray(indices, dtype=np.int64)
            m_full[ii] = m
            e_full[ii] = e
            tgt_full[ii, 0] = obs_bool[:, 0].astype(np.float32)

        return self._t(m_full), self._t(e_full), self.stabilizer_indices, self._t(tgt_full), F

    # ------------------------------------------------------------------
    # Training batch WITH auxiliary labels (A.2.3 / A.3.3) — same-shot
    # ------------------------------------------------------------------

    def generate_batch_with_aux(
        self, batch_size: int, noise_levels: list[float],
        noise_probs: list[float], temporal_K: int,
        rounds: Optional[int] = None,
    ) -> dict:
        """Generate a batch with same-shot auxiliary labels.

        One FlipSimulator run of the full experiment (with inline noiseless
        MPPs) yields the main frames, the final observable, and the noiseless
        observable at every chunk boundary.  One run per chunk-boundary
        truncated experiment (same seed → identical noise realisation for the
        shared prefix) yields the fake-ending frames and observables.
        """
        if self.uniform_rounds and rounds is None:
            T = int(self.rng.choice(self.available_rounds))
        else:
            T = rounds or self.max_rounds
        assert T % temporal_K == 0, (
            f"rounds={T} must be a multiple of temporal_K={temporal_K} "
            "(paper trains on multiples of the chunk length)"
        )

        idx = self.rng.choice(len(noise_levels), size=batch_size, p=noise_probs)
        scales = [float(noise_levels[i]) for i in idx]
        scale_groups: dict[float, list[int]] = {}
        for b, sc in enumerate(scales):
            scale_groups.setdefault(float(sc), []).append(b)

        N = self.n_spatial
        F = T + 1                     # frames
        C = T // temporal_K           # chunk boundaries (K, 2K, ..., T)
        S = C + 1                     # aux slots: C boundaries + final-frame chunk

        m_full = np.zeros((batch_size, F, N), dtype=np.float32)
        e_full = np.zeros((batch_size, F, N), dtype=np.float32)
        tgt_full = np.zeros((batch_size, 1), dtype=np.float32)
        pseudo_m = np.zeros((batch_size, S, N), dtype=np.float32)
        pseudo_e = np.zeros((batch_size, S, N), dtype=np.float32)
        pseudo_obs = np.zeros((batch_size, S), dtype=np.float32)
        nl_end = np.zeros((batch_size, S), dtype=np.float32)

        for sc, indices in scale_groups.items():
            noise = 0.0015 * sc
            bundle = self._get_bundle(noise, T, temporal_K)
            # shared seed ⇒ identical noise realisation across full/truncated
            seed = int(self.rng.integers(0, 2 ** 31))
            n_samp = len(indices)
            ii = np.asarray(indices, dtype=np.int64)

            # --- full experiment (detectors + observable + noiseless MPPs) ---
            fsim = stim.FlipSimulator(batch_size=n_samp, num_qubits=bundle.nq, seed=seed)
            fsim.do(bundle.full)
            dets = np.asarray(fsim.get_detector_flips(), dtype=bool).T
            obs = np.asarray(fsim.get_observable_flips(), dtype=bool).T
            rec = np.asarray(fsim.get_measurement_flips(), dtype=bool)
            m, e = _detectors_to_frames(dets, bundle.layout)
            m_full[ii] = m
            e_full[ii] = e
            tgt_full[ii, 0] = obs[:, 0].astype(np.float32)
            for c in range(C):
                nl_end[ii, c] = rec[bundle.mpp_rec_indices[c]].astype(np.float32)

            # --- truncated experiments (fake endings, same shot) ---
            for c in range(C):
                if c == C - 1:
                    # truncation at T is the full experiment itself
                    pm, pe, po = m[:, -1], e[:, -1], obs[:, 0]
                else:
                    fsim_t = stim.FlipSimulator(batch_size=n_samp,
                                                num_qubits=bundle.nq, seed=seed)
                    fsim_t.do(bundle.truncs[c])
                    dets_t = np.asarray(fsim_t.get_detector_flips(), dtype=bool).T
                    obs_t = np.asarray(fsim_t.get_observable_flips(), dtype=bool).T
                    m_t, e_t = _detectors_to_frames(dets_t, bundle.trunc_layouts[c])
                    pm, pe, po = m_t[:, -1], e_t[:, -1], obs_t[:, 0]
                pseudo_m[ii, c] = pm
                pseudo_e[ii, c] = pe
                pseudo_obs[ii, c] = po.astype(np.float32)

            # --- final-frame chunk: the true ending is its own fake ending ---
            pseudo_m[ii, C] = m[:, -1]
            pseudo_e[ii, C] = e[:, -1]
            pseudo_obs[ii, C] = obs[:, 0].astype(np.float32)
            nl_end[ii, C] = nl_end[ii, C - 1]

        # 4. auxiliary targets (Table S3 + A.3.3 semantics)
        nl_begin = np.zeros_like(nl_end)
        nl_begin[:, 1:] = nl_end[:, :-1]
        nl_end_b = nl_end > 0.5
        nl_begin_b = nl_begin > 0.5
        pseudo_obs_b = pseudo_obs > 0.5

        return {
            "m": self._t(m_full),
            "e": self._t(e_full),
            "i_idx": self.stabilizer_indices,
            "targets": self._t(tgt_full),
            "T": F,                                   # frames (model input length)
            "num_cycles": T,                          # cycles (LR formula A.2.6)
            "pseudo_m": self._t(pseudo_m),
            "pseudo_e": self._t(pseudo_e),
            "aux_tgt_pseudo": self._t(pseudo_obs),                        # ①
            "aux_tgt_noiseless": self._t(nl_end.astype(np.float32)),      # ②
            "aux_tgt_noiseless_diff": self._t(                            # ③
                np.logical_xor(nl_end_b, nl_begin_b).astype(np.float32)),
            "aux_tgt_noiseless_to_inter": self._t(                        # ④
                np.logical_xor(nl_begin_b, pseudo_obs_b).astype(np.float32)),
        }
