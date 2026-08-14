"""Fresh-shot surface-code data for the strict structured workflow.

The sampler deliberately wraps the same ``QCDataGeneratorTorch`` used by the
original Ising-fast trainer.  For endpoint supervision/evaluation it asks the
wrapped ``MemoryCircuitTorch`` for auxiliary frames and converts those frames
to Stim detectors from the *same physical shots*.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

from common import repo_path
from data.generator_torch import QCDataGeneratorTorch
from generate_labeled_dataset import _measurements_to_dets_and_obs
from qec.noise_model import NoiseModel
from qec.surface_code.memory_circuit import MemoryCircuit
from qec.surface_code.stim_sample_io import normalize_code_rotation


STREAM_OFFSETS = {"train": 0, "validation": 100_000_000, "test": 200_000_000}


@dataclass
class StrictBatch:
    train_x: torch.Tensor
    train_y: torch.Tensor
    basis: str
    dets_and_obs: np.ndarray | None = None


def _resolved_noise_model(noise_config: Any) -> tuple[NoiseModel | None, dict[str, Any]]:
    raw = OmegaConf.select(noise_config, "data.noise_model")
    if raw is None:
        p_error = OmegaConf.select(noise_config, "data.p_error")
        p_max = OmegaConf.select(noise_config, "data.p_max")
        p = float(p_error if p_error is not None else p_max)
        return None, {"kind": "simple", "p_error": p}
    params = OmegaConf.to_container(raw, resolve=True)
    if not isinstance(params, dict):
        raise ValueError("reference data.noise_model must be a mapping")
    model = NoiseModel.from_config_dict(params)
    return model, {
        "kind": "noise_model",
        "parameters": model.canonical_parameters(),
        "sha256": model.sha256(),
    }


def _choose(override: dict[str, Any], name: str, fallback: Any) -> Any:
    """Use an explicit strict override, including an explicit null."""
    return override[name] if name in override else fallback


class StrictSurfaceSampler:
    """Generate reproducible, non-overlapping train/validation/test streams."""

    def __init__(self, cfg: dict[str, Any], device: torch.device, *, session_seed: int | None = None):
        reference_path = repo_path(cfg["reference_config"])
        self.reference_path = reference_path
        reference = OmegaConf.load(reference_path)
        noise_config_path = repo_path(cfg.get("noise_config", cfg["reference_config"]))
        self.noise_config_path = noise_config_path
        noise_config = OmegaConf.load(noise_config_path)
        self.device = device
        self.distance = int(_choose(cfg, "distance", OmegaConf.select(reference, "distance")))
        self.n_rounds = int(_choose(cfg, "n_rounds", OmegaConf.select(reference, "n_rounds")))
        self.measure_basis = str(_choose(cfg, "measure_basis", OmegaConf.select(reference, "meas_basis"))).lower()
        self.code_rotation = normalize_code_rotation(str(
            _choose(cfg, "code_rotation", OmegaConf.select(reference, "data.code_rotation") or "O1")
        ))
        if self.measure_basis not in {"x", "z", "both", "mixed"}:
            raise ValueError("strict measure_basis must be X, Z, or both")
        self.bases = ("X", "Z") if self.measure_basis in {"both", "mixed"} else (self.measure_basis.upper(),)
        self.noise_model, self.noise_metadata = _resolved_noise_model(noise_config)
        if "p_error" in cfg:
            if self.noise_model is not None:
                raise ValueError("strict p_error override cannot be combined with a reference noise_model")
            self.noise_metadata = {"kind": "simple", "p_error": float(cfg["p_error"])}

        data = reference.get("data", {})
        if self.noise_model is None:
            p_error = cfg.get("p_error", data.get("p_error"))
            p_min = _choose(cfg, "p_min", data.get("p_min"))
            p_max = _choose(cfg, "p_max", data.get("p_max"))
        else:
            # ``p_error`` is only a structural DEM placeholder in 25-parameter
            # mode. The actual probability vector comes entirely from the
            # NoiseModel, so do not leak p_max=0.006 from the count/HE reference.
            p_error = float(self.noise_model.get_max_probability())
            p_min = p_max = None
        precomputed = _choose(cfg, "precomputed_frames_dir", data.get("precomputed_frames_dir"))
        if precomputed is not None:
            precomputed = str(repo_path(precomputed)) if not Path(str(precomputed)).is_absolute() else str(precomputed)
        seed_value = cfg.get("session_seed") if session_seed is None else session_seed
        self.session_seed = random.SystemRandom().randrange(1, 2**31) if seed_value is None else int(seed_value)

        self.generator = QCDataGeneratorTorch(
            distance=self.distance,
            n_rounds=self.n_rounds,
            p_error=p_error,
            p_min=p_min,
            p_max=p_max,
            measure_basis=self.measure_basis,
            rank=0,
            global_rank=0,
            mode="train",
            verbose=bool(cfg.get("verbose_generator", True)),
            timelike_he=bool(_choose(cfg, "timelike_he", data.get("timelike_he", True))),
            num_he_cycles=int(_choose(cfg, "num_he_cycles", data.get("num_he_cycles", 1))),
            use_weight2=bool(_choose(cfg, "use_weight2", data.get("use_weight2", False))),
            max_passes_w1=int(_choose(cfg, "max_passes_w1", data.get("max_passes_w1", 32))),
            max_passes_w2=int(_choose(cfg, "max_passes_w2", data.get("max_passes_w2", 32))),
            precomputed_frames_dir=precomputed,
            code_rotation=self.code_rotation,
            noise_model=self.noise_model,
            base_seed=self.session_seed,
            device=device,
            use_compile=bool(_choose(cfg, "use_compile", data.get("use_compile", False))),
            compile_chunk_size=int(_choose(cfg, "compile_chunk_size", data.get("compile_chunk_size", 2))),
            use_coset_search=bool(_choose(cfg, "use_coset_search", data.get("use_coset_search", False))),
            coset_max_generators=int(_choose(cfg, "coset_max_generators", data.get("coset_max_generators", 20))),
            use_dense_overlap=bool(_choose(cfg, "use_dense_overlap", data.get("use_dense_overlap", False))),
            use_parallel_spacelike=bool(_choose(
                cfg, "use_parallel_spacelike", data.get("use_parallel_spacelike", False)
            )),
        )
        p_placeholder = (
            float(self.noise_model.get_max_probability())
            if self.noise_model is not None else float(self.noise_metadata["p_error"])
        )
        self.stim_circuits: dict[str, Any] = {}
        for basis in self.bases:
            wrapper = MemoryCircuit(
                distance=self.distance,
                idle_error=p_placeholder,
                sqgate_error=p_placeholder,
                tqgate_error=p_placeholder,
                spam_error=(2.0 / 3.0) * p_placeholder,
                n_rounds=self.n_rounds,
                basis=basis,
                code_rotation=self.code_rotation,
                noise_model=self.noise_model,
                add_boundary_detectors=True,
            )
            wrapper.set_error_rates()
            self.stim_circuits[basis] = wrapper.stim_circuit

    def metadata(self, basis: str | None = None) -> dict[str, Any]:
        basis = (basis or self.bases[0]).upper()
        circuit = self.stim_circuits[basis]
        return {
            "schema_version": 1,
            "artifact": "strict_online_surface_code",
            "num_samples": None,
            "distance": self.distance,
            "n_rounds": self.n_rounds,
            "basis": basis,
            "code_rotation": self.code_rotation,
            "num_detectors": int(circuit.num_detectors),
            "num_observables": int(circuit.num_observables),
            "seed": self.session_seed,
            "noise": self.noise_metadata,
            "reference_config": str(self.reference_path),
            "noise_config": str(self.noise_config_path),
        }

    def _simulator(self, basis: str):
        if len(self.bases) == 2:
            return self.generator.sim_X if basis == "X" else self.generator.sim_Z
        return self.generator.sim

    def generate(self, *, stream: str, step: int, batch_size: int, with_endpoint: bool) -> StrictBatch:
        if stream not in STREAM_OFFSETS:
            raise ValueError(f"unknown strict stream: {stream}")
        basis = self.bases[int(step) % len(self.bases)]
        # Explicit seeds make the split disjoint and make every epoch auditable.
        seed = self.session_seed + STREAM_OFFSETS[stream] + int(step)
        simulator = self._simulator(basis)
        if not with_endpoint:
            train_x, train_y = simulator.generate_batch(batch_size=batch_size, seed=seed)
            return StrictBatch(train_x=train_x, train_y=train_y, basis=basis)
        train_x, train_y, meas_old, x_cum, z_cum = simulator.generate_batch(
            batch_size=batch_size, return_aux=True, seed=seed
        )
        dets_and_obs = _measurements_to_dets_and_obs(
            self.stim_circuits[basis], simulator.code, meas_old, x_cum, z_cum
        )
        return StrictBatch(train_x=train_x, train_y=train_y, basis=basis, dets_and_obs=dets_and_obs)


def strict_sample_counts(cfg: dict[str, Any]) -> tuple[int, int, int]:
    """Return train/validation/test counts, defaulting to the reference YAML."""
    reference = OmegaConf.load(repo_path(cfg["reference_config"]))
    train = int(cfg.get("train_num_samples_per_epoch", OmegaConf.select(reference, "train.num_samples")))
    validation = int(cfg.get("validation_num_samples", OmegaConf.select(reference, "val.num_samples")))
    test = int(cfg.get("inference_num_samples", OmegaConf.select(reference, "test.num_samples")))
    if min(train, validation, test) <= 0:
        raise ValueError("strict sample counts must be positive")
    return train, validation, test
