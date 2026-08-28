# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public batch=1 surface-decoder API."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter_ns
from typing import Any

import numpy as np
import pymatching
import stim
import torch

from .config import DecoderConfig
from .geometry import load_geometry
from .model import HTnet
from .pipeline import PreDecoderPipeline


@dataclass(frozen=True)
class StageLatencies:
    """Hot-path stage timings in microseconds."""

    validation_us: float
    host_to_device_us: float
    predecoder_us: float
    device_to_host_us: float
    pymatching_us: float
    total_us: float


@dataclass(frozen=True)
class DecodeResult:
    """One basis-specific logical-frame prediction."""

    logical_flip: int
    predecoder_flip: int
    pymatching_flip: int
    status: str
    stage_latencies: StageLatencies
    residual: np.ndarray | None = None


def _load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    raw = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(raw, dict):
        raise ValueError(
            f"Unexpected checkpoint format: expected dict, got {type(raw).__name__}"
        )
    if "model_state_dict" in raw:
        state = raw["model_state_dict"]
    elif "state_dict" in raw:
        state = raw["state_dict"]
    else:
        state = raw
    return {
        key[len("module.") :] if key.startswith("module.") else key: value
        for key, value in state.items()
    }


def _load_matcher(path: Path) -> pymatching.Matching:
    if not path.is_file():
        raise FileNotFoundError(f"Detector error model not found: {path}")
    dem = stim.DetectorErrorModel.from_file(path)
    return pymatching.Matching.from_detector_error_model(dem)


class SurfaceDecoderCore:
    """Long-lived d=9/T=9 HTnet + PyMatching decoder for one basis."""

    def __init__(
        self,
        config: DecoderConfig,
        *,
        pipeline: torch.nn.Module | None = None,
        matcher: Any | None = None,
    ):
        self.config = config
        self.device = torch.device(config.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA device {config.device!r} was requested but CUDA is unavailable"
            )
        if pipeline is None:
            model = HTnet()
            model.load_state_dict(_load_state_dict(config.checkpoint_path), strict=True)
            model.eval()
            geometry = load_geometry(config.geometry_path)
            pipeline = PreDecoderPipeline(
                model,
                geometry,
                basis=config.basis,
                distance=config.distance,
                n_rounds=config.n_rounds,
            )
        self.pipeline = pipeline.to(self.device).eval()
        self.matcher = _load_matcher(config.dem_path) if matcher is None else matcher

    def _synchronize(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _validate_detectors(self, detectors: np.ndarray) -> np.ndarray:
        if not isinstance(detectors, np.ndarray):
            raise TypeError("detectors must be a numpy.ndarray")
        if detectors.dtype != np.uint8:
            raise TypeError(f"detectors must have dtype uint8, got {detectors.dtype}")
        if detectors.ndim != 1 or detectors.shape[0] != self.config.num_detectors:
            raise ValueError(
                "detectors must have shape "
                f"({self.config.num_detectors},), got {detectors.shape}"
            )
        if np.any(detectors > 1):
            raise ValueError("detectors must contain only binary values 0 or 1")
        return np.ascontiguousarray(detectors)

    def decode_detectors(
        self,
        detectors: np.ndarray,
        *,
        return_residual: bool | None = None,
    ) -> DecodeResult:
        """Decode one detector window and return its logical-frame prediction."""
        total_start = perf_counter_ns()
        stage_start = total_start
        validated = self._validate_detectors(detectors)
        after_validation = perf_counter_ns()

        detector_tensor = torch.from_numpy(validated).unsqueeze(0).to(self.device)
        self._synchronize()
        after_h2d = perf_counter_ns()

        with torch.inference_mode():
            pipeline_output = self.pipeline(detector_tensor)
        self._synchronize()
        after_predecoder = perf_counter_ns()

        predecoder_flip = int(pipeline_output[0, 0].to("cpu").item())
        residual = np.ascontiguousarray(
            pipeline_output[0, 1:].to("cpu").numpy(),
            dtype=np.uint8,
        )
        after_d2h = perf_counter_ns()

        prediction = np.asarray(self.matcher.decode(residual)).reshape(-1)
        pymatching_flip = int(prediction[0])
        after_pymatching = perf_counter_ns()
        logical_flip = predecoder_flip ^ pymatching_flip

        keep_residual = (
            self.config.return_residual
            if return_residual is None
            else bool(return_residual)
        )
        latencies = StageLatencies(
            validation_us=(after_validation - stage_start) / 1_000.0,
            host_to_device_us=(after_h2d - after_validation) / 1_000.0,
            predecoder_us=(after_predecoder - after_h2d) / 1_000.0,
            device_to_host_us=(after_d2h - after_predecoder) / 1_000.0,
            pymatching_us=(after_pymatching - after_d2h) / 1_000.0,
            total_us=(after_pymatching - total_start) / 1_000.0,
        )
        return DecodeResult(
            logical_flip=logical_flip,
            predecoder_flip=predecoder_flip,
            pymatching_flip=pymatching_flip,
            status="ok",
            stage_latencies=latencies,
            residual=residual.copy() if keep_residual else None,
        )
