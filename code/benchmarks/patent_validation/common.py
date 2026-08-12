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
"""Small reproducibility helpers shared by patent-validation benchmarks."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

try:
    import yaml
except ImportError:  # pragma: no cover - reported by the benchmark preflight
    yaml = None


REPO_ROOT = Path(__file__).resolve().parents[3]


def load_yaml(path: Path) -> dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required to load patent-validation configs")
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a mapping in {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}) if rows else ["status"]
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows or [{"status": "no_results"}])
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _command_output(args: list[str]) -> str:
    try:
        return subprocess.run(
            args,
            cwd=REPO_ROOT,
            check=False,
            text=True,
            capture_output=True,
            timeout=15,
        ).stdout.strip()
    except Exception as exc:  # pragma: no cover - metadata only
        return f"unavailable: {exc}"


def environment_manifest(
    config_path: Path, mode: str, seed: int, *, evidence_scope: str
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "mode": mode,
        "seed": int(seed),
        "evidence_scope": evidence_scope,
        "config": str(config_path.relative_to(REPO_ROOT)),
        "config_sha256": sha256_file(config_path),
        "git_commit": _command_output(["git", "rev-parse", "HEAD"]),
        "git_status": _command_output(["git", "status", "--short"]),
        "python": os.sys.version,
        "executable": os.sys.executable,
        "prefix": os.sys.prefix,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_devices": [
            torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
        ],
        "sampling_backend": "torch_dem_gf2",
        "started_unix": time.time(),
        "completed_unix": None,
        "status": "running",
        "failures": [],
    }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def wilson_interval(
    errors: int, shots: int, z: float = 1.959963984540054
) -> tuple[float, float]:
    if shots <= 0:
        return float("nan"), float("nan")
    probability = errors / shots
    denominator = 1 + z * z / shots
    center = (probability + z * z / (2 * shots)) / denominator
    radius = (
        z
        * math.sqrt(
            probability * (1 - probability) / shots + z * z / (4 * shots * shots)
        )
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def paired_bootstrap_interval(
    left_error: np.ndarray,
    right_error: np.ndarray,
    *,
    seed: int,
    repeats: int = 1000,
) -> tuple[float, float]:
    if left_error.shape != right_error.shape or left_error.size == 0:
        return float("nan"), float("nan")
    generator = np.random.default_rng(seed)
    delta = left_error.astype(np.int8) - right_error.astype(np.int8)
    support, counts = np.unique(delta, return_counts=True)
    draws = generator.multinomial(delta.size, counts / delta.size, size=repeats)
    values = draws @ support.astype(np.float64) / delta.size
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


@dataclass(frozen=True)
class SurfaceTask:
    name: str
    after_clifford_depolarization: float
    before_round_data_depolarization: float
    before_measure_flip_probability: float
    after_reset_flip_probability: float


@dataclass
class DemBundle:
    circuit: Any
    dem: Any
    h: torch.Tensor
    logical: torch.Tensor
    probability: torch.Tensor
    detector_coordinates: dict[int, tuple[float, ...]]


def build_surface_bundle(
    distance: int, rounds: int, basis: str, task: SurfaceTask
) -> DemBundle:
    import stim

    circuit = stim.Circuit.generated(
        f"surface_code:rotated_memory_{basis.lower()}",
        distance=int(distance),
        rounds=int(rounds),
        after_clifford_depolarization=float(task.after_clifford_depolarization),
        before_round_data_depolarization=float(task.before_round_data_depolarization),
        before_measure_flip_probability=float(task.before_measure_flip_probability),
        after_reset_flip_probability=float(task.after_reset_flip_probability),
    )
    dem = circuit.detector_error_model(
        decompose_errors=True,
        approximate_disjoint_errors=True,
        ignore_decomposition_failures=True,
    ).flattened()
    columns: list[list[int]] = []
    logical: list[int] = []
    probability: list[float] = []
    for instruction in dem:
        if instruction.type != "error":
            continue
        detectors: list[int] = []
        logical_bit = 0
        for target in instruction.targets_copy():
            if target.is_relative_detector_id():
                detectors.append(int(target.val))
            elif target.is_logical_observable_id() and int(target.val) == 0:
                logical_bit ^= 1
        if not detectors and logical_bit == 0:
            continue
        columns.append(detectors)
        logical.append(logical_bit)
        probability.append(float(instruction.args_copy()[0]))
    h = torch.zeros((circuit.num_detectors, len(columns)), dtype=torch.uint8)
    for column, detectors in enumerate(columns):
        if detectors:
            h[torch.tensor(detectors, dtype=torch.long), column] = 1
    return DemBundle(
        circuit=circuit,
        dem=dem,
        h=h,
        logical=torch.tensor(logical, dtype=torch.uint8),
        probability=torch.tensor(probability, dtype=torch.float32),
        detector_coordinates={
            int(key): tuple(float(item) for item in value)
            for key, value in circuit.get_detector_coordinates().items()
        },
    )


def sample_dem_bundle(
    bundle: DemBundle,
    shots: int,
    *,
    seed: int,
    device: torch.device,
    shot_chunk: int = 4096,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample independent DEM mechanisms and return detectors/observable/faults."""

    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    h = bundle.h.to(device=device, dtype=torch.float32)
    logical = bundle.logical.to(device=device, dtype=torch.float32)
    probability = bundle.probability.to(device=device)
    detector_batches: list[torch.Tensor] = []
    observable_batches: list[torch.Tensor] = []
    fault_batches: list[torch.Tensor] = []
    remaining = int(shots)
    while remaining > 0:
        count = min(remaining, int(shot_chunk))
        faults = (
            torch.rand((count, probability.numel()), generator=generator, device=device)
            < probability
        )
        faults_float = faults.to(torch.float32)
        detector_batches.append(torch.remainder(faults_float @ h.t(), 2).to(torch.uint8).cpu())
        observable_batches.append(
            torch.remainder(faults_float @ logical, 2).to(torch.uint8).cpu()
        )
        fault_batches.append(faults.to(torch.uint8).cpu())
        remaining -= count
    return (
        torch.cat(detector_batches),
        torch.cat(observable_batches),
        torch.cat(fault_batches),
    )


def detectors_to_grid(detectors: torch.Tensor, distance: int, rounds: int) -> torch.Tensor:
    batch = detectors.shape[0]
    width = int(distance) * int(distance)
    expected = int(rounds) * (width - 1)
    if detectors.shape[1] != expected:
        raise ValueError(
            f"expected {expected} detectors for d={distance}, r={rounds}; "
            f"got {detectors.shape[1]}"
        )
    padded = torch.zeros((batch, rounds, width), dtype=torch.float32)
    padded[:, :, : width - 1] = detectors.reshape(batch, rounds, width - 1).float()
    return padded.reshape(batch, 1, rounds, distance, distance)


class CandidateCorrectionNet(torch.nn.Module):
    """Small reference model used only for the DEM-action proxy phase."""

    def __init__(self, num_candidates: int, channels: int = 32):
        super().__init__()
        blocks: list[torch.nn.Module] = []
        current = 1
        for _ in range(4):
            blocks.extend(
                [
                    torch.nn.Conv3d(current, channels, 3, padding=1),
                    torch.nn.GroupNorm(4 if channels >= 4 else 1, channels),
                    torch.nn.SiLU(),
                ]
            )
            current = channels
        self.features = torch.nn.Sequential(*blocks)
        self.head = torch.nn.Linear(channels, int(num_candidates))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        features = self.features(value).mean(dim=(2, 3, 4))
        return self.head(features)


def tasks_from_config(config: dict[str, Any]) -> list[SurfaceTask]:
    return [SurfaceTask(**item) for item in config["tasks"]]


def batch_indices(size: int, batch_size: int, *, seed: int) -> Iterable[torch.Tensor]:
    generator = torch.Generator().manual_seed(int(seed))
    permutation = torch.randperm(size, generator=generator)
    for start in range(0, size, batch_size):
        yield permutation[start : start + batch_size]


def render_results_markdown(
    path: Path,
    *,
    title: str,
    rows: list[dict[str, Any]],
    conclusion: str,
    limitations: list[str],
) -> None:
    columns = [
        key
        for key in (
            "method",
            "task",
            "basis",
            "ler",
            "ler_ci_low",
            "ler_ci_high",
            "shots",
            "residual_density",
            "topology_complexity",
            "end_to_end_us_per_shot",
        )
        if any(key in row for row in rows)
    ]
    lines = [f"# {title}", "", conclusion, ""]
    if rows and columns:
        lines.extend(
            [
                "| " + " | ".join(columns) + " |",
                "| " + " | ".join(["---"] * len(columns)) + " |",
            ]
        )
        for row in rows:
            values = []
            for column in columns:
                value = row.get(column, "")
                values.append(f"{value:.6g}" if isinstance(value, float) else str(value))
            lines.append("| " + " | ".join(values) + " |")
    else:
        lines.append("尚无成功完成的结果单元。")
    lines.extend(["", "## 局限", ""] + [f"- {item}" for item in limitations])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
