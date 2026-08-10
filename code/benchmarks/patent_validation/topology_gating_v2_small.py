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
"""Reproducible four-channel GO/NO-GO experiment for topology gating v2.

This entry point intentionally lives outside the public training workflow.  It
uses the production four-channel surface model and data/circuit machinery, but
keeps patent-only checkpoint selection, gating, ablations, and evidence output
isolated under ``outputs/patent_validation``.
"""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import subprocess
import time
from types import SimpleNamespace
from typing import Any, Iterable

import numpy as np
import torch

from benchmarks.patent_validation.common import (
    REPO_ROOT,
    atomic_json,
    load_yaml,
    paired_bootstrap_interval,
    sha256_file,
    wilson_interval,
    write_csv,
)
from data.generator_torch import QCDataGeneratorTorch
from evaluation.surface_topology_adapter import (
    SurfaceActionAdapter,
    build_surface_action_adapter,
)
from evaluation.topology_gating_v2 import (
    DATA_X,
    DATA_Z,
    MEASUREMENT_X,
    MEASUREMENT_Z,
    GateConfig,
    GateDecision,
    GateResult,
    apply_actions,
    config_from_mapping,
    evaluate_cluster,
    gate_actions,
    interaction_clusters,
    pointwise_actions,
    typed_candidates,
    workload_features,
    workload_score,
)
from model.predecoder import PreDecoderModelMemory_v1
from qec.noise_model import NoiseModel, get_training_upscaled_noise_model
from qec.precompute_dem import (
    dem_artifact_metadata_matches,
    load_dem_artifact_metadata,
    precompute_dem_bundle_surface_code,
)
from qec.surface_code.detector_input import SurfaceDetectorInputTransform
from qec.surface_code.memory_circuit import MemoryCircuit
from training.optimizers import Lion
from training.topology_loss import TopologyLossWeights, topology_joint_loss


DEFAULT_CONFIG = REPO_ROOT / "conf/experiments/patent/topology_gating_v2_small.yaml"
DEFAULT_OUTPUT = REPO_ROOT / "outputs/patent_validation/topology_gating_v2_small"
LOSS_KINDS = ("bce", "topology")
EVIDENCE_SCOPE = "phase2_single_distance_enhanced_pilot"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def resolved_config(path: Path, mode: str) -> dict[str, Any]:
    raw = load_yaml(path)
    modes = raw.pop("modes", {})
    if mode == "full":
        value = raw
    else:
        if mode not in modes:
            raise ValueError(f"mode {mode!r} is not declared in {path}")
        value = _deep_merge(raw, modes[mode])
    if value.get("evidence_scope") != EVIDENCE_SCOPE:
        raise ValueError("unexpected evidence_scope in experiment config")
    return value


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _git_output(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, check=False, capture_output=True, text=True
    ).stdout.strip()


def _identity(config: dict[str, Any], mode: str, **fields: Any) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "config_sha256": _canonical_hash(config),
        "git_commit": _git_output("rev-parse", "HEAD"),
        "mode": mode,
        **fields,
    }


def _manifest(identity: dict[str, Any], status: str = "running") -> dict[str, Any]:
    return {
        **identity,
        "evidence_scope": EVIDENCE_SCOPE,
        "status": status,
        "git_status": _git_output("status", "--short"),
        "python": os.sys.version,
        "executable": os.sys.executable,
        "prefix": os.sys.prefix,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_devices": [
            torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
        ],
        "started_unix": time.time(),
        "completed_unix": None,
        "failures": [],
    }


def _resume_complete(
    manifest_path: Path,
    identity: dict[str, Any],
    artifacts: Iterable[Path],
    resume: bool,
) -> bool:
    if not resume or not manifest_path.exists():
        return False
    previous = json.loads(manifest_path.read_text(encoding="utf-8"))
    previous_identity = {key: previous.get(key) for key in identity}
    if previous_identity != identity:
        raise RuntimeError(
            f"resume identity mismatch in {manifest_path}; use a new output directory"
        )
    if previous.get("status") != "completed":
        return False
    missing = [str(path) for path in artifacts if not path.exists()]
    if missing:
        raise RuntimeError(f"completed manifest has missing artifacts: {missing}")
    recorded_hashes = previous.get("artifact_sha256")
    if not isinstance(recorded_hashes, dict):
        raise RuntimeError(f"completed manifest has no artifact hashes: {manifest_path}")
    for path in artifacts:
        expected = recorded_hashes.get(str(path))
        if expected is None:
            raise RuntimeError(f"completed manifest has no hash for artifact: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(
                f"artifact hash mismatch for {path}: expected {expected}, got {actual}"
            )
    return True


def _artifact_hashes(paths: Iterable[Path]) -> dict[str, str]:
    """Hash completed files using their invocation-stable path spellings."""

    return {str(path): sha256_file(path) for path in paths}


def _array_sha256(*values: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in values:
        array = np.ascontiguousarray(value)
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(json.dumps(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _complete_manifest(path: Path, manifest: dict[str, Any], **values: Any) -> None:
    manifest.update(status="completed", completed_unix=time.time(), **values)
    atomic_json(path, manifest)


def _fail_manifest(path: Path, manifest: dict[str, Any], exc: BaseException) -> None:
    manifest.update(status="failed", completed_unix=time.time())
    manifest["failures"].append(
        {"type": type(exc).__name__, "message": str(exc)}
    )
    atomic_json(path, manifest)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def _noise_mapping(config: dict[str, Any], task: str) -> dict[str, float]:
    raw = deepcopy(config["noise_tasks"][task])
    parent = raw.pop("inherit", None)
    overrides = raw.pop("overrides", {})
    if parent is not None:
        raw = _noise_mapping(config, str(parent))
    raw.update(overrides)
    return {str(key): float(value) for key, value in raw.items()}


def _noise_model(config: dict[str, Any], task: str) -> NoiseModel:
    return NoiseModel.from_config_dict(_noise_mapping(config, task))


def _geometry(config: dict[str, Any]) -> tuple[int, int, str]:
    geometry = config["geometry"]
    return int(geometry["distance"]), int(geometry["rounds"]), str(
        geometry["rotation"]
    ).upper()


def _device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is unavailable")
    return device


def _model_config(config: dict[str, Any]) -> SimpleNamespace:
    distance, rounds, _ = _geometry(config)
    model = config["model"]
    return SimpleNamespace(
        distance=distance,
        n_rounds=rounds,
        model=SimpleNamespace(
            dropout_p=float(model["dropout_p"]),
            activation=str(model["activation"]),
            num_filters=[int(value) for value in model["num_filters"]],
            kernel_size=[int(value) for value in model["kernel_size"]],
            input_channels=int(model["input_channels"]),
            out_channels=int(model["out_channels"]),
        ),
    )


def _model(config: dict[str, Any], device: torch.device) -> torch.nn.Module:
    return PreDecoderModelMemory_v1(_model_config(config)).to(device)


def _state_hash(model: torch.nn.Module) -> str:
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def _save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int,
    seed: int,
    loss_kind: str,
    identity: dict[str, Any],
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": int(epoch),
            "seed": int(seed),
            "loss_kind": loss_kind,
            "identity": identity,
            "rng_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            },
        },
        temporary,
    )
    os.replace(temporary, path)
    return sha256_file(path)

def _restore_rng_state(value: dict[str, Any]) -> None:
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    torch.set_rng_state(value["torch"].cpu())
    if torch.cuda.is_available() and value.get("cuda"):
        torch.cuda.set_rng_state_all([state.cpu() for state in value["cuda"]])

def _load_checkpoint(
    path: Path, config: dict[str, Any], device: torch.device
) -> torch.nn.Module:
    payload = torch.load(path, map_location=device, weights_only=False)
    model = _model(config, device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model


def _generator(
    config: dict[str, Any],
    *,
    noise_model: NoiseModel,
    seed: int,
    device: torch.device,
    dem_dir: Path,
) -> QCDataGeneratorTorch:
    distance, rounds, rotation = _geometry(config)
    return QCDataGeneratorTorch(
        distance=distance,
        n_rounds=rounds,
        measure_basis="both",
        mode="train",
        timelike_he=True,
        num_he_cycles=1,
        use_weight2=False,
        max_passes_w1=8,
        max_passes_w2=4,
        decompose_y=False,
        precomputed_frames_dir=str(dem_dir),
        code_rotation=rotation,
        noise_model=noise_model,
        base_seed=int(seed),
        device=device,
        use_compile=False,
    )


def prepare(config: dict[str, Any], args: argparse.Namespace, output: Path) -> None:
    identity = _identity(config, args.mode, unit="prepare")
    root = output / args.mode
    dem_dir = root / "dem"
    manifest_path = dem_dir / "manifest.json"
    resolved_path = root / "resolved_config.json"
    distance, rounds, rotation = _geometry(config)
    expected = [
        dem_dir / f"surface_d{distance}_r{rounds}_{basis}_frame_predecoder.{suffix}.npz"
        for basis in ("X", "Z")
        for suffix in ("X", "Z", "p", "A")
    ]
    if _resume_complete(manifest_path, identity, [resolved_path, *expected], args.resume):
        print("[resume] prepare")
        return
    manifest = _manifest(identity)
    atomic_json(resolved_path, config)
    atomic_json(manifest_path, manifest)
    try:
        device = _device(args.device)
        registered = _noise_model(config, "t0")
        training_noise, upscale = get_training_upscaled_noise_model(
            registered, code_type="surface_code"
        )
        for basis in ("X", "Z"):
            precompute_dem_bundle_surface_code(
                distance=distance,
                n_rounds=rounds,
                basis=basis,
                code_rotation=rotation,
                p_scalar=float(training_noise.get_max_probability()),
                dem_output_dir=str(dem_dir),
                device=device,
                export=True,
                noise_model=training_noise,
            )
            prefix = dem_dir / f"surface_d{distance}_r{rounds}_{basis}_frame_predecoder"
            metadata = load_dem_artifact_metadata(prefix.with_suffix(".p.npz"))
            ok, reason = dem_artifact_metadata_matches(
                metadata,
                distance=distance,
                n_rounds=rounds,
                basis=basis,
                code_rotation=rotation,
                p_scalar=float(training_noise.get_max_probability()),
                noise_model=training_noise,
            )
            if not ok:
                raise RuntimeError(f"DEM metadata validation failed for {basis}: {reason}")
        hashes = {path.name: sha256_file(path) for path in expected}
        _complete_manifest(
            manifest_path,
            manifest,
            dem_sha256=hashes,
            artifact_sha256=_artifact_hashes([resolved_path, *expected]),
            registered_noise_sha256=registered.sha256(),
            training_noise_sha256=training_noise.sha256(),
            training_upscale=upscale,
        )
    except Exception as exc:
        _fail_manifest(manifest_path, manifest, exc)
        raise


def _loss_weights(config: dict[str, Any], loss_kind: str) -> TopologyLossWeights:
    if loss_kind == "bce":
        return TopologyLossWeights()
    value = config["training"]["topology_loss"]
    return TopologyLossWeights(
        topology=float(value["topology"]),
        logical_frame=float(value["logical_frame"]),
        calibration=float(value["calibration"]),
    )


def _validation_loss(
    model: torch.nn.Module,
    generator: QCDataGeneratorTorch,
    adapters: dict[str, SurfaceActionAdapter],
    config: dict[str, Any],
    loss_kind: str,
    *,
    epoch: int,
) -> dict[str, float]:
    training = config["training"]
    total_shots = int(training["checkpoint_validation_shots"])
    batch_size = int(training["batch_size"])
    batches = math.ceil(total_shots / batch_size)
    weights = _loss_weights(config, loss_kind)
    sums = {key: 0.0 for key in ("local", "topology", "logical_frame", "calibration", "total")}
    seen = 0
    model.eval()
    with torch.no_grad():
        for batch_index in range(batches):
            count = min(batch_size, total_shots - seen)
            step = epoch * 100000 + batch_index
            train_x, targets = generator.generate_batch(step=step, batch_size=count)
            basis = generator.get_current_basis(step)
            logits = model(train_x)
            _, components = topology_joint_loss(
                logits,
                targets,
                train_x,
                adapters[basis],
                weights=weights,
                positive_weight=float(training["positive_weight"]),
            )
            for key in sums:
                sums[key] += float(components[key].detach()) * count
            seen += count
    return {key: value / max(1, seen) for key, value in sums.items()}


def train(config: dict[str, Any], args: argparse.Namespace, output: Path) -> None:
    if args.loss_kind not in LOSS_KINDS:
        raise ValueError(f"loss_kind must be one of {LOSS_KINDS}")
    identity = _identity(
        config,
        args.mode,
        unit="train",
        seed=int(args.seed),
        loss_kind=args.loss_kind,
    )
    root = output / args.mode
    job = root / "train" / args.loss_kind / f"seed_{int(args.seed)}"
    manifest_path = job / "manifest.json"
    history_path = job / "history.csv"
    checkpoint_epochs = [int(value) for value in config["training"]["checkpoint_epochs"]]
    checkpoint_paths = [job / "checkpoints" / f"epoch_{epoch:03d}.pt" for epoch in checkpoint_epochs]
    if _resume_complete(
        manifest_path, identity, [history_path, *checkpoint_paths], args.resume
    ):
        print(f"[resume] train {args.loss_kind} seed={args.seed}")
        return
    previous_manifest = None
    if args.resume and manifest_path.exists():
        previous_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = _manifest(identity)
    job.mkdir(parents=True, exist_ok=True)
    atomic_json(manifest_path, manifest)
    try:
        device = _device(args.device)
        _seed_everything(int(args.seed))
        model = _model(config, device)
        initial_hash = _state_hash(model)
        training = config["training"]
        optimizer_cfg = training["optimizer"]
        optimizer = Lion(
            model.parameters(),
            lr=float(optimizer_cfg["learning_rate"]),
            betas=(float(optimizer_cfg["beta1"]), float(optimizer_cfg["beta2"])),
            weight_decay=float(optimizer_cfg["weight_decay"]),
        )
        registered = _noise_model(config, "t0")
        training_noise, upscale = get_training_upscaled_noise_model(
            registered, code_type="surface_code"
        )
        train_generator = _generator(
            config,
            noise_model=training_noise,
            seed=int(args.seed),
            device=device,
            dem_dir=root / "dem",
        )
        val_generator = _generator(
            config,
            noise_model=registered,
            seed=int(args.seed) + int(config["seed_offsets"]["checkpoint_validation"]),
            device=device,
            dem_dir=root / "dem",
        )
        distance, rounds, rotation = _geometry(config)
        adapters = {
            basis: build_surface_action_adapter(distance, rounds, basis, rotation)
            for basis in ("X", "Z")
        }
        weights = _loss_weights(config, args.loss_kind)
        epochs = int(training["epochs"])
        batch_size = int(training["batch_size"])
        samples = int(training["samples_per_epoch"])
        batches = math.ceil(samples / batch_size)
        accumulate = int(training["accumulate_steps"])
        history: list[dict[str, Any]] = []
        checkpoint_hashes: dict[str, str] = {}
        start_epoch = 1
        if previous_manifest is not None:
            if (
                previous_manifest.get("initial_state_sha256") is not None
                and previous_manifest["initial_state_sha256"] != initial_hash
            ):
                raise RuntimeError("resume initial-state hash mismatch")
            previous_hashes = previous_manifest.get("checkpoint_sha256", {})
            for resume_epoch in reversed(checkpoint_epochs):
                expected_hash = previous_hashes.get(str(resume_epoch))
                if expected_hash is None:
                    continue
                checkpoint_path = job / "checkpoints" / f"epoch_{resume_epoch:03d}.pt"
                if not checkpoint_path.exists():
                    raise RuntimeError(f"resume checkpoint is missing: {checkpoint_path}")
                actual_hash = sha256_file(checkpoint_path)
                if actual_hash != expected_hash:
                    raise RuntimeError(
                        f"resume checkpoint hash mismatch for {checkpoint_path}"
                    )
                payload = torch.load(
                    checkpoint_path, map_location=device, weights_only=False
                )
                if payload.get("identity") != identity:
                    raise RuntimeError(
                        f"resume checkpoint identity mismatch: {checkpoint_path}"
                    )
                if "rng_state" not in payload:
                    raise RuntimeError(
                        f"resume checkpoint has no RNG state: {checkpoint_path}"
                    )
                model.load_state_dict(payload["model_state_dict"])
                optimizer.load_state_dict(payload["optimizer_state_dict"])
                _restore_rng_state(payload["rng_state"])
                start_epoch = resume_epoch + 1
                for prior_epoch in checkpoint_epochs:
                    if prior_epoch > resume_epoch:
                        continue
                    prior_path = job / "checkpoints" / f"epoch_{prior_epoch:03d}.pt"
                    prior_hash = previous_hashes.get(str(prior_epoch))
                    if prior_hash is None or not prior_path.exists():
                        raise RuntimeError(f"incomplete trusted checkpoint chain at {prior_path}")
                    if sha256_file(prior_path) != prior_hash:
                        raise RuntimeError(f"checkpoint chain hash mismatch at {prior_path}")
                    checkpoint_hashes[str(prior_epoch)] = prior_hash
                if history_path.exists():
                    expected_history_hash = previous_manifest.get("history_sha256")
                    if (
                        expected_history_hash is None
                        or sha256_file(history_path) != expected_history_hash
                    ):
                        raise RuntimeError("resume history hash mismatch")
                    import csv

                    with history_path.open(encoding="utf-8", newline="") as handle:
                        history = [
                            row
                            for row in csv.DictReader(handle)
                            if int(row["epoch"]) <= resume_epoch
                        ]
                print(
                    f"[resume] train {args.loss_kind} seed={args.seed} "
                    f"from epoch {resume_epoch}"
                )
                break
        for epoch in range(start_epoch, epochs + 1):
            model.train()
            sums = {key: 0.0 for key in ("local", "topology", "logical_frame", "calibration", "total")}
            seen = 0
            optimizer.zero_grad(set_to_none=True)
            started = time.perf_counter()
            for batch_index in range(batches):
                count = min(batch_size, samples - seen)
                step = (epoch - 1) * batches + batch_index
                train_x, targets = train_generator.generate_batch(step=step, batch_size=count)
                basis = train_generator.get_current_basis(step)
                logits = model(train_x)
                loss, components = topology_joint_loss(
                    logits,
                    targets,
                    train_x,
                    adapters[basis],
                    weights=weights,
                    positive_weight=float(training["positive_weight"]),
                )
                (loss / accumulate).backward()
                if (batch_index + 1) % accumulate == 0 or batch_index + 1 == batches:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                for key in sums:
                    sums[key] += float(components[key].detach()) * count
                seen += count
            row: dict[str, Any] = {
                "epoch": epoch,
                "loss_kind": args.loss_kind,
                "seed": int(args.seed),
                "seconds": time.perf_counter() - started,
                **{f"train_{key}": value / max(1, seen) for key, value in sums.items()},
            }
            if epoch in checkpoint_epochs:
                row.update(
                    {
                        f"validation_{key}": value
                        for key, value in _validation_loss(
                            model,
                            val_generator,
                            adapters,
                            config,
                            args.loss_kind,
                            epoch=epoch,
                        ).items()
                    }
                )
                checkpoint_path = job / "checkpoints" / f"epoch_{epoch:03d}.pt"
                checkpoint_hashes[str(epoch)] = _save_checkpoint(
                    checkpoint_path,
                    model,
                    optimizer,
                    epoch=epoch,
                    seed=int(args.seed),
                    loss_kind=args.loss_kind,
                    identity=identity,
                )
            history.append(row)
            write_csv(history_path, history)
            if epoch in checkpoint_epochs:
                manifest.update(
                    initial_state_sha256=initial_hash,
                    checkpoint_sha256=dict(checkpoint_hashes),
                    last_checkpoint_epoch=epoch,
                    data_seeds={
                        "train": int(args.seed),
                        "validation": int(args.seed)
                        + int(config["seed_offsets"]["checkpoint_validation"]),
                    },
                    history_sha256=sha256_file(history_path),
                )
                atomic_json(manifest_path, manifest)
            print(
                f"[train] loss={args.loss_kind} seed={args.seed} epoch={epoch}/{epochs} "
                f"total={row['train_total']:.6g} seconds={row['seconds']:.1f}"
            )
        _complete_manifest(
            manifest_path,
            manifest,
            initial_state_sha256=initial_hash,
            checkpoint_sha256=checkpoint_hashes,
            training_noise_sha256=training_noise.sha256(),
            registered_noise_sha256=registered.sha256(),
            training_upscale=upscale,
            data_seeds={
                "train": int(args.seed),
                "validation": int(args.seed)
                + int(config["seed_offsets"]["checkpoint_validation"]),
            },
            data_stream_sha256=_canonical_hash(
                {
                    "seed": int(args.seed),
                    "validation_seed": int(args.seed)
                    + int(config["seed_offsets"]["checkpoint_validation"]),
                    "samples_per_epoch": samples,
                    "epochs": epochs,
                    "training_noise_sha256": training_noise.sha256(),
                    "registered_noise_sha256": registered.sha256(),
                }
            ),
            artifact_sha256=_artifact_hashes([history_path, *checkpoint_paths]),
        )
    except Exception as exc:
        _fail_manifest(manifest_path, manifest, exc)
        raise


def _sample_stim(
    config: dict[str, Any],
    *,
    task: str,
    basis: str,
    shots: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, Any]:
    distance, rounds, rotation = _geometry(config)
    noise = _noise_model(config, task)
    placeholder = float(noise.get_max_probability())
    memory = MemoryCircuit(
        distance=distance,
        idle_error=placeholder,
        sqgate_error=placeholder,
        tqgate_error=placeholder,
        spam_error=(2.0 / 3.0) * placeholder,
        n_rounds=rounds,
        basis=basis,
        code_rotation=rotation,
        noise_model=noise,
        add_boundary_detectors=True,
    )
    memory.set_error_rates()
    circuit = memory.stim_circuit
    measurements = circuit.compile_sampler(seed=int(seed)).sample(shots=int(shots))
    dets_and_obs = circuit.compile_m2d_converter().convert(
        measurements=measurements, append_observables=True
    )
    detector_count = int(circuit.num_detectors)
    detectors = np.asarray(dets_and_obs[:, :detector_count], dtype=np.uint8)
    observable = np.asarray(dets_and_obs[:, detector_count], dtype=np.uint8)
    return detectors, observable, circuit


def _predict_probabilities(
    model: torch.nn.Module,
    detectors: np.ndarray,
    adapter: SurfaceActionAdapter,
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, float]:
    transform = SurfaceDetectorInputTransform(
        distance=adapter.distance,
        rounds=adapter.rounds,
        basis=adapter.basis,
        rotation=adapter.rotation,
    )
    expected = int(transform.detector_width)
    if detectors.shape[1] != expected:
        raise ValueError(f"detector width {detectors.shape[1]} != expected {expected}")
    batches = []
    started = time.perf_counter()
    model.eval()
    with torch.no_grad():
        for start in range(0, len(detectors), int(batch_size)):
            dets = torch.as_tensor(
                detectors[start : start + int(batch_size)], dtype=torch.float32
            )
            train_x, _, _, _ = transform.build_train_x(dets)
            probabilities = torch.sigmoid(model(train_x.to(device))).cpu()
            batches.append(adapter.flatten_spatial(probabilities))
    seconds = time.perf_counter() - started
    return torch.cat(batches).numpy().astype(np.float32, copy=False), seconds


def _detector_adjacency(circuit: Any, detector_count: int) -> list[set[int]]:
    coordinates = {
        int(key): tuple(float(item) for item in value)
        for key, value in circuit.get_detector_coordinates().items()
    }
    adjacency = [set() for _ in range(detector_count)]
    for left in range(detector_count):
        left_coordinate = coordinates.get(left, ())
        for right in range(left + 1, detector_count):
            right_coordinate = coordinates.get(right, ())
            if (
                left_coordinate
                and len(left_coordinate) == len(right_coordinate)
                and sum(
                    abs(first - second)
                    for first, second in zip(left_coordinate, right_coordinate)
                )
                <= 1.01
            ):
                adjacency[left].add(right)
                adjacency[right].add(left)
    return adjacency


def _whole_cluster(
    syndrome: np.ndarray,
    probabilities: np.ndarray,
    adapter: SurfaceActionAdapter,
    adjacency: list[set[int]],
    config: GateConfig,
    valid: np.ndarray,
) -> GateResult:
    h, logical, _, action_types = adapter.numpy_maps()
    candidates = typed_candidates(
        probabilities,
        action_types,
        data_threshold=config.data_threshold,
        measurement_threshold=config.measurement_threshold,
        valid_actions=valid,
        max_candidates=config.max_candidates,
    )
    clusters = interaction_clusters(
        candidates, h, adjacency, radius=config.interaction_radius
    )
    state = syndrome.copy()
    correction = np.zeros(adapter.action_count, dtype=np.uint8)
    local_frame = np.zeros(1, dtype=np.uint8)
    decisions: list[GateDecision] = []
    for cluster in clusters:
        proposed = np.zeros(adapter.action_count, dtype=np.uint8)
        proposed[list(cluster)] = 1
        trial, logical_delta = apply_actions(state, proposed, h, logical)
        before = workload_features(state, adjacency)
        after = workload_features(trial, adjacency)
        accepted = workload_score(after, len(state), config) <= workload_score(
            before, len(state), config
        )
        if accepted:
            correction[list(cluster)] = 1
            state = trial
            local_frame ^= logical_delta
        decisions.append(
            GateDecision(
                cluster=cluster,
                selected_actions=cluster,
                accepted=accepted,
                reason="accepted" if accepted else "workload_increase",
                utility=float(
                    workload_score(before, len(state), config)
                    - workload_score(after, len(state), config)
                ),
                uncertainty=0.0,
                logical_risk=0.0,
                workload_before=before,
                workload_after=after,
                search_exact=True,
                combinations_evaluated=1,
            )
        )
    return GateResult(
        accepted_actions=correction,
        residual=state,
        local_logical_frame=local_frame,
        candidate_count=int(candidates.size),
        decisions=tuple(decisions),
    )


def _static_gate(
    syndrome: np.ndarray,
    probabilities: np.ndarray,
    adapter: SurfaceActionAdapter,
    adjacency: list[set[int]],
    config: GateConfig,
    valid: np.ndarray,
) -> GateResult:
    """Ablation that scores every cluster once against the initial state."""

    h, logical, _, action_types = adapter.numpy_maps()
    candidates = typed_candidates(
        probabilities,
        action_types,
        data_threshold=config.data_threshold,
        measurement_threshold=config.measurement_threshold,
        valid_actions=valid,
        max_candidates=config.max_candidates,
    )
    clusters = interaction_clusters(
        candidates, h, adjacency, radius=config.interaction_radius
    )
    evaluated = [
        (
            cluster,
            evaluate_cluster(
                syndrome,
                cluster,
                probabilities,
                h,
                logical,
                adjacency,
                config,
            ),
        )
        for cluster in clusters
    ]
    evaluated.sort(key=lambda item: (-item[1].utility, item[0]))
    correction = np.zeros(adapter.action_count, dtype=np.uint8)
    decisions = []
    for cluster, best in evaluated:
        accepted = bool(
            best.actions
            and np.isfinite(best.utility)
            and best.utility >= config.acceptance_threshold
            and correction.sum() + len(best.actions) <= config.max_total_actions
        )
        if accepted:
            correction[list(best.actions)] = 1
        decisions.append(
            GateDecision(
                cluster=cluster,
                selected_actions=best.actions,
                accepted=accepted,
                reason="accepted" if accepted else "not_accepted",
                utility=best.utility,
                uncertainty=best.uncertainty,
                logical_risk=best.logical_risk,
                workload_before=best.workload_before,
                workload_after=best.workload_after,
                search_exact=best.search_exact,
                combinations_evaluated=best.combinations_evaluated,
            )
        )
    residual, frame = apply_actions(syndrome, correction, h, logical)
    return GateResult(
        accepted_actions=correction,
        residual=residual,
        local_logical_frame=frame,
        candidate_count=int(candidates.size),
        decisions=tuple(decisions),
    )


def _method_gate(
    method: str,
    syndrome: np.ndarray,
    probabilities: np.ndarray,
    adapter: SurfaceActionAdapter,
    adjacency: list[set[int]],
    config: GateConfig,
    valid: np.ndarray,
) -> GateResult:
    h, logical, _, action_types = adapter.numpy_maps()
    if method == "pointwise":
        correction = pointwise_actions(
            probabilities, action_types, config, valid_actions=valid
        )
        residual, frame = apply_actions(syndrome, correction, h, logical)
        return GateResult(
            accepted_actions=correction,
            residual=residual,
            local_logical_frame=frame,
            candidate_count=int(correction.sum()),
            decisions=(),
        )
    if method == "whole_cluster_v1":
        return _whole_cluster(syndrome, probabilities, adapter, adjacency, config, valid)
    if method == "combination_v2":
        return gate_actions(
            syndrome,
            probabilities,
            action_types,
            h,
            logical,
            adjacency,
            config,
            valid_actions=valid,
        )
    if method == "no_recompute":
        return _static_gate(syndrome, probabilities, adapter, adjacency, config, valid)
    raise ValueError(f"unknown method {method!r}")


def _evaluate_method(
    *,
    method_name: str,
    gate_method: str,
    detectors: np.ndarray,
    observable: np.ndarray,
    probabilities: np.ndarray,
    adapter: SurfaceActionAdapter,
    adjacency: list[set[int]],
    matcher: Any,
    gate_config: GateConfig,
    valid: np.ndarray,
    model_seconds: float,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    started = time.perf_counter()
    residuals = []
    frames = []
    action_counts = []
    candidate_counts = []
    combinations = []
    exact = []
    for syndrome, probability in zip(detectors, probabilities, strict=True):
        result = _method_gate(
            gate_method,
            syndrome,
            probability,
            adapter,
            adjacency,
            gate_config,
            valid,
        )
        residuals.append(result.residual)
        frames.append(int(result.local_logical_frame[0]))
        action_counts.append(result.accepted_count)
        candidate_counts.append(result.candidate_count)
        combinations.append(result.combinations_evaluated)
        exact.append(result.exact_search)
    gate_seconds = time.perf_counter() - started
    residual = np.stack(residuals).astype(np.uint8, copy=False)
    frame = np.asarray(frames, dtype=np.uint8)
    decode_started = time.perf_counter()
    global_prediction = np.asarray(
        matcher.decode_batch(residual), dtype=np.uint8
    ).reshape(-1)
    decode_seconds = time.perf_counter() - decode_started
    prediction = frame ^ global_prediction
    errors = (prediction != observable).astype(np.uint8)
    low, high = wilson_interval(int(errors.sum()), len(errors))
    complexity = np.asarray(
        [workload_features(item, adjacency).component_square_sum for item in residual],
        dtype=np.float32,
    )
    density = residual.mean(axis=1, dtype=np.float64).astype(np.float32)
    row = {
        "method": method_name,
        "errors": int(errors.sum()),
        "shots": len(errors),
        "ler": float(errors.mean()),
        "ler_ci_low": low,
        "ler_ci_high": high,
        "residual_density": float(density.mean()),
        "topology_complexity": float(complexity.mean()),
        "accepted_actions_per_shot": float(np.mean(action_counts)),
        "candidate_count_per_shot": float(np.mean(candidate_counts)),
        "combinations_evaluated_per_shot": float(np.mean(combinations)),
        "exact_search_rate": float(np.mean(exact)),
        "model_us_per_shot": model_seconds * 1e6 / len(errors),
        "gate_us_per_shot": gate_seconds * 1e6 / len(errors),
        "decode_us_per_shot": decode_seconds * 1e6 / len(errors),
        "end_to_end_us_per_shot": (
            model_seconds + gate_seconds + decode_seconds
        ) * 1e6 / len(errors),
    }
    vectors = {
        f"{method_name}__errors": errors,
        f"{method_name}__action_counts": np.asarray(action_counts, dtype=np.int16),
        f"{method_name}__complexity": complexity,
        f"{method_name}__density": density,
    }
    return row, vectors


def _raw_method(
    detectors: np.ndarray,
    observable: np.ndarray,
    adjacency: list[set[int]],
    matcher: Any,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    started = time.perf_counter()
    prediction = np.asarray(matcher.decode_batch(detectors), dtype=np.uint8).reshape(-1)
    seconds = time.perf_counter() - started
    errors = (prediction != observable).astype(np.uint8)
    complexity = np.asarray(
        [workload_features(item, adjacency).component_square_sum for item in detectors],
        dtype=np.float32,
    )
    density = detectors.mean(axis=1, dtype=np.float64).astype(np.float32)
    low, high = wilson_interval(int(errors.sum()), len(errors))
    row = {
        "method": "raw_pymatching",
        "errors": int(errors.sum()),
        "shots": len(errors),
        "ler": float(errors.mean()),
        "ler_ci_low": low,
        "ler_ci_high": high,
        "residual_density": float(density.mean()),
        "topology_complexity": float(complexity.mean()),
        "accepted_actions_per_shot": 0.0,
        "candidate_count_per_shot": 0.0,
        "combinations_evaluated_per_shot": 0.0,
        "exact_search_rate": 1.0,
        "model_us_per_shot": 0.0,
        "gate_us_per_shot": 0.0,
        "decode_us_per_shot": seconds * 1e6 / len(errors),
        "end_to_end_us_per_shot": seconds * 1e6 / len(errors),
    }
    return row, {
        "raw_pymatching__errors": errors,
        "raw_pymatching__action_counts": np.zeros(len(errors), dtype=np.int16),
        "raw_pymatching__complexity": complexity,
        "raw_pymatching__density": density,
    }


def _gate_candidates(config: dict[str, Any]) -> list[GateConfig]:
    gate = config["gate"]
    values = []
    for data_threshold in gate["data_thresholds"]:
        for measurement_threshold in gate["measurement_thresholds"]:
            for logical_weight in gate["logical_risk_weights"]:
                mapping = dict(gate["defaults"])
                mapping.update(
                    data_threshold=float(data_threshold),
                    measurement_threshold=float(measurement_threshold),
                    logical_risk_weight=float(logical_weight),
                )
                values.append(config_from_mapping(mapping))
    return values


def _checkpoint_paths(
    config: dict[str, Any], root: Path, loss_kind: str, seed: int
) -> list[tuple[int, Path]]:
    return [
        (
            int(epoch),
            root
            / "train"
            / loss_kind
            / f"seed_{seed}"
            / "checkpoints"
            / f"epoch_{int(epoch):03d}.pt",
        )
        for epoch in config["training"]["checkpoint_epochs"]
    ]


def select_checkpoint(
    config: dict[str, Any], args: argparse.Namespace, output: Path
) -> None:
    identity = _identity(
        config,
        args.mode,
        unit="select",
        seed=int(args.seed),
        loss_kind=args.loss_kind,
    )
    root = output / args.mode
    job = root / "select" / args.loss_kind / f"seed_{int(args.seed)}"
    manifest_path = job / "manifest.json"
    selection_path = job / "selection.json"
    rows_path = job / "checkpoint_validation.csv"
    if _resume_complete(
        manifest_path, identity, [selection_path, rows_path], args.resume
    ):
        print(f"[resume] select {args.loss_kind} seed={args.seed}")
        return
    manifest = _manifest(identity)
    atomic_json(manifest_path, manifest)
    try:
        device = _device(args.device)
        distance, rounds, rotation = _geometry(config)
        shots_total = int(config["training"]["checkpoint_validation_shots"])
        shots = max(1, shots_total // 2)
        gate_config = config_from_mapping(config["gate"]["defaults"])
        validation_rows = []
        validation_data_sha256: dict[str, str] = {}
        validation_data_seeds: dict[str, int] = {}
        for epoch, checkpoint in _checkpoint_paths(
            config, root, args.loss_kind, int(args.seed)
        ):
            if not checkpoint.exists():
                raise FileNotFoundError(checkpoint)
            model = _load_checkpoint(checkpoint, config, device)
            basis_rows = []
            for basis_index, basis in enumerate(("X", "Z")):
                seed = (
                    int(args.seed)
                    + int(config["seed_offsets"]["checkpoint_validation"])
                    + basis_index * 10000
                )
                detectors, observable, circuit = _sample_stim(
                    config, task="t0", basis=basis, shots=shots, seed=seed
                )
                validation_data_sha256[basis] = _array_sha256(detectors, observable)
                validation_data_seeds[basis] = seed
                adapter = build_surface_action_adapter(distance, rounds, basis, rotation)
                probabilities, model_seconds = _predict_probabilities(
                    model,
                    detectors,
                    adapter,
                    device=device,
                    batch_size=int(config["training"]["batch_size"]),
                )
                import pymatching

                matcher = pymatching.Matching.from_detector_error_model(
                    circuit.detector_error_model(
                        decompose_errors=True, approximate_disjoint_errors=True
                    )
                )
                row, _ = _evaluate_method(
                    method_name="combination_v2",
                    gate_method="combination_v2",
                    detectors=detectors,
                    observable=observable,
                    probabilities=probabilities,
                    adapter=adapter,
                    adjacency=_detector_adjacency(circuit, detectors.shape[1]),
                    matcher=matcher,
                    gate_config=gate_config,
                    valid=adapter.valid_actions.numpy(),
                    model_seconds=model_seconds,
                )
                row.update(epoch=epoch, basis=basis, checkpoint=str(checkpoint))
                basis_rows.append(row)
                validation_rows.append(row)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        write_csv(rows_path, validation_rows)
        grouped = []
        for epoch, checkpoint in _checkpoint_paths(
            config, root, args.loss_kind, int(args.seed)
        ):
            epoch_rows = [row for row in validation_rows if row["epoch"] == epoch]
            grouped.append(
                (
                    sum(row["errors"] for row in epoch_rows) / sum(
                        row["shots"] for row in epoch_rows
                    ),
                    float(np.mean([row["topology_complexity"] for row in epoch_rows])),
                    float(np.mean([row["residual_density"] for row in epoch_rows])),
                    epoch,
                    checkpoint,
                )
            )
        _, _, _, selected_epoch, selected_path = min(grouped)
        selection = {
            "seed": int(args.seed),
            "loss_kind": args.loss_kind,
            "epoch": selected_epoch,
            "checkpoint": str(selected_path),
            "checkpoint_sha256": sha256_file(selected_path),
            "selection_order": ["ler", "topology_complexity", "residual_density", "epoch"],
        }
        atomic_json(selection_path, selection)
        _complete_manifest(
            manifest_path,
            manifest,
            selection=selection,
            validation_data_sha256=validation_data_sha256,
            validation_data_seeds=validation_data_seeds,
            artifact_sha256=_artifact_hashes([selection_path, rows_path]),
        )
    except Exception as exc:
        _fail_manifest(manifest_path, manifest, exc)
        raise


def _selected_model(
    config: dict[str, Any], root: Path, loss_kind: str, seed: int, device: torch.device
) -> tuple[torch.nn.Module, dict[str, Any]]:
    path = root / "select" / loss_kind / f"seed_{seed}" / "selection.json"
    selection = json.loads(path.read_text(encoding="utf-8"))
    checkpoint = Path(selection["checkpoint"])
    if sha256_file(checkpoint) != selection["checkpoint_sha256"]:
        raise RuntimeError(f"selected checkpoint hash mismatch: {checkpoint}")
    return _load_checkpoint(checkpoint, config, device), selection


def evaluate(config: dict[str, Any], args: argparse.Namespace, output: Path) -> None:
    identity = _identity(config, args.mode, unit="evaluate", seed=int(args.seed))
    root = output / args.mode
    job = root / "evaluate" / f"seed_{int(args.seed)}"
    manifest_path = job / "manifest.json"
    summary_path = job / "summary.csv"
    ablation_path = job / "ablation.csv"
    gate_path = job / "selected_gate.json"
    expected_vectors = [
        job / "vectors" / f"{task}_{basis}.npz"
        for task in config["test_shots_per_basis"]
        for basis in ("X", "Z")
    ]
    if _resume_complete(
        manifest_path,
        identity,
        [summary_path, ablation_path, gate_path, *expected_vectors],
        args.resume,
    ):
        print(f"[resume] evaluate seed={args.seed}")
        return
    manifest = _manifest(identity)
    atomic_json(manifest_path, manifest)
    try:
        device = _device(args.device)
        distance, rounds, rotation = _geometry(config)
        bce_model, bce_selection = _selected_model(
            config, root, "bce", int(args.seed), device
        )
        topology_model, topology_selection = _selected_model(
            config, root, "topology", int(args.seed), device
        )
        selected_gates: dict[str, Any] = {}
        dataset_sha256: dict[str, str] = {}
        data_seeds: dict[str, int] = {}
        for basis_index, basis in enumerate(("X", "Z")):
            seed = (
                int(args.seed)
                + int(config["seed_offsets"]["gate_validation"])
                + basis_index * 10000
            )
            detectors, observable, circuit = _sample_stim(
                config,
                task="t0",
                basis=basis,
                shots=int(config["gate_validation_shots_per_basis"]),
                seed=seed,
            )
            data_key = f"gate_validation_{basis}"
            dataset_sha256[data_key] = _array_sha256(detectors, observable)
            data_seeds[data_key] = seed
            adapter = build_surface_action_adapter(distance, rounds, basis, rotation)
            probabilities, model_seconds = _predict_probabilities(
                bce_model,
                detectors,
                adapter,
                device=device,
                batch_size=int(config["training"]["batch_size"]),
            )
            import pymatching

            matcher = pymatching.Matching.from_detector_error_model(
                circuit.detector_error_model(
                    decompose_errors=True, approximate_disjoint_errors=True
                )
            )
            adjacency = _detector_adjacency(circuit, detectors.shape[1])
            ranked = []
            for candidate in _gate_candidates(config):
                row, _ = _evaluate_method(
                    method_name="bce_combination_v2",
                    gate_method="combination_v2",
                    detectors=detectors,
                    observable=observable,
                    probabilities=probabilities,
                    adapter=adapter,
                    adjacency=adjacency,
                    matcher=matcher,
                    gate_config=candidate,
                    valid=adapter.valid_actions.numpy(),
                    model_seconds=model_seconds,
                )
                ranked.append(
                    (
                        row["ler"],
                        row["topology_complexity"],
                        row["residual_density"],
                        candidate.data_threshold,
                        candidate.measurement_threshold,
                        candidate.logical_risk_weight,
                        candidate,
                    )
                )
            selected_gates[basis] = asdict(min(ranked)[-1])
        atomic_json(gate_path, selected_gates)

        rows: list[dict[str, Any]] = []
        ablation_rows: list[dict[str, Any]] = []
        for task_index, (task, shots_value) in enumerate(
            config["test_shots_per_basis"].items()
        ):
            for basis_index, basis in enumerate(("X", "Z")):
                offset_key = f"test_{task}"
                seed = (
                    int(args.seed)
                    + int(config["seed_offsets"][offset_key])
                    + basis_index * 10000
                )
                detectors, observable, circuit = _sample_stim(
                    config,
                    task=task,
                    basis=basis,
                    shots=int(shots_value),
                    seed=seed,
                )
                data_key = f"test_{task}_{basis}"
                dataset_sha256[data_key] = _array_sha256(detectors, observable)
                data_seeds[data_key] = seed
                adapter = build_surface_action_adapter(distance, rounds, basis, rotation)
                bce_probabilities, bce_seconds = _predict_probabilities(
                    bce_model,
                    detectors,
                    adapter,
                    device=device,
                    batch_size=int(config["training"]["batch_size"]),
                )
                topology_probabilities, topology_seconds = _predict_probabilities(
                    topology_model,
                    detectors,
                    adapter,
                    device=device,
                    batch_size=int(config["training"]["batch_size"]),
                )
                import pymatching

                matcher = pymatching.Matching.from_detector_error_model(
                    circuit.detector_error_model(
                        decompose_errors=True, approximate_disjoint_errors=True
                    )
                )
                adjacency = _detector_adjacency(circuit, detectors.shape[1])
                gate_config = config_from_mapping(selected_gates[basis])
                cell_rows = []
                vectors: dict[str, np.ndarray] = {}
                raw_row, raw_vectors = _raw_method(
                    detectors, observable, adjacency, matcher
                )
                cell_rows.append(raw_row)
                vectors.update(raw_vectors)
                for method_name, gate_method in (
                    ("bce_pointwise", "pointwise"),
                    ("bce_whole_cluster_v1", "whole_cluster_v1"),
                    ("bce_combination_v2", "combination_v2"),
                ):
                    row, method_vectors = _evaluate_method(
                        method_name=method_name,
                        gate_method=gate_method,
                        detectors=detectors,
                        observable=observable,
                        probabilities=bce_probabilities,
                        adapter=adapter,
                        adjacency=adjacency,
                        matcher=matcher,
                        gate_config=gate_config,
                        valid=adapter.valid_actions.numpy(),
                        model_seconds=bce_seconds,
                    )
                    cell_rows.append(row)
                    vectors.update(method_vectors)
                row, method_vectors = _evaluate_method(
                    method_name="topology_combination_v2",
                    gate_method="combination_v2",
                    detectors=detectors,
                    observable=observable,
                    probabilities=topology_probabilities,
                    adapter=adapter,
                    adjacency=adjacency,
                    matcher=matcher,
                    gate_config=gate_config,
                    valid=adapter.valid_actions.numpy(),
                    model_seconds=topology_seconds,
                )
                cell_rows.append(row)
                vectors.update(method_vectors)
                for row in cell_rows:
                    row.update(task=task, basis=basis, seed=int(args.seed), scope="main")
                    rows.append(row)

                count = min(int(config["ablation_shots_per_cell"]), len(detectors))
                data_types = (DATA_Z, DATA_X)
                measurement_types = (MEASUREMENT_X, MEASUREMENT_Z)
                ablations = [
                    (
                        "bce_v2_no_recompute",
                        "no_recompute",
                        gate_config,
                        adapter.valid_actions.numpy(),
                    ),
                    (
                        "bce_v2_no_logical_risk",
                        "combination_v2",
                        replace(
                            gate_config,
                            logical_risk_weight=0.0,
                            max_logical_risk=float("inf"),
                        ),
                        adapter.valid_actions.numpy(),
                    ),
                    (
                        "bce_v2_data_only",
                        "combination_v2",
                        gate_config,
                        adapter.action_type_mask(data_types),
                    ),
                    (
                        "bce_v2_measurement_only",
                        "combination_v2",
                        gate_config,
                        adapter.action_type_mask(measurement_types),
                    ),
                ]
                for name, method, ablation_gate, valid in ablations:
                    ablation_row, _ = _evaluate_method(
                        method_name=name,
                        gate_method=method,
                        detectors=detectors[:count],
                        observable=observable[:count],
                        probabilities=bce_probabilities[:count],
                        adapter=adapter,
                        adjacency=adjacency,
                        matcher=matcher,
                        gate_config=ablation_gate,
                        valid=valid,
                        model_seconds=bce_seconds * count / len(detectors),
                    )
                    ablation_row.update(
                        task=task, basis=basis, seed=int(args.seed), scope="ablation"
                    )
                    ablation_rows.append(ablation_row)

                v1_error = vectors["bce_whole_cluster_v1__errors"].astype(bool)
                v2_error = vectors["bce_combination_v2__errors"].astype(bool)
                v1_count = vectors["bce_whole_cluster_v1__action_counts"]
                v2_count = vectors["bce_combination_v2__action_counts"]
                harmful = v1_error & ~v2_error & (v2_count < v1_count)
                vectors["harmful_whole_avoided"] = harmful.astype(np.uint8)
                vector_path = job / "vectors" / f"{task}_{basis}.npz"
                vector_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(vector_path, **vectors)
        write_csv(summary_path, rows)
        write_csv(ablation_path, ablation_rows)
        _complete_manifest(
            manifest_path,
            manifest,
            selected_gate=selected_gates,
            bce_selection=bce_selection,
            topology_selection=topology_selection,
            result_rows=len(rows),
            dataset_sha256=dataset_sha256,
            data_seeds=data_seeds,
            artifact_sha256=_artifact_hashes(
                [summary_path, ablation_path, gate_path, *expected_vectors]
            ),
        )
    except Exception as exc:
        _fail_manifest(manifest_path, manifest, exc)
        raise


def _rows_by_key(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str, int], dict[str, Any]]:
    return {
        (str(row["method"]), str(row["task"]), str(row["basis"]), int(row["seed"])): row
        for row in rows
    }


def _paired_summary(
    root: Path,
    seeds: list[int],
    task: str,
    basis: str,
    left: str,
    right: str,
    repeats: int,
) -> dict[str, Any]:
    left_values = []
    right_values = []
    seed_deltas = []
    for seed in seeds:
        with np.load(root / "evaluate" / f"seed_{seed}" / "vectors" / f"{task}_{basis}.npz") as data:
            left_error = data[f"{left}__errors"].astype(np.uint8)
            right_error = data[f"{right}__errors"].astype(np.uint8)
        left_values.append(left_error)
        right_values.append(right_error)
        seed_deltas.append(float(left_error.mean() - right_error.mean()))
    left_all = np.concatenate(left_values)
    right_all = np.concatenate(right_values)
    low, high = paired_bootstrap_interval(
        left_all, right_all, seed=sum(seeds) + len(left) + len(right), repeats=repeats
    )
    return {
        "task": task,
        "basis": basis,
        "left": left,
        "right": right,
        "metric": "ler",
        "shots": int(left_all.size),
        "left_errors": int(left_all.sum()),
        "right_errors": int(right_all.sum()),
        "paired_error_events": int(np.logical_or(left_all, right_all).sum()),
        "delta_ler": float(left_all.mean() - right_all.mean()),
        "delta_mean": float(left_all.mean() - right_all.mean()),
        "delta_ci_low": low,
        "delta_ci_high": high,
        "seed_deltas": seed_deltas,
    }


def _paired_metric_summary(
    root: Path,
    seeds: list[int],
    task: str,
    basis: str,
    left: str,
    right: str,
    field: str,
    repeats: int,
) -> dict[str, Any]:
    left_values = []
    right_values = []
    seed_deltas = []
    for seed in seeds:
        vector_path = (
            root / "evaluate" / f"seed_{seed}" / "vectors" / f"{task}_{basis}.npz"
        )
        with np.load(vector_path) as data:
            left_value = data[f"{left}__{field}"].astype(np.float64)
            right_value = data[f"{right}__{field}"].astype(np.float64)
        left_values.append(left_value)
        right_values.append(right_value)
        seed_deltas.append(float(np.mean(left_value - right_value)))
    delta = np.concatenate(left_values) - np.concatenate(right_values)
    support, counts = np.unique(delta, return_counts=True)
    generator = np.random.default_rng(sum(seeds) + len(left) + len(right) + len(field))
    draws = generator.multinomial(delta.size, counts / delta.size, size=repeats)
    values = np.matmul(draws, support) / delta.size
    return {
        "task": task,
        "basis": basis,
        "left": left,
        "right": right,
        "metric": field,
        "shots": int(delta.size),
        "delta_mean": float(delta.mean()),
        "delta_ci_low": float(np.quantile(values, 0.025)),
        "delta_ci_high": float(np.quantile(values, 0.975)),
        "seed_deltas": seed_deltas,
    }


def aggregate(config: dict[str, Any], args: argparse.Namespace, output: Path) -> None:
    root = output / args.mode
    seeds = [int(value) for value in config["seeds"]]
    manifests = [root / "evaluate" / f"seed_{seed}" / "manifest.json" for seed in seeds]
    completed = []
    for path in manifests:
        if path.exists():
            value = json.loads(path.read_text(encoding="utf-8"))
            completed.append(value.get("status") == "completed")
        else:
            completed.append(False)
    aggregate_dir = root / "aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    if not all(completed):
        partial_rows = []
        partial_ablations = []
        for seed, ok in zip(seeds, completed, strict=True):
            if not ok:
                continue
            with (root / "evaluate" / f"seed_{seed}" / "summary.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                partial_rows.extend(csv.DictReader(handle))
            with (root / "evaluate" / f"seed_{seed}" / "ablation.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                partial_ablations.extend(csv.DictReader(handle))
        write_csv(aggregate_dir / "summary.csv", list(partial_rows))
        write_csv(aggregate_dir / "ablation.csv", list(partial_ablations))
        write_csv(aggregate_dir / "paired_deltas.csv", [])
        decision = {
            "status": "INCOMPLETE",
            "completed_seeds": [seed for seed, ok in zip(seeds, completed, strict=True) if ok],
            "required_seeds": seeds,
        }
        atomic_json(aggregate_dir / "decisions.json", decision)
        (aggregate_dir / "results.md").write_text(
            "# 拓扑门控 v2 四通道增强型验证\n\n"
            "结论：INCOMPLETE。主实验单元未全部完成，不进行 GO/NO-GO 判定。\n",
            encoding="utf-8",
        )
        print("INCOMPLETE")
        return

    rows = []
    ablations = []
    for seed in seeds:
        import csv

        with (root / "evaluate" / f"seed_{seed}" / "summary.csv").open(
            encoding="utf-8", newline=""
        ) as handle:
            rows.extend(list(csv.DictReader(handle)))
        with (root / "evaluate" / f"seed_{seed}" / "ablation.csv").open(
            encoding="utf-8", newline=""
        ) as handle:
            ablations.extend(list(csv.DictReader(handle)))
    write_csv(aggregate_dir / "summary.csv", rows)
    write_csv(aggregate_dir / "ablation.csv", ablations)
    stats = config["statistics"]
    repeats = int(stats["bootstrap_repeats"])
    comparisons = []
    for basis in ("X", "Z"):
        for left, right in (
            ("bce_combination_v2", "bce_pointwise"),
            ("bce_combination_v2", "raw_pymatching"),
            ("bce_combination_v2", "bce_whole_cluster_v1"),
            ("topology_combination_v2", "bce_combination_v2"),
        ):
            comparisons.append(
                _paired_summary(root, seeds, "t0", basis, left, right, repeats)
            )
    ler_comparisons = list(comparisons)
    complexity_comparisons = [
        _paired_metric_summary(
            root,
            seeds,
            "t0",
            basis,
            "topology_combination_v2",
            "bce_combination_v2",
            "complexity",
            repeats,
        )
        for basis in ("X", "Z")
    ]
    comparisons.extend(complexity_comparisons)
    write_csv(aggregate_dir / "paired_deltas.csv", comparisons)
    margin = float(stats["ler_noninferiority_margin"])
    minimum_errors = int(stats["minimum_error_events"])
    enough_errors = all(
        item["paired_error_events"] >= minimum_errors
        for item in ler_comparisons
    )
    lookup = {(item["basis"], item["left"], item["right"]): item for item in ler_comparisons}
    core_safety = all(
        lookup[(basis, "bce_combination_v2", comparator)]["delta_ci_high"] <= margin
        for basis in ("X", "Z")
        for comparator in ("bce_pointwise", "raw_pymatching")
    )
    def same_direction(values: list[float]) -> bool:
        return max(values) <= 0.0 or min(values) >= 0.0

    seed_consistency = all(
        same_direction(
            lookup[(basis, "bce_combination_v2", "bce_pointwise")]["seed_deltas"]
        )
        for basis in ("X", "Z")
    )
    index = _rows_by_key(rows)
    workload_reductions: dict[str, dict[str, float]] = {}
    for metric in ("topology_complexity", "residual_density"):
        by_basis: dict[str, float] = {}
        for basis in ("X", "Z"):
            candidate = np.mean(
                [float(index[("bce_combination_v2", "t0", basis, seed)][metric]) for seed in seeds]
            )
            baseline = np.mean(
                [float(index[("bce_pointwise", "t0", basis, seed)][metric]) for seed in seeds]
            )
            by_basis[basis] = 1 - candidate / max(baseline, 1e-12)
        workload_reductions[metric] = by_basis
    workload_ok = any(
        max(by_basis.values()) >= float(stats["workload_reduction"])
        and min(by_basis.values()) >= 0.0
        for by_basis in workload_reductions.values()
    )
    core_go = enough_errors and core_safety and seed_consistency and workload_ok

    whole_safety = all(
        lookup[(basis, "bce_combination_v2", "bce_whole_cluster_v1")]["delta_ci_high"] <= margin
        for basis in ("X", "Z")
    )
    harmful_count = 0
    whole_cell_reductions = []
    for seed in seeds:
        for basis in ("X", "Z"):
            with np.load(
                root / "evaluate" / f"seed_{seed}" / "vectors" / f"t0_{basis}.npz"
            ) as data:
                harmful_count += int(data["harmful_whole_avoided"].sum())
            candidate = float(
                index[("bce_combination_v2", "t0", basis, seed)]["topology_complexity"]
            )
            baseline = float(
                index[("bce_whole_cluster_v1", "t0", basis, seed)]["topology_complexity"]
            )
            whole_cell_reductions.append(1 - candidate / max(baseline, 1e-12))
    whole_reduction = max(whole_cell_reductions)
    cluster_go = (
        enough_errors
        and whole_safety
        and harmful_count >= int(stats["minimum_harmful_whole_avoided"])
        and whole_reduction >= float(stats["whole_cluster_complexity_reduction"])
    )

    joint_safety = all(
        lookup[(basis, "topology_combination_v2", "bce_combination_v2")]["delta_ci_high"] <= margin
        for basis in ("X", "Z")
    )
    complexity_lookup = {item["basis"]: item for item in complexity_comparisons}
    clear_joint_improvements = []
    for basis in ("X", "Z"):
        ler_item = lookup[(basis, "topology_combination_v2", "bce_combination_v2")]
        if ler_item["delta_ci_high"] < 0.0:
            clear_joint_improvements.append(("ler", basis, ler_item["seed_deltas"]))
        complexity_item = complexity_lookup[basis]
        if complexity_item["delta_ci_high"] < 0.0:
            clear_joint_improvements.append(
                ("complexity", basis, complexity_item["seed_deltas"])
            )
    joint_direction_seed_count = max(
        (sum(float(delta) < 0.0 for delta in deltas) for _, _, deltas in clear_joint_improvements),
        default=0,
    )
    joint_reductions = []
    for seed in seeds:
        for basis in ("X", "Z"):
            topology_complexity = float(
                index[("topology_combination_v2", "t0", basis, seed)]["topology_complexity"]
            )
            bce_complexity = float(
                index[("bce_combination_v2", "t0", basis, seed)]["topology_complexity"]
            )
            joint_reductions.append(
                1 - topology_complexity / max(bce_complexity, 1e-12)
            )
    joint_go = (
        enough_errors
        and joint_safety
        and bool(clear_joint_improvements)
        and joint_direction_seed_count >= 2
    )
    drift_red_flags = []
    for seed in seeds:
        for basis in ("X", "Z"):
            t0 = float(index[("bce_combination_v2", "t0", basis, seed)]["ler"])
            drift = float(
                index[("bce_combination_v2", "measurement_drift_1p5", basis, seed)]["ler"]
            )
            if drift - t0 > float(stats["drift_ler_red_flag"]):
                drift_red_flags.append({"seed": seed, "basis": basis, "delta_ler": drift - t0})
    def verdict(go: bool, safety: bool) -> str:
        if go:
            return "GO"
        if not safety:
            return "NO-GO"
        if not enough_errors:
            return "INCONCLUSIVE"
        return "NO-GO"

    core_status = verdict(core_go, core_safety)
    cluster_status = verdict(cluster_go, whole_safety)
    joint_status = verdict(joint_go, joint_safety)
    overall = core_status
    if args.mode == "smoke":
        core_status = "INCONCLUSIVE"
        cluster_status = "INCONCLUSIVE"
        joint_status = "INCONCLUSIVE"
        overall = "INCONCLUSIVE"
    decisions = {
        "status": overall,
        "formal_patent_evidence": False,
        "smoke_not_effect_evidence": args.mode == "smoke",
        "reason_not_formal": "single distance (d=5) enhanced pilot",
        "core_gate": core_status,
        "cluster_combinations": cluster_status,
        "joint_loss": joint_status,
        "enough_error_events": enough_errors,
        "seed_direction_consistent": seed_consistency,
        "workload_requirement_met": workload_ok,
        "workload_reductions": workload_reductions,
        "harmful_whole_avoided": harmful_count,
        "whole_cluster_complexity_reduction": whole_reduction,
        "joint_complexity_reductions": joint_reductions,
        "joint_clear_improvements": clear_joint_improvements,
        "joint_direction_seed_count": joint_direction_seed_count,
        "drift_red_flags": drift_red_flags,
        "bypass_required": bool(drift_red_flags),
    }
    atomic_json(aggregate_dir / "decisions.json", decisions)
    lines = [
        "# 拓扑门控 v2 四通道增强型验证",
        "",
        f"结论：**{overall}**。本结果为单码距增强型 pilot，不是正式申请证据。",
        "",
        f"- 核心组合门控：{decisions['core_gate']}",
        f"- 相对旧整簇机制：{decisions['cluster_combinations']}",
        f"- 拓扑联合损失：{decisions['joint_loss']}",
        f"- 避免整簇有害提交事件：{harmful_count}",
        f"- 漂移红旗单元：{len(drift_red_flags)}",
        f"- 旁路需求：{'是' if drift_red_flags else '否'}",
        "",
        "详细数值见 `summary.csv`、`paired_deltas.csv` 和 `decisions.json`。",
    ]
    (aggregate_dir / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(overall)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--mode", choices=("full", "smoke"), default="full")
    parser.add_argument("--resume", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--device", default="cuda:0")

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--seed", required=True, type=int)
    train_parser.add_argument("--loss-kind", required=True, choices=LOSS_KINDS)
    train_parser.add_argument("--device", default="cuda:0")

    select_parser = subparsers.add_parser("select")
    select_parser.add_argument("--seed", required=True, type=int)
    select_parser.add_argument("--loss-kind", required=True, choices=LOSS_KINDS)
    select_parser.add_argument("--device", default="cuda:0")

    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--seed", required=True, type=int)
    evaluate_parser.add_argument("--device", default="cuda:0")

    subparsers.add_parser("aggregate")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    output = Path(args.output).resolve()
    config = resolved_config(config_path, args.mode)
    if args.command == "prepare":
        prepare(config, args, output)
    elif args.command == "train":
        train(config, args, output)
    elif args.command == "select":
        select_checkpoint(config, args, output)
    elif args.command == "evaluate":
        evaluate(config, args, output)
    elif args.command == "aggregate":
        aggregate(config, args, output)
    else:  # pragma: no cover
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
