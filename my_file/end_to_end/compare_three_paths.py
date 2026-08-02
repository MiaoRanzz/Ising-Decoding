#!/usr/bin/env python3
"""Compare PyMatching, a checkpoint, and paired correction labels on one corpus.

The corpus must have been written by ``generate_labeled_dataset.py``.  It is
important that the oracle action is read from that corpus: labels cannot be
recovered from detector bits after the fact.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pymatching
import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from evaluation.logical_error_rate import PreDecoderMemoryEvalModule, _build_stab_maps
from qec.noise_model import NoiseModel
from qec.surface_code.memory_circuit import MemoryCircuit
from data.predecoder_transform import dets_to_predecoder_inputs
from workflows.config_validator import apply_public_defaults_and_model, validate_public_config
from workflows.run import _load_model


DEFAULT_SETTINGS = Path(__file__).with_name("end_to_end.yaml")


class BatchActionModel(torch.nn.Module):
    """Present the stored `trainY` batch as deterministic logits to eval code."""

    def __init__(self):
        super().__init__()
        self._actions: torch.Tensor | None = None

    def set_actions(self, actions: torch.Tensor) -> None:
        self._actions = actions

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._actions is None:
            raise RuntimeError("oracle actions were not set for this batch")
        if self._actions.shape[0] != x.shape[0]:
            raise RuntimeError("oracle action batch does not match detector batch")
        # Eval threshold is zero, so these logits reproduce binary labels exactly.
        return self._actions.to(dtype=torch.float32, device=x.device).mul(2).sub(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paired three-path surface-code comparison.")
    parser.add_argument("--settings", type=Path, default=DEFAULT_SETTINGS, help="Workflow settings YAML.")
    parser.add_argument("--dataset-dir", type=Path, default=None, help="Override comparison.dataset_dir.")
    parser.add_argument(
        "--project-config", "--config", dest="project_config", type=Path, default=None,
        help="YAML used to obtain model_id (other training-only fields are ignored).",
    )
    parser.add_argument("--checkpoint", type=Path, default=None, help="Override comparison.checkpoint.")
    parser.add_argument("--output", type=Path, default=None, help="Override comparison.output.")
    parser.add_argument("--model-id", type=int, default=None, help="Override comparison.model_id.")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def _repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def resolve_settings(cli: argparse.Namespace) -> SimpleNamespace:
    settings_path = cli.settings.expanduser().resolve()
    if not settings_path.is_file():
        raise FileNotFoundError(f"settings file not found: {settings_path}")
    section = OmegaConf.to_container(OmegaConf.load(settings_path).get("comparison", {}), resolve=True)
    if not isinstance(section, dict):
        raise ValueError("settings comparison section must be a mapping")

    def pick(name: str, *, required: bool = False, fallback=None):
        value = getattr(cli, name)
        if value is None:
            value = section.get(name, fallback)
        if required and value is None:
            raise ValueError(f"missing comparison.{name} in {settings_path}")
        return value

    return SimpleNamespace(
        dataset_dir=_repo_path(pick("dataset_dir", required=True)),
        project_config=_repo_path(pick("project_config", required=True)),
        checkpoint=_repo_path(pick("checkpoint", required=True)),
        output=_repo_path(pick("output", required=True)),
        model_id=pick("model_id"),
        batch_size=int(pick("batch_size", required=True)),
        device=pick("device"),
    )


def load_corpus(dataset_dir: Path) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    metadata_path = dataset_dir / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("artifact") != "labeled_end_to_end_surface_code":
        raise ValueError("dataset is not a labeled_end_to_end_surface_code corpus")
    files = metadata.get("files", {})
    dets = np.load(dataset_dir / files.get("dets_and_obs", ""), mmap_mode="r")
    train_x = np.load(dataset_dir / files.get("train_x", ""), mmap_mode="r")
    train_y = np.load(dataset_dir / files.get("train_y", ""), mmap_mode="r")
    n = int(metadata["num_samples"])
    expected_tensor = (n, 4, int(metadata["n_rounds"]), int(metadata["distance"]), int(metadata["distance"]))
    if dets.shape[0] != n or train_x.shape != expected_tensor or train_y.shape != expected_tensor:
        raise ValueError("stored array shapes do not match metadata")
    return metadata, dets, train_x, train_y


def noise_model_from_metadata(metadata: dict[str, Any]) -> NoiseModel | None:
    noise = metadata.get("noise", {})
    if noise.get("kind") == "simple":
        return None
    if noise.get("kind") != "noise_model" or "parameters" not in noise:
        raise ValueError("unsupported or incomplete noise metadata")
    model = NoiseModel.from_config_dict(noise["parameters"])
    if noise.get("sha256") and model.sha256() != noise["sha256"]:
        raise ValueError("noise metadata fingerprint does not match stored parameters")
    return model


def build_matcher(metadata: dict[str, Any]) -> tuple[pymatching.Matching, int]:
    noise_model = noise_model_from_metadata(metadata)
    p_placeholder = float(noise_model.get_max_probability()) if noise_model else float(metadata["noise"]["p_error"])
    circuit = MemoryCircuit(
        distance=int(metadata["distance"]),
        idle_error=p_placeholder,
        sqgate_error=p_placeholder,
        tqgate_error=p_placeholder,
        spam_error=(2.0 / 3.0) * p_placeholder,
        n_rounds=int(metadata["n_rounds"]),
        basis=str(metadata["basis"]),
        code_rotation=str(metadata["code_rotation"]),
        noise_model=noise_model,
        add_boundary_detectors=True,
    )
    circuit.set_error_rates()
    dem = circuit.stim_circuit.detector_error_model(
        decompose_errors=True, approximate_disjoint_errors=True
    )
    if int(dem.num_detectors) != int(metadata["num_detectors"]):
        raise ValueError("rebuilt detector count differs from the stored corpus")
    return pymatching.Matching.from_detector_error_model(dem), int(dem.num_observables)


def build_model_cfg(args: argparse.Namespace, metadata: dict[str, Any]) -> Any:
    source_cfg = OmegaConf.load(args.project_config)
    model_id = args.model_id if args.model_id is not None else OmegaConf.select(source_cfg, "model_id")
    if model_id is None:
        raise ValueError("--model-id is required when the supplied YAML has no model_id")
    # Ising-fast experiment YAMLs contain training-only extensions such as
    # noise mixtures, which the public-config validator intentionally rejects.
    # The checkpoint architecture only needs model_id; construct a minimal
    # inference config, while the matcher itself is rebuilt from dataset metadata.
    cfg = OmegaConf.create({
        "model_id": int(model_id),
        "distance": int(metadata["distance"]),
        "n_rounds": int(metadata["n_rounds"]),
        "workflow": {"task": "inference"},
        "data": {"code_rotation": str(metadata["code_rotation"])},
    })
    spec = validate_public_config(cfg)
    cfg = apply_public_defaults_and_model(cfg, spec)
    cfg.model_checkpoint_file = str(args.checkpoint.expanduser().resolve())
    cfg.test.meas_basis_test = str(metadata["basis"])
    cfg.test.sampling_mode = "threshold"
    cfg.test.th_data = 0.0
    cfg.test.th_syn = 0.0
    return cfg


def final_failures(
    pipeline: PreDecoderMemoryEvalModule,
    matcher: pymatching.Matching,
    dets_and_obs: np.ndarray,
    num_obs: int,
    batch_size: int,
    device: torch.device,
    action_source: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run one predecoder pipeline and return final-failure/residual-weight rows."""
    total = int(dets_and_obs.shape[0])
    failures = np.zeros(total, dtype=bool)
    residual_weight = np.zeros(total, dtype=np.int32)
    model = pipeline.model
    with torch.no_grad():
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            dets_batch = np.asarray(dets_and_obs[start:end, :-num_obs], dtype=np.uint8)
            obs_batch = np.asarray(dets_and_obs[start:end, -num_obs:], dtype=np.uint8)
            if action_source is not None:
                if not isinstance(model, BatchActionModel):
                    raise TypeError("action_source requires BatchActionModel")
                actions = torch.as_tensor(action_source[start:end], dtype=torch.uint8, device=device)
                model.set_actions(actions)
            output = pipeline(torch.as_tensor(dets_batch, dtype=torch.uint8, device=device))
            pre_l = output[:, 0].to(torch.uint8).cpu().numpy().reshape(-1, 1)
            residual = output[:, 1:].to(torch.uint8).cpu().numpy()
            decoded = np.asarray(matcher.decode_batch(np.ascontiguousarray(residual)), dtype=np.uint8).reshape(obs_batch.shape)
            final_obs = np.bitwise_xor(pre_l, decoded)
            failures[start:end] = np.any(final_obs != obs_batch, axis=1)
            residual_weight[start:end] = residual.sum(axis=1, dtype=np.int32)
    return failures, residual_weight


def baseline_failures(
    matcher: pymatching.Matching, dets_and_obs: np.ndarray, num_obs: int, batch_size: int
) -> np.ndarray:
    total = int(dets_and_obs.shape[0])
    failures = np.zeros(total, dtype=bool)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        dets = np.asarray(dets_and_obs[start:end, :-num_obs], dtype=np.uint8)
        obs = np.asarray(dets_and_obs[start:end, -num_obs:], dtype=np.uint8)
        decoded = np.asarray(matcher.decode_batch(np.ascontiguousarray(dets)), dtype=np.uint8).reshape(obs.shape)
        failures[start:end] = np.any(decoded != obs, axis=1)
    return failures


def verify_input_alignment(
    metadata: dict[str, Any], dets_and_obs: np.ndarray, train_x: np.ndarray, num_obs: int, device: torch.device
) -> None:
    """Fail fast if stored labels no longer align with production preprocessing."""
    take = min(32, int(dets_and_obs.shape[0]))
    dets = torch.as_tensor(dets_and_obs[:take, :-num_obs], dtype=torch.uint8, device=device)
    rebuilt_x, _, _ = dets_to_predecoder_inputs(
        dets,
        distance=int(metadata["distance"]),
        n_rounds=int(metadata["n_rounds"]),
        basis=str(metadata["basis"]),
        code_rotation=str(metadata["code_rotation"]),
    )
    expected = np.asarray(train_x[:take], dtype=np.uint8)
    actual = rebuilt_x.to(torch.uint8).cpu().numpy()
    if not np.array_equal(actual, expected):
        raise ValueError(
            "stored trainX differs from production detector preprocessing; "
            "do not compare its trainY label through this pipeline"
        )


def summarise(failures: np.ndarray) -> dict[str, int | float]:
    errors = int(failures.sum())
    samples = int(failures.size)
    return {"logical_errors": errors, "samples": samples, "ler": errors / samples}


def main() -> None:
    args = resolve_settings(parse_args())
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")
    dataset_dir = args.dataset_dir
    metadata, dets_and_obs, train_x, train_y = load_corpus(dataset_dir)
    matcher, num_obs = build_matcher(metadata)
    if num_obs != 1:
        raise NotImplementedError("this comparison currently supports one logical observable per shot")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    verify_input_alignment(metadata, dets_and_obs, train_x, num_obs, device)
    del train_x  # It remains on disk for audits; production reconstructs it from detector bits.
    cfg = build_model_cfg(args, metadata)
    dist = SimpleNamespace(rank=0, world_size=1, device=device)
    print(f"[load] checkpoint={args.checkpoint}, device={device}")
    model = _load_model(cfg, dist).eval()
    maps = _build_stab_maps(int(metadata["distance"]), str(metadata["code_rotation"]))
    model_pipeline = PreDecoderMemoryEvalModule(model, cfg, maps, device).to(device).eval()
    oracle_model = BatchActionModel().to(device).eval()
    oracle_pipeline = PreDecoderMemoryEvalModule(oracle_model, cfg, maps, device).to(device).eval()

    print("[run] PyMatching baseline")
    pymatching_fail = baseline_failures(matcher, dets_and_obs, num_obs, args.batch_size)
    print("[run] Ising-fast checkpoint + PyMatching")
    model_fail, model_residual_weight = final_failures(
        model_pipeline, matcher, dets_and_obs, num_obs, args.batch_size, device
    )
    print("[run] paired trainY oracle + PyMatching")
    oracle_fail, oracle_residual_weight = final_failures(
        oracle_pipeline, matcher, dets_and_obs, num_obs, args.batch_size, device, action_source=train_y
    )

    helpful = pymatching_fail & ~model_fail
    harmful = ~pymatching_fail & model_fail
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    per_shot_path = output.with_suffix(".per_shot.npz")
    np.savez_compressed(
        per_shot_path,
        pymatching_failure=pymatching_fail,
        model_failure=model_fail,
        oracle_failure=oracle_fail,
        model_residual_weight=model_residual_weight,
        oracle_residual_weight=oracle_residual_weight,
    )
    report = {
        "dataset_dir": str(dataset_dir),
        "checkpoint": str(args.checkpoint),
        "model_id": int(cfg.model_id),
        "distance": int(metadata["distance"]),
        "n_rounds": int(metadata["n_rounds"]),
        "basis": str(metadata["basis"]),
        "paths": {
            "pymatching": summarise(pymatching_fail),
            "ising_fast_plus_pymatching": summarise(model_fail),
            "paired_trainy_oracle_plus_pymatching": summarise(oracle_fail),
        },
        "model_vs_pymatching": {
            "helpful_shots": int(helpful.sum()),
            "harmful_shots": int(harmful.sum()),
            "both_fail": int((pymatching_fail & model_fail).sum()),
            "both_succeed": int((~pymatching_fail & ~model_fail).sum()),
        },
        "residual_weight": {
            "model_mean": float(model_residual_weight.mean()),
            "oracle_mean": float(oracle_residual_weight.mean()),
            "oracle_nonzero_shots": int((oracle_residual_weight != 0).sum()),
        },
        "per_shot_file": str(per_shot_path),
    }
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["paths"], indent=2))
    print(f"[done] report={output}\n[done] per-shot={per_shot_path}")


if __name__ == "__main__":
    main()
