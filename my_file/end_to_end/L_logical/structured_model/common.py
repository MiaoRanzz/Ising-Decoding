"""Shared I/O and exact endpoint helpers for the structured experiment."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pymatching
import torch
from omegaconf import OmegaConf

HERE = Path(__file__).resolve().parent
L_LOGICAL_ROOT = HERE.parent
REPO_ROOT = HERE.parents[3]
CODE_ROOT = REPO_ROOT / "code"
for path in (HERE, L_LOGICAL_ROOT, CODE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from compare_three_paths import BatchActionModel, build_matcher, build_model_cfg, load_corpus
from evaluation.logical_error_rate import PreDecoderMemoryEvalModule, _build_stab_maps
from workflows.run import _load_model
from structured_actions import StructuredIsingFast, from_ising_fast


DEFAULT_SETTINGS = HERE / "settings.yaml"


def repo_path(value: str | Path) -> Path:
    value = Path(value).expanduser()
    return value if value.is_absolute() else REPO_ROOT / value


def section(settings: Path, name: str) -> dict[str, Any]:
    config = OmegaConf.load(settings)
    result = OmegaConf.to_container(config.get(name, {}), resolve=True)
    if not isinstance(result, dict):
        raise ValueError(f"{name} must be a mapping in {settings}")
    return result


def build_base_model(
    metadata: dict[str, Any], project_config: Path, base_checkpoint: Path, model_id: int | None, device: torch.device
) -> tuple[torch.nn.Module, Any]:
    args = SimpleNamespace(project_config=project_config, checkpoint=base_checkpoint, model_id=model_id)
    cfg = build_model_cfg(args, metadata)
    model = _load_model(cfg, SimpleNamespace(rank=0, world_size=1, device=device)).to(device).eval()
    return model, cfg


def build_structured_model(
    metadata: dict[str, Any], project_config: Path, base_checkpoint: Path, model_id: int | None,
    device: torch.device, structured_checkpoint: Path | None = None,
) -> tuple[StructuredIsingFast, Any]:
    source, cfg = build_base_model(metadata, project_config, base_checkpoint, model_id, device)
    model = from_ising_fast(source).to(device)
    if structured_checkpoint is not None:
        payload = torch.load(structured_checkpoint, map_location=device, weights_only=False)
        state = payload.get("state_dict", payload)
        model.load_state_dict(state)
    return model, cfg


def save_checkpoint(path: Path, model: StructuredIsingFast, **metadata: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "format": "structured_ising_fast_v1", **metadata}, path)


def endpoint_pipeline(cfg: Any, metadata: dict[str, Any], device: torch.device):
    matcher, num_obs = build_matcher(metadata)
    if num_obs != 1:
        raise NotImplementedError("structured surface experiment currently requires one observable")
    action_model = BatchActionModel().to(device).eval()
    maps = _build_stab_maps(int(metadata["distance"]), str(metadata["code_rotation"]))
    pipeline = PreDecoderMemoryEvalModule(action_model, cfg, maps, device).to(device).eval()
    return matcher, action_model, pipeline


def endpoint_outcomes(
    pipeline: PreDecoderMemoryEvalModule,
    action_model: BatchActionModel,
    matcher: pymatching.Matching,
    dets: np.ndarray,
    obs: np.ndarray,
    actions: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact logical-failure and residual-weight outcome for fixed actions."""
    if not (len(dets) == len(obs) == len(actions)):
        raise ValueError("dets, obs, and actions must have equal batch size")
    failures = np.empty(len(actions), dtype=bool)
    residual_weight = np.empty(len(actions), dtype=np.int32)
    for start in range(0, len(actions), batch_size):
        end = min(start + batch_size, len(actions))
        action_model.set_actions(torch.as_tensor(np.array(actions[start:end], dtype=np.uint8, copy=True), device=device))
        with torch.no_grad():
            output = pipeline(torch.as_tensor(np.array(dets[start:end], dtype=np.uint8, copy=True), device=device))
        pre_l = output[:, 0].to(torch.uint8).cpu().numpy().reshape(-1, 1)
        residual = output[:, 1:].to(torch.uint8).cpu().numpy()
        decoded = np.asarray(matcher.decode_batch(np.ascontiguousarray(residual)), dtype=np.uint8).reshape(obs[start:end].shape)
        failures[start:end] = np.any(np.bitwise_xor(pre_l, decoded) != obs[start:end], axis=1)
        residual_weight[start:end] = residual.sum(axis=1, dtype=np.int32)
    return failures, residual_weight


def summarize(failure: np.ndarray) -> dict[str, int | float]:
    errors, samples = int(failure.sum()), int(failure.size)
    return {"logical_errors": errors, "samples": samples, "ler": errors / samples if samples else float("nan")}


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")
