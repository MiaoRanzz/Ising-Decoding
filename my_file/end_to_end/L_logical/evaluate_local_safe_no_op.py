#!/usr/bin/env python3
"""Evaluate a trained local safe-no-op gate on its held-out risk-data split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
CODE_ROOT = REPO_ROOT / "code"
for path in (HERE, CODE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from compare_three_paths import BatchActionModel, baseline_failures, build_matcher, build_model_cfg, load_corpus
from evaluation.logical_error_rate import PreDecoderMemoryEvalModule, _build_stab_maps
from generate_local_risk_dataset import evaluate_actions
from local_safe_no_op import GateArchitecture, LocalSafeNoOpGate, apply_packet_gate, gate_features


DEFAULT_SETTINGS = HERE / "end_to_end.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate packet-level safe no-op versus PyMatching and proposal.")
    parser.add_argument("--settings", type=Path, default=DEFAULT_SETTINGS)
    parser.add_argument("--risk-dataset-dir", type=Path, default=None)
    parser.add_argument("--gate-checkpoint", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def _repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def load_settings(cli: argparse.Namespace) -> SimpleNamespace:
    settings = cli.settings.expanduser().resolve()
    section = OmegaConf.to_container(OmegaConf.load(settings).get("local_gate_evaluation", {}), resolve=True)
    if not isinstance(section, dict):
        raise ValueError("local_gate_evaluation must be a mapping")

    def pick(name: str, *, required: bool = False, fallback: Any = None) -> Any:
        # Most evaluation settings intentionally live only in YAML.
        value = getattr(cli, name, None)
        if value is None:
            value = section.get(name, fallback)
        if required and value is None:
            raise ValueError(f"missing local_gate_evaluation.{name} in {settings}")
        return value

    return SimpleNamespace(
        risk_dataset_dir=_repo_path(pick("risk_dataset_dir", required=True)),
        gate_checkpoint=_repo_path(pick("gate_checkpoint", required=True)),
        output=_repo_path(pick("output", required=True)),
        batch_size=int(pick("batch_size", required=True)),
        validation_fraction=float(pick("validation_fraction", required=True)),
        split_seed=int(pick("split_seed", required=True)),
        gate_threshold=float(pick("gate_threshold", fallback=0.0)),
        device=cli.device if cli.device is not None else pick("device"),
    )


def summarize(failure: np.ndarray) -> dict[str, int | float]:
    return {"logical_errors": int(failure.sum()), "samples": int(len(failure)), "ler": float(failure.mean())}


def main() -> None:
    args = load_settings(parse_args())
    if not args.gate_checkpoint.is_file():
        raise FileNotFoundError(f"gate checkpoint not found: {args.gate_checkpoint}")
    if not 0.0 < args.validation_fraction < 1.0:
        raise ValueError("validation_fraction must lie between zero and one")
    metadata = json.loads((args.risk_dataset_dir / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("artifact") != "local_safe_no_op_risk_dataset":
        raise ValueError("risk_dataset_dir does not contain a local safe-no-op dataset")
    source_dir = Path(metadata["source_dataset_dir"])
    source_metadata, source_dets, source_x, _ = load_corpus(source_dir)
    source_indices = np.load(args.risk_dataset_dir / metadata["files"]["source_indices"], mmap_mode="r")
    proposal_logits = np.load(args.risk_dataset_dir / metadata["files"]["proposal_logits"], mmap_mode="r")
    proposal_actions_packed = np.load(args.risk_dataset_dir / metadata["files"]["proposal_actions_packed"], mmap_mode="r")
    action_shape = tuple(int(value) for value in metadata["proposal_action_shape"])
    action_values = int(np.prod(action_shape))
    total = len(source_indices)
    rng = np.random.default_rng(args.split_seed)
    held_out = rng.random(total) < args.validation_fraction
    # The training script uses this exact predicate for its validation split.
    rows = np.flatnonzero(held_out)
    if not len(rows):
        raise RuntimeError("held-out split is empty")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = torch.load(args.gate_checkpoint, map_location=device, weights_only=False)
    gate = LocalSafeNoOpGate(GateArchitecture(**checkpoint["architecture"])).to(device).eval()
    gate.load_state_dict(checkpoint["state_dict"])
    if int(checkpoint.get("split_seed", args.split_seed)) != args.split_seed:
        raise ValueError("gate checkpoint split_seed differs from evaluation split_seed")
    if float(checkpoint.get("validation_fraction", args.validation_fraction)) != args.validation_fraction:
        raise ValueError("gate checkpoint validation_fraction differs from evaluation setting")

    matcher, num_obs = build_matcher(source_metadata)
    cfg = build_model_cfg(SimpleNamespace(
        project_config=REPO_ROOT / "conf/config_public.yaml",
        checkpoint=Path(metadata["proposal_checkpoint"]), model_id=int(metadata["proposal_model_id"])
    ), source_metadata)
    action_model = BatchActionModel().to(device).eval()
    maps = _build_stab_maps(int(source_metadata["distance"]), str(source_metadata["code_rotation"]))
    action_pipeline = PreDecoderMemoryEvalModule(action_model, cfg, maps, device).to(device).eval()
    baseline_all = np.zeros(len(rows), dtype=bool)
    proposal_all = np.zeros(len(rows), dtype=bool)
    gated_all = np.zeros(len(rows), dtype=bool)
    packet_total = 0
    packet_accepted = 0

    for start in range(0, len(rows), args.batch_size):
        end = min(start + args.batch_size, len(rows))
        risk_rows = rows[start:end]
        source_rows = np.asarray(source_indices[risk_rows], dtype=np.int64)
        dets = np.array(source_dets[source_rows, :-num_obs], dtype=np.uint8, copy=True)
        obs = np.asarray(source_dets[source_rows, -num_obs:], dtype=np.uint8)
        x = torch.as_tensor(np.array(source_x[source_rows], dtype=np.float32, copy=True), device=device)
        logits = torch.as_tensor(np.array(proposal_logits[risk_rows], dtype=np.float32, copy=True), device=device)
        proposal_np = np.unpackbits(proposal_actions_packed[risk_rows], axis=1, bitorder="little")[:, :action_values]
        proposal = torch.as_tensor(proposal_np.reshape(len(risk_rows), *action_shape), dtype=torch.bool, device=device)
        with torch.no_grad():
            gate_logits = gate(gate_features(x, logits, proposal))
        gated = apply_packet_gate(proposal, gate_logits, args.gate_threshold)
        baseline_all[start:end] = baseline_failures(matcher, np.concatenate((dets, obs), axis=1), num_obs, len(dets))
        proposal_all[start:end] = evaluate_actions(
            action_pipeline, action_model, matcher, dets, obs, proposal.to(torch.uint8).cpu().numpy(), device
        )
        gated_all[start:end] = evaluate_actions(
            action_pipeline, action_model, matcher, dets, obs, gated.to(torch.uint8).cpu().numpy(), device
        )
        active = torch.stack((proposal[:, 0] | proposal[:, 1], proposal[:, 2], proposal[:, 3]), dim=1)
        accepted = gate_logits >= args.gate_threshold
        packet_total += int(active.sum().item())
        packet_accepted += int((active & accepted).sum().item())

    report = {
        "risk_dataset_dir": str(args.risk_dataset_dir), "gate_checkpoint": str(args.gate_checkpoint),
        "held_out_shots": int(len(rows)), "gate_threshold": args.gate_threshold,
        "paths": {"pymatching": summarize(baseline_all), "proposal_plus_pymatching": summarize(proposal_all),
                  "local_safe_no_op_plus_pymatching": summarize(gated_all)},
        "packet_gate": {"active_packets": packet_total, "accepted_packets": packet_accepted,
                        "accept_coverage": packet_accepted / packet_total if packet_total else 0.0},
        "paired_comparison": {
            "gate_helpful_vs_proposal": int((proposal_all & ~gated_all).sum()),
            "gate_harmful_vs_proposal": int((~proposal_all & gated_all).sum()),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"[done] report={args.output}")


if __name__ == "__main__":
    main()
