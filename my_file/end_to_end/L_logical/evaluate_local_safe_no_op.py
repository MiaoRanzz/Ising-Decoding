#!/usr/bin/env python3
"""Select a local safe-no-op gate by validation endpoint LER, then test once.

Checkpoint and threshold selection use only the deterministic validation split.
The test split is never used for selection, so its reported LER is an
independent final estimate for this fixed risk dataset.
"""

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
from local_safe_no_op import GateArchitecture, LocalSafeNoOpGate, gate_features, split_rows


DEFAULT_SETTINGS = HERE / "end_to_end.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select a safe-no-op gate using validation endpoint LER.")
    parser.add_argument("--settings", type=Path, default=DEFAULT_SETTINGS)
    parser.add_argument("--risk-dataset-dir", type=Path, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
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
        value = getattr(cli, name, None)
        if value is None:
            value = section.get(name, fallback)
        if required and value is None:
            raise ValueError(f"missing local_gate_evaluation.{name} in {settings}")
        return value

    return SimpleNamespace(
        risk_dataset_dir=_repo_path(pick("risk_dataset_dir", required=True)),
        checkpoint_dir=_repo_path(pick("checkpoint_dir", required=True)),
        checkpoint_glob=str(pick("checkpoint_glob", fallback="epoch_*.pt")),
        output=_repo_path(pick("output", required=True)),
        batch_size=int(pick("batch_size", required=True)),
        train_fraction=float(pick("train_fraction", required=True)),
        validation_fraction=float(pick("validation_fraction", required=True)),
        split_seed=int(pick("split_seed", required=True)),
        num_thresholds=int(pick("num_thresholds", fallback=21)),
        device=cli.device if cli.device is not None else pick("device"),
    )


def summarize(failure: np.ndarray) -> dict[str, int | float]:
    return {"logical_errors": int(failure.sum()), "samples": int(len(failure)), "ler": float(failure.mean())}


def unpack_actions(packed: np.ndarray, rows: np.ndarray, action_shape: tuple[int, ...]) -> np.ndarray:
    values = int(np.prod(action_shape))
    bits = np.unpackbits(packed[rows], axis=1, bitorder="little")[:, :values]
    return np.asarray(bits.reshape(len(rows), *action_shape), dtype=bool)


def packet_activity(proposal: np.ndarray) -> np.ndarray:
    return np.stack((proposal[:, 0] | proposal[:, 1], proposal[:, 2], proposal[:, 3]), axis=1)


def apply_gate(proposal: np.ndarray, gate_logits: np.ndarray, threshold: float) -> tuple[np.ndarray, int, int]:
    accepted = gate_logits >= threshold
    gated = proposal.copy()
    gated[:, 0] &= accepted[:, 0]
    gated[:, 1] &= accepted[:, 0]
    gated[:, 2] &= accepted[:, 1]
    gated[:, 3] &= accepted[:, 2]
    active = packet_activity(proposal)
    return gated, int(active.sum()), int((active & accepted).sum())


def load_gate(path: Path, device: torch.device, args: SimpleNamespace) -> tuple[LocalSafeNoOpGate, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("artifact") != "local_safe_no_op_gate":
        raise ValueError(f"not a local safe-no-op checkpoint: {path}")
    for name in ("split_seed", "train_fraction", "validation_fraction"):
        expected = getattr(args, name)
        actual = checkpoint.get(name)
        if actual is None or (float(actual) != float(expected) if "fraction" in name else int(actual) != int(expected)):
            raise ValueError(f"checkpoint {path} has incompatible {name}: {actual!r}, expected {expected!r}")
    gate = LocalSafeNoOpGate(GateArchitecture(**checkpoint["architecture"])).to(device).eval()
    gate.load_state_dict(checkpoint["state_dict"])
    return gate, checkpoint


def infer_gate_logits(
    gate: LocalSafeNoOpGate,
    rows: np.ndarray,
    source_indices: np.ndarray,
    source_x: np.ndarray,
    proposal_logits: np.ndarray,
    proposal_actions_packed: np.ndarray,
    action_shape: tuple[int, ...],
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Cache one checkpoint's gate logits for all rows of one split."""
    output = np.empty((len(rows), 3, *action_shape[1:]), dtype=np.float16)
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            end = min(start + batch_size, len(rows))
            risk_rows = rows[start:end]
            source_rows = np.asarray(source_indices[risk_rows], dtype=np.int64)
            x = torch.as_tensor(np.array(source_x[source_rows], dtype=np.float32, copy=True), device=device)
            logits = torch.as_tensor(np.array(proposal_logits[risk_rows], dtype=np.float32, copy=True), device=device)
            proposal = torch.as_tensor(unpack_actions(proposal_actions_packed, risk_rows, action_shape), device=device)
            output[start:end] = gate(gate_features(x, logits, proposal)).to(torch.float32).cpu().numpy().astype(np.float16)
    return output


def threshold_candidates(
    gate_logits: np.ndarray,
    rows: np.ndarray,
    proposal_actions_packed: np.ndarray,
    action_shape: tuple[int, ...],
    num_thresholds: int,
) -> np.ndarray:
    scores: list[np.ndarray] = []
    for start in range(0, len(rows), 4096):
        end = min(start + 4096, len(rows))
        proposal = unpack_actions(proposal_actions_packed, rows[start:end], action_shape)
        scores.append(np.asarray(gate_logits[start:end][packet_activity(proposal)], dtype=np.float32))
    active_scores = np.concatenate(scores) if scores else np.empty(0, dtype=np.float32)
    if not len(active_scores):
        return np.asarray([-np.inf, np.inf], dtype=np.float32)
    quantiles = np.quantile(active_scores, np.linspace(0.0, 1.0, num_thresholds))
    return np.unique(np.concatenate(([-np.inf], quantiles, [0.0, np.inf]))).astype(np.float32)


def evaluate_baseline_and_proposal(
    rows: np.ndarray,
    source_indices: np.ndarray,
    source_dets: np.ndarray,
    proposal_actions_packed: np.ndarray,
    action_shape: tuple[int, ...],
    num_obs: int,
    matcher,
    action_pipeline: PreDecoderMemoryEvalModule,
    action_model: BatchActionModel,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    baseline, proposal_failure = np.zeros(len(rows), dtype=bool), np.zeros(len(rows), dtype=bool)
    for start in range(0, len(rows), batch_size):
        end = min(start + batch_size, len(rows))
        risk_rows = rows[start:end]
        source_rows = np.asarray(source_indices[risk_rows], dtype=np.int64)
        dets = np.array(source_dets[source_rows, :-num_obs], dtype=np.uint8, copy=True)
        obs = np.asarray(source_dets[source_rows, -num_obs:], dtype=np.uint8)
        proposal = unpack_actions(proposal_actions_packed, risk_rows, action_shape)
        baseline[start:end] = baseline_failures(matcher, np.concatenate((dets, obs), axis=1), num_obs, len(dets))
        proposal_failure[start:end] = evaluate_actions(
            action_pipeline, action_model, matcher, dets, obs, proposal.astype(np.uint8), device
        )
    return baseline, proposal_failure


def evaluate_threshold(
    rows: np.ndarray,
    gate_logits: np.ndarray,
    threshold: float,
    source_indices: np.ndarray,
    source_dets: np.ndarray,
    proposal_actions_packed: np.ndarray,
    action_shape: tuple[int, ...],
    num_obs: int,
    matcher,
    action_pipeline: PreDecoderMemoryEvalModule,
    action_model: BatchActionModel,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, int | float]]:
    failures = np.zeros(len(rows), dtype=bool)
    active_total = accepted_total = 0
    for start in range(0, len(rows), batch_size):
        end = min(start + batch_size, len(rows))
        risk_rows = rows[start:end]
        source_rows = np.asarray(source_indices[risk_rows], dtype=np.int64)
        dets = np.array(source_dets[source_rows, :-num_obs], dtype=np.uint8, copy=True)
        obs = np.asarray(source_dets[source_rows, -num_obs:], dtype=np.uint8)
        proposal = unpack_actions(proposal_actions_packed, risk_rows, action_shape)
        gated, active, accepted = apply_gate(proposal, gate_logits[start:end], threshold)
        failures[start:end] = evaluate_actions(
            action_pipeline, action_model, matcher, dets, obs, gated.astype(np.uint8), device
        )
        active_total += active
        accepted_total += accepted
    return failures, {
        "active_packets": active_total,
        "accepted_packets": accepted_total,
        "accept_coverage": accepted_total / active_total if active_total else 0.0,
    }


def main() -> None:
    args = load_settings(parse_args())
    if args.batch_size <= 0 or args.num_thresholds < 2:
        raise ValueError("batch_size must be positive and num_thresholds must be at least two")
    metadata = json.loads((args.risk_dataset_dir / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("artifact") != "local_safe_no_op_risk_dataset":
        raise ValueError("risk_dataset_dir does not contain a local safe-no-op dataset")
    checkpoints = sorted(args.checkpoint_dir.glob(args.checkpoint_glob))
    if not checkpoints:
        raise FileNotFoundError(f"no checkpoints match {args.checkpoint_glob!r} in {args.checkpoint_dir}")
    source_dir = Path(metadata["source_dataset_dir"])
    source_metadata, source_dets, source_x, _ = load_corpus(source_dir)
    source_indices = np.load(args.risk_dataset_dir / metadata["files"]["source_indices"], mmap_mode="r")
    proposal_logits = np.load(args.risk_dataset_dir / metadata["files"]["proposal_logits"], mmap_mode="r")
    proposal_actions_packed = np.load(args.risk_dataset_dir / metadata["files"]["proposal_actions_packed"], mmap_mode="r")
    action_shape = tuple(int(value) for value in metadata["proposal_action_shape"])
    train_rows, validation_rows, test_rows = split_rows(
        len(source_indices), args.train_fraction, args.validation_fraction, args.split_seed
    )
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    matcher, num_obs = build_matcher(source_metadata)
    cfg = build_model_cfg(SimpleNamespace(
        project_config=REPO_ROOT / "conf/config_public.yaml",
        checkpoint=Path(metadata["proposal_checkpoint"]), model_id=int(metadata["proposal_model_id"])
    ), source_metadata)
    action_model = BatchActionModel().to(device).eval()
    maps = _build_stab_maps(int(source_metadata["distance"]), str(source_metadata["code_rotation"]))
    action_pipeline = PreDecoderMemoryEvalModule(action_model, cfg, maps, device).to(device).eval()
    print(f"[select] {len(checkpoints)} checkpoint(s), {len(validation_rows)} validation shots, "
          f"{len(test_rows)} untouched test shots")
    validation_baseline, validation_proposal = evaluate_baseline_and_proposal(
        validation_rows, source_indices, source_dets, proposal_actions_packed, action_shape, num_obs, matcher,
        action_pipeline, action_model, args.batch_size, device,
    )
    candidate_reports: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    selected_gate_logits: np.ndarray | None = None
    for checkpoint_path in checkpoints:
        gate, checkpoint = load_gate(checkpoint_path, device, args)
        print(f"[select] epoch={checkpoint.get('epoch')} checkpoint={checkpoint_path.name}")
        gate_logits = infer_gate_logits(
            gate, validation_rows, source_indices, source_x, proposal_logits, proposal_actions_packed,
            action_shape, args.batch_size, device,
        )
        sweep: list[dict[str, Any]] = []
        for threshold in threshold_candidates(
            gate_logits, validation_rows, proposal_actions_packed, action_shape, args.num_thresholds
        ):
            failure, packet_metrics = evaluate_threshold(
                validation_rows, gate_logits, float(threshold), source_indices, source_dets,
                proposal_actions_packed, action_shape, num_obs, matcher, action_pipeline, action_model,
                args.batch_size, device,
            )
            sweep.append({"threshold": float(threshold), "gated": summarize(failure), "packet_gate": packet_metrics})
        best = min(sweep, key=lambda row: (row["gated"]["ler"], row["packet_gate"]["accept_coverage"]))
        candidate = {"checkpoint": str(checkpoint_path), "epoch": int(checkpoint.get("epoch", -1)),
                     "selected_validation": best, "threshold_sweep": sweep}
        candidate_reports.append(candidate)
        candidate_key = (best["gated"]["ler"], best["packet_gate"]["accept_coverage"])
        if selected is None or candidate_key < (selected["selected_validation"]["gated"]["ler"],
                                                  selected["selected_validation"]["packet_gate"]["accept_coverage"]):
            selected = candidate
            selected_gate_logits = None
        del gate_logits
    if selected is None:
        raise RuntimeError("no candidate checkpoint was selected")
    selected_path = Path(selected["checkpoint"])
    gate, _ = load_gate(selected_path, device, args)
    test_gate_logits = infer_gate_logits(
        gate, test_rows, source_indices, source_x, proposal_logits, proposal_actions_packed,
        action_shape, args.batch_size, device,
    )
    test_baseline, test_proposal = evaluate_baseline_and_proposal(
        test_rows, source_indices, source_dets, proposal_actions_packed, action_shape, num_obs, matcher,
        action_pipeline, action_model, args.batch_size, device,
    )
    selected_threshold = float(selected["selected_validation"]["threshold"])
    test_gated, test_packet_metrics = evaluate_threshold(
        test_rows, test_gate_logits, selected_threshold, source_indices, source_dets, proposal_actions_packed,
        action_shape, num_obs, matcher, action_pipeline, action_model, args.batch_size, device,
    )
    report = {
        "risk_dataset_dir": str(args.risk_dataset_dir),
        "split": {"seed": args.split_seed, "train_fraction": args.train_fraction,
                  "validation_fraction": args.validation_fraction, "train_shots": int(len(train_rows)),
                  "validation_shots": int(len(validation_rows)), "test_shots": int(len(test_rows))},
        "selection_baselines": {"pymatching": summarize(validation_baseline),
                                "proposal_plus_pymatching": summarize(validation_proposal)},
        "candidate_selection": candidate_reports,
        "selected": selected,
        "test": {
            "gate_threshold": selected_threshold,
            "paths": {"pymatching": summarize(test_baseline), "proposal_plus_pymatching": summarize(test_proposal),
                      "local_safe_no_op_plus_pymatching": summarize(test_gated)},
            "packet_gate": test_packet_metrics,
            "paired_comparison": {"gate_helpful_vs_proposal": int((test_proposal & ~test_gated).sum()),
                                  "gate_harmful_vs_proposal": int((~test_proposal & test_gated).sum())},
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"selected": selected, "test": report["test"]}, indent=2))
    print(f"[done] report={args.output}")


if __name__ == "__main__":
    main()
