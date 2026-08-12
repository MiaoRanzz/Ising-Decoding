#!/usr/bin/env python3
"""Run one pre-selected harmful-veto gate once on the test split.

This task deliberately does *not* scan checkpoints or thresholds.  It is for
the final test of a gate/threshold selected on validation data.  It also
reports whether vetoed groups are enriched for the stored one-group harmful
counterfactual label.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from omegaconf import OmegaConf

HERE = Path(__file__).resolve().parent
L_ROOT, REPO_ROOT, CODE_ROOT = HERE.parent, HERE.parents[3], HERE.parents[3] / "code"
for folder in (HERE, L_ROOT, CODE_ROOT):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from compare_three_paths import BatchActionModel, baseline_failures, build_matcher, build_model_cfg, load_corpus
from evaluation.logical_error_rate import PreDecoderMemoryEvalModule, _build_stab_maps
from packet_model.generate_local_risk_dataset import evaluate_actions
from evaluate_local_group_safe_no_op import apply, failures, scores_for, summary, unpack
from local_group_safe_no_op import GroupGateArchitecture, LocalGroupSafeNoOpGate, split_rows


DEFAULT_SETTINGS = L_ROOT / "end_to_end.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test one fixed harmful-veto group gate.")
    parser.add_argument("--settings", type=Path, default=DEFAULT_SETTINGS)
    parser.add_argument("--risk-dataset-dir", type=Path, default=None)
    parser.add_argument("--ising-fast-checkpoint", type=Path, default=None)
    parser.add_argument("--gate-checkpoint", type=Path, default=None)
    parser.add_argument("--veto-threshold", type=float, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def _repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def load_settings(cli: argparse.Namespace) -> SimpleNamespace:
    settings_path = cli.settings.expanduser().resolve()
    section = OmegaConf.to_container(OmegaConf.load(settings_path).get("group_gate_fixed_test", {}), resolve=True)
    if not isinstance(section, dict):
        raise ValueError("group_gate_fixed_test must be a mapping")

    def pick(name: str, *, required: bool = False, fallback=None):
        # Only the explicit override arguments exist on the CLI Namespace;
        # task-only values (batch size, split fractions, ...) come from YAML.
        value = getattr(cli, name, None)
        if value is None:
            value = section.get(name, fallback)
        if required and value is None:
            raise ValueError(f"missing group_gate_fixed_test.{name} in {settings_path}")
        return value

    return SimpleNamespace(
        risk_dataset_dir=_repo_path(pick("risk_dataset_dir", required=True)),
        ising_fast_checkpoint=_repo_path(pick("ising_fast_checkpoint", required=True)),
        gate_checkpoint=_repo_path(pick("gate_checkpoint", required=True)),
        veto_threshold=float(pick("veto_threshold", required=True)),
        output=_repo_path(pick("output", required=True)),
        batch_size=int(pick("batch_size", required=True)),
        train_fraction=float(pick("train_fraction", required=True)),
        validation_fraction=float(pick("validation_fraction", required=True)),
        split_seed=int(pick("split_seed", required=True)),
        progress_every_batches=int(pick("progress_every_batches", fallback=8)),
        device=cli.device if cli.device is not None else pick("device"),
    )


def _same_path(left: Path, right: Path) -> bool:
    """Compare paths without requiring the checkpoint to be loaded twice."""
    return left.expanduser().resolve() == right.expanduser().resolve()


def label_counts(labels: np.ndarray) -> dict[str, int | float]:
    labels = np.asarray(labels, dtype=np.int8)
    total = int(len(labels))
    harmful = int(np.count_nonzero(labels == -1))
    helpful = int(np.count_nonzero(labels == 1))
    neutral = int(np.count_nonzero(labels == 0))
    return {
        "groups": total,
        "harmful": harmful,
        "helpful": helpful,
        "neutral": neutral,
        "harmful_rate": harmful / max(1, total),
        "helpful_rate": helpful / max(1, total),
        "neutral_rate": neutral / max(1, total),
    }


def test_group_ids(rows: np.ndarray, shot_group_ptr: np.ndarray) -> np.ndarray:
    chunks = [np.arange(shot_group_ptr[row], shot_group_ptr[row + 1], dtype=np.int64) for row in rows]
    return np.concatenate(chunks) if chunks else np.empty(0, dtype=np.int64)


def main() -> None:
    args = load_settings(parse_args())
    if not args.gate_checkpoint.is_file():
        raise FileNotFoundError(f"gate checkpoint not found: {args.gate_checkpoint}")
    if not args.ising_fast_checkpoint.is_file():
        raise FileNotFoundError(f"Ising-fast checkpoint not found: {args.ising_fast_checkpoint}")

    metadata = json.loads((args.risk_dataset_dir / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("artifact") != "local_group_safe_no_op_risk_dataset":
        raise ValueError("risk_dataset_dir is not a local_group_safe_no_op_risk_dataset")
    expected_proposal = Path(metadata["proposal_checkpoint"])
    if not _same_path(args.ising_fast_checkpoint, expected_proposal):
        raise ValueError(
            "ising_fast_checkpoint differs from this risk dataset's proposal checkpoint; "
            "regenerate labels and retrain a gate for the new Ising-fast checkpoint"
        )

    source_indices = np.load(args.risk_dataset_dir / "source_indices.npy", mmap_mode="r")
    logits = np.load(args.risk_dataset_dir / "proposal_logits.npy", mmap_mode="r")
    packed = np.load(args.risk_dataset_dir / "proposal_actions_packed.npy", mmap_mode="r")
    shot_group_ptr = np.load(args.risk_dataset_dir / "shot_group_ptr.npy", mmap_mode="r")
    group_member_ptr = np.load(args.risk_dataset_dir / "group_member_ptr.npy", mmap_mode="r")
    group_members = np.load(args.risk_dataset_dir / "group_members.npy", mmap_mode="r")
    group_effect = np.load(args.risk_dataset_dir / "group_effect.npy", mmap_mode="r")
    source_metadata, dets_and_obs, train_x, _ = load_corpus(Path(metadata["source_dataset_dir"]))
    _, _, test_rows = split_rows(len(source_indices), args.train_fraction, args.validation_fraction, args.split_seed)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    saved = torch.load(args.gate_checkpoint, map_location=device, weights_only=False)
    if saved.get("gate_target") != "harmful_veto":
        raise ValueError("gate_checkpoint is not a harmful-veto checkpoint")
    gate = LocalGroupSafeNoOpGate(GroupGateArchitecture(**saved["architecture"])).to(device).eval()
    gate.load_state_dict(saved["state_dict"])

    matcher, num_obs = build_matcher(source_metadata)
    cfg = build_model_cfg(SimpleNamespace(
        project_config=Path(metadata["project_config"]),
        checkpoint=args.ising_fast_checkpoint,
        model_id=metadata["proposal_model_id"],
    ), source_metadata)
    maps = _build_stab_maps(int(source_metadata["distance"]), str(source_metadata["code_rotation"]))
    action_model = BatchActionModel().to(device).eval()
    pipeline = PreDecoderMemoryEvalModule(action_model, cfg, maps, device).to(device).eval()

    action_shape = tuple(metadata["proposal_action_shape"])
    proposal = unpack(packed, test_rows, action_shape)
    print(f"[test] shots={len(test_rows)}, gate={args.gate_checkpoint.name}, veto_threshold={args.veto_threshold:.6g}")
    scores = scores_for(
        gate, test_rows, train_x, source_indices, logits, shot_group_ptr, group_member_ptr, group_members,
        args.batch_size, device, "fixed-test gate scoring", args.progress_every_batches,
    )
    gated, accepted, active = apply(
        proposal, test_rows, scores, args.veto_threshold, shot_group_ptr, group_member_ptr, group_members,
    )
    print(f"[test] accepted={accepted}/{active}; vetoed={active - accepted}/{active}")

    source_rows = source_indices[test_rows]
    pymatching_failure = baseline_failures(
        matcher, np.array(dets_and_obs[source_rows], dtype=np.uint8, copy=True), num_obs, args.batch_size,
    )
    proposal_failure = failures(
        test_rows, proposal, dets_and_obs, source_indices, num_obs, pipeline, action_model, matcher,
        device, args.batch_size, "fixed-test proposal", args.progress_every_batches,
    )
    gate_failure = failures(
        test_rows, gated, dets_and_obs, source_indices, num_obs, pipeline, action_model, matcher,
        device, args.batch_size, "fixed-test harmful-veto", args.progress_every_batches,
    )

    all_group_ids = test_group_ids(test_rows, shot_group_ptr)
    veto_mask = scores[all_group_ids] >= args.veto_threshold
    all_counts = label_counts(group_effect[all_group_ids])
    veto_counts = label_counts(group_effect[all_group_ids][veto_mask])
    harmful_precision = veto_counts["harmful_rate"]
    harmful_recall = veto_counts["harmful"] / max(1, all_counts["harmful"])
    harmful_lift = harmful_precision / max(all_counts["harmful_rate"], np.finfo(float).tiny)
    enrichment = {
        "all_active_groups": all_counts,
        "vetoed_groups": veto_counts,
        "harmful_precision": harmful_precision,
        "harmful_recall": harmful_recall,
        "harmful_lift": harmful_lift,
        "helpful_false_veto_rate": veto_counts["helpful_rate"],
    }
    report = {
        "risk_dataset_dir": str(args.risk_dataset_dir),
        "ising_fast_checkpoint": str(args.ising_fast_checkpoint),
        "gate_checkpoint": str(args.gate_checkpoint),
        "gate_target": "harmful_veto",
        "veto_threshold": args.veto_threshold,
        "held_out_shots": int(len(test_rows)),
        "paths": {
            "pymatching": summary(pymatching_failure),
            "proposal_plus_pymatching": summary(proposal_failure),
            "local_group_safe_no_op_plus_pymatching": summary(gate_failure),
        },
        "group_gate": {
            "active_groups": int(active),
            "accepted_groups": int(accepted),
            "accept_coverage": accepted / max(1, active),
            "vetoed_groups": int(active - accepted),
            "veto_coverage": (active - accepted) / max(1, active),
        },
        "counterfactual_label_enrichment": enrichment,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"[done] report={args.output}")


if __name__ == "__main__":
    main()
