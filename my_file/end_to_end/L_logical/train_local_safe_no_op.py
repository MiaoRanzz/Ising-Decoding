#!/usr/bin/env python3
"""Train the local packet-level safe-no-op gate on counterfactual labels."""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from local_safe_no_op import GateArchitecture, LocalSafeNoOpGate, NO_PACKET_LABEL, gate_features


DEFAULT_SETTINGS = HERE / "end_to_end.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a local counterfactual safe-no-op gate.")
    parser.add_argument("--settings", type=Path, default=DEFAULT_SETTINGS)
    parser.add_argument("--risk-dataset-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def _repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def load_settings(cli: argparse.Namespace) -> SimpleNamespace:
    settings_path = cli.settings.expanduser().resolve()
    section = OmegaConf.to_container(OmegaConf.load(settings_path).get("local_gate_training", {}), resolve=True)
    if not isinstance(section, dict):
        raise ValueError("local_gate_training must be a mapping")

    def pick(name: str, *, required: bool = False, fallback: Any = None) -> Any:
        # Only a small set of settings have command-line overrides.  The rest
        # must safely fall through to the task's YAML section.
        value = getattr(cli, name, None)
        if value is None:
            value = section.get(name, fallback)
        if required and value is None:
            raise ValueError(f"missing local_gate_training.{name} in {settings_path}")
        return value

    architecture_cfg = section.get("architecture", {})
    if not isinstance(architecture_cfg, dict):
        raise ValueError("local_gate_training.architecture must be a mapping")
    architecture = GateArchitecture(**architecture_cfg)
    return SimpleNamespace(
        risk_dataset_dir=_repo_path(pick("risk_dataset_dir", required=True)),
        output_dir=_repo_path(pick("output_dir", required=True)),
        batch_size=int(pick("batch_size", required=True)),
        epochs=int(pick("epochs", required=True)),
        learning_rate=float(pick("learning_rate", required=True)),
        weight_decay=float(pick("weight_decay", fallback=0.0)),
        validation_fraction=float(pick("validation_fraction", fallback=0.2)),
        split_seed=int(pick("split_seed", fallback=20260803)),
        max_positive_weight=float(pick("max_positive_weight", fallback=100.0)),
        num_workers=int(pick("num_workers", fallback=0)),
        device=cli.device if cli.device is not None else pick("device"),
        architecture=architecture,
    )


class RiskDataset(Dataset):
    """Read source trainX and generated proposal/risk tensors without copying all data."""

    def __init__(self, risk_dir: Path, rows: np.ndarray):
        metadata = json.loads((risk_dir / "metadata.json").read_text(encoding="utf-8"))
        if metadata.get("artifact") != "local_safe_no_op_risk_dataset":
            raise ValueError("risk_dataset_dir does not contain a local_safe_no_op_risk_dataset")
        self.metadata = metadata
        self.rows = np.asarray(rows, dtype=np.int64)
        source_dir = Path(metadata["source_dataset_dir"])
        source_metadata = json.loads((source_dir / "metadata.json").read_text(encoding="utf-8"))
        self.train_x = np.load(source_dir / source_metadata["files"]["train_x"], mmap_mode="r")
        self.source_indices = np.load(risk_dir / metadata["files"]["source_indices"], mmap_mode="r")
        self.logits = np.load(risk_dir / metadata["files"]["proposal_logits"], mmap_mode="r")
        self.action_packed = np.load(risk_dir / metadata["files"]["proposal_actions_packed"], mmap_mode="r")
        self.effect = np.load(risk_dir / metadata["files"]["packet_effect"], mmap_mode="r")
        self.action_shape = tuple(int(value) for value in metadata["proposal_action_shape"])
        if (len(self.source_indices) != len(self.logits) or self.effect.shape[0] != len(self.logits)
                or self.action_packed.shape[0] != len(self.logits)):
            raise ValueError("risk dataset arrays disagree on sample count")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = int(self.rows[item])
        source_row = int(self.source_indices[row])
        x = torch.from_numpy(np.array(self.train_x[source_row], dtype=np.float32, copy=True))
        logits = torch.from_numpy(np.array(self.logits[row], dtype=np.float32, copy=True))
        actions_np = np.unpackbits(self.action_packed[row], bitorder="little")[:int(np.prod(self.action_shape))]
        actions = torch.from_numpy(np.array(actions_np.reshape(self.action_shape), dtype=np.bool_, copy=True))
        effect = torch.from_numpy(np.array(self.effect[row], dtype=np.int8, copy=True))
        target = (effect == 1).to(torch.float32)
        mask = effect != NO_PACKET_LABEL
        return gate_features(x.unsqueeze(0), logits.unsqueeze(0), actions.unsqueeze(0)).squeeze(0), target, mask


def count_labels(effect: np.ndarray, rows: np.ndarray, batch_size: int) -> dict[str, int]:
    counts = {"helpful": 0, "neutral": 0, "harmful": 0, "active": 0}
    for start in range(0, len(rows), batch_size):
        block = np.asarray(effect[rows[start:start + batch_size]], dtype=np.int8)
        counts["helpful"] += int((block == 1).sum())
        counts["neutral"] += int((block == 0).sum())
        counts["harmful"] += int((block == -1).sum())
    counts["active"] = counts["helpful"] + counts["neutral"] + counts["harmful"]
    return counts


def run_epoch(model: LocalSafeNoOpGate, loader: DataLoader, device: torch.device,
              positive_weight: float, optimizer: torch.optim.Optimizer | None = None) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "labels": 0, "true_positive": 0, "predicted_positive": 0, "correct": 0}
    for features, target, mask in loader:
        features, target, mask = features.to(device), target.to(device), mask.to(device)
        with torch.set_grad_enabled(training):
            logits = model(features)
            loss_values = F.binary_cross_entropy_with_logits(
                logits, target, reduction="none", pos_weight=torch.tensor(positive_weight, device=device)
            )
            mask_f = mask.to(torch.float32)
            label_count = int(mask.sum().item())
            if label_count == 0:
                continue
            loss = (loss_values * mask_f).sum() / mask_f.sum()
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        predicted = logits >= 0.0
        totals["loss"] += float(loss.detach().item()) * label_count
        totals["labels"] += label_count
        totals["true_positive"] += int((target.bool() & mask).sum().item())
        totals["predicted_positive"] += int((predicted & mask).sum().item())
        totals["correct"] += int(((predicted == target.bool()) & mask).sum().item())
    if totals["labels"] == 0:
        raise RuntimeError("split has no proposed packets to train or validate")
    return {
        "loss": totals["loss"] / totals["labels"],
        "labels": totals["labels"],
        "helpful_rate": totals["true_positive"] / totals["labels"],
        "accept_rate": totals["predicted_positive"] / totals["labels"],
        "accuracy": totals["correct"] / totals["labels"],
    }


def main() -> None:
    args = load_settings(parse_args())
    if args.batch_size <= 0 or args.epochs <= 0:
        raise ValueError("batch_size and epochs must be positive")
    if not 0.0 < args.validation_fraction < 1.0:
        raise ValueError("validation_fraction must lie between zero and one")
    metadata = json.loads((args.risk_dataset_dir / "metadata.json").read_text(encoding="utf-8"))
    total = int(metadata["num_samples"])
    rng = np.random.default_rng(args.split_seed)
    validation = rng.random(total) < args.validation_fraction
    if not validation.any() or validation.all():
        raise RuntimeError("split unexpectedly has an empty training or validation side")
    train_rows, valid_rows = np.flatnonzero(~validation), np.flatnonzero(validation)
    effect = np.load(args.risk_dataset_dir / metadata["files"]["packet_effect"], mmap_mode="r")
    train_counts = count_labels(effect, train_rows, args.batch_size)
    valid_counts = count_labels(effect, valid_rows, args.batch_size)
    if train_counts["helpful"] == 0:
        raise RuntimeError("training split has no helpful packet labels")
    positive_weight = min(args.max_positive_weight,
                          (train_counts["active"] - train_counts["helpful"]) / train_counts["helpful"])

    random.seed(args.split_seed)
    np.random.seed(args.split_seed)
    torch.manual_seed(args.split_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.split_seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    train_set, valid_set = RiskDataset(args.risk_dataset_dir, train_rows), RiskDataset(args.risk_dataset_dir, valid_rows)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              pin_memory=device.type == "cuda")
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                              pin_memory=device.type == "cuda")
    model = LocalSafeNoOpGate(args.architecture).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    best_loss = float("inf")
    print(json.dumps({"device": str(device), "train_labels": train_counts, "validation_labels": valid_counts,
                      "positive_weight": positive_weight}, indent=2))
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, device, positive_weight, optimizer)
        valid_metrics = run_epoch(model, valid_loader, device, positive_weight)
        row = {"epoch": epoch, "train": train_metrics, "validation": valid_metrics}
        history.append(row)
        print(json.dumps(row))
        checkpoint = {
            "schema_version": 1,
            "artifact": "local_safe_no_op_gate",
            "state_dict": model.state_dict(),
            "architecture": asdict(args.architecture),
            "risk_dataset_dir": str(args.risk_dataset_dir.resolve()),
            "split_seed": args.split_seed,
            "validation_fraction": args.validation_fraction,
            "positive_weight": positive_weight,
            "epoch": epoch,
        }
        torch.save(checkpoint, args.output_dir / "last.pt")
        if valid_metrics["loss"] < best_loss:
            best_loss = valid_metrics["loss"]
            torch.save(checkpoint, args.output_dir / "best.pt")
    report = {
        "risk_dataset_dir": str(args.risk_dataset_dir), "output_dir": str(args.output_dir),
        "architecture": asdict(args.architecture), "split": {"seed": args.split_seed,
        "validation_fraction": args.validation_fraction, "train_shots": int(len(train_rows)),
        "validation_shots": int(len(valid_rows))}, "label_counts": {"train": train_counts, "validation": valid_counts},
        "positive_weight": positive_weight, "history": history,
    }
    (args.output_dir / "training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[done] best={args.output_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
