#!/usr/bin/env python3
"""Train the local packet-level safe-no-op gate on counterfactual labels."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
L_LOGICAL_ROOT = HERE.parent
REPO_ROOT = HERE.parents[3]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from local_safe_no_op import GateArchitecture, LocalSafeNoOpGate, NO_PACKET_LABEL, gate_features, split_rows


DEFAULT_SETTINGS = L_LOGICAL_ROOT / "end_to_end.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a local counterfactual safe-no-op gate.")
    parser.add_argument("--settings", type=Path, default=DEFAULT_SETTINGS)
    parser.add_argument("--risk-dataset-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--log-every-batches", type=int, default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def _repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def load_settings(cli: argparse.Namespace) -> SimpleNamespace:
    settings_path = cli.settings.expanduser().resolve()
    section = OmegaConf.to_container(OmegaConf.load(settings_path).get("packet_gate_training", {}), resolve=True)
    if not isinstance(section, dict):
        raise ValueError("packet_gate_training must be a mapping")

    def pick(name: str, *, required: bool = False, fallback: Any = None) -> Any:
        # Only a small set of settings have command-line overrides.  The rest
        # must safely fall through to the task's YAML section.
        value = getattr(cli, name, None)
        if value is None:
            value = section.get(name, fallback)
        if required and value is None:
            raise ValueError(f"missing packet_gate_training.{name} in {settings_path}")
        return value

    architecture_cfg = section.get("architecture", {})
    if not isinstance(architecture_cfg, dict):
        raise ValueError("packet_gate_training.architecture must be a mapping")
    architecture = GateArchitecture(**architecture_cfg)
    return SimpleNamespace(
        risk_dataset_dir=_repo_path(pick("risk_dataset_dir", required=True)),
        output_dir=_repo_path(pick("output_dir", required=True)),
        batch_size=int(pick("batch_size", required=True)),
        epochs=int(pick("epochs", required=True)),
        learning_rate=float(pick("learning_rate", required=True)),
        lr_scheduler=str(pick("lr_scheduler", fallback="cosine")).lower(),
        min_learning_rate=float(pick("min_learning_rate", fallback=0.0)),
        weight_decay=float(pick("weight_decay", fallback=0.0)),
        train_fraction=float(pick("train_fraction", fallback=0.7)),
        validation_fraction=float(pick("validation_fraction", fallback=0.15)),
        split_seed=int(pick("split_seed", fallback=20260803)),
        neutral_per_informative=float(pick("neutral_per_informative", fallback=10.0)),
        max_positive_weight=float(pick("max_positive_weight", fallback=100.0)),
        checkpoint_interval_epochs=int(pick("checkpoint_interval_epochs", fallback=1)),
        num_workers=int(pick("num_workers", fallback=0)),
        log_every_batches=int(pick("log_every_batches", fallback=20)),
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

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = int(self.rows[item])
        source_row = int(self.source_indices[row])
        x = torch.from_numpy(np.array(self.train_x[source_row], dtype=np.float32, copy=True))
        logits = torch.from_numpy(np.array(self.logits[row], dtype=np.float32, copy=True))
        actions_np = np.unpackbits(self.action_packed[row], bitorder="little")[:int(np.prod(self.action_shape))]
        actions = torch.from_numpy(np.array(actions_np.reshape(self.action_shape), dtype=np.bool_, copy=True))
        effect = torch.from_numpy(np.array(self.effect[row], dtype=np.int8, copy=True))
        return gate_features(x.unsqueeze(0), logits.unsqueeze(0), actions.unsqueeze(0)).squeeze(0), effect


def count_labels(effect: np.ndarray, rows: np.ndarray, batch_size: int) -> dict[str, int]:
    counts = {"helpful": 0, "neutral": 0, "harmful": 0, "active": 0}
    for start in range(0, len(rows), batch_size):
        block = np.asarray(effect[rows[start:start + batch_size]], dtype=np.int8)
        counts["helpful"] += int((block == 1).sum())
        counts["neutral"] += int((block == 0).sum())
        counts["harmful"] += int((block == -1).sum())
    counts["active"] = counts["helpful"] + counts["neutral"] + counts["harmful"]
    return counts


def _format_duration(seconds: float) -> str:
    """Human-readable non-negative duration for progress logs."""
    seconds = max(0, int(round(seconds)))
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f"{hours:d}h{minutes:02d}m{seconds:02d}s" if hours else f"{minutes:d}m{seconds:02d}s"


def _loss_targets_and_mask(
    effect: torch.Tensor, neutral_keep_probability: float, training: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep every endpoint-changing label and subsample neutral positions only."""
    target = (effect == 1).to(torch.float32)
    informative = (effect == 1) | (effect == -1)
    neutral = effect == 0
    if training:
        keep_neutral = torch.rand_like(target) < neutral_keep_probability
        mask = informative | (neutral & keep_neutral)
    else:
        # This is diagnostic BCE only. Model selection is done by endpoint LER
        # in evaluate_local_safe_no_op.py, not by this all-active loss.
        mask = effect != NO_PACKET_LABEL
    return target, mask


def run_epoch(
    model: LocalSafeNoOpGate,
    loader: DataLoader,
    device: torch.device,
    positive_weight: float,
    optimizer: torch.optim.Optimizer | None = None,
    *,
    epoch: int | None = None,
    total_epochs: int | None = None,
    log_every_batches: int = 0,
    neutral_keep_probability: float = 1.0,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "labels": 0, "true_positive": 0, "predicted_positive": 0, "correct": 0}
    batches = len(loader)
    epoch_started = time.perf_counter()
    window_loss = 0.0
    window_labels = 0
    previous_window_loss: float | None = None
    processed_shots = 0
    for batch_index, (features, effect) in enumerate(loader, start=1):
        batch_shots = int(features.shape[0])
        features, effect = features.to(device), effect.to(device)
        target, mask = _loss_targets_and_mask(effect, neutral_keep_probability, training)
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
        window_loss += float(loss.detach().item()) * label_count
        window_labels += label_count
        processed_shots += batch_shots

        if training and log_every_batches > 0 and (batch_index % log_every_batches == 0 or batch_index == batches):
            elapsed = time.perf_counter() - epoch_started
            per_batch = elapsed / batch_index
            epoch_eta = per_batch * (batches - batch_index)
            # This is intentionally labelled train-only: validation time has
            # not yet been measured for the current epoch.
            future_train_batches = (batches - batch_index) + max(0, (total_epochs or 1) - (epoch or 1)) * batches
            total_train_eta = per_batch * future_train_batches
            window_average = window_loss / window_labels if window_labels else float("nan")
            delta = "n/a" if previous_window_loss is None else f"{window_average - previous_window_loss:+.5f}"
            samples_per_second = processed_shots / elapsed if elapsed else 0.0
            finish = datetime.now() + timedelta(seconds=total_train_eta)
            print(
                f"[Train Epoch {epoch}/{total_epochs}] Batch {batch_index}/{batches} | "
                f"Loss: {window_average:.5f} | Delta: {delta} | "
                f"LR: {optimizer.param_groups[0]['lr']:.3e} | "
                f"Throughput: {samples_per_second:.1f} shots/s | "
                f"Elapsed: {_format_duration(elapsed)} | "
                f"Epoch ETA: {_format_duration(epoch_eta)} | "
                f"Train ETA: {_format_duration(total_train_eta)} | "
                f"Est. finish: {finish:%Y-%m-%d %H:%M:%S}",
                flush=True,
            )
            previous_window_loss = window_average
            window_loss = 0.0
            window_labels = 0
    if totals["labels"] == 0:
        raise RuntimeError("split has no proposed packets to train or validate")
    return {
        "loss": totals["loss"] / totals["labels"],
        "labels": totals["labels"],
        "helpful_rate": totals["true_positive"] / totals["labels"],
        "accept_rate": totals["predicted_positive"] / totals["labels"],
        "accuracy": totals["correct"] / totals["labels"],
        "elapsed_seconds": time.perf_counter() - epoch_started,
    }


def main() -> None:
    args = load_settings(parse_args())
    if (args.batch_size <= 0 or args.epochs <= 0 or args.learning_rate <= 0 or args.log_every_batches <= 0
            or args.neutral_per_informative < 0 or args.checkpoint_interval_epochs <= 0):
        raise ValueError("batch_size, epochs, learning_rate, log_every_batches, and checkpoint_interval_epochs must be positive; "
                         "neutral_per_informative must be non-negative")
    if args.lr_scheduler not in {"none", "cosine"}:
        raise ValueError("lr_scheduler must be either 'none' or 'cosine'")
    if not 0.0 <= args.min_learning_rate <= args.learning_rate:
        raise ValueError("min_learning_rate must lie between zero and learning_rate")
    metadata = json.loads((args.risk_dataset_dir / "metadata.json").read_text(encoding="utf-8"))
    total = int(metadata["num_samples"])
    train_rows, valid_rows, test_rows = split_rows(
        total, args.train_fraction, args.validation_fraction, args.split_seed
    )
    effect = np.load(args.risk_dataset_dir / metadata["files"]["packet_effect"], mmap_mode="r")
    train_counts = count_labels(effect, train_rows, args.batch_size)
    valid_counts = count_labels(effect, valid_rows, args.batch_size)
    if train_counts["helpful"] == 0:
        raise RuntimeError("training split has no helpful packet labels")
    neutral_keep_probability = min(
        1.0,
        args.neutral_per_informative * (train_counts["helpful"] + train_counts["harmful"])
        / max(1, train_counts["neutral"]),
    )
    expected_negative = train_counts["harmful"] + train_counts["neutral"] * neutral_keep_probability
    positive_weight = min(args.max_positive_weight, expected_negative / train_counts["helpful"])

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
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.min_learning_rate
        ) if args.lr_scheduler == "cosine" else None
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    best_loss = float("inf")
    print(json.dumps({
        "device": str(device), "train_labels": train_counts, "validation_labels": valid_counts,
        "test_shots": int(len(test_rows)), "positive_weight": positive_weight,
        "neutral_per_informative": args.neutral_per_informative,
        "neutral_keep_probability": neutral_keep_probability,
        "lr_scheduler": args.lr_scheduler,
        "min_learning_rate": args.min_learning_rate,
    }, indent=2))
    training_started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        print(f"[Train Epoch {epoch}/{args.epochs}] starting: {len(train_loader)} batches, "
              f"{len(train_set)} shots", flush=True)
        train_metrics = run_epoch(
            model, train_loader, device, positive_weight, optimizer,
            epoch=epoch, total_epochs=args.epochs, log_every_batches=args.log_every_batches,
            neutral_keep_probability=neutral_keep_probability,
        )
        validation_started = time.perf_counter()
        valid_metrics = run_epoch(model, valid_loader, device, positive_weight)
        validation_seconds = time.perf_counter() - validation_started
        total_elapsed = time.perf_counter() - training_started
        average_epoch_seconds = total_elapsed / epoch
        total_eta = average_epoch_seconds * (args.epochs - epoch)
        estimated_finish = datetime.now() + timedelta(seconds=total_eta)
        row = {"epoch": epoch, "train": train_metrics, "validation": valid_metrics,
               "validation_seconds": validation_seconds, "total_elapsed_seconds": total_elapsed,
               "total_eta_seconds": total_eta}
        history.append(row)
        print(
            f"[Epoch {epoch}/{args.epochs}] "
            f"train_loss={train_metrics['loss']:.5f} | valid_bce_diagnostic={valid_metrics['loss']:.5f} | "
            f"train_time={_format_duration(train_metrics['elapsed_seconds'])} | "
            f"valid_time={_format_duration(validation_seconds)} | "
            f"Total ETA: {_format_duration(total_eta)} | "
            f"Est. finish: {estimated_finish:%Y-%m-%d %H:%M:%S}",
            flush=True,
        )
        checkpoint = {
            "schema_version": 1,
            "artifact": "local_safe_no_op_gate",
            "state_dict": model.state_dict(),
            "architecture": asdict(args.architecture),
            "risk_dataset_dir": str(args.risk_dataset_dir.resolve()),
            "split_seed": args.split_seed,
            "train_fraction": args.train_fraction,
            "validation_fraction": args.validation_fraction,
            "positive_weight": positive_weight,
            "neutral_per_informative": args.neutral_per_informative,
            "neutral_keep_probability": neutral_keep_probability,
            "lr_scheduler": args.lr_scheduler,
            "min_learning_rate": args.min_learning_rate,
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "epoch": epoch,
        }
        torch.save(checkpoint, args.output_dir / "last.pt")
        if epoch % args.checkpoint_interval_epochs == 0:
            torch.save(checkpoint, args.output_dir / f"epoch_{epoch:03d}.pt")
        if valid_metrics["loss"] < best_loss:
            best_loss = valid_metrics["loss"]
            torch.save(checkpoint, args.output_dir / "best_bce.pt")
        if scheduler is not None:
            scheduler.step()
    report = {
        "risk_dataset_dir": str(args.risk_dataset_dir), "output_dir": str(args.output_dir),
        "architecture": asdict(args.architecture), "split": {"seed": args.split_seed,
        "train_fraction": args.train_fraction, "validation_fraction": args.validation_fraction,
        "train_shots": int(len(train_rows)), "validation_shots": int(len(valid_rows)),
        "test_shots": int(len(test_rows))}, "label_counts": {"train": train_counts, "validation": valid_counts},
        "neutral_sampling": {"neutral_per_informative": args.neutral_per_informative,
        "neutral_keep_probability": neutral_keep_probability}, "positive_weight": positive_weight,
        "lr_scheduler": args.lr_scheduler, "min_learning_rate": args.min_learning_rate, "history": history,
    }
    (args.output_dir / "training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[done] endpoint selection candidates={args.output_dir / 'epoch_*.pt'}")
    print(f"[done] diagnostic BCE best={args.output_dir / 'best_bce.pt'}")


if __name__ == "__main__":
    main()
