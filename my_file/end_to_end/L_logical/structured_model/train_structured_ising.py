#!/usr/bin/env python3
"""Train the integrated structured-action Ising-fast model.

Set ``structured_training.phase`` in ``settings.yaml`` to ``oracle`` first.
After generating a teacher corpus, change it to ``teacher``; the configured
``resume_checkpoint`` is then used to optimize the combined oracle and
endpoint-teacher objective.  This script intentionally has no command-line
configuration switches.
"""
from __future__ import annotations

import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from common import DEFAULT_SETTINGS, build_structured_model, repo_path, save_checkpoint, section
from compare_three_paths import load_corpus
from structured_actions import structured_cross_entropy


def split_rows(total: int, train_fraction: float, validation_fraction: float, seed: int):
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1 or train_fraction + validation_fraction >= 1:
        raise ValueError("train/validation fractions must be positive and sum to less than one")
    order = np.random.default_rng(seed).permutation(total)
    a, b = int(total * train_fraction), int(total * (train_fraction + validation_fraction))
    return np.sort(order[:a]), np.sort(order[a:b]), np.sort(order[b:])


class ActionDataset(Dataset):
    def __init__(self, train_x: np.ndarray, train_y: np.ndarray, rows: np.ndarray,
                 teacher_actions: np.ndarray | None = None, teacher_mask: np.ndarray | None = None,
                 teacher_row_lookup: dict[int, int] | None = None):
        self.train_x, self.train_y, self.rows = train_x, train_y, rows
        self.teacher_actions, self.teacher_mask, self.lookup = teacher_actions, teacher_mask, teacher_row_lookup

    def __len__(self): return len(self.rows)

    def __getitem__(self, index: int):
        row = int(self.rows[index])
        values = [
            torch.from_numpy(np.array(self.train_x[row], dtype=np.float32, copy=True)),
            torch.from_numpy(np.array(self.train_y[row], dtype=np.uint8, copy=True)),
        ]
        if self.teacher_actions is not None:
            teacher_row = self.lookup[row]
            values.extend((
                torch.from_numpy(np.array(self.teacher_actions[teacher_row], dtype=np.uint8, copy=True)),
                torch.from_numpy(np.array(self.teacher_mask[teacher_row], dtype=np.bool_, copy=True)),
            ))
        return tuple(values)


def run_epoch(model, loader, optimizer, device, endpoint_weight: float) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    sums = {"loss": 0.0, "oracle": 0.0, "endpoint": 0.0, "count": 0}
    for batch in loader:
        train_x, train_y = batch[0].to(device), batch[1].to(device)
        with torch.set_grad_enabled(training):
            logits = model(train_x)
            oracle = structured_cross_entropy(logits, train_y).total
            endpoint = logits.new_zeros(())
            if len(batch) == 4:
                teacher_actions, teacher_mask = batch[2].to(device), batch[3].to(device)
                endpoint = structured_cross_entropy(logits, teacher_actions, teacher_mask).total
            loss = oracle + endpoint_weight * endpoint
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        n = int(train_x.shape[0])
        sums["loss"] += float(loss.detach()) * n
        sums["oracle"] += float(oracle.detach()) * n
        sums["endpoint"] += float(endpoint.detach()) * n
        sums["count"] += n
    return {key: value / sums["count"] for key, value in sums.items() if key != "count"}


def main() -> None:
    settings = DEFAULT_SETTINGS.resolve()
    model_cfg, workflow = section(settings, "structured_model"), section(settings, "structured_training")
    phase = str(workflow.get("phase", "")).lower()
    if phase not in {"oracle", "teacher"}:
        raise ValueError("structured_training.phase must be 'oracle' or 'teacher'")
    cfg = section(settings, "structured_oracle_training" if phase == "oracle" else "structured_teacher_training")
    get = lambda name, default=None: cfg.get(name, default)
    dataset_dir = repo_path(model_cfg["dataset_dir"])
    base_checkpoint = repo_path(model_cfg["base_checkpoint"])
    project_config = repo_path(model_cfg["project_config"])
    model_id = model_cfg.get("model_id")
    output_dir = repo_path(get("output_dir"))
    epochs, batch_size = int(get("epochs")), int(get("batch_size"))
    learning_rate = float(get("learning_rate"))
    endpoint_weight = float(cfg.get("endpoint_weight", 0.0))
    device = torch.device(get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    resume_value = workflow.get("resume_checkpoint")
    if phase == "teacher" and resume_value is None:
        raise ValueError("teacher phase requires structured_training.resume_checkpoint")
    metadata, _, train_x, train_y = load_corpus(dataset_dir)
    resume = repo_path(resume_value) if resume_value else None
    model, _ = build_structured_model(metadata, project_config, base_checkpoint, model_id, device, resume)
    seed = int(cfg.get("split_seed", 12345))
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    train_rows, validation_rows, test_rows = split_rows(
        len(train_x), float(cfg.get("train_fraction", 0.70)), float(cfg.get("validation_fraction", 0.15)), seed
    )
    teacher_actions = teacher_mask = lookup = None
    if phase == "teacher":
        teacher_dir = repo_path(cfg["teacher_dataset_dir"])
        teacher_actions = np.load(teacher_dir / "teacher_actions.npy", mmap_mode="r")
        teacher_mask = np.load(teacher_dir / "teacher_mask.npy", mmap_mode="r")
        source_rows = np.load(teacher_dir / "source_indices.npy", mmap_mode="r")
        lookup = {int(row): i for i, row in enumerate(source_rows)}
        train_rows = np.asarray([row for row in train_rows if int(row) in lookup], dtype=np.int64)
        validation_rows = np.asarray([row for row in validation_rows if int(row) in lookup], dtype=np.int64)
        if not len(train_rows) or not len(validation_rows):
            raise ValueError("teacher corpus does not overlap the configured train/validation split")
    train_set = ActionDataset(train_x, train_y, train_rows, teacher_actions, teacher_mask, lookup)
    valid_set = ActionDataset(train_x, train_y, validation_rows, teacher_actions, teacher_mask, lookup)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=int(cfg.get("num_workers", 0)))
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=int(cfg.get("num_workers", 0)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=float(cfg.get("weight_decay", 1e-5)))
    output_dir.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    print(f"[setup] phase={phase} device={device} train={len(train_set)} validation={len(valid_set)} test={len(test_rows)}")
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        train_metrics = run_epoch(model, train_loader, optimizer, device, endpoint_weight)
        with torch.no_grad():
            valid_metrics = run_epoch(model, valid_loader, None, device, endpoint_weight)
        payload = {"phase": phase, "epoch": epoch, "base_checkpoint": str(base_checkpoint), "settings": str(settings),
                   "train": train_metrics, "validation": valid_metrics, "endpoint_weight": endpoint_weight}
        save_checkpoint(output_dir / f"epoch_{epoch:03d}.pt", model, **payload)
        if valid_metrics["loss"] < best:
            best = valid_metrics["loss"]
            save_checkpoint(output_dir / "best.pt", model, **payload)
        print(f"[epoch {epoch:03d}] train={train_metrics} validation={valid_metrics} elapsed={time.perf_counter()-started:.1f}s")


if __name__ == "__main__":
    main()
