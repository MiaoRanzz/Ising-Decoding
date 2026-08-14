#!/usr/bin/env python3
"""Train the integrated structured-action Ising-fast model.

Set ``structured_training.phase`` in ``settings.yaml`` to ``oracle`` first.
In offline mode, generate a teacher corpus before changing it to ``teacher``.
In strict mode, teacher actions are generated online from the frozen resume
checkpoint.  Both modes optimize the same oracle/endpoint objective and this
script intentionally has no command-line configuration switches.
"""
from __future__ import annotations

import random
import time
import copy
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from common import (DEFAULT_SETTINGS, build_base_model, build_structured_model, endpoint_pipeline,
                    repo_path, save_checkpoint, section)
from compare_three_paths import load_corpus
from generate_endpoint_teacher import endpoint_teacher_batch
from strict_data import StrictSurfaceSampler, strict_sample_counts
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


def run_epoch(model, loader, optimizer, device, endpoint_weight: float, *, epoch: int,
              total_epochs: int, log_every_batches: int) -> dict[str, float]:
    """Run one epoch and emit live batch-level progress for training/validation."""
    training = optimizer is not None
    model.train(training)
    sums = {"loss": 0.0, "oracle": 0.0, "endpoint": 0.0, "count": 0}
    stage = "train" if training else "validation"
    started = time.perf_counter()
    window = {"loss": 0.0, "oracle": 0.0, "endpoint": 0.0, "count": 0}
    total_batches = len(loader)
    for batch_index, batch in enumerate(loader, start=1):
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
        window["loss"] += float(loss.detach()) * n
        window["oracle"] += float(oracle.detach()) * n
        window["endpoint"] += float(endpoint.detach()) * n
        window["count"] += n
        if batch_index % log_every_batches == 0 or batch_index == total_batches:
            elapsed = time.perf_counter() - started
            samples_per_second = sums["count"] / elapsed if elapsed else 0.0
            remaining = (total_batches - batch_index) * elapsed / batch_index if batch_index else 0.0
            lr = optimizer.param_groups[0]["lr"] if training else 0.0
            print(
                f"[{stage} epoch {epoch:03d}/{total_epochs:03d}] "
                f"batch {batch_index}/{total_batches} "
                f"loss={window['loss'] / window['count']:.5f} "
                f"oracle={window['oracle'] / window['count']:.5f} "
                f"endpoint={window['endpoint'] / window['count']:.5f} "
                f"lr={lr:.3e} throughput={samples_per_second:.1f} shots/s "
                f"epoch_eta={remaining:.0f}s",
                flush=True,
            )
            window = {"loss": 0.0, "oracle": 0.0, "endpoint": 0.0, "count": 0}
    return {key: value / sums["count"] for key, value in sums.items() if key != "count"}


def run_strict_epoch(
    model,
    sampler: StrictSurfaceSampler,
    optimizer,
    device: torch.device,
    endpoint_weight: float,
    *,
    epoch: int,
    total_epochs: int,
    num_samples: int,
    batch_size: int,
    stream: str,
    log_every_batches: int,
    teacher_policy=None,
    teacher_contexts: dict | None = None,
    teacher_cfg: dict | None = None,
) -> dict[str, float]:
    """Run an epoch whose shots are generated once and never reused."""
    training = optimizer is not None
    model.train(training)
    total_batches = math.ceil(num_samples / batch_size)
    stage = "train" if training else "validation"
    sums = {"loss": 0.0, "oracle": 0.0, "endpoint": 0.0, "count": 0, "changed": 0, "improved": 0}
    window = {"loss": 0.0, "oracle": 0.0, "endpoint": 0.0, "count": 0}
    started = time.perf_counter()
    epoch_step_base = (epoch - 1) * total_batches
    for batch_index in range(1, total_batches + 1):
        count = min(batch_size, num_samples - (batch_index - 1) * batch_size)
        fresh = sampler.generate(
            stream=stream,
            step=epoch_step_base + batch_index - 1,
            batch_size=count,
            with_endpoint=teacher_policy is not None,
        )
        train_x = fresh.train_x.to(device=device, dtype=torch.float32)
        train_y = fresh.train_y.to(device=device, dtype=torch.uint8)
        teacher_actions = teacher_mask = None
        teacher_stats = {"changed": 0, "endpoint_improved": 0}
        if teacher_policy is not None:
            if fresh.dets_and_obs is None or teacher_contexts is None or teacher_cfg is None:
                raise RuntimeError("strict teacher phase requires endpoint data and contexts")
            with torch.no_grad():
                proposal_logits = teacher_policy(train_x)
            matcher, action_model, pipeline = teacher_contexts[fresh.basis]
            teacher_actions, teacher_mask, teacher_stats = endpoint_teacher_batch(
                proposal_logits,
                fresh.dets_and_obs,
                pipeline=pipeline,
                action_model=action_model,
                matcher=matcher,
                device=device,
                cfg=teacher_cfg,
            )
        with torch.set_grad_enabled(training):
            logits = model(train_x)
            oracle = structured_cross_entropy(logits, train_y).total
            endpoint = logits.new_zeros(())
            if teacher_actions is not None:
                endpoint = structured_cross_entropy(logits, teacher_actions, teacher_mask).total
            loss = oracle + endpoint_weight * endpoint
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        sums["count"] += count
        sums["changed"] += teacher_stats["changed"]
        sums["improved"] += teacher_stats["endpoint_improved"]
        for name, value in (("loss", loss), ("oracle", oracle), ("endpoint", endpoint)):
            scalar = float(value.detach())
            sums[name] += scalar * count
            window[name] += scalar * count
        window["count"] += count
        if batch_index % log_every_batches == 0 or batch_index == total_batches:
            elapsed = time.perf_counter() - started
            remaining = (total_batches - batch_index) * elapsed / batch_index
            lr = optimizer.param_groups[0]["lr"] if training else 0.0
            print(
                f"[{stage} strict epoch {epoch:03d}/{total_epochs:03d}] "
                f"batch {batch_index}/{total_batches} basis={fresh.basis} "
                f"loss={window['loss']/window['count']:.5f} "
                f"oracle={window['oracle']/window['count']:.5f} "
                f"endpoint={window['endpoint']/window['count']:.5f} "
                f"teacher_changed={sums['changed']} teacher_improved={sums['improved']} "
                f"lr={lr:.3e} throughput={sums['count']/elapsed:.1f} shots/s epoch_eta={remaining:.0f}s",
                flush=True,
            )
            window = {"loss": 0.0, "oracle": 0.0, "endpoint": 0.0, "count": 0}
    result = {name: sums[name] / sums["count"] for name in ("loss", "oracle", "endpoint")}
    if teacher_policy is not None:
        result.update({"teacher_changed": sums["changed"], "teacher_endpoint_improved": sums["improved"]})
    return result


def run_strict_training(
    settings: Path,
    model_cfg: dict,
    cfg: dict,
    teacher_generation_cfg: dict,
    *,
    phase: str,
) -> None:
    strict_cfg = section(settings, "structured_strict_data")
    base_checkpoint = repo_path(model_cfg["base_checkpoint"])
    project_config = repo_path(model_cfg["project_config"])
    device = torch.device(cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    sampler = StrictSurfaceSampler(strict_cfg, device)
    metadata = sampler.metadata()
    resume_value = cfg.get("strict_resume_checkpoint", cfg.get("resume_checkpoint"))
    if phase == "teacher" and resume_value is None:
        raise ValueError("teacher phase requires structured_teacher_training.resume_checkpoint")
    resume = repo_path(resume_value) if resume_value else None
    model, _ = build_structured_model(
        metadata, project_config, base_checkpoint, model_cfg.get("model_id"), device, resume
    )
    teacher_policy = copy.deepcopy(model).eval() if phase == "teacher" else None
    if teacher_policy is not None:
        teacher_policy.requires_grad_(False)
    teacher_contexts = None
    if phase == "teacher":
        teacher_contexts = {}
        for basis in sampler.bases:
            basis_metadata = sampler.metadata(basis)
            unused_model, endpoint_cfg = build_base_model(
                basis_metadata, project_config, base_checkpoint, model_cfg.get("model_id"), device
            )
            del unused_model
            teacher_contexts[basis] = endpoint_pipeline(endpoint_cfg, basis_metadata, device)

    epochs, batch_size = int(cfg["epochs"]), int(cfg["batch_size"])
    train_samples, validation_samples, _ = strict_sample_counts(strict_cfg)
    endpoint_weight = float(cfg.get("endpoint_weight", 0.0))
    log_every = int(cfg.get("log_every_batches", 25))
    if min(epochs, batch_size, log_every) <= 0:
        raise ValueError("epochs, batch_size, and log_every_batches must be positive")
    random.seed(sampler.session_seed)
    np.random.seed(sampler.session_seed % (2**32))
    torch.manual_seed(sampler.session_seed)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg.get("weight_decay", 1e-5))
    )
    output_dir = repo_path(cfg.get("strict_output_dir", cfg["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    print(
        f"[setup] phase={phase} data_mode=strict device={device} seed={sampler.session_seed} "
        f"train_per_epoch={train_samples} validation_per_epoch={validation_samples} bases={sampler.bases}",
        flush=True,
    )
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        train_metrics = run_strict_epoch(
            model, sampler, optimizer, device, endpoint_weight,
            epoch=epoch, total_epochs=epochs, num_samples=train_samples, batch_size=batch_size,
            stream="train", log_every_batches=log_every, teacher_policy=teacher_policy,
            teacher_contexts=teacher_contexts, teacher_cfg=teacher_generation_cfg,
        )
        valid_metrics = run_strict_epoch(
            model, sampler, None, device, endpoint_weight,
            epoch=epoch, total_epochs=epochs, num_samples=validation_samples, batch_size=batch_size,
            stream="validation", log_every_batches=log_every, teacher_policy=teacher_policy,
            teacher_contexts=teacher_contexts, teacher_cfg=teacher_generation_cfg,
        )
        payload = {
            "phase": phase, "data_mode": "strict", "epoch": epoch,
            "base_checkpoint": str(base_checkpoint), "settings": str(settings),
            "strict_reference_config": str(sampler.reference_path),
            "strict_train_samples_per_epoch": train_samples,
            "strict_validation_samples_per_epoch": validation_samples,
            "strict_session_seed": sampler.session_seed, "train": train_metrics,
            "validation": valid_metrics, "endpoint_weight": endpoint_weight,
        }
        save_checkpoint(output_dir / f"epoch_{epoch:03d}.pt", model, **payload)
        if valid_metrics["loss"] < best:
            best = valid_metrics["loss"]
            save_checkpoint(output_dir / "best.pt", model, **payload)
        print(
            f"[epoch {epoch:03d}] train={train_metrics} validation={valid_metrics} "
            f"elapsed={time.perf_counter()-started:.1f}s",
            flush=True,
        )


def main() -> None:
    settings = DEFAULT_SETTINGS.resolve()
    model_cfg, workflow = section(settings, "structured_model"), section(settings, "structured_training")
    phase = str(workflow.get("phase", "")).lower()
    if phase not in {"oracle", "teacher"}:
        raise ValueError("structured_training.phase must be 'oracle' or 'teacher'")
    cfg = section(settings, "structured_oracle_training" if phase == "oracle" else "structured_teacher_training")
    data_mode = str(model_cfg.get("data_mode", "offline")).lower()
    if data_mode not in {"offline", "strict"}:
        raise ValueError("structured_model.data_mode must be 'offline' or 'strict'")
    if data_mode == "strict":
        run_strict_training(
            settings, model_cfg, cfg, section(settings, "structured_teacher_generation"), phase=phase
        )
        return
    get = lambda name, default=None: cfg.get(name, default)
    dataset_dir = repo_path(model_cfg["dataset_dir"])
    base_checkpoint = repo_path(model_cfg["base_checkpoint"])
    project_config = repo_path(model_cfg["project_config"])
    model_id = model_cfg.get("model_id")
    output_dir = repo_path(get("output_dir"))
    epochs, batch_size = int(get("epochs")), int(get("batch_size"))
    learning_rate = float(get("learning_rate"))
    log_every_batches = int(cfg.get("log_every_batches", 25))
    if log_every_batches <= 0:
        raise ValueError("log_every_batches must be positive")
    endpoint_weight = float(cfg.get("endpoint_weight", 0.0))
    device = torch.device(get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    resume_value = cfg.get("resume_checkpoint")
    if phase == "teacher" and resume_value is None:
        raise ValueError("teacher phase requires structured_teacher_training.resume_checkpoint")
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
    print(f"[setup] phase={phase} data_mode=offline device={device} train={len(train_set)} validation={len(valid_set)} test={len(test_rows)}")
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        train_metrics = run_epoch(
            model, train_loader, optimizer, device, endpoint_weight,
            epoch=epoch, total_epochs=epochs, log_every_batches=log_every_batches,
        )
        with torch.no_grad():
            valid_metrics = run_epoch(
                model, valid_loader, None, device, endpoint_weight,
                epoch=epoch, total_epochs=epochs, log_every_batches=log_every_batches,
            )
        payload = {"phase": phase, "data_mode": "offline", "epoch": epoch,
                   "base_checkpoint": str(base_checkpoint), "settings": str(settings),
                   "train": train_metrics, "validation": valid_metrics, "endpoint_weight": endpoint_weight}
        save_checkpoint(output_dir / f"epoch_{epoch:03d}.pt", model, **payload)
        if valid_metrics["loss"] < best:
            best = valid_metrics["loss"]
            save_checkpoint(output_dir / "best.pt", model, **payload)
        print(f"[epoch {epoch:03d}] train={train_metrics} validation={valid_metrics} elapsed={time.perf_counter()-started:.1f}s")


if __name__ == "__main__":
    main()
