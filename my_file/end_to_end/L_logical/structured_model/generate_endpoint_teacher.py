#!/usr/bin/env python3
"""Generate endpoint-selected structured teacher actions from a frozen policy.

For each shot, candidates retain the current structured action, remove one or
two uncertain active groups, and optionally use all-no-op.  The exact
predecoder + PyMatching endpoint selects the lowest-cost candidate.  Only
positions actually changed by that winning candidate receive teacher loss.
"""
from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import torch

from common import (DEFAULT_SETTINGS, build_structured_model, endpoint_outcomes, endpoint_pipeline,
                    repo_path, section, write_json)
from compare_three_paths import load_corpus
from group_model.local_group_safe_no_op import GroupingConfig, build_groups
from structured_actions import actions_from_logits, packet_mask_from_action_difference


def remove_groups(actions: np.ndarray, groups: list[np.ndarray]) -> np.ndarray:
    output = actions.copy()
    for group in groups:
        for packet, t, y, x in group:
            if packet == 0:
                output[0, t, y, x] = False
                output[1, t, y, x] = False
            else:
                output[packet + 1, t, y, x] = False
    return output


def group_margin(group: np.ndarray, logits: np.ndarray) -> float:
    """Smaller argmax margin means a more uncertain group and is searched first."""
    families = ((0, 4), (4, 6), (6, 8))
    margins = []
    for packet, t, y, x in group:
        values = logits[families[int(packet)][0]:families[int(packet)][1], t, y, x]
        top_two = np.partition(values, -2)[-2:]
        margins.append(float(top_two.max() - top_two.min()))
    return float(np.mean(margins)) if margins else float("inf")


def candidates_for_shot(actions: np.ndarray, logits: np.ndarray, grouping: GroupingConfig,
                        max_groups: int, include_pairs: bool, include_all_no_op: bool):
    groups = build_groups(actions, grouping)
    groups.sort(key=lambda group: group_margin(group, logits))
    groups = groups[:max_groups]
    candidates = [actions]
    for group in groups:
        candidates.append(remove_groups(actions, [group]))
    if include_pairs:
        for left, right in itertools.combinations(groups, 2):
            candidates.append(remove_groups(actions, [left, right]))
    if include_all_no_op:
        candidates.append(np.zeros_like(actions, dtype=bool))
    return candidates


def main() -> None:
    settings = DEFAULT_SETTINGS.resolve()
    model_cfg, cfg = section(settings, "structured_model"), section(settings, "structured_teacher_generation")
    dataset_dir = repo_path(model_cfg["dataset_dir"])
    project_config, base_checkpoint = repo_path(model_cfg["project_config"]), repo_path(model_cfg["base_checkpoint"])
    checkpoint = repo_path(cfg["checkpoint"])
    output_dir = repo_path(cfg["output_dir"])
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is non-empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    device_name = cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    metadata, dets_and_obs, train_x, _ = load_corpus(dataset_dir)
    total = len(train_x)
    requested = cfg.get("num_samples")
    if requested is None:
        source_rows = np.arange(total, dtype=np.int64)
    else:
        requested = int(requested)
        if not 0 < requested <= total:
            raise ValueError("num_samples must be in [1, source corpus size]")
        source_rows = np.sort(np.random.default_rng(int(cfg.get("seed", 12345))).choice(total, requested, replace=False))
    model, endpoint_cfg = build_structured_model(
        metadata, project_config, base_checkpoint, model_cfg.get("model_id"), device, checkpoint
    )
    model.eval()
    matcher, action_model, pipeline = endpoint_pipeline(endpoint_cfg, metadata, device)
    shape = tuple(int(x) for x in train_x.shape[1:])
    count = len(source_rows)
    teacher_actions = np.lib.format.open_memmap(output_dir / "teacher_actions.npy", mode="w+", dtype=np.uint8,
                                                 shape=(count, *shape))
    teacher_mask = np.lib.format.open_memmap(output_dir / "teacher_mask.npy", mode="w+", dtype=np.bool_,
                                              shape=(count, 3, *shape[1:]))
    grouping = GroupingConfig(**dict(cfg.get("grouping", {})))
    proposal_batch_size, endpoint_batch_size = int(cfg["proposal_batch_size"]), int(cfg["endpoint_batch_size"])
    max_groups = int(cfg.get("max_groups_per_shot", 3))
    failure_weight, residual_weight, action_weight = (float(cfg.get(name, default)) for name, default in (
        ("logical_failure_weight", 1000.0), ("residual_weight", 1.0), ("action_weight", 0.01)
    ))
    changed, endpoint_improved = 0, 0
    for start in range(0, count, proposal_batch_size):
        end = min(start + proposal_batch_size, count)
        rows = source_rows[start:end]
        with torch.no_grad():
            logits = model(torch.as_tensor(np.array(train_x[rows], dtype=np.float32, copy=True), device=device))
            base_actions = actions_from_logits(logits).cpu().numpy()
        logits_np = logits.float().cpu().numpy()
        candidate_actions: list[np.ndarray] = []
        candidate_shots: list[int] = []
        candidate_ranges: list[tuple[int, int]] = []
        for local_row, (base, logit) in enumerate(zip(base_actions, logits_np)):
            begin = len(candidate_actions)
            options = candidates_for_shot(base, logit, grouping, max_groups, bool(cfg.get("include_pairs", True)),
                                          bool(cfg.get("include_all_no_op", True)))
            candidate_actions.extend(options)
            candidate_shots.extend([local_row] * len(options))
            candidate_ranges.append((begin, len(candidate_actions)))
        stacked_actions = np.asarray(candidate_actions, dtype=np.uint8)
        local_indices = np.asarray(candidate_shots, dtype=np.int64)
        dets = np.asarray(dets_and_obs[rows, :-1], dtype=np.uint8)
        obs = np.asarray(dets_and_obs[rows, -1:], dtype=np.uint8)
        failures, residuals = endpoint_outcomes(
            pipeline, action_model, matcher, dets[local_indices], obs[local_indices], stacked_actions, device,
            endpoint_batch_size,
        )
        costs = failure_weight * failures.astype(np.float64) + residual_weight * residuals + action_weight * stacked_actions.sum(axis=(1, 2, 3, 4))
        for local_row, (begin, finish) in enumerate(candidate_ranges):
            # The baseline is stored first, so exact ties intentionally retain it.
            best = begin + int(np.argmin(costs[begin:finish]))
            base, teacher = base_actions[local_row], stacked_actions[best].astype(bool)
            teacher_actions[start + local_row] = teacher
            teacher_mask[start + local_row] = packet_mask_from_action_difference(
                torch.from_numpy(base[None]), torch.from_numpy(teacher[None])
            ).squeeze(0).numpy()
            changed += int(np.any(teacher != base))
            endpoint_improved += int(costs[best] < costs[begin])
        print(f"[teacher] {end}/{count} shots; changed={changed}; endpoint_improved={endpoint_improved}")
    del teacher_actions, teacher_mask
    np.save(output_dir / "source_indices.npy", source_rows)
    write_json(output_dir / "metadata.json", {
        "artifact": "structured_endpoint_teacher_v1", "source_dataset_dir": str(dataset_dir.resolve()),
        "base_checkpoint": str(base_checkpoint.resolve()), "structured_checkpoint": str(checkpoint.resolve()),
        "num_samples": count, "shape": list(shape), "changed_shots": changed,
        "endpoint_improved_shots": endpoint_improved, "cost": {
            "logical_failure_weight": failure_weight, "residual_weight": residual_weight, "action_weight": action_weight,
        }, "grouping": grouping.__dict__, "max_groups_per_shot": max_groups,
    })
    print(f"[done] wrote endpoint teacher corpus: {output_dir}")


if __name__ == "__main__":
    main()
