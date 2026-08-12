#!/usr/bin/env python3
"""Select a structured checkpoint no-op bias on validation and report held-out LER."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from common import (DEFAULT_SETTINGS, build_structured_model, endpoint_outcomes, endpoint_pipeline,
                    repo_path, section, summarize, write_json)
from compare_three_paths import load_corpus
from structured_actions import actions_from_logits
from train_structured_ising import split_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path, default=DEFAULT_SETTINGS)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def actions_for_rows(model, train_x: np.ndarray, rows: np.ndarray, device: torch.device, batch_size: int, bias: float):
    result = []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            part = rows[start:start + batch_size]
            logits = model(torch.as_tensor(np.array(train_x[part], dtype=np.float32, copy=True), device=device))
            result.append(actions_from_logits(logits, bias).cpu().numpy().astype(np.uint8))
    return np.concatenate(result, axis=0)


def main() -> None:
    cli = parse_args()
    settings = cli.settings.resolve()
    model_cfg, cfg = section(settings, "structured_model"), section(settings, "structured_evaluation")
    dataset_dir = repo_path(model_cfg["dataset_dir"])
    base_checkpoint, project_config = repo_path(model_cfg["base_checkpoint"]), repo_path(model_cfg["project_config"])
    checkpoint, output = repo_path(cli.checkpoint or cfg["checkpoint"]), repo_path(cli.output or cfg["output"])
    device = torch.device(cli.device or cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    metadata, dets_and_obs, train_x, _ = load_corpus(dataset_dir)
    model, endpoint_cfg = build_structured_model(metadata, project_config, base_checkpoint, model_cfg.get("model_id"), device, checkpoint)
    model.eval()
    matcher, action_model, pipeline = endpoint_pipeline(endpoint_cfg, metadata, device)
    _, validation_rows, test_rows = split_rows(
        len(train_x), float(cfg.get("train_fraction", .70)), float(cfg.get("validation_fraction", .15)), int(cfg.get("split_seed", 12345))
    )
    batch_size = int(cfg.get("batch_size", 256))
    candidates = [float(value) for value in cfg.get("no_op_biases", [0.0])]
    validation = []
    for bias in candidates:
        actions = actions_for_rows(model, train_x, validation_rows, device, batch_size, bias)
        failed, residual = endpoint_outcomes(pipeline, action_model, matcher,
                                             np.asarray(dets_and_obs[validation_rows, :-1], dtype=np.uint8),
                                             np.asarray(dets_and_obs[validation_rows, -1:], dtype=np.uint8), actions,
                                             device, batch_size)
        validation.append({"no_op_bias": bias, **summarize(failed), "mean_residual_weight": float(residual.mean()),
                           "mean_action_count": float(actions.sum(axis=(1, 2, 3, 4)).mean())})
    selected = min(validation, key=lambda item: (item["logical_errors"], item["mean_residual_weight"], item["mean_action_count"]))
    bias = float(selected["no_op_bias"])
    test_actions = actions_for_rows(model, train_x, test_rows, device, batch_size, bias)
    test_failure, test_residual = endpoint_outcomes(pipeline, action_model, matcher,
                                                    np.asarray(dets_and_obs[test_rows, :-1], dtype=np.uint8),
                                                    np.asarray(dets_and_obs[test_rows, -1:], dtype=np.uint8), test_actions,
                                                    device, batch_size)
    report = {"checkpoint": str(checkpoint), "selected_on_validation": selected, "validation_candidates": validation,
              "held_out_test": {**summarize(test_failure), "mean_residual_weight": float(test_residual.mean()),
                                  "mean_action_count": float(test_actions.sum(axis=(1, 2, 3, 4)).mean())}}
    write_json(output, report)
    print(report)


if __name__ == "__main__":
    main()
