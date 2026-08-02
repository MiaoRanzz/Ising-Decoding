#!/usr/bin/env python3
"""Validate a whole-shot safe-no-op gate on a paired end-to-end corpus.

The script does not retrain Ising-fast or change its decoder.  It extracts
compact shot-level confidence features from the checkpoint logits, then uses
the existing paired PyMatching/model outcomes to simulate this rule:

    low confidence -> raw detectors -> PyMatching (no-op)
    high confidence -> existing Ising-fast -> PyMatching

Threshold selection uses only a deterministic validation split.  The selected
threshold is evaluated on the held-out split, avoiding test-set tuning.
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
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from compare_three_paths import _load_model, build_model_cfg, load_corpus
from training.precision import match_input_to_model_memory_format


DEFAULT_SETTINGS = HERE / "end_to_end.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract confidence and validate whole-shot safe no-op.")
    parser.add_argument("--settings", type=Path, default=DEFAULT_SETTINGS)
    parser.add_argument("--score", default=None, help="Override safe_no_op.score.")
    parser.add_argument("--device", default=None, help="Override comparison/safe_no_op device.")
    return parser.parse_args()


def _repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def load_settings(cli: argparse.Namespace) -> SimpleNamespace:
    settings_path = cli.settings.expanduser().resolve()
    cfg = OmegaConf.load(settings_path)
    comparison = OmegaConf.to_container(cfg.get("comparison", {}), resolve=True)
    safe = OmegaConf.to_container(cfg.get("safe_no_op", {}), resolve=True)
    if not isinstance(comparison, dict) or not isinstance(safe, dict):
        raise ValueError("comparison and safe_no_op settings must both be mappings")

    def required(section: dict[str, Any], name: str) -> Any:
        value = section.get(name)
        if value is None:
            raise ValueError(f"missing {name} in {settings_path}")
        return value

    comparison_output = _repo_path(required(comparison, "output"))
    per_shot_raw = safe.get("per_shot_file")
    per_shot_file = (
        _repo_path(per_shot_raw)
        if per_shot_raw is not None
        else comparison_output.with_suffix(".per_shot.npz")
    )
    return SimpleNamespace(
        dataset_dir=_repo_path(required(comparison, "dataset_dir")),
        project_config=_repo_path(required(comparison, "project_config")),
        checkpoint=_repo_path(required(comparison, "checkpoint")),
        model_id=comparison.get("model_id"),
        batch_size=int(required(comparison, "batch_size")),
        device=cli.device if cli.device is not None else safe.get("device", comparison.get("device")),
        comparison_output=comparison_output,
        per_shot_file=per_shot_file,
        confidence_output=_repo_path(required(safe, "confidence_output")),
        output=_repo_path(required(safe, "output")),
        score=str(cli.score if cli.score is not None else safe.get("score", "positive_margin_q10")),
        num_confidence_bins=int(safe.get("num_confidence_bins", 10)),
        num_thresholds=int(safe.get("num_thresholds", 101)),
        validation_fraction=float(safe.get("validation_fraction", 0.5)),
        split_seed=int(safe.get("split_seed", 20260802)),
    )


def load_outcomes(path: Path, expected_samples: int) -> tuple[np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"per-shot comparison file not found: {path}")
    payload = np.load(path)
    try:
        baseline = np.asarray(payload["pymatching_failure"], dtype=bool)
        model = np.asarray(payload["model_failure"], dtype=bool)
    finally:
        payload.close()
    if baseline.shape != (expected_samples,) or model.shape != (expected_samples,):
        raise ValueError("per-shot outcomes do not match the configured dataset")
    return baseline, model


def extract_confidence(
    model: torch.nn.Module,
    train_x: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Return compact per-shot scores; no full logits are stored."""
    total = int(train_x.shape[0])
    q10 = np.empty(total, dtype=np.float32)
    mean = np.empty(total, dtype=np.float32)
    minimum = np.empty(total, dtype=np.float32)
    action_count = np.empty(total, dtype=np.int32)
    data_action_count = np.empty(total, dtype=np.int32)
    syndrome_action_count = np.empty(total, dtype=np.int32)

    model.eval()
    with torch.no_grad():
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            inputs_np = np.array(train_x[start:end], dtype=np.float32, copy=True)
            inputs = torch.as_tensor(inputs_np, dtype=torch.float32, device=device)
            inputs = match_input_to_model_memory_format(inputs, model)
            logits = model(inputs)
            if logits.ndim != 5 or logits.shape[1] != 4:
                raise ValueError(f"expected model logits (B, 4, T, D, D), got {tuple(logits.shape)}")
            flat_logits = logits.reshape(logits.shape[0], -1)
            selected = flat_logits >= 0.0
            margins = flat_logits.abs()
            counts = selected.sum(dim=1)
            # NaNs allow vectorized quantiles while retaining shots with no
            # predicted actions as an explicit special case.
            selected_margins = margins.masked_fill(~selected, float("nan"))
            q_values = torch.nanquantile(selected_margins, 0.10, dim=1)
            mean_values = torch.nanmean(selected_margins, dim=1)
            min_values = torch.nan_to_num(selected_margins, nan=float("inf")).min(dim=1).values
            no_action = counts == 0
            q_values = torch.where(no_action, torch.full_like(q_values, float("inf")), q_values)
            mean_values = torch.where(no_action, torch.full_like(mean_values, float("inf")), mean_values)
            min_values = torch.where(no_action, torch.full_like(min_values, float("inf")), min_values)
            selected_5d = logits >= 0.0

            q10[start:end] = q_values.cpu().numpy()
            mean[start:end] = mean_values.cpu().numpy()
            minimum[start:end] = min_values.cpu().numpy()
            action_count[start:end] = counts.cpu().numpy().astype(np.int32)
            data_action_count[start:end] = selected_5d[:, :2].sum(dim=(1, 2, 3, 4)).cpu().numpy().astype(np.int32)
            syndrome_action_count[start:end] = selected_5d[:, 2:].sum(dim=(1, 2, 3, 4)).cpu().numpy().astype(np.int32)
            if start == 0 or end == total:
                print(f"[confidence] {end}/{total} shots")

    return {
        "positive_margin_q10": q10,
        "positive_margin_mean": mean,
        "positive_margin_min": minimum,
        "action_count": action_count,
        "data_action_count": data_action_count,
        "syndrome_action_count": syndrome_action_count,
    }


def outcome_metrics(baseline: np.ndarray, model: np.ndarray, apply: np.ndarray) -> dict[str, int | float]:
    gated = np.where(apply, model, baseline)
    helpful = baseline & ~model
    harmful = ~baseline & model
    return {
        "logical_errors": int(gated.sum()),
        "ler": float(gated.mean()),
        "apply_shots": int(apply.sum()),
        "apply_coverage": float(apply.mean()),
        "harmful_blocked": int((harmful & ~apply).sum()),
        "harmful_blocked_rate": float((harmful & ~apply).sum() / harmful.sum()) if harmful.any() else 0.0,
        "helpful_retained": int((helpful & apply).sum()),
        "helpful_retained_rate": float((helpful & apply).sum() / helpful.sum()) if helpful.any() else 0.0,
    }


def confidence_bins(
    score: np.ndarray, baseline: np.ndarray, model: np.ndarray, bins: int
) -> list[dict[str, Any]]:
    finite = np.isfinite(score)
    rows: list[dict[str, Any]] = []
    helpful = baseline & ~model
    harmful = ~baseline & model
    if finite.any():
        edges = np.unique(np.quantile(score[finite], np.linspace(0.0, 1.0, bins + 1)))
        for index, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
            include = (score >= low) & ((score < high) if index + 1 < len(edges) - 1 else (score <= high))
            count = int(include.sum())
            if count:
                rows.append({
                    "kind": "confidence_bin",
                    "lower": float(low),
                    "upper": float(high),
                    "shots": count,
                    "helpful_shots": int((helpful & include).sum()),
                    "harmful_shots": int((harmful & include).sum()),
                    "helpful_rate": float((helpful & include).sum() / count),
                    "harmful_rate": float((harmful & include).sum() / count),
                })
    no_action = ~finite
    if no_action.any():
        count = int(no_action.sum())
        rows.append({
            "kind": "no_predicted_action",
            "shots": count,
            "helpful_shots": int((helpful & no_action).sum()),
            "harmful_shots": int((harmful & no_action).sum()),
            "helpful_rate": float((helpful & no_action).sum() / count),
            "harmful_rate": float((harmful & no_action).sum() / count),
        })
    return rows


def main() -> None:
    args = load_settings(parse_args())
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")
    if not 0.0 < args.validation_fraction < 1.0:
        raise ValueError("validation_fraction must lie strictly between 0 and 1")
    metadata, _, train_x, _ = load_corpus(args.dataset_dir)
    total = int(metadata["num_samples"])
    baseline, model_failure = load_outcomes(args.per_shot_file, total)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # Reuse exactly the same model configuration construction as the comparison.
    report_model_id = None
    if args.comparison_output.is_file():
        report_model_id = json.loads(args.comparison_output.read_text(encoding="utf-8")).get("model_id")
    model_args = SimpleNamespace(
        project_config=args.project_config,
        checkpoint=args.checkpoint,
        model_id=args.model_id if args.model_id is not None else report_model_id,
    )
    cfg = build_model_cfg(model_args, metadata)
    print(f"[load] checkpoint={args.checkpoint}, device={device}")
    model = _load_model(cfg, SimpleNamespace(rank=0, world_size=1, device=device)).eval()
    scores = extract_confidence(model, train_x, args.batch_size, device)
    if args.score not in scores:
        raise ValueError(f"unknown score {args.score!r}; choices: {sorted(scores)}")
    score = scores[args.score]
    args.confidence_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.confidence_output, **scores)

    rng = np.random.default_rng(args.split_seed)
    validation = rng.random(total) < args.validation_fraction
    if not validation.any() or validation.all():
        raise RuntimeError("split unexpectedly produced an empty validation or test subset")
    test = ~validation
    finite_validation = score[validation & np.isfinite(score)]
    if finite_validation.size == 0:
        raise RuntimeError("selected score has no finite values on the validation split")
    thresholds = np.unique(np.quantile(finite_validation, np.linspace(0.0, 1.0, args.num_thresholds)))
    thresholds = np.concatenate(([-np.inf], thresholds, [np.inf]))
    sweep: list[dict[str, Any]] = []
    for threshold in thresholds:
        # A shot with no predicted correction already behaves as no-op.  Keep
        # it out of gate coverage instead of counting it as a model application.
        apply = np.isfinite(score) & (score >= threshold)
        if np.isposinf(threshold):
            apply = np.zeros(total, dtype=bool)  # exact all-shot no-op endpoint
        val_metrics = outcome_metrics(baseline[validation], model_failure[validation], apply[validation])
        test_metrics = outcome_metrics(baseline[test], model_failure[test], apply[test])
        sweep.append({"threshold": float(threshold), "validation": val_metrics, "test": test_metrics})

    # Lowest validation LER wins; in a tie prefer applying the model more often.
    best = min(sweep, key=lambda row: (row["validation"]["ler"], -row["validation"]["apply_coverage"]))
    base_test = {"logical_errors": int(baseline[test].sum()), "ler": float(baseline[test].mean())}
    model_test = {"logical_errors": int(model_failure[test].sum()), "ler": float(model_failure[test].mean())}
    report = {
        "dataset_dir": str(args.dataset_dir),
        "checkpoint": str(args.checkpoint),
        "model_id": int(cfg.model_id),
        "score": args.score,
        "confidence_file": str(args.confidence_output),
        "per_shot_file": str(args.per_shot_file),
        "split": {"validation_fraction": args.validation_fraction, "seed": args.split_seed,
                  "validation_shots": int(validation.sum()), "test_shots": int(test.sum())},
        "confidence_bins_all_shots": confidence_bins(score, baseline, model_failure, args.num_confidence_bins),
        "test_baselines": {"pymatching": base_test, "ising_fast_plus_pymatching": model_test},
        "selected_gate": best,
        "threshold_sweep": sweep,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"test_baselines": report["test_baselines"], "selected_gate": best}, indent=2))
    print(f"[done] confidence={args.confidence_output}\n[done] report={args.output}")


if __name__ == "__main__":
    main()
