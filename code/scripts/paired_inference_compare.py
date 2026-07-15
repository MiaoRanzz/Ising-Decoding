#!/usr/bin/env python3
"""Paired inference comparison on one shared inference dataset.

This script compares pure PyMatching with one or more predecoder models on the
same generated samples for each measurement basis. It is intentionally separate
from the Hydra workflow so the standard train/inference entry points stay
unchanged.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pymatching
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

CODE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_ROOT.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from scripts.config_paths import config_path  # noqa: E402
from data.factory import DatapipeFactory  # noqa: E402
from evaluation.logical_error_rate import (  # noqa: E402
    PreDecoderMemoryEvalModule,
    _build_stab_maps,
)
from training.utils import dict_to_device  # noqa: E402
from workflows.config_validator import (  # noqa: E402
    apply_public_defaults_and_model,
    validate_public_config,
)
from workflows.run import _load_model  # noqa: E402


@dataclass(frozen=True)
class ModelSpec:
    name: str
    model_id: int
    checkpoint: Path


def parse_model_spec(value: str) -> ModelSpec:
    parts = value.split(":", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "--model must be formatted as name:model_id:/path/to/checkpoint.pt"
        )
    name, model_id_raw, checkpoint_raw = parts
    if not name:
        raise argparse.ArgumentTypeError("model name must not be empty")
    try:
        model_id = int(model_id_raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid model_id: {model_id_raw}") from exc
    checkpoint = Path(checkpoint_raw).expanduser()
    if not checkpoint.is_absolute():
        checkpoint = REPO_ROOT / checkpoint
    return ModelSpec(name=name, model_id=model_id, checkpoint=checkpoint)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare PyMatching and replaceable predecoder models on identical samples."
    )
    parser.add_argument("--config-name", default="config_domestic")
    parser.add_argument("--distance", type=int, default=9)
    parser.add_argument("--n-rounds", type=int, default=9)
    parser.add_argument("--num-samples", type=int, default=262144)
    parser.add_argument("--latency-num-samples", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--basis",
        choices=("both", "X", "Z"),
        default="both",
        help="Measurement basis to evaluate.",
    )
    parser.add_argument(
        "--model",
        action="append",
        type=parse_model_spec,
        required=True,
        help="Repeatable model spec: name:model_id:/path/to/checkpoint.pt",
    )
    parser.add_argument(
        "--output",
        default="outputs/paired_inference_compare/fast_vs_fastopt.json",
        help="JSON output path. A CSV summary is written next to it.",
    )
    return parser.parse_args()


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_cfg(args: argparse.Namespace, model: ModelSpec, basis: str) -> Any:
    cfg_path = config_path(args.config_name)
    cfg = OmegaConf.load(cfg_path)
    cfg.model_id = model.model_id
    cfg.distance = args.distance
    cfg.n_rounds = args.n_rounds
    cfg.workflow.task = "inference"

    spec = validate_public_config(cfg)
    cfg = apply_public_defaults_and_model(cfg, spec)
    cfg.model_checkpoint_file = str(model.checkpoint)
    cfg.test.meas_basis_test = basis
    cfg.test.num_samples = int(args.num_samples)
    cfg.test.latency_num_samples = int(args.latency_num_samples)
    cfg.test.batch_size = int(args.batch_size)
    cfg.test.dataloader_num_workers = int(args.num_workers)
    return cfg


def make_dataset(cfg: Any, seed: int):
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        set_all_seeds(seed)
        return DatapipeFactory.create_datapipe_inference(cfg)
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


def time_single_shot(matcher: pymatching.Matching, syndromes: np.ndarray, n_rounds: int) -> float:
    n_rounds = max(int(n_rounds), 1)
    if syndromes.size == 0:
        return float("nan")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    warmup_n = min(50, len(syndromes))
    for i in range(warmup_n):
        matcher.decode(np.asarray(syndromes[i], dtype=np.uint8))

    times = []
    for row in syndromes:
        start = time.perf_counter()
        matcher.decode(np.asarray(row, dtype=np.uint8))
        times.append(time.perf_counter() - start)
    return float(np.mean(times) / n_rounds * 1e6)


def build_matcher(dataset) -> tuple[pymatching.Matching, int]:
    circuit = dataset.circ.stim_circuit
    det_model = circuit.detector_error_model(decompose_errors=True, approximate_disjoint_errors=True)
    return pymatching.Matching.from_detector_error_model(det_model), int(circuit.num_observables)


def evaluate_pymatching(
    matcher: pymatching.Matching,
    dets_and_obs: np.ndarray,
    num_obs: int,
    latency_samples: int,
    n_rounds: int,
) -> dict[str, float | int]:
    dets = np.ascontiguousarray(dets_and_obs[:, :-num_obs], dtype=np.uint8)
    obs = np.ascontiguousarray(dets_and_obs[:, -num_obs:], dtype=np.uint8)
    pred = matcher.decode_batch(dets).reshape(obs.shape)
    errors = int((pred != obs).sum())
    total = int(obs.shape[0])
    latency_rows = dets[: min(latency_samples, len(dets))]
    return {
        "logical_errors": errors,
        "samples": total,
        "ler": float(errors / total) if total else float("nan"),
        "latency_us_per_round": time_single_shot(matcher, latency_rows, n_rounds),
    }


def evaluate_model(
    model: torch.nn.Module,
    cfg: Any,
    dataset,
    matcher: pymatching.Matching,
    num_obs: int,
    device: torch.device,
    latency_samples: int,
    n_rounds: int,
) -> dict[str, float | int]:
    maps = _build_stab_maps(int(cfg.distance), getattr(cfg, "rotation", "XV"))
    module = PreDecoderMemoryEvalModule(model, cfg, maps, device).to(device)
    module.eval()
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.test.batch_size),
        shuffle=False,
        num_workers=int(cfg.test.dataloader_num_workers),
        pin_memory=(device.type == "cuda"),
    )

    logical_errors = 0
    total = 0
    residual_chunks: list[np.ndarray] = []
    residual_count = 0

    with torch.no_grad():
        for batch in loader:
            batch = dict_to_device(batch, device)
            dets_and_obs = batch["dets_and_obs"]
            dets_only = dets_and_obs[:, :-num_obs]
            gt_obs = dets_and_obs[:, -num_obs:].to(torch.int64).cpu()

            output = module(dets_only)
            pre_l = output[:, 0].to(torch.int64).cpu()
            residual = output[:, 1:].to(torch.uint8).cpu().numpy()
            pred_obs = torch.from_numpy(matcher.decode_batch(residual)).reshape(gt_obs.shape)
            final_l = (pre_l.reshape(gt_obs.shape) + pred_obs).remainder(2)

            logical_errors += int((final_l != gt_obs).sum().item())
            total += int(gt_obs.shape[0])

            if residual_count < latency_samples:
                take = min(latency_samples - residual_count, residual.shape[0])
                residual_chunks.append(np.ascontiguousarray(residual[:take], dtype=np.uint8))
                residual_count += take

    residual_rows = (
        np.concatenate(residual_chunks, axis=0) if residual_chunks else np.empty((0, 0), dtype=np.uint8)
    )
    return {
        "logical_errors": logical_errors,
        "samples": total,
        "ler": float(logical_errors / total) if total else float("nan"),
        "latency_us_per_round": time_single_shot(matcher, residual_rows, n_rounds),
    }


def mean_metric(rows: list[dict[str, Any]], name: str) -> float:
    values = [float(row[name]) for row in rows if row.get(name) is not None]
    return float(np.mean(values)) if values else float("nan")


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = REPO_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for spec in args.model:
        if not spec.checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found for {spec.name}: {spec.checkpoint}")

    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    dist = SimpleNamespace(rank=0, world_size=1, device=device)
    bases = ["X", "Z"] if args.basis == "both" else [args.basis]

    model_cfgs = {spec.name: build_cfg(args, spec, basis=bases[0]) for spec in args.model}
    models = {}
    for spec in args.model:
        print(f"[load] {spec.name}: model_id={spec.model_id}, checkpoint={spec.checkpoint}")
        model = _load_model(model_cfgs[spec.name], dist)
        model.eval()
        models[spec.name] = model

    rows: list[dict[str, Any]] = []
    for basis_index, basis in enumerate(bases):
        dataset_cfg = build_cfg(args, args.model[0], basis=basis)
        dataset_seed = int(args.seed) + basis_index
        print(f"[data] basis={basis}, seed={dataset_seed}, samples={args.num_samples}")
        dataset = make_dataset(dataset_cfg, dataset_seed)
        matcher, num_obs = build_matcher(dataset)
        dets_and_obs = np.asarray(dataset.dets_and_obs, dtype=np.uint8)

        baseline = evaluate_pymatching(
            matcher,
            dets_and_obs,
            num_obs,
            int(args.latency_num_samples),
            int(args.n_rounds),
        )
        baseline_row = {
            "basis": basis,
            "method": "pymatching",
            "model_id": "",
            "checkpoint": "",
            **baseline,
            "speedup_vs_pymatching": 1.0,
        }
        rows.append(baseline_row)
        print(
            f"[result] {basis} pymatching ler={baseline['ler']:.6f}, "
            f"latency={baseline['latency_us_per_round']:.3f} us/round"
        )

        for spec in args.model:
            cfg = build_cfg(args, spec, basis=basis)
            result = evaluate_model(
                models[spec.name],
                cfg,
                dataset,
                matcher,
                num_obs,
                device,
                int(args.latency_num_samples),
                int(args.n_rounds),
            )
            speedup = float(baseline["latency_us_per_round"]) / float(result["latency_us_per_round"])
            row = {
                "basis": basis,
                "method": spec.name,
                "model_id": spec.model_id,
                "checkpoint": str(spec.checkpoint),
                **result,
                "speedup_vs_pymatching": speedup,
            }
            rows.append(row)
            print(
                f"[result] {basis} {spec.name} ler={result['ler']:.6f}, "
                f"latency={result['latency_us_per_round']:.3f} us/round, speedup={speedup:.3f}x"
            )

    methods = sorted({row["method"] for row in rows})
    summary = []
    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        summary.append(
            {
                "method": method,
                "ler_avg": mean_metric(method_rows, "ler"),
                "latency_us_per_round_avg": mean_metric(method_rows, "latency_us_per_round"),
                "speedup_vs_pymatching_avg": mean_metric(method_rows, "speedup_vs_pymatching"),
            }
        )

    payload = {
        "config_name": args.config_name,
        "distance": args.distance,
        "n_rounds": args.n_rounds,
        "num_samples": args.num_samples,
        "latency_num_samples": args.latency_num_samples,
        "seed": args.seed,
        "device": str(device),
        "rows": rows,
        "summary": summary,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    csv_path = output_path.with_suffix(".csv")
    fieldnames = [
        "basis",
        "method",
        "model_id",
        "logical_errors",
        "samples",
        "ler",
        "latency_us_per_round",
        "speedup_vs_pymatching",
        "checkpoint",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})

    print(f"[write] {output_path}")
    print(f"[write] {csv_path}")


if __name__ == "__main__":
    main()
