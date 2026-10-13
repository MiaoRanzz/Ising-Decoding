#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Learn a 25-parameter Ising-Decoding noise model from Google QEC data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from noise_learning.google_qec import GoogleQECDataset, fit_noise_model
from qec.noise_model import NoiseModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Google benchmark zip or extracted root.")
    parser.add_argument("--distance", type=int, action="append", dest="distances")
    parser.add_argument(
        "--experiment-key", action="append", dest="experiment_keys",
        help="Exact archive experiment key; may be repeated.",
    )
    parser.add_argument("--basis", choices=("X", "Z"), action="append", dest="bases")
    parser.add_argument("--rounds", type=int, action="append")
    parser.add_argument("--max-experiments", type=int, default=4)
    parser.add_argument("--max-shots", type=int, default=50_000)
    parser.add_argument("--max-pair-moments", type=int, default=256)
    parser.add_argument("--max-nfev", type=int, default=80)
    parser.add_argument("--prior-strength", type=float, default=0.05)
    parser.add_argument("--initial-p", type=float, default=1e-3)
    parser.add_argument(
        "--initial-model", type=Path,
        help="Resume from a YAML or JSON fit result containing noise_model.",
    )
    parser.add_argument("--min-probability", type=float, default=1e-5)
    parser.add_argument("--max-probability", type=float, default=3e-2)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--training-config",
        type=Path,
        help="Also emit a complete training YAML derived from this base config.",
    )
    parser.add_argument("--training-config-output", type=Path)
    parser.add_argument("--target-distance", type=int, help="Target distance for generated config.")
    parser.add_argument("--target-rounds", type=int, help="Target rounds for generated config.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset = GoogleQECDataset(args.source)
    keys = (
        tuple(args.experiment_keys)
        if args.experiment_keys
        else dataset.select(
            distances=args.distances,
            bases=args.bases,
            rounds=args.rounds,
            max_experiments=args.max_experiments,
        )
    )
    if not keys:
        raise SystemExit("no Google QEC experiments matched the requested filters")
    print("Selected experiments:")
    for key in keys:
        print(f"  {key}")
    experiments = [dataset.load(key, max_shots=args.max_shots) for key in keys]
    if args.initial_model:
        initial_payload = yaml.safe_load(args.initial_model.read_text())
        initial_model = NoiseModel.from_config_dict(initial_payload["noise_model"])
    else:
        initial_model = NoiseModel.from_si1000(args.initial_p)
    result = fit_noise_model(
        experiments,
        initial=initial_model,
        max_nfev=args.max_nfev,
        prior_strength=args.prior_strength,
        min_probability=args.min_probability,
        max_probability=args.max_probability,
        max_pair_moments=args.max_pair_moments,
    )
    payload = result.to_dict()
    payload["provenance"] = {
        "source": str(args.source.resolve()),
        "source_size_bytes": args.source.stat().st_size if args.source.is_file() else None,
        "experiment_keys": list(keys),
        "max_shots_per_experiment": args.max_shots,
        "max_pair_moments_per_experiment": args.max_pair_moments,
        "initial_p": args.initial_p,
        "initial_model": str(args.initial_model.resolve()) if args.initial_model else None,
        "initial_model_sha256": initial_model.sha256(),
        "prior_strength": args.prior_strength,
        "min_probability": args.min_probability,
        "max_probability": args.max_probability,
        "max_nfev": args.max_nfev,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix.lower() == ".json":
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    else:
        args.output.write_text(yaml.safe_dump(payload, sort_keys=False))
    print(f"Wrote learned noise model: {args.output}")
    print(f"sha256={payload['noise_model_sha256']} rank={result.jacobian_rank}/25")

    if args.training_config:
        target = args.training_config_output
        if target is None:
            raise SystemExit("--training-config-output is required with --training-config")
        config = yaml.safe_load(args.training_config.read_text())
        config.setdefault("data", {})["noise_model"] = payload["noise_model"]
        if args.target_distance is not None:
            config["distance"] = int(args.target_distance)
        if args.target_rounds is not None:
            config["n_rounds"] = int(args.target_rounds)
        # Hardware adaptation must train at the learned physical scale.  The
        # default public pipeline otherwise upscales sparse models to 0.006.
        config["data"]["skip_noise_upscaling"] = True
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(yaml.safe_dump(config, sort_keys=False))
        print(f"Wrote training config: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
