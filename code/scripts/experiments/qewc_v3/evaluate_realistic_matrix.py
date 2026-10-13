#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluate QEWC v3 checkpoints on five frozen task configurations."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
CODE_ROOT = REPO_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from scripts.qadapt_example_utils import InferenceJob, parse_gpus, run_jobs  # noqa: E402


PAIRED_SCRIPT = REPO_ROOT / "code" / "scripts" / "paired_inference_compare.py"
TASKS = {
    "T0": REPO_ROOT / "conf" / "experiments" / "qewc_v3" / "realistic" / "t0_base.yaml",
    "T1": REPO_ROOT / "conf" / "experiments" / "qewc_v3" / "realistic" / "t1_meas_1p5.yaml",
    "T2": REPO_ROOT / "conf" / "experiments" / "qewc_v3" / "realistic" / "t2_cnot_1p5.yaml",
    "T3": REPO_ROOT / "conf" / "experiments" / "qewc_v3" / "realistic" / "t3_idle_1p5.yaml",
    "T4": REPO_ROOT / "conf" / "experiments" / "qewc_v3" / "realistic" / "t4_z_bias_1p5.yaml",
}


@dataclass(frozen=True)
class Checkpoint:
    name: str
    path: Path


def parse_checkpoint(value: str) -> Checkpoint:
    name, separator, raw_path = value.partition(":")
    if not separator or not name.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("--checkpoint must be NAME:PATH")
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return Checkpoint(name.strip(), path)



def parse_task_config(value: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition(":")
    if not separator or not name.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("--task must be NAME:CONFIG_PATH")
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return name.strip(), path

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_job(args: argparse.Namespace, task: str, config: Path) -> InferenceJob:
    output = args.output_dir / f"{task}.json"
    command = [
        str(args.python),
        "-u",
        str(PAIRED_SCRIPT),
        "--config-file",
        str(config),
        "--distance",
        "9",
        "--n-rounds",
        "9",
        "--num-samples",
        str(args.num_samples),
        "--latency-num-samples",
        str(args.latency_num_samples),
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        "0",
        "--seed",
        str(args.seed),
        "--basis",
        "both",
        "--device",
        "cuda:0",
        "--output",
        str(output),
    ]
    sample_root = getattr(args, "sample_root", None)
    if sample_root is not None:
        command.extend(("--stim-samples-dir", str(sample_root / task)))
    for checkpoint in args.checkpoint:
        command.extend(
            (
                "--model",
                f"{checkpoint.name}:{args.model_id}:{checkpoint.path}:{config}",
            )
        )
    for baseline, candidate in combinations(args.checkpoint, 2):
        command.extend(("--paired-comparison", f"{candidate.name}:{baseline.name}"))
    return InferenceJob(label=task, command=tuple(command), output_path=output)


def write_summary(args: argparse.Namespace) -> None:
    model_names = [checkpoint.name for checkpoint in args.checkpoint]
    rows = {name: [] for name in model_names}
    paired = {}
    test_streams = {}
    for task in args.tasks:
        payload = json.loads((args.output_dir / f"{task}.json").read_text(encoding="utf-8"))
        by_method = {row["method"]: row for row in payload["summary"]}
        for name in model_names:
            rows[name].append(float(by_method[name]["ler_avg"]))
        paired[task] = payload["paired_comparisons"]
        test_streams[task] = payload.get("sample_streams", {})
    manifest = {
        "track": args.track,
        "task_columns": list(args.tasks),
        "seed": args.seed,
        "samples_per_basis": args.num_samples,
        "batch_size": args.batch_size,
        "model_id": args.model_id,
        "checkpoints": [
            {"name": item.name, "path": str(item.path), "sha256": sha256(item.path)}
            for item in args.checkpoint
        ],
        "configs": {
            task: {"path": str(path), "sha256": sha256(path)}
            for task, path in args.tasks.items()
        },
        "fixed_sample_root": str(args.sample_root) if getattr(args, "sample_root", None) else None,
        "ler_rows": rows,
        "paired_comparisons": paired,
        "test_streams": test_streams,
    }
    (args.output_dir / "evaluation_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-root", type=Path)
    parser.add_argument("--checkpoint", action="append", type=parse_checkpoint, required=True)
    parser.add_argument("--track", default="realistic")
    parser.add_argument("--task", action="append", type=parse_task_config)
    parser.add_argument("--num-samples", type=int, default=65536)
    parser.add_argument("--latency-num-samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument(
        "--model-id",
        type=int,
        default=111,
        help="decoder model id passed to paired_inference_compare (HTNet=111, Ising-Fast=1)",
    )
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--parallelism", type=int, default=2)
    parser.add_argument(
        "--python",
        type=Path,
        default=Path("/root/miniconda3/envs/ising-decoding/bin/python"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/qewc_v3/evaluations/realistic_pilot_c0_c1"),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if len(args.checkpoint) < 1:
        parser.error("at least one checkpoint is required")
    names = [item.name for item in args.checkpoint]
    if len(names) != len(set(names)):
        parser.error("checkpoint names must be unique")
    task_items = args.task or list(TASKS.items())
    if args.sample_root is not None and not args.sample_root.is_absolute():
        args.sample_root = REPO_ROOT / args.sample_root
    if len(task_items) != 5 or {name for name, _ in task_items} != set(TASKS):
        parser.error("--task must provide T0,T1,T2,T3,T4 exactly once")
    args.tasks = dict(task_items)
    args.output_dir = args.output_dir if args.output_dir.is_absolute() else REPO_ROOT / args.output_dir
    return args


def main() -> None:
    args = parse_args()
    for task, config in args.tasks.items():
        if not config.is_file():
            raise FileNotFoundError(f"{task}: {config}")
    if args.sample_root is not None:
        for task in args.tasks:
            for basis in ("X", "Z"):
                samples = args.sample_root / task / f"samples_{basis}.dets"
                metadata = args.sample_root / task / f"metadata_{basis}.json"
                if not samples.is_file() or not metadata.is_file():
                    raise FileNotFoundError(f"missing fixed stream for {task}/{basis}")
    for item in args.checkpoint:
        if not item.path.is_file():
            raise FileNotFoundError(item.path)
    jobs = [build_job(args, task, config) for task, config in args.tasks.items()]
    run_jobs(
        jobs,
        gpus=parse_gpus(args.gpus),
        parallelism=args.parallelism,
        resume=args.resume,
        dry_run=args.dry_run,
    )
    if not args.dry_run:
        write_summary(args)
        print(f"[QEWC v3] evaluation manifest: {args.output_dir / 'evaluation_manifest.json'}")


if __name__ == "__main__":
    main()
