#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Launch one gated QEWC-v4 formal run from the confirmed 16-run matrix."""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[4]
MANIFEST_PATH = REPO_ROOT / "experiment" / "qewc-v4" / "experiment_manifest.yaml"
RUN_CONTINUAL = REPO_ROOT / "code" / "scripts" / "experiments" / "qewc_v4" / "run_continual.py"
RUN_MIX = REPO_ROOT / "code" / "scripts" / "experiments" / "qewc_v4" / "run_mix.py"
EXPECTED_PREFIX = Path("/root/miniconda3/envs/ising-decoding")
FORBIDDEN_LAMBDA = 43.87539484535376


@dataclass(frozen=True)
class Architecture:
    key: str
    short: str
    config_dir: str
    base_config: str
    checkpoint_prefix: str
    mix_config: str


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    architecture: str
    method: str
    order_kind: str


ARCHITECTURES = {
    "htnet_a101": Architecture(
        "htnet_a101", "H", "experiments/qewc_v4/htnet",
        "experiments/qewc_v4/htnet/T0", "HTnet",
        "experiments/qewc_v4/htnet/mix_T0_T4",
    ),
    "ising_fast": Architecture(
        "ising_fast", "I", "experiments/qewc_v4/ising",
        "experiments/qewc_v4/ising/t0_base", "PreDecoderModelMemory_v1",
        "experiments/qewc_v4/ising/mix_T0_T4",
    ),
}


RUNS = {
    "H_T0": RunSpec("H_T0", "htnet_a101", "t0_only", "t0_stages"),
    "H_SEQ": RunSpec("H_SEQ", "htnet_a101", "seq", "high_conflict"),
    "H_EWC": RunSpec("H_EWC", "htnet_a101", "ewc", "high_conflict"),
    "H_REPLAY": RunSpec("H_REPLAY", "htnet_a101", "replay", "high_conflict"),
    "H_MIX_TOTAL": RunSpec("H_MIX_TOTAL", "htnet_a101", "mix_total", "joint"),
    "H_MIX_EXPOSURE": RunSpec("H_MIX_EXPOSURE", "htnet_a101", "mix_exposure", "joint"),
    "H_REV_SEQ": RunSpec("H_REV_SEQ", "htnet_a101", "seq", "reverse"),
    "H_REV_EWC": RunSpec("H_REV_EWC", "htnet_a101", "ewc", "reverse"),
    "I_T0": RunSpec("I_T0", "ising_fast", "t0_only", "t0_stages"),
    "I_SEQ": RunSpec("I_SEQ", "ising_fast", "seq", "high_conflict"),
    "I_EWC": RunSpec("I_EWC", "ising_fast", "ewc", "high_conflict"),
    "I_REPLAY": RunSpec("I_REPLAY", "ising_fast", "replay", "high_conflict"),
    "I_MIX_TOTAL": RunSpec("I_MIX_TOTAL", "ising_fast", "mix_total", "joint"),
    "I_MIX_EXPOSURE": RunSpec("I_MIX_EXPOSURE", "ising_fast", "mix_exposure", "joint"),
    "I_REV_SEQ": RunSpec("I_REV_SEQ", "ising_fast", "seq", "reverse"),
    "I_REV_EWC": RunSpec("I_REV_EWC", "ising_fast", "ewc", "reverse"),
}


def _calibration(manifest, section: str, architecture: str):
    block = manifest.get(section)
    if block is None:
        return None
    architectures = block.get("architectures")
    if architectures is None:
        return None
    return architectures.get(architecture)


def assert_frozen(manifest) -> None:
    required_freezes = (
        "design_confirmed", "stage0_endpoints", "legacy_epoch_semantics",
        "stress_tasks_and_orders", "budget_B", "ewc_parameters",
        "replay_parameters", "evaluation_protocols", "success_criteria",
    )
    missing = [name for name in required_freezes if not bool(manifest.freeze.get(name, False))]
    if missing:
        raise RuntimeError(f"formal runs blocked; manifest fields are not frozen: {missing}")
    budget = manifest.budget_pilot.get("selected_B")
    if budget is None or int(budget) <= 0:
        raise RuntimeError("formal runs blocked; selected_B is unset")
    for architecture in ARCHITECTURES:
        ewc = _calibration(manifest, "ewc_calibration", architecture)
        replay = _calibration(manifest, "replay_calibration", architecture)
        if (
            ewc is None
            or ewc.get("selected_lambda") is None
            or ewc.get("fisher_samples") is None
            or ewc.get("initial_fisher_path") is None
        ):
            raise RuntimeError(f"formal runs blocked; EWC calibration unset for {architecture}")
        if float(ewc.selected_lambda) == FORBIDDEN_LAMBDA:
            raise RuntimeError("the forbidden V3 EWC lambda cannot be used in V4")
        if (
            replay is None
            or replay.get("selected_ratio") is None
            or replay.get("capacity_samples") is None
            or replay.get("target_bytes") is None
        ):
            raise RuntimeError(f"formal runs blocked; Replay calibration unset for {architecture}")


def _stage0(manifest, architecture: str) -> Path:
    return REPO_ROOT / str(manifest.stage0[architecture].endpoint.path)


def _task_config(architecture: Architecture, task: str) -> str:
    if task == "T0":
        return architecture.base_config
    return f"{architecture.config_dir}/{task}"


def _order(manifest, kind: str) -> list[str]:
    if kind == "high_conflict":
        return [str(task) for task in manifest.stress_selection.high_conflict_order]
    if kind == "reverse":
        return [str(task) for task in manifest.stress_selection.reverse_order]
    if kind == "t0_stages":
        return ["T0_stage1", "T0_stage2", "T0_stage3", "T0_stage4"]
    raise ValueError(f"unsupported order kind: {kind}")


def build_command(spec: RunSpec, manifest, *, gpu: str, resume: bool) -> list[str]:
    architecture = ARCHITECTURES[spec.architecture]
    python = str(EXPECTED_PREFIX / "bin" / "python")
    budget = int(manifest.budget_pilot.selected_B)
    epoch = manifest.legacy_epoch
    run_name = f"qewc_v4/formal/{spec.run_id}"
    if spec.method in {"mix_total", "mix_exposure"}:
        total_epochs = (4 if spec.method == "mix_total" else 5) * budget
        command = [
            python, "-u", str(RUN_MIX),
            "--method", spec.method,
            "--run-name", run_name,
            "--config-name", architecture.mix_config,
            "--stage0", str(_stage0(manifest, spec.architecture)),
            "--legacy-epochs-per-task", str(budget),
            "--num-continual-tasks", "4",
            "--total-legacy-epochs", str(total_epochs),
            "--checkpoint-count", str(4 if spec.method == "mix_total" else 5),
            "--requested-samples-per-epoch", str(epoch.requested_samples),
            "--batch-size", str(epoch.micro_batch_size),
            "--accumulate-steps", str(epoch.accumulate_steps),
            "--checkpoint-prefix", architecture.checkpoint_prefix,
            "--validation-samples", "65536",
            "--seed", str(manifest.seeds.lifelong_train),
            "--gpu", gpu,
            "--python", python,
        ]
    else:
        order = _order(manifest, spec.order_kind)
        command = [
            python, "-u", str(RUN_CONTINUAL),
            "--method", spec.method,
            "--run-name", run_name,
            "--base-task", f"T0:{_task_config(architecture, 'T0')}",
        ]
        if spec.method == "t0_only":
            for alias in order:
                command.extend(("--task", f"{alias}:{_task_config(architecture, 'T0')}"))
        else:
            for task in order:
                command.extend(("--task", f"{task}:{_task_config(architecture, task)}"))
        command.extend((
            "--order", ",".join(order), "--expected-stages", "4",
            "--stage0", str(_stage0(manifest, spec.architecture)),
            "--epochs-per-task", str(budget),
            "--requested-samples-per-epoch", str(epoch.requested_samples),
            "--batch-size", str(epoch.micro_batch_size),
            "--accumulate-steps", str(epoch.accumulate_steps),
            "--checkpoint-prefix", architecture.checkpoint_prefix,
            "--validation-samples", "65536",
            "--seed", str(manifest.seeds.lifelong_train),
            "--gpu", gpu, "--python", python,
        ))
        if spec.method == "ewc":
            calibration = _calibration(manifest, "ewc_calibration", spec.architecture)
            command.extend((
                "--positive-weight", str(calibration.get("positive_weight", 1.0)),
                "--fisher-samples", str(calibration.fisher_samples),
                "--fisher-batch-size", str(calibration.get("fisher_batch_size", 2048)),
                "--fisher-per-sample-chunk-size", str(calibration.get("per_sample_chunk_size", 16)),
                "--fisher-seed", str(calibration.get("fisher_seed", 20260912)),
                "--initial-fisher", str(REPO_ROOT / str(calibration.initial_fisher_path)),
                "--ewc-lambda", str(calibration.selected_lambda),
                "--ewc-gamma", str(manifest.ewc_calibration.gamma),
            ))
        elif spec.method == "replay":
            calibration = _calibration(manifest, "replay_calibration", spec.architecture)
            command.extend((
                "--replay-dir", str(REPO_ROOT / "outputs" / run_name / "replay"),
                "--replay-capacity", str(calibration.capacity_samples),
                "--replay-ratio", str(calibration.selected_ratio),
                "--replay-target-bytes", str(calibration.target_bytes),
                "--replay-byte-tolerance", str(calibration.get("byte_tolerance", 0.05)),
                "--replay-seed", str(calibration.get("seed", 20260913)),
            ))
    if resume:
        command.append("--resume")
    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", choices=tuple(RUNS), required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if Path(sys.prefix).resolve() != EXPECTED_PREFIX.resolve():
        raise RuntimeError(
            f"wrong Python environment: executable={sys.executable}, prefix={sys.prefix}"
        )
    manifest = OmegaConf.load(args.manifest)
    assert_frozen(manifest)
    command = build_command(RUNS[args.run_id], manifest, gpu=args.gpu, resume=args.resume)
    print("[QEWC v4 formal] " + " ".join(command), flush=True)
    if not args.dry_run:
        subprocess.run(command, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
