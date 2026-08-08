#!/usr/bin/env python3
"""Create counterfactual packet-risk labels for local safe no-op training.

For every packet proposed by a frozen Ising-fast checkpoint, this script runs
the complete ``predecoder -> PyMatching`` path twice conceptually: once with
the original proposal and once with only that packet removed.  The stored
effect is ``failure_without_packet - failure_with_proposal``:

* ``+1``: packet is locally helpful (removing it creates a logical failure);
* ``-1``: packet is locally harmful (removing it fixes a logical failure);
* `` 0``: no endpoint change;
* ``-2``: the packet was not proposed, hence has no gate target.

The source corpus's ``train_y`` is deliberately not altered.  It remains the
supervision for the original proposal model; this exporter adds supervision
for a separate, frozen-proposal safe-no-op gate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pymatching
import torch
from omegaconf import OmegaConf

HERE = Path(__file__).resolve().parent
L_LOGICAL_ROOT = HERE.parent
REPO_ROOT = HERE.parents[3]
CODE_ROOT = REPO_ROOT / "code"
for path in (HERE, L_LOGICAL_ROOT, CODE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from compare_three_paths import BatchActionModel, _load_model, build_matcher, build_model_cfg, load_corpus
from evaluation.logical_error_rate import PreDecoderMemoryEvalModule, _build_stab_maps
from local_safe_no_op import NO_PACKET_LABEL, PACKET_NAMES, packet_activity, proposal_actions_from_logits
from training.precision import match_input_to_model_memory_format


DEFAULT_SETTINGS = L_LOGICAL_ROOT / "end_to_end.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate local counterfactual safe-no-op labels.")
    parser.add_argument("--settings", type=Path, default=DEFAULT_SETTINGS)
    parser.add_argument("--source-dataset-dir", type=Path, default=None)
    parser.add_argument("--project-config", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--model-id", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Optional deterministic subset; null means every source shot.")
    parser.add_argument("--proposal-batch-size", type=int, default=None)
    parser.add_argument("--counterfactual-batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def _repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def resolve_settings(cli: argparse.Namespace) -> SimpleNamespace:
    settings_path = cli.settings.expanduser().resolve()
    cfg = OmegaConf.load(settings_path)
    section = OmegaConf.to_container(cfg.get("packet_risk_generation", {}), resolve=True)
    if not isinstance(section, dict):
        raise ValueError("packet_risk_generation must be a mapping")

    def pick(name: str, *, required: bool = False, fallback: Any = None) -> Any:
        value = getattr(cli, name)
        if value is None:
            value = section.get(name, fallback)
        if required and value is None:
            raise ValueError(f"missing packet_risk_generation.{name} in {settings_path}")
        return value

    return SimpleNamespace(
        source_dataset_dir=_repo_path(pick("source_dataset_dir", required=True)),
        project_config=_repo_path(pick("project_config", required=True)),
        checkpoint=_repo_path(pick("checkpoint", required=True)),
        model_id=pick("model_id"),
        output_dir=_repo_path(pick("output_dir", required=True)),
        num_samples=pick("num_samples"),
        proposal_batch_size=int(pick("proposal_batch_size", required=True)),
        counterfactual_batch_size=int(pick("counterfactual_batch_size", required=True)),
        seed=int(pick("seed", fallback=20260803)),
        device=pick("device"),
    )


def evaluate_actions(
    pipeline: PreDecoderMemoryEvalModule,
    action_model: BatchActionModel,
    matcher: pymatching.Matching,
    dets: np.ndarray,
    obs: np.ndarray,
    actions: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Run a fixed correction tensor through the real postprocess + matcher."""
    action_model.set_actions(torch.as_tensor(np.array(actions, dtype=np.uint8, copy=True), device=device))
    with torch.no_grad():
        result = pipeline(torch.as_tensor(np.array(dets, dtype=np.uint8, copy=True), device=device))
    pre_l = result[:, 0].to(torch.uint8).cpu().numpy().reshape(-1, 1)
    residual = result[:, 1:].to(torch.uint8).cpu().numpy()
    decoded = np.asarray(matcher.decode_batch(np.ascontiguousarray(residual)), dtype=np.uint8).reshape(obs.shape)
    return np.any(np.bitwise_xor(pre_l, decoded) != obs, axis=1)


def counterfactual_effects(
    pipeline: PreDecoderMemoryEvalModule,
    action_model: BatchActionModel,
    matcher: pymatching.Matching,
    dets: np.ndarray,
    obs: np.ndarray,
    proposal_actions: np.ndarray,
    base_failure: np.ndarray,
    output: np.ndarray,
    counterfactual_batch_size: int,
    device: torch.device,
) -> int:
    """Fill one proposal batch's dense packet-effect labels and return work count."""
    activity = packet_activity(torch.as_tensor(proposal_actions, dtype=torch.bool)).cpu().numpy()
    candidates = np.argwhere(activity)  # local_shot, packet, round, row, column
    if not len(candidates):
        return 0
    for start in range(0, len(candidates), counterfactual_batch_size):
        entries = candidates[start:start + counterfactual_batch_size]
        local_rows = entries[:, 0]
        cf_actions = np.array(proposal_actions[local_rows], dtype=np.uint8, copy=True)
        for packet in range(len(PACKET_NAMES)):
            chosen = entries[:, 1] == packet
            if not np.any(chosen):
                continue
            rows = np.nonzero(chosen)[0]
            rounds, ys, xs = entries[chosen, 2], entries[chosen, 3], entries[chosen, 4]
            if packet == 0:
                cf_actions[rows, 0, rounds, ys, xs] = 0
                cf_actions[rows, 1, rounds, ys, xs] = 0
            else:
                cf_actions[rows, packet + 1, rounds, ys, xs] = 0
        without_failure = evaluate_actions(
            pipeline, action_model, matcher, dets[local_rows], obs[local_rows], cf_actions, device
        )
        effect = without_failure.astype(np.int8) - base_failure[local_rows].astype(np.int8)
        output[entries[:, 0], entries[:, 1], entries[:, 2], entries[:, 3], entries[:, 4]] = effect
    return int(len(candidates))


def main() -> None:
    args = resolve_settings(parse_args())
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"proposal checkpoint not found: {args.checkpoint}")
    if args.proposal_batch_size <= 0 or args.counterfactual_batch_size <= 0:
        raise ValueError("batch sizes must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {args.output_dir}")

    source_metadata, dets_and_obs, train_x, _ = load_corpus(args.source_dataset_dir)
    total_source = int(source_metadata["num_samples"])
    requested = total_source if args.num_samples is None else int(args.num_samples)
    if not 0 < requested <= total_source:
        raise ValueError(f"num_samples must be in [1, {total_source}]")
    rng = np.random.default_rng(args.seed)
    indices = np.arange(total_source, dtype=np.int64) if requested == total_source else np.sort(
        rng.choice(total_source, size=requested, replace=False).astype(np.int64)
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    np.save(args.output_dir / "source_indices.npy", indices)

    matcher, num_obs = build_matcher(source_metadata)
    if num_obs != 1:
        raise NotImplementedError("local risk exporter currently supports one logical observable")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    cfg = build_model_cfg(SimpleNamespace(
        project_config=args.project_config, checkpoint=args.checkpoint, model_id=args.model_id
    ), source_metadata)
    print(f"[load] proposal={args.checkpoint}, device={device}")
    proposal_model = _load_model(cfg, SimpleNamespace(rank=0, world_size=1, device=device)).eval()
    maps = _build_stab_maps(int(source_metadata["distance"]), str(source_metadata["code_rotation"]))
    action_model = BatchActionModel().to(device).eval()
    action_pipeline = PreDecoderMemoryEvalModule(action_model, cfg, maps, device).to(device).eval()

    tensor_shape = (requested, 4, int(source_metadata["n_rounds"]), int(source_metadata["distance"]), int(source_metadata["distance"]))
    packet_shape = (requested, len(PACKET_NAMES), *tensor_shape[2:])
    logits_store = np.lib.format.open_memmap(args.output_dir / "proposal_logits.npy", mode="w+", dtype=np.float16,
                                              shape=tensor_shape)
    action_bits = int(np.prod(tensor_shape[1:]))
    actions_store = np.lib.format.open_memmap(
        args.output_dir / "proposal_actions_packed.npy", mode="w+", dtype=np.uint8,
        shape=(requested, (action_bits + 7) // 8),
    )
    effect_store = np.lib.format.open_memmap(args.output_dir / "packet_effect.npy", mode="w+", dtype=np.int8,
                                              shape=packet_shape)
    effect_store.fill(NO_PACKET_LABEL)
    evaluated_packets = 0

    with torch.no_grad():
        for start in range(0, requested, args.proposal_batch_size):
            end = min(start + args.proposal_batch_size, requested)
            source_rows = indices[start:end]
            x_np = np.array(train_x[source_rows], dtype=np.float32, copy=True)
            x = torch.as_tensor(x_np, dtype=torch.float32, device=device)
            logits = proposal_model(match_input_to_model_memory_format(x, proposal_model))
            logits_np = logits.to(torch.float32).cpu().numpy()
            actions_np = proposal_actions_from_logits(logits).to(torch.uint8).cpu().numpy()
            logits_store[start:end] = logits_np.astype(np.float16)
            actions_store[start:end] = np.packbits(actions_np.reshape(end - start, -1), axis=1, bitorder="little")
            dets = np.array(dets_and_obs[source_rows, :-num_obs], dtype=np.uint8, copy=True)
            obs = np.asarray(dets_and_obs[source_rows, -num_obs:], dtype=np.uint8)
            base_failure = evaluate_actions(action_pipeline, action_model, matcher, dets, obs, actions_np, device)
            evaluated_packets += counterfactual_effects(
                action_pipeline, action_model, matcher, dets, obs, actions_np, base_failure,
                effect_store[start:end], args.counterfactual_batch_size, device,
            )
            print(f"[counterfactual] {end}/{requested} shots; {evaluated_packets} active packets")

    del logits_store, actions_store, effect_store
    metadata = {
        "schema_version": 1,
        "artifact": "local_safe_no_op_risk_dataset",
        "source_dataset_dir": str(args.source_dataset_dir.resolve()),
        "source_dataset_artifact": source_metadata["artifact"],
        "source_num_samples": total_source,
        "num_samples": requested,
        "distance": int(source_metadata["distance"]),
        "n_rounds": int(source_metadata["n_rounds"]),
        "basis": str(source_metadata["basis"]),
        "code_rotation": str(source_metadata["code_rotation"]),
        "proposal_checkpoint": str(args.checkpoint.resolve()),
        "proposal_model_id": int(cfg.model_id),
        "packet_names": list(PACKET_NAMES),
        "effect_definition": "failure_without_packet - failure_with_full_proposal",
        "labels": {"helpful": 1, "neutral": 0, "harmful": -1, "not_proposed": NO_PACKET_LABEL},
        "evaluated_packets": evaluated_packets,
        "seed": args.seed,
        "files": {
            "source_indices": "source_indices.npy",
            "proposal_logits": "proposal_logits.npy",
            "proposal_actions_packed": "proposal_actions_packed.npy",
            "packet_effect": "packet_effect.npy",
        },
        "proposal_action_shape": list(tensor_shape[1:]),
        "proposal_action_pack_bitorder": "little",
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[done] wrote local packet-risk dataset to {args.output_dir}")


if __name__ == "__main__":
    main()
