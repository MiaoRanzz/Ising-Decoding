"""
Pretrain GoogleDecoder or CompactGoogleDecoder on SI1000 detector error models (DEMs).

Combines train_teacher_si1000_dem.py and train_compact_si1000_dem.py into a single script
with a --model_arch flag to choose between 'teacher' and 'compact'.

Example:
    python -m src.training.train_si1000_dem --test --model_arch teacher
    python -m src.training.train_si1000_dem --test --model_arch compact
    python -m src.training.train_si1000_dem \
        --data_dir pretrain_si1000 --distance 3 --basis Z --model_arch compact
"""

import argparse
import math
import os
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from lion_pytorch import Lion
from torch.nn.parallel import DistributedDataParallel as DDP

from src.data.si1000_dem import SI1000DEMDataset
from src.models.AQ2 import GoogleDecoder as AQ2Decoder, AUX_WEIGHTS
from src.training.ler import ler_fit_with_std
from src.training.seed import seed_everything
from src.training.checkpoint import save_checkpoint

from src.training.distributed import (
    is_distributed,
    is_main_process,
    resolve_device,
    setup_distributed,
)
from src.training.common import (
    gather_rng_states,
    restore_rng_states,
)
from src.training.ema import ExponentialMovingAverage


def default_si1000_lr(distance: int) -> float:
    return 2e-5

def compute_aq2_lr(base_lr: float, n_stabilizers: int, num_rounds: int) -> float:
    lr = base_lr
    lr *= 0.8 ** (math.log2(n_stabilizers / 8))
    lr *= 2.0 ** (math.log2(num_rounds / 24))
    return lr


def get_distance_curriculum_weights(
    examples_seen: int,
    total_examples: int,
    distances: list[int],
) -> np.ndarray:
    """Compute curriculum weights for distance sampling.

    Progressively shifts from small to large code distances over training.
    Early in training the model sees mostly small distances; later the
    distribution approaches uniform across all distances.

    Uses a sigmoid schedule: the "target" distance moves from the minimum
    toward the mean distance as training progresses, and distances closer
    to the target get higher probability via a Gaussian kernel.

    Returns:
        Normalized probability array, same length and order as distances.
    """
    if len(distances) <= 1:
        return np.ones(len(distances)) / len(distances)

    sorted_d = sorted(distances)
    indices = [sorted_d.index(d) for d in distances]

    progress = min(1.0, examples_seen / max(total_examples, 1))

    # Target distance: start at min, asymptote toward the max (A.2.4)
    min_d = sorted_d[0]
    max_d = sorted_d[-1]
    peak = min_d + (max_d - min_d) * progress

    # Gaussian weights around the current peak
    sigma = 2.0
    w_c = 3.0
    weights = []
    for d in sorted_d:
        g = math.exp(-0.5 * ((d - peak) / sigma) ** 2)
        weights.append(1.0 + w_c * g)
    total = sum(weights)
    probs_sorted = np.array([w / total for w in weights], dtype=np.float64)

    # Reorder back to the caller's distance order
    return probs_sorted[indices]


def compute_dem_validation_ler(
    model: torch.nn.Module,
    dataset: SI1000DEMDataset,
    device: torch.device,
    max_rounds: int,
    num_samples: int,
    batch_size: int,
    min_rounds: int = 10,
    max_error_rate: float = 0.5,
    p_eval: float = 1.0,
) -> tuple[float, float, float, float]:
    """Compute LER on validation set using DEM samples."""
    model.eval()

    # Use available rounds directly for SI1000 (sparse round distribution)
    # For example, SI1000 has [1, 10, 30, 50, 70, ..., 250]
    eval_rounds = sorted(dataset.available_rounds)
    # Optionally limit to a subset for speed (e.g., every other available round)
    if len(eval_rounds) > 8:
        eval_rounds = eval_rounds[::2]  # every other available round
        if max_rounds not in eval_rounds and max_rounds in dataset.available_rounds:
            eval_rounds.append(max_rounds)

    error_rates = []
    round_details = []
    with torch.no_grad():
        for t in eval_rounds:
            total = 0
            errors = 0
            while total < num_samples:
                bs = min(batch_size, num_samples - total)
                m, e, i_idx, targets, _ = dataset.generate_batch(
                    batch_size=bs,
                    rounds=t,
                    scale_factor=p_eval,
                )
                logits = model(m.to(device), e.to(device), i_idx.to(device), current_distance=dataset.distance)
                preds = (torch.sigmoid(logits) > 0.5).float().cpu().numpy().reshape(-1)
                y = targets.cpu().numpy().reshape(-1)
                errors += int(np.sum(preds != y))
                total += bs
            err_rate = errors / total
            error_rates.append(err_rate)
            round_details.append((t, err_rate, errors, total))

    # Print detailed per-round results
    print("  Per-round validation results:", flush=True)
    for rounds, err_rate, errors, total in round_details:
        print(f"  Round {rounds:3d}: error_rate={err_rate:.4f} ({errors}/{total})", flush=True)

    model.train()
    fit_rounds = []
    fit_error_rates = []
    for rounds, err in zip(eval_rounds, error_rates):
        if rounds >= min_rounds and err <= max_error_rate:
            fit_rounds.append(rounds)
            fit_error_rates.append(err)

    if not fit_rounds:
        return 0.5, 0.0, 0.0, 0.0

    rounds_arr = np.asarray(fit_rounds, dtype=np.float64)
    error_arr = np.asarray(fit_error_rates, dtype=np.float64)
    # ler_fit_with_std returns (epsilon, r_squared, intercept, std_intercept)
    return ler_fit_with_std(rounds_arr, error_arr, min_round=min_rounds)


def train(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Train AlphaQubit 2 on SI1000 detector error models."
    )
    parser.add_argument(
        "--model_arch",
        type=str,
        default="aq2",
        choices=["aq2"],
        help="Model architecture: 'aq2'(AlphaQubit 2)",
    )
    parser.add_argument("--data_dir", type=str, default="pretrain_si1000", help="Path to SI1000 DEM folder")
    parser.add_argument("--distances", type=int, nargs="+", default=[3, 5, 7, 9, 11], help="List of code distances")
    parser.add_argument("--basis", type=str, default=None, help="Optional basis filter: X or Z")
    parser.add_argument("--rounds", type=int, default=None, help="Optional fixed rounds; otherwise mix available rounds")
    parser.add_argument("--batch_size", type=int, default=256, help="Initial global batch size")
    parser.add_argument("--final_batch_size", type=int, default=1024, help="Final global batch size after ramp-up")
    parser.add_argument(
        "--batch_size_change_examples",
        type=int,
        default=4_000_000,
        help="Number of seen examples before switching from initial to final batch size",
    )
    parser.add_argument(
        "--total_examples",
        type=int,
        default=2_000_000_000,
        help="Maximum number of training examples.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Optional cap on optimization steps (overrides total_examples if set).",
    )
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument(
        "--log_dir",
        type=str,
        default=None,
        help="TensorBoard log directory (rank0 only). Default: runs/si1000_dem_pretrain or runs/si1000_dem_pretrain_compact",
    )
    parser.add_argument(
        "--tb_log_every",
        type=int,
        default=50,
        help="TensorBoard logging cadence (in steps).",
    )
    parser.add_argument("--eval_every_examples", type=int, default=1_000_000)
    parser.add_argument(
        "--eval_every_steps",
        type=int,
        default=0,
        help="If >0, evaluate every N optimizer steps (overrides eval_every_examples).",
    )
    p_base = 0.0015
    parser.add_argument("--eval_num_samples", type=int, default=50_000)
    parser.add_argument("--eval_batch_size", type=int, default=1_024)
    parser.add_argument("--eval_min_rounds", type=int, default=10, help="Minimum rounds to include in LER fit")
    parser.add_argument("--eval_max_error_rate", type=float, default=0.4, help="Maximum error rate to include in LER fit")
    parser.add_argument("--noise_levels", type=float, nargs="+", default=[p / p_base for p in [0.001, 0.0015, 0.002, 0.0025, 0.003, 0.0035, 0.004]],help="Noise levels p for fixed distribution sampling (Table S1)")
    parser.add_argument("--noise_weights", type=float, nargs="+", default=[1, 5, 2, 3, 3, 2, 2], help="Relative weights for noise levels (Table S1)")
    parser.add_argument("--p_eval", type=float, default=1.0, help="Noise scale factor for primary validation (p=0.0015 nominal)")
    parser.add_argument(
        "--p_eval2",
        type=float,
        default=2.0,
        help="Noise scale factor for secondary validation track (default 2.0 = p=0.003 high-noise proxy; paper dev practice: more events, lower selection variance).",
    )
    parser.add_argument("--save_every_steps", type=int, default=10_000)
    parser.add_argument(
        "--keep_last_checkpoints",
        type=int,
        default=0,
        help="If >0, keep only the most recent N periodic checkpoints.",
    )
    parser.add_argument(
        "--no_early_stopping",
        action="store_true",
        help="Disable early stopping (default: enabled).",
    )
    parser.add_argument("--patience_evals", type=int, default=20)
    parser.add_argument("--min_delta", type=float, default=0.0)
    parser.add_argument(
        "--no_fit_quality_filter",
        action="store_true",
        help="Disable LER-fit quality filter (default: enabled).",
    )
    parser.add_argument("--ema_decay", type=float, default=0.9999, help="EMA asymptotic decay")
    parser.add_argument("--ema_warmup_steps", type=int, default=10000, help="EMA decay warmup steps")
    parser.add_argument(
        "--no_ema",
        action="store_true",
        help="Use raw weights for evaluation/checkpoint selection (ablates EMA; paper did not use EMA).",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="Checkpoint save path. Default: checkpoints/pretrained_si1000/si1000_pretrained.pth or checkpoints/pretrained_compact_si1000/compact_si1000_pretrained.pth",
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Resume training from a checkpoint.",
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for Python/NumPy/PyTorch")

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device type: auto, cuda, mps, or cpu",
    )
    parser.add_argument("--test", action="store_true", help="Quick smoke test (few steps, small model/batch)")
    args = parser.parse_args(argv)

    # Set defaults based on model architecture
    if args.log_dir is None:
        args.log_dir = "runs/aq2_si1000_pretrain"
    
    if args.save_path is None:
        args.save_path = "checkpoints/aq2_si1000/aq2_pretrained.pth"

    if args.device == "auto":
        if torch.cuda.is_available():
            device_type = "cuda"
        elif torch.backends.mps.is_available():
            device_type = "mps"
        else:
            device_type = "cpu"
    else:
        device_type = args.device

    rank = 0
    world_size = 1
    local_rank = 0
    if device_type in {"cuda", "cpu"}:
        try:
            rank, world_size, local_rank = setup_distributed(device_type)
        except Exception:
            rank, world_size, local_rank = 0, 1, 0

    if args.test:
        args.total_examples = 256
        args.steps = 10
        args.batch_size = 8
        args.final_batch_size = 8
        args.batch_size_change_examples = 0
        args.eval_every_examples = 0
        args.eval_every_steps = 2
        args.eval_num_samples = 128
        args.eval_batch_size = 64
        args.save_every_steps = 2
        args.patience_evals = 3


    seed_everything(args.seed + rank, deterministic=True)
    rng = np.random.default_rng(args.seed + rank)

    noise_levels = args.noise_levels
    noise_probs = np.array(args.noise_weights, dtype=np.float64)
    noise_probs = noise_probs / noise_probs.sum()

    device = resolve_device(device_type)
    base_lr = args.learning_rate if args.learning_rate is not None else default_si1000_lr(args.distances[0])

    writer = None
    if is_main_process():
        try:
            from torch.utils.tensorboard import SummaryWriter

            model_suffix = "_aq2"
            distances_str = "_".join(str(d) for d in args.distances)
            run_name = f"d{distances_str}_b{args.basis or 'mix'}{model_suffix}"
            log_dir = os.path.join(args.log_dir, run_name)
            writer = SummaryWriter(log_dir=log_dir)
            writer.add_text(
                "config",
                "\n".join(
                    [
                        f"model_arch={args.model_arch}",
                        f"distances={args.distances}",
                        f"basis={args.basis}",
                        f"seed={args.seed}",
                        f"learning_rate={base_lr}",
                        f"batch_size={args.batch_size}",
                        f"final_batch_size={args.final_batch_size}",
                        f"noise_levels={args.noise_levels}",
                        f"noise_weights={args.noise_weights}",
                        f"p_eval={args.p_eval}",
                        f"total_examples={args.total_examples}",
                    ]
                ),
            )
        except Exception as exc:
            print(f"[TB] Disabled (failed to initialize SummaryWriter): {exc}", flush=True)
            writer = None

    # Create one dataset per distance for training and validation
    datasets: dict[int, SI1000DEMDataset] = {}
    val_datasets: dict[int, SI1000DEMDataset] = {}
    for d in args.distances:
        datasets[d] = SI1000DEMDataset(
            data_dir=args.data_dir,
            distance=d,
            basis=args.basis,
            rounds=args.rounds,
            shuffle_sources=True,
            uniform_rounds=True,
            seed=args.seed + rank,
        )
        val_datasets[d] = SI1000DEMDataset(
            data_dir=args.data_dir,
            distance=d,
            basis=args.basis,
            rounds=args.rounds,
            shuffle_sources=False,
            uniform_rounds=True,
            seed=args.seed + 1,
        )

    # Use max frames across all distances for the model's temporal dimension
    # (model stores this for documentation; T_frames=T+1 per paper A.1.1 layout)
    max_rounds_all = max(ds.max_frames for ds in datasets.values())

    if is_main_process():
        for d in args.distances:
            ds = datasets[d]
            print(f"[DATA] d={d} available_rounds={ds.available_rounds}, "
                  f"max_rounds={ds.max_rounds}",
                  flush=True)

    # Create model based on architecture
    model = AQ2Decoder(
        distances=args.distances,
        num_rounds=max_rounds_all,
        d_model=64 if args.test else 512,
        nhead=2 if args.test else 16,
        dim_feedforward=256 if args.test else 1024,
        dropout=0.1,
        attn_key_size=32,
        temporal_K=6,
    ).to(device)


    model_to_track = model
    if is_distributed():
        # Compact/Micro have hidden_projector that is unused during baseline training
        model = DDP(model, device_ids=[local_rank] if device_type == "cuda" else None)
        model_to_track = model.module

    if is_main_process():
        num_params = sum(p.numel() for p in model_to_track.parameters() if p.requires_grad)
        model_name_map = {"aq2": "AlphaQubit2"}
        model_name = model_name_map.get(args.model_arch, "Model")
        print(f"[MODEL] {model_name} trainable_params={num_params:,}", flush=True)
        if writer is not None:
            writer.add_scalar("meta/trainable_params", float(num_params), 0)

    optimizer = Lion(
        model.parameters(),
        lr=base_lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.99),
    )
    criterion = nn.BCEWithLogitsLoss()
    ema = ExponentialMovingAverage(
        model_to_track,
        decay=args.ema_decay,
        warmup_steps=args.ema_warmup_steps,
    )

    model.train()
    examples_seen = 0
    start_step = 1

    if args.steps is None:
        max_steps = int(math.ceil(args.total_examples / args.batch_size))
    else:
        max_steps = int(args.steps)
    total_examples = int(args.total_examples) if args.steps is None else int(args.steps * args.batch_size)

    next_eval_at = int(args.eval_every_examples) if args.eval_every_examples > 0 else total_examples + 1
    
    # Track two separate best checkpoints: p_eval=1.0 (primary) and p_eval2 (secondary)
    best_ler_p1 = float("inf")
    best_meta_p1 = {}
    num_bad_evals_p1 = 0

    best_ler_p2 = float("inf")
    best_meta_p2 = {}
    num_bad_evals_p2 = 0
    
    periodic_paths: list[str] = []

    if args.resume_from:
        if not os.path.exists(args.resume_from):
            raise FileNotFoundError(f"--resume_from not found: {args.resume_from}")

        ckpt = torch.load(args.resume_from, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model_to_track.load_state_dict(ckpt["model_state_dict"], strict=False)
        else:
            model_to_track.load_state_dict(ckpt, strict=False)

        if isinstance(ckpt, dict) and "optimizer_state_dict" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except Exception:
                if is_main_process():
                    print("[RESUME] Warning: failed to load optimizer state.", flush=True)

        if isinstance(ckpt, dict) and "ema_state_dict" in ckpt:
            try:
                ema.load_state_dict(ckpt["ema_state_dict"])
                ema.to_device(device)
            except Exception:
                if is_main_process():
                    print("[RESUME] Warning: failed to load EMA state.", flush=True)

        training_state = ckpt.get("training_state", {}) if isinstance(ckpt, dict) else {}
        step0 = int(training_state.get("step", ckpt.get("step", 0) if isinstance(ckpt, dict) else 0))
        examples_seen = int(training_state.get("examples_seen", ckpt.get("examples_seen", 0) if isinstance(ckpt, dict) else 0))
        
        # Load both checkpoint states
        best_ler_p1 = float(training_state.get("best_ler_p1", best_ler_p1))
        best_meta_p1 = dict(training_state.get("best_meta_p1", {}))
        num_bad_evals_p1 = int(training_state.get("num_bad_evals_p1", 0))
        
        best_ler_p2 = float(training_state.get("best_ler_p2", best_ler_p2))
        best_meta_p2 = dict(training_state.get("best_meta_p2", {}))
        num_bad_evals_p2 = int(training_state.get("num_bad_evals_p2", 0))
        
        start_step = max(1, step0 + 1)

        restore_rng_states(
            ckpt.get("rng_states_by_rank") if isinstance(ckpt, dict) else None,
            datasets,
            val_datasets,
            rng,
            device_type,
        )

        if is_main_process():
            print(f"[RESUME] Loaded {args.resume_from} (step={step0}, examples_seen={examples_seen:,})", flush=True)
        if is_distributed():
            dist.barrier()

        # Adjust next_eval_at based on where we are after resuming
        if args.eval_every_examples > 0:
            next_eval_at = ((examples_seen // args.eval_every_examples) + 1) * args.eval_every_examples

    for step in range(start_step, max_steps + 1):
        global_batch = args.final_batch_size if examples_seen >= args.batch_size_change_examples else args.batch_size
        batch_per_gpu = int(math.ceil(global_batch / world_size))
        effective_batch = batch_per_gpu * world_size
        accum_steps = max(1, 1024 // effective_batch)  # paper A.2.5 target

        # Distance curriculum: select current code distance for this batch
        curriculum_weights = get_distance_curriculum_weights(examples_seen, total_examples, args.distances)
        if is_main_process():
            current_d = int(rng.choice(args.distances, p=curriculum_weights))
        else:
            current_d = args.distances[0]
        if is_distributed():
            d_tensor = torch.tensor([current_d], device=device, dtype=torch.long)
            dist.broadcast(d_tensor, src=0)
            current_d = int(d_tensor.item())

        train_dataset = datasets[current_d]

        # Per-sample independent noise + auxiliary labels (A.2.1, A.2.3, A.3.3)
        batch_data = train_dataset.generate_batch_with_aux(
            batch_size=batch_per_gpu,
            noise_levels=noise_levels,
            noise_probs=noise_probs,
            temporal_K=model_to_track.K,
            rounds=args.rounds,
        )
        m = batch_data["m"].to(device)
        e = batch_data["e"].to(device)
        i_idx = batch_data["i_idx"].to(device)
        targets = batch_data["targets"].to(device)               # (B, 1)
        T = batch_data["T"]
        B, _, N = m.shape

        # Pseudo-terminated stabilizer measurements (input to aux heads ①④)
        pseudo_m = batch_data["pseudo_m"].to(device)             # (B, C, N)
        pseudo_e = batch_data["pseudo_e"].to(device)             # (B, C, N)

        # Auxiliary labels per head (Table S3 ordering)
        aux_tgts = [
            batch_data["aux_tgt_pseudo"].to(device),             # ① pseudo_intermediate
            batch_data["aux_tgt_noiseless"].to(device),          # ② noiseless
            batch_data["aux_tgt_noiseless_diff"].to(device),     # ③ noiseless_diff
            batch_data["aux_tgt_noiseless_to_inter"].to(device), # ④ noiseless_to_inter_diff
        ]

        # A.2.2: per-sample stabilizer representation dropout
        # 80% of samples drop 50% of stabilizer representations (BERT-style).
        # Vectorized implementation avoids Python loops and GPU syncs.
        do_mask = (torch.rand(B, 1, 1, device=device) < 0.8).float()
        rand_mask = (torch.rand(B, T, N, device=device) > 0.5).float()
        mask = (1 - do_mask) + do_mask * rand_mask  # 20% keep-all, 80% random 50%

        # Dynamic LR: A.2.6
        n_stabs = train_dataset.n_spatial
        scaled_base_lr = compute_aq2_lr(base_lr, n_stabs, batch_data["num_cycles"])
        warmup_frac = min(1.0, max(1, examples_seen) / 1_000_000)
        current_lr = scaled_base_lr * warmup_frac
        for pg in optimizer.param_groups:
            pg["lr"] = current_lr

        if (step - start_step) % accum_steps == 0:
            optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device_type, enabled=(device_type == 'cuda'), dtype=torch.bfloat16):
            logits, aux_preds = model(
                m, e, i_idx, current_distance=current_d,
                mask=mask, return_aux=True,
                pseudo_measurements=pseudo_m,
                pseudo_detection_events=pseudo_e,
            )

            # Main final-observable loss (weight 1.2, Table S3)
            loss = 1.2 * criterion(logits, targets)

            # Auxiliary losses (A.2.3, Table S3)
            # ① pseudo_intermediate (w=1)   ② noiseless (w=1)
            # ③ noiseless_diff (w=1)        ④ noiseless_to_inter_diff (w=8)
            aux_losses: list[torch.Tensor] = []
            for aux_logits, aux_tgt, w in zip(aux_preds, aux_tgts, AUX_WEIGHTS):
                aux_l = criterion(aux_logits, aux_tgt)
                aux_losses.append(aux_l.detach())
                loss = loss + w * aux_l

        (loss / accum_steps).backward()

        if (step - start_step + 1) % accum_steps == 0 or step == max_steps:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            ema.update(model_to_track)

        if step % args.log_every == 0 or step == 1:
            if is_main_process():
                aux_str = " ".join(
                    f"aux{i}={aux_losses[i].item():.4f}" for i in range(4)
                )
                print(
                    f"step={step} examples={examples_seen:,} batch={effective_batch * accum_steps} "
                    f"d={current_d} lr={current_lr:.2e} frames={T} cycles={batch_data.get('num_cycles', T - 1)} "
                    f"loss={loss.item():.6f} | {aux_str}",
                    flush=True,
                )
            if writer is not None and is_main_process() and (step == 1 or step % int(args.tb_log_every) == 0):
                writer.add_scalar("train/loss", float(loss.item()), step)
                writer.add_scalar("train/lr", float(current_lr), step)
                writer.add_scalar("train/examples_seen", float(examples_seen), step)
                writer.add_scalar("train/frames", float(T), step)
                writer.add_scalar("train/distance", float(current_d), step)

        examples_seen += effective_batch
        if examples_seen >= total_examples:
            break

        # Periodic checkpoint
        should_save = False
        if args.save_every_steps > 0 and (step % args.save_every_steps == 0):
            should_save = True
        # Force save at step 1000
        if step == 1000:
            should_save = True
        
        if should_save:
            if is_distributed():
                dist.barrier()
            rng_states = gather_rng_states(datasets, val_datasets, rng, device_type)
            if is_main_process():
                root, ext = os.path.splitext(args.save_path)
                periodic_path = f"{root}_step{step}{ext or '.pth'}"
                save_checkpoint(
                    periodic_path,
                    model_state=model_to_track.state_dict(),
                    ema_state=ema.state_dict(),
                    optimizer_state=optimizer.state_dict(),
                    meta={
                        "model_arch": args.model_arch,
                        "distances": args.distances,
                        "basis": args.basis,
                        "rounds": args.rounds,
                        "seed": args.seed,
                        "step": step,
                        "examples_seen": examples_seen,
                        "training_state": {
                            "step": step,
                            "examples_seen": examples_seen,
                            "best_ler_p1": best_ler_p1,
                            "best_meta_p1": best_meta_p1,
                            "num_bad_evals_p1": num_bad_evals_p1,
                            "best_ler_p2": best_ler_p2,
                            "best_meta_p2": best_meta_p2,
                            "num_bad_evals_p2": num_bad_evals_p2,
                        },
                        "rng_states_by_rank": rng_states,
                    },
                )
                print(f"[CKPT] Saved periodic checkpoint at step={step} -> {periodic_path}", flush=True)
                periodic_paths.append(periodic_path)
                if args.keep_last_checkpoints > 0:
                    while len(periodic_paths) > args.keep_last_checkpoints:
                        old = periodic_paths.pop(0)
                        try:
                            os.remove(old)
                        except FileNotFoundError:
                            pass
            if is_distributed():
                dist.barrier()

        # Evaluation
        should_eval = False
        if args.eval_every_steps and args.eval_every_steps > 0:
            should_eval = (step % int(args.eval_every_steps) == 0)
        elif args.eval_every_examples > 0:
            should_eval = examples_seen >= next_eval_at
        
        # Force evaluation at step 1000
        if step == 1000:
            should_eval = True

        if should_eval:
            if is_distributed():
                dist.barrier()
            rng_states = gather_rng_states(datasets, val_datasets, rng, device_type)
            stop_tensor = torch.zeros(1, device=device)
            ler_p1 = 0.5
            ler_p2 = 0.5
            r2_p1 = 0.0
            r2_p2 = 0.0
            intercept_p1 = 0.0
            std_intercept_p1 = 0.0
            intercept_p2 = 0.0
            std_intercept_p2 = 0.0

            if is_main_process():
                if not args.no_ema:
                    ema.apply_shadow(model_to_track)
                try:
                    # Evaluate all training distances
                    ler_by_d_p1: dict[int, float] = {}
                    ler_by_d_p2: dict[int, float] = {}
                    for eval_d in args.distances:
                        eval_ds = val_datasets[eval_d]
                        ler_p1, r2_p1, intercept_p1, std_intercept_p1 = compute_dem_validation_ler(
                            model=model_to_track,
                            dataset=eval_ds,
                            device=device,
                            max_rounds=eval_ds.max_rounds,
                            num_samples=args.eval_num_samples,
                            batch_size=args.eval_batch_size,
                            min_rounds=args.eval_min_rounds,
                            max_error_rate=args.eval_max_error_rate,
                            p_eval=1.0,
                        )
                        ler_p2, r2_p2, intercept_p2, std_intercept_p2 = compute_dem_validation_ler(
                            model=model_to_track,
                            dataset=eval_ds,
                            device=device,
                            max_rounds=eval_ds.max_rounds,
                            num_samples=args.eval_num_samples,
                            batch_size=args.eval_batch_size,
                            min_rounds=args.eval_min_rounds,
                            max_error_rate=args.eval_max_error_rate,
                            p_eval=args.p_eval2,
                        )
                        ler_by_d_p1[eval_d] = ler_p1
                        ler_by_d_p2[eval_d] = ler_p2
                        print(f"  [VAL d={eval_d}] p=1.0 ler={ler_p1:.6g} r2={r2_p1:.3f}  |  p={args.p_eval2:.1f} ler={ler_p2:.6g} r2={r2_p2:.3f}", flush=True)

                    # Use largest distance for checkpoint selection
                    eval_d = args.distances[-1]
                    ler_p1 = ler_by_d_p1[eval_d]
                    ler_p2 = ler_by_d_p2[eval_d]
                finally:
                    if not args.no_ema:
                        ema.restore(model_to_track)

                # Process p_eval=1.0 results
                valid_fit_p1 = True
                reject_reason_p1 = []
                if not args.no_fit_quality_filter:
                    if r2_p1 <= 0.9:
                        valid_fit_p1 = False
                        reject_reason_p1.append(f"R²={r2_p1:.3f}≤0.9")
                    intercept_threshold_p1 = max(-0.02, -std_intercept_p1)
                    if intercept_p1 <= intercept_threshold_p1:
                        valid_fit_p1 = False
                        reject_reason_p1.append(f"intercept={intercept_p1:.4f}≤{intercept_threshold_p1:.4f}")

                improved_p1 = valid_fit_p1 and (ler_p1 < best_ler_p1 - args.min_delta)

                # Process p_eval2 results
                valid_fit_p2 = True
                reject_reason_p2 = []
                if not args.no_fit_quality_filter:
                    if r2_p2 <= 0.9:
                        valid_fit_p2 = False
                        reject_reason_p2.append(f"R²={r2_p2:.3f}≤0.9")
                    intercept_threshold_p2 = max(-0.02, -std_intercept_p2)
                    if intercept_p2 <= intercept_threshold_p2:
                        valid_fit_p2 = False
                        reject_reason_p2.append(f"intercept={intercept_p2:.4f}≤{intercept_threshold_p2:.4f}")

                improved_p2 = valid_fit_p2 and (ler_p2 < best_ler_p2 - args.min_delta)

                # Save checkpoint for p_eval=1.0
                if improved_p1:
                    best_ler_p1 = ler_p1
                    best_meta_p1 = {
                        "best_step": step,
                        "best_examples_seen": examples_seen,
                        "best_ler": ler_p1,
                        "best_r2": r2_p1,
                        "best_intercept": intercept_p1,
                        "p_eval": 1.0,
                    }
                    num_bad_evals_p1 = 0

                    root, ext = os.path.splitext(args.save_path)
                    save_path_p1 = f"{root}_p1.0{ext or '.pth'}"
                    sd = dict(model_to_track.state_dict())
                    if not args.no_ema:
                        sd.update({k: v for k, v in ema.shadow.items()})
                    save_checkpoint(
                        save_path_p1,
                        model_state=sd,
                        ema_state=ema.state_dict(),
                        optimizer_state=optimizer.state_dict(),
                        meta={
                            "model_arch": args.model_arch,
                            "distances": args.distances,
                            "basis": args.basis,
                            "rounds": args.rounds,
                            "seed": args.seed,
                            "step": step,
                            "examples_seen": examples_seen,
                            "p_eval": 1.0,
                            "weights": "ema" if not args.no_ema else "raw",
                            "training_state": {
                                "step": step,
                                "examples_seen": examples_seen,
                                "best_ler_p1": best_ler_p1,
                                "best_meta_p1": best_meta_p1,
                                "num_bad_evals_p1": num_bad_evals_p1,
                                "best_ler_p2": best_ler_p2,
                                "best_meta_p2": best_meta_p2,
                                "num_bad_evals_p2": num_bad_evals_p2,
                            },
                            "rng_states_by_rank": rng_states,
                        },
                    )
                    print(
                        f"[VAL p=1.0] examples={examples_seen:,} ler={ler_p1:.6g} r2={r2_p1:.3f} intercept={intercept_p1:.4f}±{std_intercept_p1:.4f} -> BEST saved",
                        flush=True,
                    )
                else:
                    num_bad_evals_p1 += 1
                    reason_str = f" (rejected: {', '.join(reject_reason_p1)})" if reject_reason_p1 else ""
                    print(
                        f"[VAL p=1.0] examples={examples_seen:,} ler={ler_p1:.6g} r2={r2_p1:.3f} intercept={intercept_p1:.4f}±{std_intercept_p1:.4f} "
                        f"(best={best_ler_p1:.6g}) patience={num_bad_evals_p1}/{args.patience_evals}{reason_str}",
                        flush=True,
                    )

                # Save checkpoint for p_eval2
                if improved_p2:
                    best_ler_p2 = ler_p2
                    best_meta_p2 = {
                        "best_step": step,
                        "best_examples_seen": examples_seen,
                        "best_ler": ler_p2,
                        "best_r2": r2_p2,
                        "best_intercept": intercept_p2,
                        "p_eval": args.p_eval2,
                    }
                    num_bad_evals_p2 = 0

                    root, ext = os.path.splitext(args.save_path)
                    p2_label = f"_p{args.p_eval2:.1f}".replace(".", "_")
                    save_path_p2 = f"{root}{p2_label}{ext or '.pth'}"
                    sd = dict(model_to_track.state_dict())
                    if not args.no_ema:
                        sd.update({k: v for k, v in ema.shadow.items()})
                    save_checkpoint(
                        save_path_p2,
                        model_state=sd,
                        ema_state=ema.state_dict(),
                        optimizer_state=optimizer.state_dict(),
                        meta={
                            "model_arch": args.model_arch,
                            "distances": args.distances,
                            "basis": args.basis,
                            "rounds": args.rounds,
                            "seed": args.seed,
                            "step": step,
                            "examples_seen": examples_seen,
                            "p_eval": args.p_eval2,
                            "weights": "ema" if not args.no_ema else "raw",
                            "training_state": {
                                "step": step,
                                "examples_seen": examples_seen,
                                "best_ler_p1": best_ler_p1,
                                "best_meta_p1": best_meta_p1,
                                "num_bad_evals_p1": num_bad_evals_p1,
                                "best_ler_p2": best_ler_p2,
                                "best_meta_p2": best_meta_p2,
                                "num_bad_evals_p2": num_bad_evals_p2,
                            },
                            "rng_states_by_rank": rng_states,
                        },
                    )
                    print(
                        f"[VAL p={args.p_eval2:.1f}] examples={examples_seen:,} ler={ler_p2:.6g} r2={r2_p2:.3f} intercept={intercept_p2:.4f}±{std_intercept_p2:.4f} -> BEST saved",
                        flush=True,
                    )
                else:
                    num_bad_evals_p2 += 1
                    reason_str = f" (rejected: {', '.join(reject_reason_p2)})" if reject_reason_p2 else ""
                    print(
                        f"[VAL p={args.p_eval2:.1f}] examples={examples_seen:,} ler={ler_p2:.6g} r2={r2_p2:.3f} intercept={intercept_p2:.4f}±{std_intercept_p2:.4f} "
                        f"(best={best_ler_p2:.6g}) patience={num_bad_evals_p2}/{args.patience_evals}{reason_str}",
                        flush=True,
                    )

                # TensorBoard logging
                if writer is not None:
                    for d in args.distances:
                        writer.add_scalar(f"val/d{d}_ler_p1.0", float(ler_by_d_p1[d]), step)
                        writer.add_scalar(f"val/d{d}_ler_p{args.p_eval2:.1f}", float(ler_by_d_p2[d]), step)
                    writer.add_scalar(f"val/ler_p1.0", float(ler_p1), step)
                    writer.add_scalar(f"val/r2_p1.0", float(r2_p1), step)
                    writer.add_scalar(f"val/best_ler_p1.0", float(best_ler_p1), step)
                    writer.add_scalar(f"val/ler_p{args.p_eval2:.1f}", float(ler_p2), step)
                    writer.add_scalar(f"val/r2_p{args.p_eval2:.1f}", float(r2_p2), step)
                    writer.add_scalar(f"val/best_ler_p{args.p_eval2:.1f}", float(best_ler_p2), step)

                if not (args.eval_every_steps and args.eval_every_steps > 0):
                    next_eval_at += int(args.eval_every_examples)

                # Early stopping decision: broadcast to all ranks
                if not args.no_early_stopping:
                    if num_bad_evals_p1 >= args.patience_evals and num_bad_evals_p2 >= args.patience_evals:
                        stop_tensor.fill_(1)

            # --- all ranks synchronize on early-stop decision ---
            if is_distributed():
                dist.broadcast(stop_tensor, src=0)
            if stop_tensor.item():
                if is_main_process():
                    print(f"Early stopping: no improvement for {args.patience_evals} evals (both p1.0 and p{args.p_eval2:.1f}).", flush=True)
                break

            if is_distributed():
                dist.barrier()

    if best_meta_p1 or best_meta_p2:
        if is_main_process():
            if best_meta_p1:
                root, ext = os.path.splitext(args.save_path)
                save_path_p1 = f"{root}_p1.0{ext or '.pth'}"
                print(f"Finished: best_ler_p1.0={best_ler_p1:.6g} saved at {save_path_p1}", flush=True)
            if best_meta_p2:
                root, ext = os.path.splitext(args.save_path)
                p2_label = f"_p{args.p_eval2:.1f}".replace(".", "_")
                save_path_p2 = f"{root}{p2_label}{ext or '.pth'}"
                print(f"Finished: best_ler_p{args.p_eval2:.1f}={best_ler_p2:.6g} saved at {save_path_p2}", flush=True)
    else:
        if is_main_process():
            os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model_to_track.state_dict(),
                    "model_arch": args.model_arch,
                    "distances": args.distances,
                    "basis": args.basis,
                    "rounds": args.rounds,
                    "seed": args.seed,
                },
                args.save_path,
            )
            print(f"Saved checkpoint to {args.save_path}", flush=True)

    if writer is not None:
        writer.flush()
        writer.close()


if __name__ == "__main__":
    train()
