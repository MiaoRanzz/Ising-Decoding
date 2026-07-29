# QAdapt seq+EWC training and inference examples

These five entry points cover the supported QAdapt sequential+EWC workflow.
Run every command from the repository root. Training and inference outputs are
written below `outputs/`; Google data is written below `benchmarks/google_qec/`.
Both locations are intentionally ignored by Git.

QAdapt uses `HTnet` (`model_id: 111`). New training runs save model checkpoints
as `HTnet.0.<epoch>.pt`.

## Training

```bash
# T0 -> T4 sequential training with a Fisher snapshot after T0-T3.
bash code/examples/train_seq_ewc.sh
```

The training launcher accepts `PREDECODER_PYTHON`, `CUDA_VISIBLE_DEVICES`,
`GPUS`, `EXPERIMENT_NAME`, `DISTANCE`, `N_ROUNDS`, `EPOCHS_PER_TASK`, and
`DRY_RUN=1`. Its EWC controls are `EWC_LAMBDA`, `EWC_FISHER_SAMPLES`,
`EWC_FISHER_BATCH_SIZE`, `EWC_FISHER_SEED`, and `RECOMPUTE_FISHER=1`.

## Inference

```bash
# Evaluate the QAdapt seq+EWC checkpoint on T0-T4 simulation tasks.
python code/examples/infer_t0_t4.py --gpus 0 --resume

# Evaluate the checkpoint on the fixed axis-mix OOD grid.
python code/examples/infer_ood.py --gpus 0,1 --parallelism 2 --resume

# Download and extract the smallest Google QEC surface-code archive.
python code/examples/download_google_benchmark.py --extract

# Evaluate downloaded Google data.
python code/examples/infer_google_benchmark.py --gpus 0 --resume
```

The three inference commands accept `--checkpoint`, `--num-samples`, `--basis`,
`--seed`, `--gpus`, `--parallelism`, `--output-dir`, `--resume`, and
`--dry-run`. QAdapt results are evaluated alongside a PyMatching-only reference
on the same samples.

OOD YAML files are generated at runtime under
`outputs/generated_configs/ood/`; only the five reusable T0-T4 configs under
`conf/examples/qadapt/` are source-controlled.
