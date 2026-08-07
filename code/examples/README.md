# QAdapt and Ising-Fast-T0-E100 release workflows

These entry points cover the two released surface-code pre-decoders. Run every
command from the repository root.

| Release name | Model ID | Training recipe | Final training checkpoint |
|---|---:|---|---|
| `qadapt` | 111 (`HTnet`) | T0→T4 sequential training, 20 epochs per task, EWC after T0–T3 | `HTnet.0.100.pt` |
| `ising_fast_t0_e100` | 1 (`PreDecoderModelMemory_v1`) | Random initialization, T0 only, 100 epochs, no EWC | `PreDecoderModelMemory_v1.0.100.pt` |

Training directories contain resume checkpoints and optimizer state. They are
local run artifacts, not release assets. Publish only the two final weight
files, together with each repository's model card, license, notice, checksum,
and machine-readable config produced below.

## Training

```bash
# QAdapt: T0 -> T4 sequential training with Fisher snapshots after T0-T3.
bash code/examples/train_seq_ewc.sh

# Ising-Fast: random initialization, T0 only, exactly 100 epochs by default.
bash code/examples/train_ising_fast_t0.sh
```

Both launchers accept `PREDECODER_PYTHON`, `CUDA_VISIBLE_DEVICES`, `GPUS`,
`EXPERIMENT_NAME`, `DISTANCE`, `N_ROUNDS`, and `DRY_RUN=1`. QAdapt additionally
accepts `EPOCHS_PER_TASK`, `EWC_LAMBDA`, `EWC_FISHER_SAMPLES`,
`EWC_FISHER_BATCH_SIZE`, `EWC_FISHER_SEED`, and `RECOMPUTE_FISHER=1`.

`train_ising_fast_t0.sh` starts from random weights. Set `FRESH_START=0` only
when resuming the same output directory. It does not fine-tune NVIDIA's
published Ising-Fast checkpoint.

## Export the two final release files

```bash
QADAPT_CHECKPOINT=/path/to/HTnet.0.100.pt \
ISING_FAST_T0_CHECKPOINT=/path/to/PreDecoderModelMemory_v1.0.100.pt \
OUTPUT_DIR=release_models \
  bash code/examples/export_release_models.sh
```

This creates two standalone Hugging Face staging directories:

```text
release_models/qadapt/
  README.md  LICENSE  NOTICE  config.json  evaluation.json
  .gitattributes  SHA256SUMS
  qadapt.safetensors
release_models/ising-fast-t0-e100/
  README.md  LICENSE  NOTICE  config.json  evaluation.json
  .gitattributes  SHA256SUMS
  ising-fast-t0-e100.safetensors
```

The exporter records SHA-256 hashes. It exports fp32 by default; set
`EXPORT_FP16=1` only if the fp16 artifacts have been re-evaluated. Do not upload
the training directory, optimizer checkpoints, Fisher snapshots, logs, or
`best_model/` history to Hugging Face. Before upload, replace
`REPLACE_WITH_YOUR_RELEASE_TAG` in both `config.json` files and model cards.

## Downloaded model layout

The two Hugging Face repository IDs are intentionally not hard-coded here.
After creating them, download each final file into any local directory:

```bash
hf download <hf-org>/QAdapt qadapt.safetensors --local-dir models/qadapt
hf download <hf-org>/Ising-Fast-T0-E100 ising-fast-t0-e100.safetensors \
  --local-dir models/ising-fast-t0-e100
```

Use one repeatable `--model name:model_id:path` argument per model. Both `.pt`
and `.safetensors` are accepted.

## Which inference examples are public

The paper and the model release have different evidence scopes:

| Entry point | Release status | Purpose |
|---|---|---|
| `infer_ood.py` | Required | Reproduce the paper's selected 110-configuration synthetic OOD protocol. |
| `infer_willow.py` | Required | Reproduce the paper's zero-shot Willow protocol at d=5/7 and ten rounds. |
| `download_google_benchmark.py` | Required helper | Acquire and verify the public Willow data used by `infer_willow.py`. |
| `infer_t0_t4.py` | Required for the model release | Evaluate the final released checkpoints on T0--T4; it can also select only T0 at d=7/9. |
| `infer_google_benchmark.py` | Optional compatibility entry point | Run arbitrary Google QEC distances and round counts; it is not the paper-default launcher. |

The paper's T0 architecture table compares T0-only best checkpoints
(`Ising-Fast` epoch 53 and `HTNet` epoch 89). Those are not the two final
release checkpoints documented here. Running the released QAdapt
sequential+EWC checkpoint on T0 is useful release evaluation, but it does not
reproduce the paper's T0 architecture table and must not be labelled as such.

## Simulation and OOD inference

```bash
MODELS=(
  --model qadapt:111:models/qadapt/qadapt.safetensors
  --model ising_fast_t0_e100:1:models/ising-fast-t0-e100/ising-fast-t0-e100.safetensors
)

# Evaluate both models and PyMatching on T0-T4.
python code/examples/infer_t0_t4.py "${MODELS[@]}" --gpus 0 --resume

# Select the paper's T0 geometries (release weights, not the architecture-only
# checkpoints used in the paper's T0 table).
python code/examples/infer_t0_t4.py "${MODELS[@]}" \
  --tasks T0 --distances 7,9 --gpus 0 --resume

# Reproduce the paper's 11 x 5 x 2 = 110 selected OOD configurations.
python code/examples/infer_ood.py "${MODELS[@]}" \
  --gpus 0,1 --parallelism 2 --resume
```

OOD YAML files are generated under `outputs/generated_configs/ood/`. Result
JSON/CSV files are written under `outputs/examples/released_models/`; every
model and the PyMatching-only baseline consume the same samples. The paper
defaults are distances 7 and 9, nine rounds, and multipliers
1.2/1.5/2.0/2.5/3.0. Override `--distances` or `--multipliers` only for an
explicitly labelled extended evaluation.

The earlier technical report's broader 297-configuration grid is still
available as an explicitly labelled extension:

```bash
python code/examples/infer_ood.py "${MODELS[@]}" \
  --distances 5,7,9 \
  --multipliers 1.2,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0 \
  --output-dir outputs/examples/released_models/ood_technical_report \
  --gpus 0,1 --parallelism 2 --resume
```

## Google Willow 105Q inference

```bash
# Download and extract the 105-qubit surface-code archive (~5.32 GiB).
python code/examples/download_google_benchmark.py --extract

# Evaluate both released models on Willow hardware samples.
python code/examples/infer_willow.py "${MODELS[@]}" \
  --num-samples 0 --gpus 0 --resume
```

`infer_willow.py` defaults to the paper protocol: d=5/7, X/Z bases, ten rounds,
all available shots (400,000 pooled shots at d=5 and 100,000 at d=7). Passing
`--num-samples 0` makes the all-shot choice explicit. It uses the first selected
GPU; model-level parallelism is not used because all models are evaluated on
each identical hardware case. `infer_google_benchmark.py` remains an optional
general entry point and retains its d=3/5/7, r13 calibration defaults.

The inference launchers also accept `--num-samples`, `--basis`, `--batch-size`,
`--gpus`, `--output-dir`, `--resume`, and `--dry-run`. Simulation launchers
additionally accept `--seed`, `--num-workers`, and `--parallelism`.
