#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Sequential no-EWC training for Ising fast R9 over the five domestic noise tasks.
#
# Default behavior:
#   - uses GPUs 0,1,2,3
#   - runs T0 -> T1 -> T2 -> T3 -> T4 in one experiment directory
#   - trains to cumulative epochs 20/40/60/80/100, so each task contributes 20 epochs
#
# Example:
#   nohup bash code/scripts/run_ising_fast_seq_noewc.sh \
#     > logs/log_ising_domestic_fast_seq_noewc_r9.log 2>&1 &
#
# Useful overrides:
#   CUDA_VISIBLE_DEVICES=4,5,6,7 bash code/scripts/run_ising_fast_seq_noewc.sh
#   EXPERIMENT_NAME=my_ising_seq_noewc bash code/scripts/run_ising_fast_seq_noewc.sh
#   PREDECODER_PYTHON=/path/to/python bash code/scripts/run_ising_fast_seq_noewc.sh
#   DRY_RUN=1 bash code/scripts/run_ising_fast_seq_noewc.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export CUDA_VISIBLE_DEVICES

if [ -z "${GPUS:-}" ]; then
  GPUS="$(python3 - <<'PY'
import os
visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
print(len([x for x in visible.split(",") if x.strip()]) or 1)
PY
)"
fi
export GPUS

PREDECODER_PYTHON="${PREDECODER_PYTHON:-/root/miniconda3/envs/ising-decoding/bin/python}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-ising_domestic_fast_seq_noewc_r9}"
DISTANCE="${DISTANCE:-9}"
N_ROUNDS="${N_ROUNDS:-9}"
DRY_RUN="${DRY_RUN:-0}"

configs=(
  config_domestic_ising_fast_seq_noewc_t0_base
  config_domestic_ising_fast_seq_noewc_t1_meas_1p5
  config_domestic_ising_fast_seq_noewc_t2_cnot_1p5
  config_domestic_ising_fast_seq_noewc_t3_idle_1p5
  config_domestic_ising_fast_seq_noewc_t4_z_bias_1p5
)

tasks=(
  T0_base
  T1_meas_1p5
  T2_cnot_1p5
  T3_idle_1p5
  T4_z_bias_1p5
)

target_epochs=(20 40 60 80 100)
fresh_start=(1 0 0 0 0)

echo "[ising-seq-noewc] $(date -u +%F_%T) starting Ising fast sequential no-EWC"
echo "[ising-seq-noewc] repo=${REPO_ROOT}"
echo "[ising-seq-noewc] experiment=${EXPERIMENT_NAME}"
echo "[ising-seq-noewc] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} GPUS=${GPUS}"
echo "[ising-seq-noewc] distance=${DISTANCE} n_rounds=${N_ROUNDS}"
echo "[ising-seq-noewc] dry_run=${DRY_RUN}"

for i in "${!configs[@]}"; do
  echo "[ising-seq-noewc] $(date -u +%F_%T) stage ${tasks[$i]} start: config=${configs[$i]} target_epochs=${target_epochs[$i]} fresh=${fresh_start[$i]}"

  if [ "${DRY_RUN}" = "1" ]; then
    echo "[ising-seq-noewc] dry-run skip local_run: config=${configs[$i]} target_epochs=${target_epochs[$i]} fresh=${fresh_start[$i]}"
    continue
  fi

  PREDECODER_TRAIN_EPOCHS="${target_epochs[$i]}" \
  PREDECODER_PYTHON="${PREDECODER_PYTHON}" \
  CONFIG_NAME="${configs[$i]}" \
  WORKFLOW=train \
  EXPERIMENT_NAME="${EXPERIMENT_NAME}" \
  FRESH_START="${fresh_start[$i]}" \
  bash code/scripts/local_run.sh "${DISTANCE}" "${N_ROUNDS}"

  echo "[ising-seq-noewc] $(date -u +%F_%T) stage ${tasks[$i]} complete"
done

echo "[ising-seq-noewc] $(date -u +%F_%T) sequential no-EWC complete"
