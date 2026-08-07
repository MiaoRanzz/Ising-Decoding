#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES
if [ -z "${GPUS:-}" ]; then
  GPUS="$(awk -F, '{print NF}' <<<"${CUDA_VISIBLE_DEVICES}")"
fi
export GPUS

PREDECODER_PYTHON="${PREDECODER_PYTHON:-python3}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qadapt_seq_replay}"
DISTANCE="${DISTANCE:-9}"
N_ROUNDS="${N_ROUNDS:-9}"
EPOCHS_PER_TASK="${EPOCHS_PER_TASK:-20}"
REPLAY_DIR="${PREDECODER_REPLAY_DIR:-outputs/${EXPERIMENT_NAME}/replay}"
REPLAY_CAPACITY="${PREDECODER_REPLAY_CAPACITY:-65536}"
REPLAY_RATIO="${PREDECODER_REPLAY_RATIO:-0.5}"
REPLAY_LAMBDA="${PREDECODER_REPLAY_LAMBDA:-1.0}"
REPLAY_SEED="${PREDECODER_REPLAY_SEED:-12345}"
DRY_RUN="${DRY_RUN:-0}"

configs=(
  examples/qadapt/config_qadapt_t0_base
  examples/qadapt/config_qadapt_t1_meas_1p5
  examples/qadapt/config_qadapt_t2_cnot_1p5
  examples/qadapt/config_qadapt_t3_idle_1p5
  examples/qadapt/config_qadapt_t4_z_bias_1p5
)
tasks=(T0_base T1_meas_1p5 T2_cnot_1p5 T3_idle_1p5 T4_z_bias_1p5)

for i in "${!configs[@]}"; do
  target_epochs="$(( (i + 1) * EPOCHS_PER_TASK ))"
  fresh_start=0
  if [ "${i}" -eq 0 ]; then fresh_start=1; fi

  echo "[seq+replay] stage=${tasks[$i]} config=${configs[$i]} target_epochs=${target_epochs} replay_dir=${REPLAY_DIR}"
  if [ "${DRY_RUN}" = "1" ]; then
    echo "[dry-run] WORKFLOW=train EXPERIMENT_NAME=${EXPERIMENT_NAME} PREDECODER_REPLAY_ENABLED=1 PREDECODER_REPLAY_TASK_ID=${tasks[$i]} PREDECODER_REPLAY_DIR=${REPLAY_DIR} PREDECODER_REPLAY_CAPACITY=${REPLAY_CAPACITY} PREDECODER_REPLAY_RATIO=${REPLAY_RATIO} PREDECODER_REPLAY_LAMBDA=${REPLAY_LAMBDA} PREDECODER_TRAIN_EPOCHS=${target_epochs} FRESH_START=${fresh_start} bash code/scripts/local_run.sh ${DISTANCE} ${N_ROUNDS}"
  else
    PREDECODER_EWC_ENABLED=0 \
    PREDECODER_REPLAY_ENABLED=1 \
    PREDECODER_REPLAY_TASK_ID="${tasks[$i]}" \
    PREDECODER_REPLAY_DIR="${REPLAY_DIR}" \
    PREDECODER_REPLAY_CAPACITY="${REPLAY_CAPACITY}" \
    PREDECODER_REPLAY_RATIO="${REPLAY_RATIO}" \
    PREDECODER_REPLAY_LAMBDA="${REPLAY_LAMBDA}" \
    PREDECODER_REPLAY_SEED="${REPLAY_SEED}" \
    PREDECODER_TRAIN_EPOCHS="${target_epochs}" \
    PREDECODER_PYTHON="${PREDECODER_PYTHON}" \
    CONFIG_NAME="${configs[$i]}" \
    WORKFLOW=train \
    EXPERIMENT_NAME="${EXPERIMENT_NAME}" \
    FRESH_START="${fresh_start}" \
      bash code/scripts/local_run.sh "${DISTANCE}" "${N_ROUNDS}"
  fi
done
