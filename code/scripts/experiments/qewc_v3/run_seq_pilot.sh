#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_EXE="${PREDECODER_PYTHON:-/root/miniconda3/envs/ising-decoding/bin/python}"
STAGE0="${STAGE0_CHECKPOINT:-outputs/htnet_ablation_t0_a101_s4m_e100_seed12345_a100_skipup_v1/models/best_model/HTnet.0.97.pt}"
RUN_NAME="${RUN_NAME:-qewc_v3/pilot_realistic_seq_t1_1k}"
STEPS="${STEPS_PER_TASK:-1000}"
BATCH_SIZE="${BATCH_SIZE:-512}"
VAL_SAMPLES="${VAL_SAMPLES:-8192}"
SEED="${CONTINUAL_SEED:-20260907}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"
DRY_RUN="${DRY_RUN:-0}"

if [ ! -x "${PYTHON_EXE}" ]; then
  echo "Python executable is missing or not executable: ${PYTHON_EXE}" >&2
  exit 2
fi
if [ ! -f "${STAGE0}" ]; then
  echo "Stage-0 checkpoint is missing: ${STAGE0}" >&2
  exit 2
fi
if [ "${STEPS}" -le 0 ] || [ "${BATCH_SIZE}" -le 0 ]; then
  echo "STEPS_PER_TASK and BATCH_SIZE must be positive integers" >&2
  exit 2
fi

TRAIN_SAMPLES="$((STEPS * BATCH_SIZE))"
OUTPUT_DIR="outputs/${RUN_NAME}"
METADATA_DIR="${OUTPUT_DIR}/metadata"
mkdir -p "${METADATA_DIR}"

CONFIG_NAME="experiments/qewc_v3/realistic/t1_meas_1p5"
CONFIG_FILE="conf/${CONFIG_NAME}.yaml"
{
  echo "kind=qewc_v3_seq_pilot"
  echo "track=realistic"
  echo "stage=T1"
  echo "method=SEQ"
  echo "seed=${SEED}"
  echo "optimizer_steps=${STEPS}"
  echo "batch_size=${BATCH_SIZE}"
  echo "train_samples=${TRAIN_SAMPLES}"
  echo "world_size=1"
  echo "stage0_checkpoint=${STAGE0}"
  echo "stage0_sha256=$(sha256sum "${STAGE0}" | awk '{print $1}')"
  echo "config=${CONFIG_FILE}"
  echo "config_sha256=$(sha256sum "${CONFIG_FILE}" | awk '{print $1}')"
  echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "${METADATA_DIR}/launch.env"

command=(bash code/scripts/local_run.sh 9 9)
echo "[qewc-v3] run=${RUN_NAME} stage=T1 method=SEQ steps=${STEPS} seed=${SEED}"
if [ "${DRY_RUN}" = "1" ]; then
  echo "[qewc-v3] dry-run: TRAIN_SAMPLES=${TRAIN_SAMPLES} BATCH_SIZE=${BATCH_SIZE} VAL_SAMPLES=${VAL_SAMPLES}"
  exit 0
fi

CUDA_VISIBLE_DEVICES="${GPU}" \
GPUS=1 \
PREDECODER_PYTHON="${PYTHON_EXE}" \
PREDECODER_SEED="${SEED}" \
PREDECODER_TRAIN_EPOCHS=1 \
PREDECODER_TRAIN_SAMPLES="${TRAIN_SAMPLES}" \
PREDECODER_BATCH_SIZE="${BATCH_SIZE}" \
PREDECODER_ACCUMULATE_STEPS=1 \
PREDECODER_VAL_SAMPLES="${VAL_SAMPLES}" \
PREDECODER_VALIDATION_LER=0 \
PREDECODER_EWC_ENABLED=0 \
PREDECODER_REPLAY_ENABLED=0 \
CONFIG_NAME="${CONFIG_NAME}" \
WORKFLOW=train \
EXPERIMENT_NAME="${RUN_NAME}" \
FRESH_START=1 \
EXTRA_PARAMS="++init_model_checkpoint_file=${STAGE0}" \
  "${command[@]}"

ENDPOINT="${OUTPUT_DIR}/models/HTnet.0.1.pt"
if [ ! -f "${ENDPOINT}" ]; then
  echo "Expected fixed endpoint was not created: ${ENDPOINT}" >&2
  exit 3
fi
{
  echo "endpoint=${ENDPOINT}"
  echo "endpoint_sha256=$(sha256sum "${ENDPOINT}" | awk '{print $1}')"
  echo "completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} >> "${METADATA_DIR}/launch.env"
echo "[qewc-v3] completed endpoint=${ENDPOINT}"
