#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PREDECODER_PYTHON="${PREDECODER_PYTHON:-python3}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
GPUS="${GPUS:-1}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-ising_fast_t0_e100}"
DISTANCE="${DISTANCE:-9}"
N_ROUNDS="${N_ROUNDS:-9}"
EPOCHS="${EPOCHS:-100}"
FRESH_START="${FRESH_START:-1}"
DRY_RUN="${DRY_RUN:-0}"

if [ "${DRY_RUN}" = "1" ]; then
  echo "[dry-run] Train Ising-Fast (model_id=1) from scratch on T0 for ${EPOCHS} epochs"
  echo "[dry-run] output=outputs/${EXPERIMENT_NAME}/models/PreDecoderModelMemory_v1.0.${EPOCHS}.pt"
  echo "[dry-run] CONFIG_NAME=examples/qadapt/config_qadapt_t0_base EXPERIMENT_NAME=${EXPERIMENT_NAME} PREDECODER_TRAIN_EPOCHS=${EPOCHS} EXTRA_PARAMS=model_id=1 FRESH_START=${FRESH_START} bash code/scripts/local_run.sh ${DISTANCE} ${N_ROUNDS}"
  exit 0
fi

export PREDECODER_PYTHON CUDA_VISIBLE_DEVICES GPUS
CONFIG_NAME=examples/qadapt/config_qadapt_t0_base \
EXPERIMENT_NAME="${EXPERIMENT_NAME}" \
PREDECODER_TRAIN_EPOCHS="${EPOCHS}" \
PREDECODER_EWC_ENABLED=0 \
EXTRA_PARAMS="model_id=1${EXTRA_PARAMS:+ ${EXTRA_PARAMS}}" \
FRESH_START="${FRESH_START}" \
WORKFLOW=train \
  bash code/scripts/local_run.sh "${DISTANCE}" "${N_ROUNDS}"

echo "Final release candidate: outputs/${EXPERIMENT_NAME}/models/PreDecoderModelMemory_v1.0.${EPOCHS}.pt"
