#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES
if [ -z "${GPUS:-}" ]; then
  GPUS="$(awk -F, '{print NF}' <<<"${CUDA_VISIBLE_DEVICES}")"
fi
export GPUS

PREDECODER_PYTHON="${PREDECODER_PYTHON:-python3}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qadapt_seq_ewc}"
DISTANCE="${DISTANCE:-9}"
N_ROUNDS="${N_ROUNDS:-9}"
EPOCHS_PER_TASK="${EPOCHS_PER_TASK:-20}"
EWC_DIR="${EWC_DIR:-outputs/${EXPERIMENT_NAME}/ewc}"
EWC_LAMBDA="${EWC_LAMBDA:-100}"
EWC_FISHER_SAMPLES="${EWC_FISHER_SAMPLES:-65536}"
EWC_FISHER_BATCH_SIZE="${EWC_FISHER_BATCH_SIZE:-2048}"
EWC_FISHER_SEED="${EWC_FISHER_SEED:-12345}"
FISHER_CUDA_VISIBLE_DEVICES="${FISHER_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES%%,*}}"
RECOMPUTE_FISHER="${RECOMPUTE_FISHER:-0}"
DRY_RUN="${DRY_RUN:-0}"

configs=(
  examples/qadapt/config_qadapt_t0_base
  examples/qadapt/config_qadapt_t1_meas_1p5
  examples/qadapt/config_qadapt_t2_cnot_1p5
  examples/qadapt/config_qadapt_t3_idle_1p5
  examples/qadapt/config_qadapt_t4_z_bias_1p5
)
tasks=(T0_base T1_meas_1p5 T2_cnot_1p5 T3_idle_1p5 T4_z_bias_1p5)

select_latest_checkpoint() {
  PYTHONPATH="${REPO_ROOT}/code:${PYTHONPATH:-}" "${PREDECODER_PYTHON}" -m scripts.compute_ewc_fisher \
    --help >/dev/null
  PYTHONPATH="${REPO_ROOT}/code:${PYTHONPATH:-}" "${PREDECODER_PYTHON}" -c \
    'from pathlib import Path; from scripts.compute_ewc_fisher import select_model_checkpoint; import sys; print(select_model_checkpoint(Path(sys.argv[1])))' "$1"
}

compute_fisher() {
  local index="$1"
  local task_name="${tasks[$index]}"
  local fisher_path="${EWC_DIR}/task_$(printf '%03d' "${index}")_${task_name}.pt"
  if [ "${DRY_RUN}" = "1" ]; then
    echo "[dry-run] fisher task=${task_name} config=${configs[$index]} output=${fisher_path} samples=${EWC_FISHER_SAMPLES} batch_size=${EWC_FISHER_BATCH_SIZE} seed=${EWC_FISHER_SEED} recompute=${RECOMPUTE_FISHER} cuda_visible_devices=${FISHER_CUDA_VISIBLE_DEVICES}"
    return
  fi
  if [ -f "${fisher_path}" ] && [ "${RECOMPUTE_FISHER}" != "1" ]; then
    echo "[seq+ewc] fisher exists, skip: ${fisher_path}"
    return
  fi
  mkdir -p "${EWC_DIR}"
  local checkpoint
  checkpoint="$(select_latest_checkpoint "outputs/${EXPERIMENT_NAME}/models")"
  CUDA_VISIBLE_DEVICES="${FISHER_CUDA_VISIBLE_DEVICES}" \
  PYTHONPATH="${REPO_ROOT}/code:${PYTHONPATH:-}" \
  "${PREDECODER_PYTHON}" -u code/scripts/compute_ewc_fisher.py \
    --config-name "${configs[$index]}" \
    --checkpoint "${checkpoint}" \
    --output "${fisher_path}" \
    --task-name "${task_name}" \
    --num-samples "${EWC_FISHER_SAMPLES}" \
    --batch-size "${EWC_FISHER_BATCH_SIZE}" \
    --seed "${EWC_FISHER_SEED}" \
    --device cuda:0
}

for i in "${!configs[@]}"; do
  target_epochs="$(( (i + 1) * EPOCHS_PER_TASK ))"
  fresh_start=0
  ewc_enabled=1
  if [ "${i}" -eq 0 ]; then
    fresh_start=1
    ewc_enabled=0
  fi
  echo "[seq+ewc] stage=${tasks[$i]} config=${configs[$i]} target_epochs=${target_epochs} ewc=${ewc_enabled}"
  if [ "${DRY_RUN}" = "1" ]; then
    echo "[dry-run] WORKFLOW=train EXPERIMENT_NAME=${EXPERIMENT_NAME} PREDECODER_EWC_ENABLED=${ewc_enabled} PREDECODER_EWC_DIR=${EWC_DIR} PREDECODER_EWC_LAMBDA=${EWC_LAMBDA} PREDECODER_TRAIN_EPOCHS=${target_epochs} FRESH_START=${fresh_start} bash code/scripts/local_run.sh ${DISTANCE} ${N_ROUNDS}"
  else
    PREDECODER_EWC_ENABLED="${ewc_enabled}" \
    PREDECODER_EWC_DIR="${EWC_DIR}" \
    PREDECODER_EWC_LAMBDA="${EWC_LAMBDA}" \
    PREDECODER_TRAIN_EPOCHS="${target_epochs}" \
    PREDECODER_PYTHON="${PREDECODER_PYTHON}" \
    CONFIG_NAME="${configs[$i]}" \
    WORKFLOW=train \
    EXPERIMENT_NAME="${EXPERIMENT_NAME}" \
    FRESH_START="${fresh_start}" \
      bash code/scripts/local_run.sh "${DISTANCE}" "${N_ROUNDS}"
  fi
  if [ "${i}" -lt 4 ]; then compute_fisher "${i}"; fi
done
