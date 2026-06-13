#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Sequential + EWC training for ST-Fusion-R9-X over five domestic noise tasks.
#
# Default behavior:
#   - waits for the ST-Fusion-R9-X mixed-noise experiment to finish
#   - uses GPUs 0,1,2,3 after the wait target exits
#   - runs T0 -> T1 -> T2 -> T3 -> T4 in one experiment directory
#   - trains to cumulative epochs 20/40/60/80/100, so each task contributes 20 epochs
#   - computes diagonal Fisher after T0/T1/T2/T3 and loads all saved Fisher states later
#
# Example:
#   setsid nohup bash code/scripts/run_stfusion_r9_x_seq_ewc.sh \
#     > logs/log_ising_domestic_fast_opt_stfusion_r9_x_seq_ewc_queue.log 2>&1 < /dev/null &
#
# Useful overrides:
#   CUDA_VISIBLE_DEVICES=4,5,6,7 bash code/scripts/run_stfusion_r9_x_seq_ewc.sh
#   WAIT_FOR_COMPLETION=0 bash code/scripts/run_stfusion_r9_x_seq_ewc.sh
#   EWC_LAMBDA=50 EWC_FISHER_SAMPLES=32768 bash code/scripts/run_stfusion_r9_x_seq_ewc.sh

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
EXPERIMENT_NAME="${EXPERIMENT_NAME:-ising_domestic_fast_opt_stfusion_r9_x_seq_ewc}"
DISTANCE="${DISTANCE:-9}"
N_ROUNDS="${N_ROUNDS:-9}"
EWC_DIR="${EWC_DIR:-outputs/${EXPERIMENT_NAME}/ewc}"
EWC_LAMBDA="${EWC_LAMBDA:-100}"
EWC_FISHER_SAMPLES="${EWC_FISHER_SAMPLES:-65536}"
EWC_FISHER_BATCH_SIZE="${EWC_FISHER_BATCH_SIZE:-2048}"
EWC_FISHER_SEED="${EWC_FISHER_SEED:-12345}"
FISHER_CUDA_VISIBLE_DEVICES="${FISHER_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES%%,*}}"
WAIT_FOR_COMPLETION="${WAIT_FOR_COMPLETION:-1}"
WAIT_FOR_EXPERIMENT="${WAIT_FOR_EXPERIMENT:-ising_domestic_fast_opt_stfusion_r9_x_mixed_noise}"
WAIT_POLL_SECONDS="${WAIT_POLL_SECONDS:-3600}"
RECOMPUTE_FISHER="${RECOMPUTE_FISHER:-0}"

configs=(
  config_domestic_fast_opt_stfusion_r9_x_seq_ewc_t0_base
  config_domestic_fast_opt_stfusion_r9_x_seq_ewc_t1_meas_1p5
  config_domestic_fast_opt_stfusion_r9_x_seq_ewc_t2_cnot_1p5
  config_domestic_fast_opt_stfusion_r9_x_seq_ewc_t3_idle_1p5
  config_domestic_fast_opt_stfusion_r9_x_seq_ewc_t4_z_bias_1p5
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

is_wait_target_running() {
  ps -eo pid,args \
    | grep -F "${WAIT_FOR_EXPERIMENT}" \
    | grep -v grep \
    | grep -v "run_stfusion_r9_x_seq_ewc.sh" \
    >/dev/null
}

select_latest_model_checkpoint() {
  local model_dir="$1"
  PYTHONPATH="${REPO_ROOT}/code:${PYTHONPATH:-}" "${PREDECODER_PYTHON}" - "$model_dir" <<'PY'
from pathlib import Path
import sys
from scripts.compute_ewc_fisher import select_model_checkpoint
print(select_model_checkpoint(Path(sys.argv[1])))
PY
}

compute_stage_fisher() {
  local idx="$1"
  local task_name="${tasks[$idx]}"
  local config_name="${configs[$idx]}"
  local model_dir="outputs/${EXPERIMENT_NAME}/models"
  local checkpoint
  checkpoint="$(select_latest_model_checkpoint "${model_dir}")"
  local fisher_path="${EWC_DIR}/task_$(printf '%03d' "${idx}")_${task_name}.pt"

  if [ -f "${fisher_path}" ] && [ "${RECOMPUTE_FISHER}" != "1" ]; then
    echo "[seq-ewc] $(date -u +%F_%T) fisher exists, skip: ${fisher_path}"
    return 0
  fi

  mkdir -p "${EWC_DIR}"
  echo "[seq-ewc] $(date -u +%F_%T) fisher start: task=${task_name} checkpoint=${checkpoint} output=${fisher_path}"
  CUDA_VISIBLE_DEVICES="${FISHER_CUDA_VISIBLE_DEVICES}" \
  PYTHONPATH="${REPO_ROOT}/code:${PYTHONPATH:-}" \
  "${PREDECODER_PYTHON}" -u code/scripts/compute_ewc_fisher.py \
    --config-name "${config_name}" \
    --checkpoint "${checkpoint}" \
    --output "${fisher_path}" \
    --task-name "${task_name}" \
    --num-samples "${EWC_FISHER_SAMPLES}" \
    --batch-size "${EWC_FISHER_BATCH_SIZE}" \
    --seed "${EWC_FISHER_SEED}" \
    --device cuda:0
  echo "[seq-ewc] $(date -u +%F_%T) fisher complete: ${fisher_path}"
}

echo "[seq-ewc] $(date -u +%F_%T) starting ST-Fusion-R9-X sequential + EWC"
echo "[seq-ewc] repo=${REPO_ROOT}"
echo "[seq-ewc] experiment=${EXPERIMENT_NAME}"
echo "[seq-ewc] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} GPUS=${GPUS}"
echo "[seq-ewc] fisher CUDA_VISIBLE_DEVICES=${FISHER_CUDA_VISIBLE_DEVICES}"
echo "[seq-ewc] ewc_dir=${EWC_DIR} lambda=${EWC_LAMBDA} fisher_samples=${EWC_FISHER_SAMPLES}"
echo "[seq-ewc] distance=${DISTANCE} n_rounds=${N_ROUNDS}"

if [ "${WAIT_FOR_COMPLETION}" = "1" ]; then
  while is_wait_target_running; do
    echo "[seq-ewc] $(date -u +%F_%T) waiting for ${WAIT_FOR_EXPERIMENT}; next check in ${WAIT_POLL_SECONDS}s"
    sleep "${WAIT_POLL_SECONDS}"
  done
  echo "[seq-ewc] $(date -u +%F_%T) wait target is not running; starting sequential + EWC"
fi

for i in "${!configs[@]}"; do
  ewc_enabled=0
  if [ "${i}" -gt 0 ]; then
    ewc_enabled=1
  fi

  echo "[seq-ewc] $(date -u +%F_%T) stage ${tasks[$i]} start: config=${configs[$i]} target_epochs=${target_epochs[$i]} fresh=${fresh_start[$i]} ewc=${ewc_enabled}"

  PREDECODER_EWC_ENABLED="${ewc_enabled}" \
  PREDECODER_EWC_DIR="${EWC_DIR}" \
  PREDECODER_EWC_LAMBDA="${EWC_LAMBDA}" \
  PREDECODER_TRAIN_EPOCHS="${target_epochs[$i]}" \
  PREDECODER_PYTHON="${PREDECODER_PYTHON}" \
  CONFIG_NAME="${configs[$i]}" \
  WORKFLOW=train \
  EXPERIMENT_NAME="${EXPERIMENT_NAME}" \
  FRESH_START="${fresh_start[$i]}" \
  bash code/scripts/local_run.sh "${DISTANCE}" "${N_ROUNDS}"

  echo "[seq-ewc] $(date -u +%F_%T) stage ${tasks[$i]} complete"

  if [ "${i}" -lt 4 ]; then
    compute_stage_fisher "${i}"
  fi
done

echo "[seq-ewc] $(date -u +%F_%T) sequential + EWC complete"
