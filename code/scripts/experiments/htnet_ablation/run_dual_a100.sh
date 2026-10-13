#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)"
PYTHON_BIN="${PREDECODER_PYTHON:-/root/miniconda3/envs/ising-decoding/bin/python}"
FRESH_START="${FRESH_START:-1}"

GPU0_CELLS=(a000 a011 a101 a110)
GPU1_CELLS=(a001 a010 a100 a111)

run_cell() {
  local gpu="$1"
  local cell="$2"
  local config_name="examples/qadapt/htnet_ablation/config_htnet_ablation_${cell}"
  local experiment_name="htnet_ablation_t0_${cell}_s4m_e100_seed12345_a100_skipup_v1"
  local output_dir="${REPO_ROOT}/outputs/${experiment_name}"

  if [[ "${FRESH_START}" == "1" && -d "${output_dir}/models" ]]; then
    echo "[refuse] ${experiment_name}: existing models directory; choose a new name or resume with FRESH_START=0" >&2
    return 2
  fi

  echo "[start] gpu=${gpu} cell=${cell} experiment=${experiment_name}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
  GPUS=1 \
  PREDECODER_PYTHON="${PYTHON_BIN}" \
  CONFIG_NAME="${config_name}" \
  EXPERIMENT_NAME="${experiment_name}" \
  PREDECODER_SEED=12345 \
  PREDECODER_TRAIN_EPOCHS=100 \
  PREDECODER_TRAIN_SAMPLES=4194304 \
  PREDECODER_VAL_SAMPLES=65536 \
  PREDECODER_TEST_SAMPLES=262144 \
  PREDECODER_VALIDATION_LER=1 \
  PREDECODER_DISABLE_SDR=0 \
  PREDECODER_EWC_ENABLED=0 \
  PREDECODER_BATCH_SIZE=2048 \
  PREDECODER_ACCUMULATE_STEPS=2 \
  FRESH_START="${FRESH_START}" \
  bash "${REPO_ROOT}/code/scripts/local_run.sh"
}

run_queue() {
  local gpu="$1"
  shift
  local cell
  for cell in "$@"; do
    run_cell "${gpu}" "${cell}"
  done
}

cd "${REPO_ROOT}"
mkdir -p logs
run_queue 0 "${GPU0_CELLS[@]}" &
pid0=$!
run_queue 1 "${GPU1_CELLS[@]}" &
pid1=$!

set +e
wait "${pid0}"
status0=$?
wait "${pid1}"
status1=$?
set -e

if [[ "${status0}" -ne 0 || "${status1}" -ne 0 ]]; then
  echo "[failed] gpu0_status=${status0} gpu1_status=${status1}" >&2
  exit 1
fi

echo "[done] all eight HTNet ablation cells completed"
