#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)"
PYTHON_BIN="${PREDECODER_PYTHON:-/root/miniconda3/envs/ising-decoding/bin/python}"
OUTPUT_JSON="${ABLATION_OUTPUT_JSON:-${REPO_ROOT}/outputs/htnet_ablation/htnet_t0_three_factor_s1048576_seed12345.json}"
CELLS=(a000 a001 a010 a011 a100 a101 a110 a111)
MODEL_ARGS=()

select_best_checkpoint() {
  local cell="$1"
  local experiment_name="htnet_ablation_t0_${cell}_s4m_e100_seed12345_a100_skipup_v1"
  local checkpoint_dir="${REPO_ROOT}/outputs/${experiment_name}/models/best_model"
  local candidate
  local candidate_name
  local epoch
  local best_epoch=-1
  local best_path=""

  for candidate in "${checkpoint_dir}"/HTnet.0.*.pt; do
    [[ -e "${candidate}" ]] || continue
    candidate_name="${candidate##*/HTnet.0.}"
    epoch="${candidate_name%.pt}"
    [[ "${epoch}" =~ ^[0-9]+$ ]] || continue
    if (( epoch > best_epoch )); then
      best_epoch="${epoch}"
      best_path="${candidate}"
    fi
  done

  if [[ -z "${best_path}" ]]; then
    echo "[missing] no HTnet best checkpoint for ${cell}: ${checkpoint_dir}" >&2
    return 2
  fi
  echo "${best_path}"
}

cd "${REPO_ROOT}"
mkdir -p "$(dirname -- "${OUTPUT_JSON}")" logs

for cell in "${CELLS[@]}"; do
  checkpoint="$(select_best_checkpoint "${cell}")"
  config="${REPO_ROOT}/conf/examples/qadapt/htnet_ablation/config_htnet_ablation_${cell}.yaml"
  MODEL_ARGS+=(--model "${cell}:111:${checkpoint}:${config}")
done

PYTHONPATH="${REPO_ROOT}/code" "${PYTHON_BIN}" -u \
  "${REPO_ROOT}/code/scripts/paired_inference_compare.py" \
  --config-file "${REPO_ROOT}/conf/examples/qadapt/htnet_ablation/config_htnet_ablation_a000.yaml" \
  "${MODEL_ARGS[@]}" \
  --distance 9 \
  --n-rounds 9 \
  --num-samples 1048576 \
  --latency-num-samples 1048576 \
  --latency-repeats 5 \
  --batch-size 2048 \
  --num-workers 0 \
  --seed 12345 \
  --basis both \
  --device cuda:0 \
  --three-factor-cells a000,a001,a010,a011,a100,a101,a110,a111 \
  --output "${OUTPUT_JSON}"
