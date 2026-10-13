#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/code${PYTHONPATH:+:${PYTHONPATH}}"

PREDECODER_PYTHON="${PREDECODER_PYTHON:-/root/miniconda3/envs/ising-decoding/bin/python}"
PATENT_GPUS="${PATENT_GPUS:-0,1,2}"
PATENT_CPU_WORKERS="${PATENT_CPU_WORKERS:-30}"
PATENT_LATENCY_CPU="${PATENT_LATENCY_CPU:-27}"
PATENT_MAX_HOURS="${PATENT_MAX_HOURS:-8}"
PATENT_MODE="${PATENT_MODE:-full}"
PATENT_RESUME="${PATENT_RESUME:-1}"
PATENT_DRY_RUN="${PATENT_DRY_RUN:-0}"
PATENT_OUTPUT="${PATENT_OUTPUT:-${REPO_ROOT}/outputs/patent_validation/topology_gating_v3_latency}"
PATENT_CONFIG="${PATENT_CONFIG:-${REPO_ROOT}/conf/experiments/patent/topology_gating_v3_latency.yaml}"
PATENT_START_STAGE="${PATENT_START_STAGE:-screen}"
PATENT_CHECKPOINT="${PATENT_CHECKPOINT:-/mnt/public/miaoran/Ising-Decoding-storage/outputs/ising_domestic_fast/models/best_model/PreDecoderModelMemory_v1.0.53.pt}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TORCH_NUM_THREADS=1

if [[ "${PATENT_START_STAGE}" != "screen" && "${PATENT_START_STAGE}" != "freeze" && "${PATENT_START_STAGE}" != "test" && "${PATENT_START_STAGE}" != "latency" && "${PATENT_START_STAGE}" != "aggregate" ]]; then
  echo "PATENT_START_STAGE must be screen, freeze, test, latency, or aggregate" >&2
  exit 2
fi
if [[ "${PATENT_MODE}" != "full" && "${PATENT_MODE}" != "smoke" ]]; then
  echo "PATENT_MODE must be full or smoke" >&2
  exit 2
fi
if [[ ! "${PATENT_MAX_HOURS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "PATENT_MAX_HOURS must be a positive integer" >&2
  exit 2
fi
if [[ ! "${PATENT_CPU_WORKERS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "PATENT_CPU_WORKERS must be a positive integer" >&2
  exit 2
fi
if [[ ! "${PATENT_LATENCY_CPU}" =~ ^[0-9]+$ ]]; then
  echo "PATENT_LATENCY_CPU must be a non-negative integer" >&2
  exit 2
fi

IFS=',' read -r -a GPU_IDS <<< "${PATENT_GPUS}"
if [[ "${#GPU_IDS[@]}" -lt 1 || "${#GPU_IDS[@]}" -gt 3 ]]; then
  echo "PATENT_GPUS must contain one to three GPU ids" >&2
  exit 2
fi

SEEDS=(20260811 20260812 20260813)
if [[ "${PATENT_MODE}" == "smoke" ]]; then
  SEEDS=(20260811)
fi
MODULE="benchmarks.patent_validation.topology_gating_v3_latency"
BASE_ARGS=(
  --config "${PATENT_CONFIG}"
  --output "${PATENT_OUTPUT}"
  --checkpoint "${PATENT_CHECKPOINT}"
  --mode "${PATENT_MODE}"
)
if [[ "${PATENT_RESUME}" == "1" ]]; then
  BASE_ARGS+=(--resume)
fi

echo "S07 v3 batch-throughput-aligned pure-inference validation"
echo "  mode=${PATENT_MODE}"
echo "  checkpoint=${PATENT_CHECKPOINT}"
echo "  gpus=${PATENT_GPUS}"
echo "  cpu_workers_per_seed=${PATENT_CPU_WORKERS}"
echo "  latency_cpu=${PATENT_LATENCY_CPU}"
echo "  start_stage=${PATENT_START_STAGE}"
echo "  seeds=${SEEDS[*]}"
echo "  output=${PATENT_OUTPUT}"

if [[ "${PATENT_DRY_RUN}" == "1" ]]; then
  for stage in screen validate test; do
    for index in "${!SEEDS[@]}"; do
      gpu="${GPU_IDS[$((index % ${#GPU_IDS[@]}))]}"
      echo "[dry-run] ${stage} seed=${SEEDS[index]} gpu=${gpu} cpu_workers=${PATENT_CPU_WORKERS}"
    done
    if [[ "${stage}" == "screen" ]]; then
      echo "[dry-run] freeze-screen"
    elif [[ "${stage}" == "validate" ]]; then
      echo "[dry-run] CPU${PATENT_LATENCY_CPU} pinned decode_batch throughput freeze"
    fi
  done
  echo "[dry-run] CPU${PATENT_LATENCY_CPU} pinned decode_batch throughput"
  echo "[dry-run] aggregate"
  exit 0
fi

if [[ ! -x "${PREDECODER_PYTHON}" ]]; then
  echo "Python is not executable: ${PREDECODER_PYTHON}" >&2
  exit 2
fi
if [[ ! -f "${PATENT_CHECKPOINT}" ]]; then
  echo "checkpoint not found: ${PATENT_CHECKPOINT}" >&2
  exit 2
fi
if ! command -v taskset >/dev/null; then
  echo "taskset is required for registered batch-throughput measurement" >&2
  exit 2
fi

"${PREDECODER_PYTHON}" -c 'import sys, numpy, torch, yaml, stim, pymatching; print(f"[preflight] executable={sys.executable} prefix={sys.prefix} cuda={torch.cuda.is_available()} devices={torch.cuda.device_count()}"); assert sys.prefix == "/root/miniconda3/envs/ising-decoding"; assert torch.cuda.is_available()'

mkdir -p "${PATENT_OUTPUT}/${PATENT_MODE}/logs"
START_SECONDS="${SECONDS}"
MAX_SECONDS=$((PATENT_MAX_HOURS * 3600))
LAUNCH_CUTOFF=$((MAX_SECONDS - 1800))
if [[ "${LAUNCH_CUTOFF}" -le 0 ]]; then
  echo "PATENT_MAX_HOURS must leave at least 30 minutes for aggregation" >&2
  exit 2
fi
PIDS=()

cleanup_children() {
  if [[ "${#PIDS[@]}" -gt 0 ]]; then
    kill "${PIDS[@]}" 2>/dev/null || true
    wait "${PIDS[@]}" 2>/dev/null || true
  fi
}
trap cleanup_children INT TERM

remaining_before_cutoff() {
  local remaining=$((LAUNCH_CUTOFF - (SECONDS - START_SECONDS)))
  if [[ "${remaining}" -le 0 ]]; then
    return 1
  fi
  echo "${remaining}"
}

run_parallel_stage() {
  local stage="$1"
  local remaining
  if ! remaining="$(remaining_before_cutoff)"; then
    return 124
  fi
  PIDS=()
  for index in "${!SEEDS[@]}"; do
    local seed="${SEEDS[index]}"
    local gpu="${GPU_IDS[$((index % ${#GPU_IDS[@]}))]}"
    local log="${PATENT_OUTPUT}/${PATENT_MODE}/logs/${stage}_seed_${seed}.log"
    echo "[launch] ${stage} seed=${seed} gpu=${gpu}"
    CUDA_VISIBLE_DEVICES="${gpu}" timeout --signal=TERM --kill-after=60 "${remaining}s"       "${PREDECODER_PYTHON}" -m "${MODULE}" "${BASE_ARGS[@]}"       "${stage}" --seed "${seed}" --device cuda:0       --cpu-workers "${PATENT_CPU_WORKERS}" >"${log}" 2>&1 &
    PIDS+=("$!")
  done
  local status=0
  for pid in "${PIDS[@]}"; do
    if ! wait "${pid}"; then
      status=1
    fi
  done
  PIDS=()
  return "${status}"
}

run_serial_stage() {
  local stage="$1"
  local remaining
  if ! remaining="$(remaining_before_cutoff)"; then
    return 124
  fi
  local log="${PATENT_OUTPUT}/${PATENT_MODE}/logs/${stage}.log"
  echo "[launch] ${stage}"
  timeout --signal=TERM --kill-after=60 "${remaining}s"     "${PREDECODER_PYTHON}" -m "${MODULE}" "${BASE_ARGS[@]}"     "${stage}" >"${log}" 2>&1
}

run_isolated_stage() {
  local stage="$1"
  local remaining
  if ! remaining="$(remaining_before_cutoff)"; then
    return 124
  fi
  local log="${PATENT_OUTPUT}/${PATENT_MODE}/logs/${stage}.log"
  if [[ "${PATENT_MODE}" == "smoke" ]]; then
    echo "[launch] smoke affinity ${stage} cpu=${PATENT_LATENCY_CPU}"
    timeout --signal=TERM --kill-after=60 "${remaining}s" taskset -c "${PATENT_LATENCY_CPU}" "${PREDECODER_PYTHON}" -m "${MODULE}" "${BASE_ARGS[@]}" "${stage}" --latency-cpu "${PATENT_LATENCY_CPU}" >"${log}" 2>&1
  else
    echo "[launch] CPU-pinned decode_batch throughput ${stage} cpu=${PATENT_LATENCY_CPU}"
    timeout --signal=TERM --kill-after=60 "${remaining}s" taskset -c "${PATENT_LATENCY_CPU}" "${PREDECODER_PYTHON}" -m "${MODULE}" "${BASE_ARGS[@]}" "${stage}" --latency-cpu "${PATENT_LATENCY_CPU}" --require-isolation >"${log}" 2>&1
  fi
}

status=0
if [[ "${PATENT_START_STAGE}" == "screen" ]]; then
  run_parallel_stage screen || status=$?
  if [[ "${status}" -eq 0 ]]; then run_serial_stage freeze-screen || status=$?; fi
  if [[ "${status}" -eq 0 ]]; then run_parallel_stage validate || status=$?; fi
fi
if [[ "${status}" -eq 0 && ( "${PATENT_START_STAGE}" == "screen" || "${PATENT_START_STAGE}" == "freeze" ) ]]; then
  run_isolated_stage freeze || status=$?
fi
if [[ "${status}" -eq 0 && ( "${PATENT_START_STAGE}" == "screen" || "${PATENT_START_STAGE}" == "freeze" || "${PATENT_START_STAGE}" == "test" ) ]]; then
  run_parallel_stage test || status=$?
fi
if [[ "${status}" -eq 0 && "${PATENT_START_STAGE}" != "aggregate" ]]; then
  if run_isolated_stage latency; then
    :
  else
    echo "[retry] batch-throughput stage failed (including sentinel drift); retrying once"
    if ! run_isolated_stage latency; then
      status=$?
      [[ "${status}" -eq 0 ]] && status=1
    fi
  fi
fi

"${PREDECODER_PYTHON}" -m "${MODULE}" "${BASE_ARGS[@]}" aggregate
if [[ "${status}" -ne 0 ]]; then
  echo "Validation incomplete; inspect ${PATENT_OUTPUT}/${PATENT_MODE}/logs" >&2
  exit "${status}"
fi
echo "Completed: ${PATENT_OUTPUT}/${PATENT_MODE}/aggregate/results.md"
