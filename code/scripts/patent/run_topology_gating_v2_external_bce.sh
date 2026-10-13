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
PATENT_MODE="${PATENT_MODE:-full}"
PATENT_RESUME="${PATENT_RESUME:-1}"
PATENT_DRY_RUN="${PATENT_DRY_RUN:-0}"
PATENT_MAX_HOURS="${PATENT_MAX_HOURS:-4}"
PATENT_OUTPUT="${PATENT_OUTPUT:-${REPO_ROOT}/outputs/patent_validation/topology_gating_v2_external_bce}"
PATENT_CONFIG="${PATENT_CONFIG:-${REPO_ROOT}/conf/experiments/patent/topology_gating_v2_external_bce.yaml}"
PATENT_CHECKPOINT="${PATENT_CHECKPOINT:-/mnt/public/miaoran/Ising-Decoding-storage/outputs/ising_domestic_fast/models/best_model/PreDecoderModelMemory_v1.0.53.pt}"

# Each gate worker is single-threaded; otherwise BLAS/OpenMP can multiply the
# requested process count and oversubscribe the host.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

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
if [[ ! -f "${PATENT_CHECKPOINT}" ]]; then
  echo "checkpoint not found: ${PATENT_CHECKPOINT}" >&2
  exit 2
fi

IFS=',' read -r -a GPU_IDS <<< "${PATENT_GPUS}"
if [[ "${#GPU_IDS[@]}" -lt 1 || "${#GPU_IDS[@]}" -gt 3 ]]; then
  echo "PATENT_GPUS must contain between one and three GPU ids" >&2
  exit 2
fi

SEEDS=(20260811 20260812 20260813)
if [[ "${PATENT_MODE}" == "smoke" ]]; then
  SEEDS=(20260811)
fi
MODULE="benchmarks.patent_validation.topology_gating_v2_external_bce"
BASE_ARGS=(
  --config "${PATENT_CONFIG}"
  --output "${PATENT_OUTPUT}"
  --checkpoint "${PATENT_CHECKPOINT}"
  --mode "${PATENT_MODE}"
)
if [[ "${PATENT_RESUME}" == "1" ]]; then
  BASE_ARGS+=(--resume)
fi

echo "External BCE topology-gating comparison"
echo "  mode=${PATENT_MODE}"
echo "  checkpoint=${PATENT_CHECKPOINT}"
echo "  gpus=${PATENT_GPUS}"
echo "  cpu_workers_per_seed=${PATENT_CPU_WORKERS}"
echo "  seeds=${SEEDS[*]}"
echo "  output=${PATENT_OUTPUT}"

if [[ "${PATENT_DRY_RUN}" == "1" ]]; then
  for index in "${!SEEDS[@]}"; do
    gpu="${GPU_IDS[$((index % ${#GPU_IDS[@]}))]}"
    echo "[dry-run] evaluate seed=${SEEDS[index]} gpu=${gpu} cpu_workers=${PATENT_CPU_WORKERS}"
  done
  echo "[dry-run] aggregate"
  exit 0
fi

"${PREDECODER_PYTHON}" -c 'import sys, numpy, torch, yaml, stim, pymatching; print(f"[preflight] executable={sys.executable} prefix={sys.prefix} torch={torch.__version__} cuda={torch.cuda.is_available()} devices={torch.cuda.device_count()}"); assert torch.cuda.is_available()'

mkdir -p "${PATENT_OUTPUT}/${PATENT_MODE}/logs"
PIDS=()
START_SECONDS="${SECONDS}"
MAX_SECONDS=$((PATENT_MAX_HOURS * 3600))

cleanup_children() {
  if [[ "${#PIDS[@]}" -gt 0 ]]; then
    kill "${PIDS[@]}" 2>/dev/null || true
    wait "${PIDS[@]}" 2>/dev/null || true
  fi
}
trap cleanup_children INT TERM

for index in "${!SEEDS[@]}"; do
  seed="${SEEDS[index]}"
  gpu="${GPU_IDS[$((index % ${#GPU_IDS[@]}))]}"
  remaining=$((MAX_SECONDS - (SECONDS - START_SECONDS)))
  if [[ "${remaining}" -lt 1 ]]; then
    echo "wall-clock budget exhausted before seed ${seed}" >&2
    break
  fi
  log="${PATENT_OUTPUT}/${PATENT_MODE}/logs/evaluate_seed_${seed}.log"
  echo "[launch] seed=${seed} gpu=${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" timeout --signal=TERM --kill-after=60 "${remaining}s" \
    "${PREDECODER_PYTHON}" -m "${MODULE}" "${BASE_ARGS[@]}" \
      evaluate --seed "${seed}" --device cuda:0 \
      --cpu-workers "${PATENT_CPU_WORKERS}" >"${log}" 2>&1 &
  PIDS+=("$!")
done

status=0
for pid in "${PIDS[@]}"; do
  if wait "${pid}"; then
    :
  else
    status=$?
  fi
done
PIDS=()

"${PREDECODER_PYTHON}" -m "${MODULE}" "${BASE_ARGS[@]}" aggregate
if [[ "${status}" -ne 0 ]]; then
  exit "${status}"
fi
echo "Completed: ${PATENT_OUTPUT}/${PATENT_MODE}/aggregate/results.md"
