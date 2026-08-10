#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/code${PYTHONPATH:+:${PYTHONPATH}}"

PREDECODER_PYTHON="${PREDECODER_PYTHON:-python3}"
PATENT_GPUS="${PATENT_GPUS:-0,1,2,3}"
PATENT_MAX_HOURS="${PATENT_MAX_HOURS:-8}"
PATENT_RESUME="${PATENT_RESUME:-1}"
PATENT_DRY_RUN="${PATENT_DRY_RUN:-0}"
PATENT_MODE="${PATENT_MODE:-full}"
PATENT_OUTPUT="${PATENT_OUTPUT:-${REPO_ROOT}/outputs/patent_validation/topology_gating_v2_small}"
PATENT_CONFIG="${PATENT_CONFIG:-${REPO_ROOT}/conf/experiments/patent/topology_gating_v2_small.yaml}"

if [[ ! "${PATENT_MAX_HOURS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "PATENT_MAX_HOURS must be a positive integer" >&2
  exit 2
fi
if [[ "${PATENT_MODE}" != "full" && "${PATENT_MODE}" != "smoke" ]]; then
  echo "PATENT_MODE must be full or smoke" >&2
  exit 2
fi

IFS=',' read -r -a GPU_IDS <<< "${PATENT_GPUS}"
if [ "${#GPU_IDS[@]}" -lt 1 ] || [ "${#GPU_IDS[@]}" -gt 4 ]; then
  echo "PATENT_GPUS must contain between one and four comma-separated GPU ids" >&2
  exit 2
fi

SEEDS=(20260807 20260808 20260809)
if [ "${PATENT_MODE}" = "smoke" ]; then
  SEEDS=(20260807)
fi
LOSSES=(bce topology)
MODULE="benchmarks.patent_validation.topology_gating_v2_small"
BASE_ARGS=(--config "${PATENT_CONFIG}" --output "${PATENT_OUTPUT}" --mode "${PATENT_MODE}")
if [ "${PATENT_RESUME}" = "1" ]; then
  BASE_ARGS+=(--resume)
fi

echo "Topology-gating v2 four-channel experiment"
echo "  mode=${PATENT_MODE}"
echo "  python=${PREDECODER_PYTHON}"
echo "  gpus=${PATENT_GPUS}"
echo "  max_hours=${PATENT_MAX_HOURS}"
echo "  output=${PATENT_OUTPUT}"
echo "  seeds=${SEEDS[*]}"

if [ "${PATENT_DRY_RUN}" = "1" ]; then
  echo "[dry-run] prepare DEM on physical GPU ${GPU_IDS[0]}"
  dry_slot=0
  for seed in "${SEEDS[@]}"; do
    for loss_kind in "${LOSSES[@]}"; do
      gpu_id="${GPU_IDS[$((dry_slot % ${#GPU_IDS[@]}))]}"
      echo "[dry-run] train seed=${seed} loss=${loss_kind} gpu=${gpu_id}"
      dry_slot=$((dry_slot + 1))
    done
  done
  dry_slot=0
  for seed in "${SEEDS[@]}"; do
    for loss_kind in "${LOSSES[@]}"; do
      gpu_id="${GPU_IDS[$((dry_slot % ${#GPU_IDS[@]}))]}"
      echo "[dry-run] select seed=${seed} loss=${loss_kind} gpu=${gpu_id}"
      dry_slot=$((dry_slot + 1))
    done
  done
  dry_slot=0
  for seed in "${SEEDS[@]}"; do
    gpu_id="${GPU_IDS[$((dry_slot % ${#GPU_IDS[@]}))]}"
    echo "[dry-run] evaluate seed=${seed} gpu=${gpu_id}"
    dry_slot=$((dry_slot + 1))
  done
  echo "[dry-run] aggregate"
  exit 0
fi

"${PREDECODER_PYTHON}" - <<'PY'
import sys
if sys.version_info < (3, 11):
    raise SystemExit(f"Python >=3.11 is required, got {sys.version}")
import numpy, torch, yaml, stim, pymatching  # noqa: F401
print(f"[preflight] executable={sys.executable}")
print(f"[preflight] prefix={sys.prefix}")
print(f"[preflight] torch={torch.__version__} cuda={torch.cuda.is_available()} devices={torch.cuda.device_count()}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is required for the four-channel experiment")
PY

mkdir -p "${PATENT_OUTPUT}/${PATENT_MODE}/logs"
START_SECONDS="${SECONDS}"
MAX_SECONDS=$((PATENT_MAX_HOURS * 3600))
LAUNCH_CUTOFF=$((MAX_SECONDS - 1800))
if [ "${LAUNCH_CUTOFF}" -lt 1 ]; then
  LAUNCH_CUTOFF=1
fi

PIDS=()

cleanup_children() {
  if [ "${#PIDS[@]}" -gt 0 ]; then
    kill "${PIDS[@]}" 2>/dev/null || true
    wait "${PIDS[@]}" 2>/dev/null || true
  fi
}
trap cleanup_children INT TERM

remaining_seconds() {
  echo $((MAX_SECONDS - (SECONDS - START_SECONDS)))
}

can_launch_batch() {
  [ $((SECONDS - START_SECONDS)) -lt "${LAUNCH_CUTOFF}" ]
}

run_aggregate() {
  local aggregate_seconds
  aggregate_seconds="$(remaining_seconds)"
  if [ "${aggregate_seconds}" -lt 1 ]; then
    aggregate_seconds=1
  fi
  timeout --signal=TERM --kill-after=60 "${aggregate_seconds}s" \
    "${PREDECODER_PYTHON}" -m "${MODULE}" "${BASE_ARGS[@]}" aggregate
}

aggregate_partial() {
  run_aggregate || true
}

run_one() {
  local gpu_id="$1"
  local label="$2"
  shift 2
  local task_seconds=$((LAUNCH_CUTOFF - (SECONDS - START_SECONDS)))
  if [ "${task_seconds}" -lt 1 ]; then
    echo "Launch cutoff reached before ${label}." >&2
    return 3
  fi
  local log_path="${PATENT_OUTPUT}/${PATENT_MODE}/logs/${label}.log"
  echo "[launch] ${label} gpu=${gpu_id} remaining=$(remaining_seconds)s"
  CUDA_VISIBLE_DEVICES="${gpu_id}" \
    timeout --signal=TERM --kill-after=60 "${task_seconds}s" \
    "${PREDECODER_PYTHON}" -m "${MODULE}" "${BASE_ARGS[@]}" "$@" \
      --device cuda:0 >"${log_path}" 2>&1 &
  PIDS+=("$!")
}

wait_batch() {
  local count="${#PIDS[@]}"
  local completed=0
  local status=0
  while [ "${completed}" -lt "${count}" ]; do
    if wait -n; then
      completed=$((completed + 1))
    else
      status=$?
      echo "A patent-validation child process failed with status ${status}." >&2
      cleanup_children
      PIDS=()
      aggregate_partial
      return "${status}"
    fi
  done
  PIDS=()
}

if ! can_launch_batch; then
  aggregate_partial
  exit 3
fi
if CUDA_VISIBLE_DEVICES="${GPU_IDS[0]}" \
  timeout --signal=TERM --kill-after=60 "${LAUNCH_CUTOFF}s" \
  "${PREDECODER_PYTHON}" -m "${MODULE}" "${BASE_ARGS[@]}" prepare --device cuda:0; then
  :
else
  status=$?
  aggregate_partial
  exit "${status}"
fi

TRAIN_TASKS=()
SELECT_TASKS=()
for seed in "${SEEDS[@]}"; do
  for loss_kind in "${LOSSES[@]}"; do
    TRAIN_TASKS+=("${seed}:${loss_kind}")
    SELECT_TASKS+=("${seed}:${loss_kind}")
  done
done

run_task_batches() {
  local stage="$1"
  shift
  local tasks=("$@")
  local offset=0
  while [ "${offset}" -lt "${#tasks[@]}" ]; do
    if ! can_launch_batch; then
      echo "Launch cutoff reached before ${stage} batch." >&2
      aggregate_partial
      return 3
    fi
    PIDS=()
    local slot=0
    while [ "${slot}" -lt "${#GPU_IDS[@]}" ] && [ $((offset + slot)) -lt "${#tasks[@]}" ]; do
      local task="${tasks[$((offset + slot))]}"
      local seed="${task%%:*}"
      local loss_kind="${task##*:}"
      run_one \
        "${GPU_IDS[${slot}]}" \
        "${stage}_${loss_kind}_seed_${seed}" \
        "${stage}" --seed "${seed}" --loss-kind "${loss_kind}"
      slot=$((slot + 1))
    done
    wait_batch
    offset=$((offset + slot))
  done
}

run_task_batches train "${TRAIN_TASKS[@]}"
run_task_batches select "${SELECT_TASKS[@]}"

EVALUATE_TASKS=()
for seed in "${SEEDS[@]}"; do
  EVALUATE_TASKS+=("${seed}:evaluate")
done

offset=0
while [ "${offset}" -lt "${#EVALUATE_TASKS[@]}" ]; do
  if ! can_launch_batch; then
    echo "Launch cutoff reached before evaluate batch." >&2
    aggregate_partial
    exit 3
  fi
  PIDS=()
  slot=0
  while [ "${slot}" -lt "${#GPU_IDS[@]}" ] && [ $((offset + slot)) -lt "${#EVALUATE_TASKS[@]}" ]; do
    task="${EVALUATE_TASKS[$((offset + slot))]}"
    seed="${task%%:*}"
    run_one "${GPU_IDS[${slot}]}" "evaluate_seed_${seed}" evaluate --seed "${seed}"
    slot=$((slot + 1))
  done
  wait_batch
  offset=$((offset + slot))
done

run_aggregate
echo "Completed. Results: ${PATENT_OUTPUT}/${PATENT_MODE}/aggregate/results.md"
