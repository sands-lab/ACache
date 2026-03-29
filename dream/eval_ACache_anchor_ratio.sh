#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="${SCRIPT_DIR}"
if [[ ! -f "${RUN_DIR}/eval_ACache.py" && -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/eval_ACache.py" ]]; then
  RUN_DIR="${SLURM_SUBMIT_DIR}"
fi

COMMON_SCRIPT=""
candidate_paths=(
  "${SCRIPT_DIR}/../acache_anchor_ratio_common.sh"
  "${RUN_DIR}/../acache_anchor_ratio_common.sh"
  "${PWD}/../acache_anchor_ratio_common.sh"
  "${PWD}/acache_anchor_ratio_common.sh"
)
for candidate in "${candidate_paths[@]}"; do
  if [[ -f "${candidate}" ]]; then
    COMMON_SCRIPT="$(cd "$(dirname "${candidate}")" && pwd)/$(basename "${candidate}")"
    break
  fi
done

if [[ -z "${COMMON_SCRIPT}" ]]; then
  echo "Failed to locate acache_anchor_ratio_common.sh." >&2
  echo "Checked relative to script dir, run dir, and current working directory." >&2
  exit 1
fi

if [[ ! -f "${RUN_DIR}/eval_ACache.py" ]]; then
  RUN_DIR="$(cd "$(dirname "${COMMON_SCRIPT}")/dream" && pwd)"
fi

source "${COMMON_SCRIPT}"

run_acache_anchor_ratio_sweep \
  --run-dir "${RUN_DIR}" \
  --model "dream_acache" \
  --model-args-prefix "pretrained=Dream-org/Dream-v0-Instruct-7B,gen_length=256,steps=256,block_length=32,threshold=0.9,show_speed=True" \
  -- "$@"
