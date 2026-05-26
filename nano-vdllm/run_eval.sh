#!/bin/bash

set -euo pipefail

usage() {
  local prog_name="$1"
  local script_name="./$(basename "${prog_name}")"
  cat <<EOF
Usage: ${script_name} --dataset {mbpp|gsm8k} (--acache|--baseline) [--model {llada|dream}] [--seed SEED] [--num-fewshot N] [--batch-size N] [--anchor-ratio R] [--profile] [--no-profile] [-- EXTRA_EVAL_ARGS...]

Examples:
  ${script_name} --dataset mbpp --baseline
  ${script_name} --dataset mbpp --acache --anchor-ratio 0.2 --profile
  ${script_name} --model dream --dataset mbpp --acache --anchor-ratio 0.2 --profile
EOF
}

task_requires_unsafe_code() {
  local tasks="${1:-}"
  local task
  local IFS=','
  read -r -a task_list <<< "${tasks}"
  for task in "${task_list[@]}"; do
    task="${task//[[:space:]]/}"
    case "${task}" in
      mbpp|mbpp_*)
        return 0
        ;;
    esac
  done
  return 1
}

seed=0
dataset=""
model_choice="llada"
num_fewshot=2
batch_size=16
anchor_ratio=0.2
acache=""
profile_timing=false
extra_eval_args=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset|--tasks)
      if [[ $# -lt 2 ]]; then
        usage "$0"
        exit 1
      fi
      dataset="$2"
      shift 2
      ;;
    --model|--model-type|--model_type)
      if [[ $# -lt 2 ]]; then
        usage "$0"
        exit 1
      fi
      model_choice="$2"
      shift 2
      ;;
    --acache)
      acache=True
      shift
      ;;
    --baseline|--no-acache)
      acache=False
      shift
      ;;
    --seed)
      if [[ $# -lt 2 ]]; then
        usage "$0"
        exit 1
      fi
      seed="$2"
      shift 2
      ;;
    --num-fewshot|--num_fewshot)
      if [[ $# -lt 2 ]]; then
        usage "$0"
        exit 1
      fi
      num_fewshot="$2"
      shift 2
      ;;
    --batch-size|--batch_size)
      if [[ $# -lt 2 ]]; then
        usage "$0"
        exit 1
      fi
      batch_size="$2"
      shift 2
      ;;
    --anchor-ratio|--anchor_ratio)
      if [[ $# -lt 2 ]]; then
        usage "$0"
        exit 1
      fi
      anchor_ratio="$2"
      shift 2
      ;;
    --profile|--profile-timing|--profile_timing)
      profile_timing=true
      shift
      ;;
    --no-profile|--no-profile-timing|--no_profile_timing)
      profile_timing=false
      shift
      ;;
    --help|-h)
      usage "$0"
      exit 0
      ;;
    --)
      shift
      while [[ $# -gt 0 ]]; do
        extra_eval_args+=("$1")
        shift
      done
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage "$0"
      exit 1
      ;;
  esac
done

case "${dataset}" in
  mbpp|gsm8k)
    ;;
  "")
    echo "Missing required --dataset {mbpp|gsm8k}." >&2
    usage "$0"
    exit 1
    ;;
  *)
    echo "dataset must be one of: mbpp, gsm8k. Got: ${dataset}" >&2
    exit 1
    ;;
esac
case "${model_choice}" in
  llada|dream)
    ;;
  *)
    echo "model must be one of: llada, dream. Got: ${model_choice}" >&2
    exit 1
    ;;
esac
case "${acache}" in
  True|False)
    ;;
  "")
    echo "Missing required mode: pass one of --acache or --baseline." >&2
    usage "$0"
    exit 1
    ;;
esac

if ! [[ "${seed}" =~ ^-?[0-9]+$ ]]; then
  echo "Seed must be an integer, got: ${seed}" >&2
  exit 1
fi
if ! [[ "${num_fewshot}" =~ ^[0-9]+$ ]]; then
  echo "num_fewshot must be a non-negative integer, got: ${num_fewshot}" >&2
  exit 1
fi
if ! [[ "${batch_size}" =~ ^[0-9]+$ ]] || [[ "${batch_size}" -eq 0 ]]; then
  echo "batch_size must be a positive integer, got: ${batch_size}" >&2
  exit 1
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="${SLURM_SUBMIT_DIR:-${script_dir}}"
if [[ ! -f "${repo_dir}/eval_llada.py" ]]; then
  repo_dir="${script_dir}"
fi
if [[ ! -f "${repo_dir}/eval_llada.py" ]]; then
  echo "Failed to locate eval_llada.py. Checked ${repo_dir}." >&2
  exit 1
fi
cd "${repo_dir}"

export HF_DATASETS_TRUST_REMOTE_CODE=true
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  export HF_METRICS_CACHE="${HF_METRICS_CACHE:-/tmp/hf_metrics_${SLURM_JOB_ID}}"
  export HF_EVALUATE_CACHE="${HF_EVALUATE_CACHE:-/tmp/hf_evaluate_${SLURM_JOB_ID}}"
  export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/tmp/hf_datasets_${SLURM_JOB_ID}}"
  mkdir -p "${HF_METRICS_CACHE}" "${HF_EVALUATE_CACHE}" "${HF_DATASETS_CACHE}"
fi
if [[ -n "${TMPDIR:-}" && ! -d "${TMPDIR}" ]]; then
  export TMPDIR=/tmp
fi

task_eval_args=()
if task_requires_unsafe_code "${dataset}"; then
  export HF_ALLOW_CODE_EVAL=1
  task_eval_args+=(--confirm_run_unsafe_code)
fi

case "${model_choice}" in
  dream)
    eval_model_name="dream_dist"
    model_args="model_path=Dream-org/Dream-v0-Instruct-7B,gen_length=256,block_length=32,recompute_batch_size=4,threshold=0.9,show_speed=True,acache=${acache}"
    ;;
  llada)
    eval_model_name="llada_dist"
    model_args="model_path=GSAI-ML/LLaDA-8B-Instruct,gen_length=256,recompute_batch_size=4,show_speed=True,acache=${acache}"
    ;;
esac
if [[ "${acache}" == "True" ]]; then
  model_args+=",anchor_ratio=${anchor_ratio}"
fi
if [[ "${profile_timing}" == "true" ]]; then
  model_args+=",profile_timing=True"
fi

if [[ -n "${EVAL_PYTHON:-}" ]]; then
  python_bin="${EVAL_PYTHON}"
elif [[ -x "${HOME}/miniforge3/envs/ACache/bin/python" ]]; then
  python_bin="${HOME}/miniforge3/envs/ACache/bin/python"
else
  python_bin="$(command -v python)"
fi

echo "Running model=${model_choice}, dataset=${dataset}, acache=${acache}, seed=${seed}, num_fewshot=${num_fewshot}, batch_size=${batch_size}, anchor_ratio=${anchor_ratio}, profile_timing=${profile_timing}, python=${python_bin}"
"${python_bin}" eval_llada.py \
  --seed "${seed}" \
  --tasks "${dataset}" \
  --num_fewshot "${num_fewshot}" \
  --batch_size "${batch_size}" \
  "${task_eval_args[@]}" \
  "${extra_eval_args[@]}" \
  --model "${eval_model_name}" \
  --model_args "${model_args}"
