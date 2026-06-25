#!/usr/bin/env bash
set -euo pipefail

usage() {
  local prog_name="$1"
  echo "Usage: ${prog_name} [--seed SEED] [--dataset DATASET] [--num-fewshot N] [--affix-type {prefix|infix|suffix}] [--prefix|--infix|--suffix] [--drop-non-anchor] [--score-metric {l2|l1|relative_l2}] [--keys-only] [--confirm-run-unsafe-code|--no-confirm-run-unsafe-code] [SEED] [DATASET]"
}

task_requires_unsafe_code() {
  local tasks="${1:-}"
  local task
  local IFS=','
  read -r -a task_list <<< "${tasks}"
  for task in "${task_list[@]}"; do
    task="${task//[[:space:]]/}"
    if [[ -z "${task}" ]]; then
      continue
    fi
    case "${task}" in
      mbpp|mbpp_*)
        return 0
        ;;
    esac
  done
  return 1
}

find_accelerate_bin() {
  local accelerate_bin="${ACCELERATE_BIN:-$(command -v accelerate || true)}"
  if [[ -z "${accelerate_bin}" && -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/accelerate" ]]; then
    accelerate_bin="${CONDA_PREFIX}/bin/accelerate"
  fi
  if [[ -z "${accelerate_bin}" && -x "${HOME}/miniforge3/envs/ACache/bin/accelerate" ]]; then
    accelerate_bin="${HOME}/miniforge3/envs/ACache/bin/accelerate"
  fi
  printf '%s\n' "${accelerate_bin}"
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="${SCRIPT_DIR}"
if [[ ! -f "${RUN_DIR}/eval_CacheBlend_ACache.py" && -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/eval_CacheBlend_ACache.py" ]]; then
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

if [[ ! -f "${RUN_DIR}/eval_CacheBlend_ACache.py" ]]; then
  RUN_DIR="$(cd "$(dirname "${COMMON_SCRIPT}")/llada" && pwd)"
fi

source "${COMMON_SCRIPT}"

EVAL_SCRIPT="${RUN_DIR}/eval_CacheBlend_ACache.py"
if [[ ! -f "${EVAL_SCRIPT}" ]]; then
  echo "Failed to locate eval_CacheBlend_ACache.py. Checked: ${EVAL_SCRIPT}" >&2
  exit 1
fi

if [[ -n "${TMPDIR:-}" && ! -d "${TMPDIR}" ]]; then
  export TMPDIR=/tmp
fi

cd "${RUN_DIR}"

seed=0
dataset="gsm8k"
num_fewshot=4
drop_non_anchor=false
affix_type="prefix"
score_metric="l2"
include_values=true
confirm_run_unsafe_code=false
positional_args=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --seed)
      if [[ $# -lt 2 ]]; then
        usage "$0"
        exit 1
      fi
      seed="$2"
      shift 2
      ;;
    --dataset)
      if [[ $# -lt 2 ]]; then
        usage "$0"
        exit 1
      fi
      dataset="$2"
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
    --affix-type|--affix_type)
      if [[ $# -lt 2 ]]; then
        usage "$0"
        exit 1
      fi
      affix_type="$2"
      shift 2
      ;;
    --prefix)
      affix_type="prefix"
      shift
      ;;
    --infix)
      affix_type="infix"
      shift
      ;;
    --suffix)
      affix_type="suffix"
      shift
      ;;
    --drop|--drop-non-anchor)
      drop_non_anchor=true
      shift
      ;;
    --score-metric|--score_metric)
      if [[ $# -lt 2 ]]; then
        usage "$0"
        exit 1
      fi
      score_metric="$2"
      shift 2
      ;;
    --keys-only|--key-only)
      include_values=false
      shift
      ;;
    --confirm-run-unsafe-code|--confirm_run_unsafe_code)
      confirm_run_unsafe_code=true
      shift
      ;;
    --no-confirm-run-unsafe-code|--no_confirm_run_unsafe_code)
      confirm_run_unsafe_code=false
      shift
      ;;
    --help|-h)
      usage "$0"
      exit 0
      ;;
    --)
      shift
      while [[ $# -gt 0 ]]; do
        positional_args+=("$1")
        shift
      done
      ;;
    -*)
      echo "Unknown option: $1" >&2
      usage "$0"
      exit 1
      ;;
    *)
      positional_args+=("$1")
      shift
      ;;
  esac
done

if [[ ${#positional_args[@]} -gt 2 ]]; then
  usage "$0"
  exit 1
fi

if [[ ${#positional_args[@]} -ge 1 ]]; then
  if [[ "${positional_args[0]}" =~ ^-?[0-9]+$ ]]; then
    seed="${positional_args[0]}"
  else
    dataset="${positional_args[0]}"
  fi
fi

if [[ ${#positional_args[@]} -eq 2 ]]; then
  dataset="${positional_args[1]}"
fi

if ! [[ "${seed}" =~ ^-?[0-9]+$ ]]; then
  echo "Seed must be an integer, got: ${seed}" >&2
  exit 1
fi

if ! [[ "${num_fewshot}" =~ ^[0-9]+$ ]]; then
  echo "num_fewshot must be a non-negative integer, got: ${num_fewshot}" >&2
  exit 1
fi

if [[ "${affix_type}" != "prefix" && "${affix_type}" != "infix" && "${affix_type}" != "suffix" ]]; then
  echo "affix_type must be one of: prefix, infix, suffix. Got: ${affix_type}" >&2
  exit 1
fi

if [[ "${score_metric}" != "l2" && "${score_metric}" != "l1" && "${score_metric}" != "relative_l2" ]]; then
  echo "score_metric must be one of: l2, l1, relative_l2. Got: ${score_metric}" >&2
  exit 1
fi

if [[ -z "${dataset}" ]]; then
  echo "Dataset must be non-empty." >&2
  exit 1
fi

original_dataset="${dataset}"
dataset="$(acache_normalize_tasks "${dataset}")"
if [[ -z "${dataset}" ]]; then
  echo "Dataset must be non-empty." >&2
  exit 1
fi
if [[ "${dataset}" != "${original_dataset}" ]]; then
  echo "[eval_ACache] normalized --dataset=${original_dataset} to --tasks=${dataset}."
fi
if acache_tasks_are_babilong "${dataset}"; then
  echo "[eval_ACache] using gen_length=${BABILONG_GENERATION_LENGTH},steps=${BABILONG_GENERATION_LENGTH} for ${dataset} from the shell script."
fi
task_eval_args=()

acache_require_unsafe_code_confirmation "${dataset}" "${confirm_run_unsafe_code}" "--confirm-run-unsafe-code"

accelerate_bin="$(find_accelerate_bin)"
if [[ ! -x "${accelerate_bin:-}" ]]; then
  echo "Failed to locate the 'accelerate' executable." >&2
  echo "Activate the ACache conda environment, or set ACCELERATE_BIN explicitly." >&2
  exit 1
fi

anchor_ratios=(0.0 0.1 0.2 0.3 0.5 1.0)
for anchor_ratio in "${anchor_ratios[@]}"; do
  model_args="model_path=GSAI-ML/LLaDA-8B-Instruct,gen_length=256,steps=256,block_length=32,threshold=0.9,show_speed=True,affix_type=${affix_type},anchor_ratio=${anchor_ratio},selection_mode=top,cacheblend_score_metric=${score_metric},cacheblend_include_values=${include_values}"
  if [[ "${drop_non_anchor}" == "true" ]]; then
    model_args+=",drop_non_anchor=True"
  fi
  model_args="$(acache_apply_generation_length_defaults "${dataset}" "${model_args}")"

  extra_eval_args=()
  if [[ "${confirm_run_unsafe_code}" == "true" ]]; then
    extra_eval_args+=(--confirm_run_unsafe_code)
    export HF_ALLOW_CODE_EVAL=1
  fi

  echo "Running LLaDA CacheBlend-style selector with seed=${seed}, dataset=${dataset}, num_fewshot=${num_fewshot}, affix_type=${affix_type}, anchor_ratio=${anchor_ratio}, score_metric=${score_metric}, include_values=${include_values}, drop_non_anchor=${drop_non_anchor}, confirm_run_unsafe_code=${confirm_run_unsafe_code}"
  "${accelerate_bin}" launch "${EVAL_SCRIPT}" \
    --seed "${seed}" \
    --tasks "${dataset}" \
    --num_fewshot "${num_fewshot}" \
    --trust_remote_code \
    "${task_eval_args[@]}" \
    "${extra_eval_args[@]}" \
    --model llada_cacheblend_acache \
    --model_args "${model_args}" \
    --output_path "evals_results/cacheblend_hkvd_${score_metric}_$([[ "${include_values}" == "true" ]] && echo kv || echo k)_anchor_top_${anchor_ratio}_$([[ "${drop_non_anchor}" == "true" ]] && echo dropna || echo keepna)_${num_fewshot}shot_${affix_type}_seed_${seed}"
done
