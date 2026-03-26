#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ ! -f "${RUN_DIR}/eval_ACache.py" && -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/eval_ACache.py" ]]; then
  RUN_DIR="${SLURM_SUBMIT_DIR}"
fi

EVAL_SCRIPT="${RUN_DIR}/eval_ACache.py"
if [[ ! -f "${EVAL_SCRIPT}" ]]; then
  echo "Failed to locate eval_ACache.py. Checked: ${EVAL_SCRIPT}" >&2
  echo "If you submit with sbatch, run it from the script directory or pass --chdir to sbatch." >&2
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

usage() {
  echo "Usage: $0 [--seed SEED] [--dataset DATASET] [--num-fewshot N] [--affix-type {prefix|infix|suffix}] [--prefix|--infix|--suffix] [--drop-non-anchor] [SEED] [DATASET]"
}

positional_args=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --seed)
      if [[ $# -lt 2 ]]; then
        usage
        exit 1
      fi
      seed="$2"
      shift 2
      ;;
    --dataset)
      if [[ $# -lt 2 ]]; then
        usage
        exit 1
      fi
      dataset="$2"
      shift 2
      ;;
    --num-fewshot|--num_fewshot)
      if [[ $# -lt 2 ]]; then
        usage
        exit 1
      fi
      num_fewshot="$2"
      shift 2
      ;;
    --affix-type|--affix_type)
      if [[ $# -lt 2 ]]; then
        usage
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
    --help|-h)
      usage
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
      echo "Unknown option: $1"
      usage
      exit 1
      ;;
    *)
      positional_args+=("$1")
      shift
      ;;
  esac
done

if [[ ${#positional_args[@]} -gt 2 ]]; then
  usage
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
  echo "Seed must be an integer, got: ${seed}"
  exit 1
fi

if ! [[ "${num_fewshot}" =~ ^[0-9]+$ ]]; then
  echo "num_fewshot must be a non-negative integer, got: ${num_fewshot}"
  exit 1
fi

if [[ "${affix_type}" != "prefix" && "${affix_type}" != "infix" && "${affix_type}" != "suffix" ]]; then
  echo "affix_type must be one of: prefix, infix, suffix. Got: ${affix_type}"
  exit 1
fi

if [[ -z "${dataset}" ]]; then
  echo "Dataset must be non-empty."
  exit 1
fi

ACCELERATE_BIN="${ACCELERATE_BIN:-$(command -v accelerate || true)}"
if [[ -z "${ACCELERATE_BIN}" && -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/accelerate" ]]; then
  ACCELERATE_BIN="${CONDA_PREFIX}/bin/accelerate"
fi
if [[ -z "${ACCELERATE_BIN}" && -x "${HOME}/miniforge3/envs/ACache/bin/accelerate" ]]; then
  ACCELERATE_BIN="${HOME}/miniforge3/envs/ACache/bin/accelerate"
fi
if [[ ! -x "${ACCELERATE_BIN:-}" ]]; then
  echo "Failed to locate the 'accelerate' executable." >&2
  echo "Activate the ACache conda environment, or set ACCELERATE_BIN explicitly." >&2
  exit 1
fi

anchor_ratios=(0.0 0.1 0.2 0.3 0.5 1.0)

for anchor_ratio in "${anchor_ratios[@]}"; do
  model_args="model_path=GSAI-ML/LLaDA-8B-Instruct,gen_length=256,steps=256,block_length=32,threshold=0.9,show_speed=True,affix_type=${affix_type},anchor_ratio=${anchor_ratio},selection_mode=top"
  if [[ "${drop_non_anchor}" == "true" ]]; then
    model_args+=",drop_non_anchor=True"
  fi

  ratio_tag=$(printf '%g' "${anchor_ratio}")
  non_anchor_mode="keepna"
  if [[ "${drop_non_anchor}" == "true" ]]; then
    non_anchor_mode="dropna"
  fi
  output_path="evals_results/anchor_top_${ratio_tag}_${non_anchor_mode}_${num_fewshot}shot_${affix_type}_seed_${seed}"

  echo "Running with seed=${seed}, dataset=${dataset}, num_fewshot=${num_fewshot}, affix_type=${affix_type}, anchor_ratio=${anchor_ratio}, drop_non_anchor=${drop_non_anchor}"
  "${ACCELERATE_BIN}" launch "${EVAL_SCRIPT}" \
    --seed "${seed}" \
    --tasks "${dataset}" \
    --num_fewshot "${num_fewshot}" \
    --output_path "${output_path}" \
    --trust_remote_code \
    --model llada_acache \
    --model_args "${model_args}"
done
