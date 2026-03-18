#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

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

anchor_ratios=(0.0 0.1 0.2 0.3 0.5 1.0)
model="Dream-org/Dream-v0-Instruct-7B"

for anchor_ratio in "${anchor_ratios[@]}"; do
  model_args="pretrained=${model},gen_length=256,steps=256,block_length=32,threshold=0.9,show_speed=True,affix_type=${affix_type},anchor_ratio=${anchor_ratio},selection_mode=top"
  if [[ "${drop_non_anchor}" == "true" ]]; then
    model_args+=",drop_non_anchor=True"
  fi

  echo "Running with seed=${seed}, dataset=${dataset}, num_fewshot=${num_fewshot}, affix_type=${affix_type}, anchor_ratio=${anchor_ratio}, drop_non_anchor=${drop_non_anchor}"
  accelerate launch eval_ACache.py \
    --seed "${seed}" \
    --tasks "${dataset}" \
    --num_fewshot "${num_fewshot}" \
    --trust_remote_code \
    --model dream_acache \
    --model_args "${model_args}"
done
