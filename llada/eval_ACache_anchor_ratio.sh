#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

seed=0

if [[ $# -gt 0 ]]; then
  case "$1" in
    --seed)
      if [[ $# -lt 2 ]]; then
        echo "Usage: $0 [--seed SEED]"
        exit 1
      fi
      seed="$2"
      shift 2
      ;;
    *)
      seed="$1"
      shift
      ;;
  esac
fi

if [[ $# -ne 0 ]]; then
  echo "Usage: $0 [--seed SEED]"
  exit 1
fi

anchor_ratios=(0.0 0.1 0.2 0.3 0.5 1.0)

for anchor_ratio in "${anchor_ratios[@]}"; do
  echo "Running with seed=${seed}, anchor_ratio=${anchor_ratio}"
  accelerate launch eval_ACache.py \
    --seed "${seed}" \
    --tasks gsm8k \
    --num_fewshot 3 \
    --trust_remote_code \
    --model llada_acache \
    --model_args "model_path=GSAI-ML/LLaDA-8B-Instruct,gen_length=256,steps=256,block_length=32,threshold=0.9,show_speed=True,affix_type=prefix,anchor_ratio=${anchor_ratio},selection_mode=top"
done
