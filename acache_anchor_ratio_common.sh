#!/usr/bin/env bash

acache_anchor_ratio_usage() {
  local prog_name="$1"
  echo "Usage: ${prog_name} [--seed SEED] [--dataset DATASET] [--num-fewshot N] [--affix-type {prefix|infix|suffix}] [--prefix|--infix|--suffix] [--drop-non-anchor] [--confirm-run-unsafe-code|--no-confirm-run-unsafe-code] [SEED] [DATASET]"
}

BABILONG_GENERATION_LENGTH=8

acache_normalize_task_name() {
  local task="${1:-}"
  task="${task//[[:space:]]/}"
  printf '%s\n' "${task}"
}

acache_normalize_tasks() {
  local tasks="${1:-}"
  local task normalized joined=""
  local IFS=','
  read -r -a task_list <<< "${tasks}"
  for task in "${task_list[@]}"; do
    normalized="$(acache_normalize_task_name "${task}")"
    if [[ -z "${normalized}" ]]; then
      continue
    fi
    if [[ -n "${joined}" ]]; then
      joined+=",${normalized}"
    else
      joined="${normalized}"
    fi
  done
  printf '%s\n' "${joined}"
}

acache_tasks_are_babilong() {
  local tasks
  tasks="$(acache_normalize_tasks "${1:-}")"
  if [[ -z "${tasks}" ]]; then
    return 1
  fi

  local task
  local IFS=','
  read -r -a task_list <<< "${tasks}"
  for task in "${task_list[@]}"; do
    case "${task}" in
      babilong)
        ;;
      *)
        return 1
        ;;
    esac
  done
  return 0
}

acache_set_model_arg() {
  local model_args="${1:-}"
  local key="$2"
  local value="$3"
  local arg="${key}=${value}"

  if [[ -z "${model_args}" ]]; then
    printf '%s\n' "${arg}"
    return 0
  fi

  local part piece updated="" replaced=false
  local IFS=','
  read -r -a parts <<< "${model_args}"
  for part in "${parts[@]}"; do
    piece="${part//[[:space:]]/}"
    if [[ -z "${piece}" ]]; then
      continue
    fi
    if [[ "${piece}" == "${key}="* ]]; then
      if [[ "${replaced}" == "false" ]]; then
        if [[ -n "${updated}" ]]; then
          updated+=",${arg}"
        else
          updated="${arg}"
        fi
        replaced=true
      fi
      continue
    fi
    if [[ -n "${updated}" ]]; then
      updated+=",${piece}"
    else
      updated="${piece}"
    fi
  done

  if [[ "${replaced}" == "false" ]]; then
    if [[ -n "${updated}" ]]; then
      updated+=",${arg}"
    else
      updated="${arg}"
    fi
  fi
  printf '%s\n' "${updated}"
}

acache_apply_generation_length_defaults() {
  local tasks="${1:-}"
  local model_args="${2:-}"

  if acache_tasks_are_babilong "${tasks}"; then
    model_args="$(acache_set_model_arg "${model_args}" "gen_length" "${BABILONG_GENERATION_LENGTH}")"
    model_args="$(acache_set_model_arg "${model_args}" "steps" "${BABILONG_GENERATION_LENGTH}")"
    model_args="$(acache_set_model_arg "${model_args}" "block_length" "${BABILONG_GENERATION_LENGTH}")"
  fi
  printf '%s\n' "${model_args}"
}

acache_locate_eval_script() {
  local run_dir="$1"
  if [[ ! -f "${run_dir}/eval_ACache.py" && -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/eval_ACache.py" ]]; then
    run_dir="${SLURM_SUBMIT_DIR}"
  fi

  local eval_script="${run_dir}/eval_ACache.py"
  if [[ ! -f "${eval_script}" ]]; then
    echo "Failed to locate eval_ACache.py. Checked: ${eval_script}" >&2
    echo "If you submit with sbatch, run it from the script directory or pass --chdir to sbatch." >&2
    return 1
  fi

  printf '%s\n' "${eval_script}"
}

acache_find_accelerate_bin() {
  local accelerate_bin="${ACCELERATE_BIN:-$(command -v accelerate || true)}"
  if [[ -z "${accelerate_bin}" && -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/accelerate" ]]; then
    accelerate_bin="${CONDA_PREFIX}/bin/accelerate"
  fi
  if [[ -z "${accelerate_bin}" && -x "${HOME}/miniforge3/envs/ACache/bin/accelerate" ]]; then
    accelerate_bin="${HOME}/miniforge3/envs/ACache/bin/accelerate"
  fi
  printf '%s\n' "${accelerate_bin}"
}

acache_task_requires_unsafe_code() {
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

acache_require_unsafe_code_confirmation() {
  local tasks="${1:-}"
  local confirmed="${2:-false}"
  local confirm_flag="${3:---confirm-run-unsafe-code}"

  if acache_task_requires_unsafe_code "${tasks}" && [[ "${confirmed}" != "true" ]]; then
    echo "Task(s) '${tasks}' may execute code during evaluation." >&2
    echo "Review the task and model code, then re-run with ${confirm_flag} if you trust them." >&2
    return 1
  fi
  return 0
}

run_acache_anchor_ratio_sweep() {
  local run_dir=""
  local eval_model=""
  local model_args_prefix=""

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --run-dir)
        run_dir="$2"
        shift 2
        ;;
      --model)
        eval_model="$2"
        shift 2
        ;;
      --model-args-prefix)
        model_args_prefix="$2"
        shift 2
        ;;
      --)
        shift
        break
        ;;
      *)
        echo "Unknown internal option: $1" >&2
        return 2
        ;;
    esac
  done

  if [[ -z "${run_dir}" || -z "${eval_model}" || -z "${model_args_prefix}" ]]; then
    echo "run_acache_anchor_ratio_sweep requires --run-dir, --model, and --model-args-prefix." >&2
    return 2
  fi

  run_dir="$(cd "${run_dir}" && pwd)"
  local eval_script
  eval_script="$(acache_locate_eval_script "${run_dir}")" || return $?

  if [[ -n "${TMPDIR:-}" && ! -d "${TMPDIR}" ]]; then
    export TMPDIR=/tmp
  fi

  run_dir="$(cd "$(dirname "${eval_script}")" && pwd)"
  cd "${run_dir}"

  local seed=0
  local dataset="gsm8k"
  local num_fewshot=4
  local drop_non_anchor=false
  local affix_type="prefix"
  local confirm_run_unsafe_code=false
  local -a positional_args=()

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --seed)
        if [[ $# -lt 2 ]]; then
          acache_anchor_ratio_usage "$0"
          return 1
        fi
        seed="$2"
        shift 2
        ;;
      --dataset)
        if [[ $# -lt 2 ]]; then
          acache_anchor_ratio_usage "$0"
          return 1
        fi
        dataset="$2"
        shift 2
        ;;
      --num-fewshot|--num_fewshot)
        if [[ $# -lt 2 ]]; then
          acache_anchor_ratio_usage "$0"
          return 1
        fi
        num_fewshot="$2"
        shift 2
        ;;
      --affix-type|--affix_type)
        if [[ $# -lt 2 ]]; then
          acache_anchor_ratio_usage "$0"
          return 1
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
      --confirm-run-unsafe-code|--confirm_run_unsafe_code)
        confirm_run_unsafe_code=true
        shift
        ;;
      --no-confirm-run-unsafe-code|--no_confirm_run_unsafe_code)
        confirm_run_unsafe_code=false
        shift
        ;;
      --help|-h)
        acache_anchor_ratio_usage "$0"
        return 0
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
        acache_anchor_ratio_usage "$0"
        return 1
        ;;
      *)
        positional_args+=("$1")
        shift
        ;;
    esac
  done

  if [[ ${#positional_args[@]} -gt 2 ]]; then
    acache_anchor_ratio_usage "$0"
    return 1
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
    return 1
  fi

  if ! [[ "${num_fewshot}" =~ ^[0-9]+$ ]]; then
    echo "num_fewshot must be a non-negative integer, got: ${num_fewshot}" >&2
    return 1
  fi

  if [[ "${affix_type}" != "prefix" && "${affix_type}" != "infix" && "${affix_type}" != "suffix" ]]; then
    echo "affix_type must be one of: prefix, infix, suffix. Got: ${affix_type}" >&2
    return 1
  fi

  if [[ -z "${dataset}" ]]; then
    echo "Dataset must be non-empty." >&2
    return 1
  fi

  local original_dataset="${dataset}"
  dataset="$(acache_normalize_tasks "${dataset}")"
  if [[ -z "${dataset}" ]]; then
    echo "Dataset must be non-empty." >&2
    return 1
  fi
  if [[ "${dataset}" != "${original_dataset}" ]]; then
    echo "[eval_ACache] normalized --dataset=${original_dataset} to --tasks=${dataset}."
  fi
  if acache_tasks_are_babilong "${dataset}"; then
    echo "[eval_ACache] using gen_length=${BABILONG_GENERATION_LENGTH},steps=${BABILONG_GENERATION_LENGTH} for ${dataset} from the shell script."
  fi
  local -a task_eval_args=()

  acache_require_unsafe_code_confirmation "${dataset}" "${confirm_run_unsafe_code}" "--confirm-run-unsafe-code" || return $?

  local accelerate_bin
  accelerate_bin="$(acache_find_accelerate_bin)"
  if [[ ! -x "${accelerate_bin:-}" ]]; then
    echo "Failed to locate the 'accelerate' executable." >&2
    echo "Activate the ACache conda environment, or set ACCELERATE_BIN explicitly." >&2
    return 1
  fi

  local -a anchor_ratios=(0.0 0.1 0.2 0.3 0.5 1.0)
  local anchor_ratio
  for anchor_ratio in "${anchor_ratios[@]}"; do
    local model_args="${model_args_prefix},affix_type=${affix_type},anchor_ratio=${anchor_ratio},selection_mode=top"
    if [[ "${drop_non_anchor}" == "true" ]]; then
      model_args+=",drop_non_anchor=True"
    fi
    model_args="$(acache_apply_generation_length_defaults "${dataset}" "${model_args}")"

    local -a extra_eval_args=()
    if [[ "${confirm_run_unsafe_code}" == "true" ]]; then
      extra_eval_args+=(--confirm_run_unsafe_code)
      export HF_ALLOW_CODE_EVAL=1
    fi

    echo "Running with seed=${seed}, dataset=${dataset}, num_fewshot=${num_fewshot}, affix_type=${affix_type}, anchor_ratio=${anchor_ratio}, drop_non_anchor=${drop_non_anchor}, confirm_run_unsafe_code=${confirm_run_unsafe_code}"
    "${accelerate_bin}" launch "${eval_script}" \
      --seed "${seed}" \
      --tasks "${dataset}" \
      --num_fewshot "${num_fewshot}" \
      --trust_remote_code \
      "${task_eval_args[@]}" \
      "${extra_eval_args[@]}" \
      --model "${eval_model}" \
      --model_args "${model_args}"
  done
}
