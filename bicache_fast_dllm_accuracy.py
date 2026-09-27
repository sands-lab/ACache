#!/usr/bin/env python3
"""Accuracy evaluation for the validated BiCache + threshold Fast-dLLM path.

This runner deliberately shares the prompt construction and engine adapter
with ``bicache_fast_dllm_system.py``.  It adds only postprocessing/scoring and
request-level audit output.  Each request is still generated independently at
batch size one with the official BiCache cache lookup, boundary hidden-state
restore, deep-prefix recomputation, and periodic refresh path.
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
import time
from pathlib import Path

import torch

from babilong_exclusion import verified_plan

from bicache_fast_dllm_system import (
    BLOCK_LENGTH,
    CACHE_BUDGET,
    DECODING_THRESHOLD,
    DEFAULT_OFFICIAL_ROOT,
    GEN_LENGTH,
    INTRA_REQUEST_CACHE_UPDATE_INTERVAL,
    MASK_ID,
    MODEL_ID,
    REMASKING,
    SEED,
    TEMPERATURE,
    AuditedFullTokenFastdLLMEngine,
    _build_full_request_ids,
    _import_official,
    _load_model_and_tokenizer,
    _load_policy,
    _load_requests,
    _make_prompt_builder,
)


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "bicache_fast_dllm_threshold_accuracy_results"
LAYER_COUNT = 32


def _load_task(dataset: str):
    """Load the same lm-eval task metric used by the existing system runs."""

    from lm_eval.tasks import TaskManager

    manager = TaskManager(include_path=str(REPO_ROOT / "lm_eval_tasks"))
    task_name = dataset
    return manager.load_task_or_group([task_name])[task_name]


def _task_filter_name(dataset: str) -> str:
    if dataset == "gsm8k":
        return "flexible-extract"
    if dataset == "mbpp":
        return "none"
    return "remove_whitespace"


def _filter_one(task, filter_name: str, response: str, doc: dict) -> str:
    """Apply one lm-eval filter pipeline exactly as the evaluator does."""

    ensemble = next((item for item in task._filters if item.name == filter_name), None)
    if ensemble is None:
        names = [item.name for item in task._filters]
        raise RuntimeError(f"Missing lm-eval filter {filter_name!r}; available={names}")

    responses = [[response]]
    docs = [doc]
    for filter_factory in ensemble.filters:
        responses = list(filter_factory().apply(responses, docs))
    if len(responses) != 1:
        raise RuntimeError(f"Unexpected filtered response shape: {responses!r}")
    return str(responses[0])


def _stop_tokens(task, dataset: str) -> list[str]:
    generation_kwargs = getattr(task.config, "generation_kwargs", {}) or {}
    stop_tokens = generation_kwargs.get("until", [])
    if stop_tokens is None:
        stop_tokens = []
    elif isinstance(stop_tokens, str):
        stop_tokens = [stop_tokens]
    else:
        stop_tokens = list(stop_tokens)
    if dataset == "mbpp" and "[DONE]" not in stop_tokens:
        stop_tokens.append("[DONE]")
    return [str(token) for token in stop_tokens]


def _decode_like_existing_eval(tokenizer, generated_ids, stop_tokens: list[str]) -> str:
    """Copy the postprocessing in nano-vdllm/eval_llada.py."""

    answer = tokenizer.decode(generated_ids, skip_special_tokens=False)
    for stop_seq in stop_tokens:
        if stop_seq in answer:
            answer = answer.split(stop_seq)[0]
    answer_ids = torch.tensor(tokenizer(answer)["input_ids"])
    return tokenizer.decode(answer_ids, skip_special_tokens=True)


def _validate_task_alignment(dataset: str, rows: list[dict], task_docs: list[dict]) -> None:
    if len(rows) != len(task_docs):
        raise RuntimeError(
            f"Dataset/task length mismatch for {dataset}: {len(rows)} vs {len(task_docs)}"
        )
    for index, (row, doc) in enumerate(zip(rows, task_docs)):
        if dataset == "gsm8k":
            same = row.get("question") == doc.get("question")
            key = "question"
        elif dataset == "mbpp":
            same = row.get("task_id") == doc.get("task_id")
            key = "task_id"
        else:
            same = (
                row.get("input") == doc.get("input")
                and row.get("question") == doc.get("question")
                and row.get("target") == doc.get("target")
            )
            key = "input/question/target"
        if not same:
            raise RuntimeError(
                f"Dataset/task order mismatch at request {index} ({key}): "
                f"workload={row.get(key)!r}, metric_task={doc.get(key)!r}"
            )


def _policy_depth(policy: dict[int, int], ratio: int) -> int | None:
    eligible_keys = [int(key) for key in policy if int(key) <= int(ratio)]
    return int(policy[max(eligible_keys)]) if eligible_keys else None


def _run(args) -> dict:
    (
        LLaDAEvalHarness,
        set_seed,
        FastdLLMLLaDAEngine,
        _,
        LLaDAModelLM,
    ) = _import_official(Path(args.official_root).resolve())
    policy, policy_metadata = _load_policy(Path(args.policy_path))

    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("BiCache accuracy requires CUDA; no CUDA device is visible")

    model, tokenizer = _load_model_and_tokenizer(args.model_path, args.device, LLaDAModelLM)
    builder = _make_prompt_builder(
        dataset_name=args.dataset,
        shots=args.shots,
        tokenizer=tokenizer,
        LLaDAEvalHarness=LLaDAEvalHarness,
        set_seed=set_seed,
        generation_length=args.gen_length,
        seed=args.seed,
    )

    rows = [dict(row) for row in _load_requests(args.dataset)]
    task = _load_task(args.dataset)
    task_docs = [dict(task.eval_docs[index]) for index in range(len(rows))]
    _validate_task_alignment(args.dataset, rows, task_docs)
    source_indices = list(range(len(rows)))
    exclusion = None
    if args.dataset == "babilong":
        selected = [item["index"] for item in builder.sampled_fewshot_examples]
        exclusion = verified_plan(rows, args.shots, args.seed, selected)
        excluded = set(exclusion["excluded_indices"])
        source_indices = [index for index in source_indices if index not in excluded]
        rows = [rows[index] for index in source_indices]
        task_docs = [task_docs[index] for index in source_indices]
        print(f"BABILONG_EXCLUSION {json.dumps(exclusion, sort_keys=True)}", flush=True)
    if args.limit is not None:
        count = min(int(args.limit), len(rows))
        rows, task_docs, source_indices = rows[:count], task_docs[:count], source_indices[:count]
    prompts, F, min_prompt_len, max_prompt_len = _build_full_request_ids(
        builder, args.dataset, rows
    )
    filter_name = _task_filter_name(args.dataset)
    stop_tokens = _stop_tokens(task, args.dataset)

    engine = AuditedFullTokenFastdLLMEngine(
        FastdLLMLLaDAEngine,
        device=args.device,
        model=model,
        tokenizer=tokenizer,
        policy=policy,
        cache_budget=CACHE_BUDGET,
        block_length=args.block_length,
        interval=INTRA_REQUEST_CACHE_UPDATE_INTERVAL,
    )

    warmup_started = time.perf_counter()
    if not args.skip_warmup:
        engine.warm_up(1000, args.warmup_steps)
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    warmup_seconds = time.perf_counter() - warmup_started

    requests_path = Path(args.requests_output)
    requests_path.parent.mkdir(parents=True, exist_ok=True)
    request_records = []
    started = time.perf_counter()
    with requests_path.open("w", encoding="utf-8") as request_file:
        for index, (row, doc, full_ids) in enumerate(zip(rows, task_docs, prompts)):
            generated = engine.generate_from_full_ids(
                full_ids,
                F,
                gen_length=args.gen_length,
                mask_id=MASK_ID,
                temperature=TEMPERATURE,
                threshold=DECODING_THRESHOLD,
                remasking=REMASKING,
            )
            generated_ids = generated[-args.gen_length :].detach().cpu().tolist()
            answer = _decode_like_existing_eval(tokenizer, generated_ids, stop_tokens)
            filtered_answer = _filter_one(task, filter_name, answer, doc)
            score_payload = task.process_results(doc, [filtered_answer])
            metric_name, metric_value = next(iter(score_payload.items()))
            metric_value = float(metric_value)

            if len(engine.request_records) != index + 1:
                raise RuntimeError("Engine request audit record count diverged from generation count")
            engine_record = dict(engine.request_records[-1])
            engine_record.update(
                {
                    "request_index": index,
                    "source_index": source_indices[index],
                    "dataset_key": row.get("task_id", index),
                    "equivalent_anchor_ratio": (
                        LAYER_COUNT - int(engine_record["selected_b"])
                    )
                    / LAYER_COUNT,
                    "score": metric_value,
                    "metric": metric_name,
                    "generated_answer": answer,
                    "filtered_answer": filtered_answer,
                }
            )
            request_records.append(engine_record)
            request_file.write(json.dumps(engine_record, ensure_ascii=False) + "\n")
            request_file.flush()
            del generated

    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    generation_seconds = time.perf_counter() - started

    if not request_records:
        raise RuntimeError("No accuracy requests were completed")

    b_values = [int(record["selected_b"]) for record in request_records]
    r_values = [int(record["prefix_ratio_r"]) for record in request_records]
    ratio_values = [(LAYER_COUNT - b) / LAYER_COUNT for b in b_values]
    b_histogram = {
        str(value): int(count)
        for value, count in sorted(collections.Counter(b_values).items())
    }
    ratio_histogram = {
        f"{value:.10f}": int(count)
        for value, count in sorted(collections.Counter(ratio_values).items())
    }
    scores = [float(record["score"]) for record in request_records]

    threshold_audit = {
        **engine.threshold_audit,
        "threshold": DECODING_THRESHOLD,
        "remasking": REMASKING,
        "fixed_transfer_quota_active": False,
        "fixed_step_schedule_active": False,
        "threshold_stop_active": True,
        "threshold_semantics_verified": (
            engine.threshold_audit["selection_calls"] > 0
            and engine.threshold_audit["selected_tokens"] == len(rows) * args.gen_length
            and engine.threshold_audit["selected_tokens"]
            == engine.threshold_audit["threshold_selected_tokens"]
            + engine.threshold_audit["fallback_selected_tokens"]
            and engine.threshold_audit["stopping_events"]
            == len(rows) * (args.gen_length // args.block_length)
        ),
    }

    policy_checks = []
    for record in request_records:
        expected_b = _policy_depth(policy, int(record["prefix_ratio_r"]))
        if expected_b is None or expected_b != int(record["selected_b"]):
            policy_checks.append(
                {
                    "request_index": record["request_index"],
                    "r": record["prefix_ratio_r"],
                    "observed_b": record["selected_b"],
                    "expected_b": expected_b,
                }
            )

    refresh_marker_histogram = {
        str(value): int(count)
        for value, count in sorted(
            collections.Counter(record["official_refresh_markers"] for record in request_records).items()
        )
    }
    deep_refresh_histogram = {
        str(value): int(count)
        for value, count in sorted(
            collections.Counter(
                record["official_deep_prefix_refresh_updates"] for record in request_records
            ).items()
        )
    }
    validation = {
        "all_requests_completed": len(request_records) == len(rows),
        "full_request_prefix_invariant": True,
        "prefix_cache_source": "full_input_ids[:F]",
        "separately_tokenized_prefix_used_for_cache": False,
        "dynamic_policy_lookup_active": len(engine.lookup_counts) > 0,
        "policy_floor_lookup_matches": not policy_checks,
        "official_shallow_prefix_kv_reuse": engine.inter_request_cache_lookups == len(rows),
        "official_boundary_hidden_state_restore": engine.boundary_hidden_state_restore_events > 0,
        "official_deep_prefix_recomputation": engine.deep_prefix_recompute_events == len(rows) * (args.gen_length // args.block_length),
        "official_deep_prefix_refresh_active": engine.deep_prefix_refresh_updates > 0,
        "threshold_semantics_verified": threshold_audit["threshold_semantics_verified"],
        "metric_scores_complete": len(scores) == len(rows),
        "policy_mismatches": policy_checks,
    }
    validation["all_valid"] = (
        validation["all_requests_completed"]
        and validation["full_request_prefix_invariant"]
        and validation["prefix_cache_source"] == "full_input_ids[:F]"
        and validation["separately_tokenized_prefix_used_for_cache"] is False
        and validation["dynamic_policy_lookup_active"]
        and validation["policy_floor_lookup_matches"]
        and validation["official_shallow_prefix_kv_reuse"]
        and validation["official_boundary_hidden_state_restore"]
        and validation["official_deep_prefix_recomputation"]
        and validation["official_deep_prefix_refresh_active"]
        and validation["threshold_semantics_verified"]
        and validation["metric_scores_complete"]
        and not policy_checks
    )
    if not validation["all_valid"]:
        raise RuntimeError(f"Accuracy validation failed: {validation}")

    result = {
        "dataset": args.dataset,
        "seed": int(args.seed),
        "shots": args.shots,
        "batch_size": 1,
        "model": args.model_path,
        "requests": len(request_records),
        "accuracy": statistics.fmean(scores),
        "accuracy_percent": 100.0 * statistics.fmean(scores),
        "accuracy_metric": f"{metric_name},{filter_name}",
        "shared_prefix_length_F": int(F),
        "observed_prefix_ratio_r_range": [min(r_values), max(r_values)],
        "observed_prefix_ratio_r_values": sorted(set(r_values)),
        "selected_b_mean": statistics.fmean(b_values),
        "selected_b_min": min(b_values),
        "selected_b_max": max(b_values),
        "selected_b_histogram": b_histogram,
        "equivalent_anchor_ratio_definition": "(32 - b_i) / 32",
        "equivalent_anchor_ratio_mean": statistics.fmean(ratio_values),
        "equivalent_anchor_ratio_median": statistics.median(ratio_values),
        "equivalent_anchor_ratio_histogram": ratio_histogram,
        "layer_count_L": LAYER_COUNT,
        "average_nfe": engine.total_nfe / len(request_records),
        "total_nfe": int(engine.total_nfe),
        "cache_forward_nfe": int(engine.cache_forward_nfe),
        "decode_forward_nfe": int(engine.decode_forward_nfe),
        "average_decode_iterations": engine.total_decode_iterations / len(request_records),
        "total_decode_iterations": int(engine.total_decode_iterations),
        "refresh_count": {
            "official_intra_request_refresh_markers": sum(
                record["official_refresh_markers"] for record in request_records
            ),
            "official_intra_request_refresh_marker_histogram": refresh_marker_histogram,
            "official_deep_prefix_refresh_updates": int(engine.deep_prefix_refresh_updates),
            "official_deep_prefix_refresh_update_histogram": deep_refresh_histogram,
            "official_refresh_interval": INTRA_REQUEST_CACHE_UPDATE_INTERVAL,
        },
        "generated_length": args.gen_length,
        "block_length": args.block_length,
        "temperature": TEMPERATURE,
        "mask_id": MASK_ID,
        "remasking": REMASKING,
        "decoding_threshold": DECODING_THRESHOLD,
        "intra_request_cache_update_interval": INTRA_REQUEST_CACHE_UPDATE_INTERVAL,
        "profile_threshold": 0.97,
        "cache_budget": CACHE_BUDGET,
        "prompt_length_min": min_prompt_len,
        "prompt_length_max": max_prompt_len,
        "fewshot_indices": [item["index"] for item in builder.sampled_fewshot_examples],
        "babilong_exclusion": exclusion,
        "stop_tokens": stop_tokens,
        "policy_path": str(Path(args.policy_path).resolve()),
        "policy_metadata": policy_metadata,
        "official_commit": args.official_commit,
        "requests_output": str(requests_path.resolve()),
        "warmup_seconds": warmup_seconds,
        "generation_seconds": generation_seconds,
        "request_nfe_histogram": {
            str(value): int(count)
            for value, count in sorted(collections.Counter(record["request_nfe"] for record in request_records).items())
        },
        "threshold_decoding": threshold_audit,
        "validation": validation,
        "implementation_deviations": [
            "The ACache full-request token sequence full_input_ids[:F] is supplied directly; separately tokenized prefix_affix_token_ids and BiCache retokenized boundaries are not used for cache construction.",
            "The official fixed-step/fixed-transfer loop is replaced by the existing Nano-vDLLM threshold=0.9 selection and block stopping logic; all BiCache cache and refresh mechanisms remain official.",
            "For BABILong, prompt construction is taken from llada/eval_ACache.py via acache_eval_shared.ACacheEvalHarnessMixin; the Nano-vDLLM BABILong prompt builder is not used.",
            "Only lm-eval postprocessing and task scoring are added around the validated throughput engine.",
        ],
        "prompt_source": getattr(builder, "prompt_source", "nano-vdllm/eval_llada.py"),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("BICACHE_ACCURACY_RESULT=" + json.dumps(result, sort_keys=True), flush=True)
    return result


def _parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-root", default=str(DEFAULT_OFFICIAL_ROOT))
    parser.add_argument("--official-commit", default="82a8d0122807daf58ceacd6a121201b8355f46c1")
    parser.add_argument("--model-path", default=MODEL_ID)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--policy-path", required=True)
    parser.add_argument("--dataset", choices=("gsm8k", "mbpp", "babilong"), required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--shots", type=int, choices=(1,), default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--gen-length", type=int, default=GEN_LENGTH)
    parser.add_argument("--block-length", type=int, default=BLOCK_LENGTH)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--skip-warmup", action="store_true")
    parser.add_argument("--output", required=True)
    parser.add_argument("--requests-output", required=True)
    return parser


def main():
    args = _parser().parse_args()
    if args.block_length <= 0:
        raise ValueError("block-length must be positive")
    if args.gen_length <= 0 or args.gen_length % args.block_length != 0:
        raise ValueError("gen-length must be positive and divisible by block length")
    _run(args)


if __name__ == "__main__":
    main()
