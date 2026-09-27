#!/usr/bin/env python3
"""Dream 1-shot accuracy through the official BiCache + Fast-dLLM adapter.

Prompt construction is borrowed from the existing Dream ACache harness only
to reproduce its workload.  The model, cache lookup, profiler policy, hidden
state boundary, deep-prefix recomputation, periodic refresh, and decoding loop
come from the official BiCache checkout in ``--official-root``.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from babilong_exclusion import verified_plan


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_OFFICIAL_ROOT = Path(__file__).resolve().parent / "third_party" / "BiCache"
MODEL_ID = "Dream-org/Dream-v0-Instruct-7B"
THRESHOLD = 0.9
INTER_REQUEST_REFRESH_INTERVAL = 16
TEMPERATURE = 0.0
MASK_ID = 151666


def _import_official(official_root: Path):
    official_root = official_root.resolve()
    sys.path.insert(0, str(official_root))
    from bicache import FastdLLMDreamEngine  # type: ignore
    from model.dream import DreamModel  # type: ignore
    return FastdLLMDreamEngine, DreamModel


def _load_model_and_tokenizer(model_path: str, device: str, DreamModel):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model = DreamModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to(device).eval()
    return model, tokenizer


class DreamACachePromptBuilder:
    """The prefix portion of ``dream/eval_ACache.py`` without loading its model."""

    def _uses_chat_template_for_prompts(self) -> bool:
        return True

    def _build_prefix_input_ids(self, question):
        input_ids, _, _, _, _ = self._build_input_ids_with_affix(
            question,
            self.default_affix_state,
        )
        return input_ids


def _make_prompt_builder(
    dataset: str,
    tokenizer,
    seed: int,
    generation_length: int,
    shots: int = 1,
):
    from acache_eval_shared import ACacheEvalHarnessMixin, set_seed

    class _Builder(ACacheEvalHarnessMixin, DreamACachePromptBuilder):
        pass

    builder = _Builder.__new__(_Builder)
    set_seed(int(seed))
    builder.seed = int(seed)
    builder.fewshot_num_examples = int(shots)
    builder.shared_prefix_text = ""
    builder.shared_prefix_extra_text = ""
    builder.shared_prefix_role = "system"
    builder.shared_prefix_use_chat_template = True
    builder.affix_type = "prefix"
    builder.gen_length = int(generation_length)
    builder.mask_id = MASK_ID
    builder.tokenizer = tokenizer

    if dataset == "gsm8k":
        builder.fewshot_dataset_path = "gsm8k"
        builder.fewshot_dataset_name = "main"
        builder.fewshot_split = "train"
        builder.fewshot_question_key = "question"
        builder.fewshot_answer_key = "answer"
        builder.prompt_style = "gsm8k"
    elif dataset == "mbpp":
        builder.fewshot_dataset_path = "google-research-datasets/mbpp"
        builder.fewshot_dataset_name = "full"
        builder.fewshot_split = "prompt"
        builder.fewshot_question_key = "text"
        builder.fewshot_answer_key = "code"
        builder.prompt_style = "mbpp"
    elif dataset == "babilong":
        builder.fewshot_dataset_path = "RMT-team/babilong-1k-samples"
        builder.fewshot_dataset_name = "0k"
        builder.fewshot_split = "qa1"
        builder.fewshot_question_key = "question"
        builder.fewshot_answer_key = "target"
        builder.prompt_style = "babilong_qa1"
    else:
        raise ValueError(f"Unsupported Dream workload dataset: {dataset}")

    prefix_messages, sampled_examples = builder._build_prefix_fewshot_messages(
        state_name="default"
    )
    # This is deliberately the separately tokenized affix used by the
    # existing ACache harness to define F.  The generation runner below still
    # takes its cache tokens from full_input_ids[:F].
    prefix_affix_token_ids = builder._tokenize_chat_messages(
        prefix_messages,
        add_generation_prompt=False,
    )
    affix_state = builder._make_empty_affix_state(state_name="default")
    affix_state.update(
        {
            "sampled_fewshot_examples": sampled_examples,
            "prefix_fewshot_messages": prefix_messages,
            "prefix_affix_token_ids": prefix_affix_token_ids,
            "affix_length": len(prefix_affix_token_ids),
        }
    )
    builder.default_affix_state = affix_state
    builder.sampled_fewshot_examples = sampled_examples
    builder.prefix_fewshot_messages = prefix_messages
    builder.prefix_affix_token_ids = [int(token) for token in prefix_affix_token_ids]
    builder.prefix_affix_len = len(prefix_affix_token_ids)
    builder.prompt_source = "dream/eval_ACache.py prompt mixin"
    return builder


def _load_requests(dataset: str):
    from datasets import load_dataset

    if dataset == "gsm8k":
        return [dict(row) for row in load_dataset("gsm8k", "main", split="test")]
    if dataset == "mbpp":
        return [
            dict(row)
            for row in load_dataset(
                "google-research-datasets/mbpp", "full", split="test"
            )
        ]
    if dataset == "babilong":
        return [
            dict(row)
            for row in load_dataset(
                "RMT-team/babilong-1k-samples", "0k", split="qa1"
            )
        ]
    raise ValueError(f"Unsupported Dream workload dataset: {dataset}")


def _build_full_request_ids(builder, dataset: str, rows: list[dict]):
    F = int(builder.prefix_affix_len)
    if F <= 0:
        raise RuntimeError("The 1-shot Dream ACache prefix is empty; cannot run BiCache.")

    full_requests = []
    reference_prefix = None
    separate_mismatches = []
    min_length = None
    max_length = None
    for request_index, row in enumerate(rows):
        req = SimpleNamespace(doc=row, args=(), task_name=dataset)
        question = builder._extract_question_text(req)
        full_ids = [int(token) for token in builder._build_prefix_input_ids(question)]
        if len(full_ids) < F:
            raise RuntimeError(
                f"Request {request_index} has {len(full_ids)} tokens, below F={F}."
            )
        current_prefix = tuple(full_ids[:F])
        if reference_prefix is None:
            reference_prefix = current_prefix
        elif current_prefix != reference_prefix:
            mismatch = next(
                (
                    position
                    for position, (left, right) in enumerate(
                        zip(reference_prefix, current_prefix)
                    )
                    if left != right
                ),
                min(len(reference_prefix), len(current_prefix)),
            )
            raise RuntimeError(
                "Request-dependent token inside full_input_ids[:F]: "
                f"request={request_index}, position={mismatch}."
            )

        separately_tokenized = builder.prefix_affix_token_ids
        mismatch_position = next(
            (
                position
                for position, (left, right) in enumerate(
                    zip(full_ids[:F], separately_tokenized)
                )
                if left != right
            ),
            None,
        )
        if mismatch_position is None and len(full_ids[:F]) != len(separately_tokenized):
            mismatch_position = min(len(full_ids[:F]), len(separately_tokenized))
        if mismatch_position is not None:
            separate_mismatches.append(
                {
                    "request_index": request_index,
                    "position": int(mismatch_position),
                    "full_token": (
                        int(full_ids[mismatch_position])
                        if mismatch_position < len(full_ids[:F])
                        else None
                    ),
                    "separate_prefix_token": (
                        int(separately_tokenized[mismatch_position])
                        if mismatch_position < len(separately_tokenized)
                        else None
                    ),
                }
            )

        full_requests.append(full_ids)
        min_length = len(full_ids) if min_length is None else min(min_length, len(full_ids))
        max_length = len(full_ids) if max_length is None else max(max_length, len(full_ids))

    if reference_prefix is None:
        raise RuntimeError("No requests were loaded for the Dream workload.")
    return {
        "full_input_ids": full_requests,
        "F": F,
        "min_prompt_length": min_length,
        "max_prompt_length": max_length,
        "full_prefix_identical": True,
        "separately_tokenized_prefix_length": len(builder.prefix_affix_token_ids),
        "separately_tokenized_prefix_exact_match": not separate_mismatches,
        "separately_tokenized_prefix_mismatch_count": len(separate_mismatches),
        "separately_tokenized_prefix_first_mismatch": (
            separate_mismatches[0] if separate_mismatches else None
        ),
        "separately_tokenized_prefix_difference_reason": (
            "none; separately tokenized prefix equals full_input_ids[:F]"
            if not separate_mismatches
            else "the separately tokenized affix is retained only to define F; "
            "BiCache uses the true full-request slice full_input_ids[:F]"
        ),
    }


def _load_task(dataset: str):
    # lm-eval imports MBPP's optional code-eval metric while loading the task,
    # even though this workload uses the ``none`` response filter.  The
    # existing evaluation environment explicitly enables that import.
    os.environ.setdefault("HF_ALLOW_CODE_EVAL", "1")
    from lm_eval.tasks import TaskManager

    manager = TaskManager(include_path=str(REPO_ROOT / "lm_eval_tasks"))
    return manager.load_task_or_group([dataset])[dataset]


def _validate_task_alignment(dataset: str, rows: list[dict], docs: list[dict]):
    if len(rows) != len(docs):
        raise RuntimeError(f"{dataset}: workload/task lengths differ: {len(rows)} vs {len(docs)}")
    for index, (row, doc) in enumerate(zip(rows, docs)):
        if dataset == "gsm8k":
            same = row.get("question") == doc.get("question")
            label = "question"
        elif dataset == "mbpp":
            same = row.get("task_id") == doc.get("task_id")
            label = "task_id"
        else:
            same = (
                row.get("input") == doc.get("input")
                and row.get("question") == doc.get("question")
                and row.get("target") == doc.get("target")
            )
            label = "input/question/target"
        if not same:
            raise RuntimeError(
                f"{dataset}: workload/task order mismatch at request {index} ({label})."
            )


def _filter_one(task, filter_name: str, response: str, doc: dict) -> str:
    ensemble = next((item for item in task._filters if item.name == filter_name), None)
    if ensemble is None:
        raise RuntimeError(f"Task filter {filter_name!r} is unavailable for {task}")
    responses = [[response]]
    for filter_factory in ensemble.filters:
        responses = list(filter_factory().apply(responses, [doc]))
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


def _decode_like_dream(tokenizer, generated_ids, stop_tokens: list[str]) -> str:
    answer = tokenizer.decode(generated_ids, skip_special_tokens=False)
    eos_token = getattr(tokenizer, "eos_token", None)
    if isinstance(eos_token, str) and eos_token:
        answer = answer.split(eos_token)[0]
    for stop_seq in stop_tokens:
        if stop_seq in answer:
            answer = answer.split(stop_seq)[0]
    return answer


def _policy_depth(policy: dict[int, int], ratio: int):
    eligible = [key for key in policy if int(key) <= int(ratio)]
    return int(policy[max(eligible)]) if eligible else None


def _default_config(dataset: str):
    if dataset == "babilong":
        return {"gen_length": 8, "steps": 8, "block_length": 8}
    return {"gen_length": 256, "steps": 128, "block_length": 32}


def _run(args):
    if abs(float(args.threshold) - THRESHOLD) > 1e-12:
        raise ValueError("This validated Dream BiCache runner requires threshold=0.9.")
    if abs(float(args.temperature) - TEMPERATURE) > 1e-12:
        raise ValueError("This validated Dream workload requires temperature=0.0.")

    config = _default_config(args.dataset)
    gen_length = config["gen_length"] if args.gen_length is None else int(args.gen_length)
    steps = config["steps"] if args.steps is None else int(args.steps)
    block_length = config["block_length"] if args.block_length is None else int(args.block_length)
    if gen_length % block_length != 0 or steps % (gen_length // block_length) != 0:
        raise ValueError(
            f"Invalid Dream Fast-dLLM settings: gen_length={gen_length}, "
            f"steps={steps}, block_length={block_length}."
        )
    if args.smoke and args.limit != 1:
        raise ValueError("The smoke validation must run exactly one request.")

    FastdLLMDreamEngine, DreamModel = _import_official(Path(args.official_root))
    policy_payload = json.loads(Path(args.policy_path).read_text(encoding="utf-8"))
    policy = {int(key): int(value) for key, value in policy_payload["policy"].items()}
    if policy_payload.get("threshold") != 0.97:
        raise RuntimeError("Dream policy was not profiled at the official cosine threshold 0.97.")

    model, tokenizer = _load_model_and_tokenizer(args.model_path, args.device, DreamModel)
    layer_count = int(getattr(model.config, "num_hidden_layers", 0))
    if layer_count <= 0:
        raise RuntimeError("Loaded Dream model config has no valid num_hidden_layers.")
    builder = _make_prompt_builder(
        args.dataset,
        tokenizer,
        args.seed,
        gen_length,
        shots=args.shots,
    )
    rows = _load_requests(args.dataset)
    task = _load_task(args.dataset)
    task_docs = [dict(doc) for doc in task.eval_docs]
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
    prompt_payload = _build_full_request_ids(builder, args.dataset, rows)
    full_requests = prompt_payload.pop("full_input_ids")
    _validate_task_alignment(args.dataset, rows, task_docs)
    filter_name = {
        "gsm8k": "flexible-extract",
        "mbpp": "none",
        "babilong": "remove_whitespace",
    }[args.dataset]
    stop_tokens = _stop_tokens(task, args.dataset)

    engine = FastdLLMDreamEngine(
        device=args.device,
        model=model,
        tokenizer=tokenizer,
        number_of_inter_request_caching_layer=policy,
        intra_request_cache_update_interval=int(args.interval),
        cache_budget=int(args.cache_budget),
        show_speed=False,
        block_length=block_length,
        threshold=float(args.threshold),
    )

    requests_output = Path(args.requests_output or str(args.output).replace(".json", "_requests.jsonl"))
    requests_output.parent.mkdir(parents=True, exist_ok=True)
    records = []
    started = time.perf_counter()
    with requests_output.open("w", encoding="utf-8") as handle:
        for request_index, (row, doc, full_ids) in enumerate(
            zip(rows, task_docs, full_requests)
        ):
            input_tensor = torch.tensor(full_ids, dtype=torch.long, device=args.device)
            generated, nfe, audit = engine.generate_token_ids(
                input_tensor,
                int(prompt_payload["F"]),
                steps=steps,
                gen_length=gen_length,
                mask_id=int(args.mask_id),
                temperature=float(args.temperature),
                block_length=block_length,
                threshold=float(args.threshold),
            )
            generated_ids = generated[-gen_length:].detach().cpu().tolist()
            answer = _decode_like_dream(tokenizer, generated_ids, stop_tokens)
            filtered_answer = _filter_one(task, filter_name, answer, doc)
            score_payload = task.process_results(doc, [filtered_answer])
            metric_name, metric_value = next(iter(score_payload.items()))
            if int(audit.get("layer_count", -1)) != layer_count:
                raise RuntimeError(
                    "Dream BiCache audit layer count disagrees with the loaded model config: "
                    f"audit={audit.get('layer_count')}, config={layer_count}."
                )
            record = {
                "request_index": request_index,
                "source_index": source_indices[request_index],
                "dataset_key": row.get("task_id", request_index),
                "F": int(prompt_payload["F"]),
                "prefix_ratio_r": int(audit["prefix_ratio_r"]),
                "selected_b": int(audit["selected_b"]),
                "equivalent_anchor_ratio": (layer_count - int(audit["selected_b"]))
                / layer_count,
                "nfe": int(nfe),
                "periodic_refresh_count": int(audit["periodic_refresh_events"]),
                "refresh_markers": int(audit["refresh_markers"]),
                "audit": audit,
                "score": float(metric_value),
                "metric": metric_name,
                "generated_answer": answer,
                "filtered_answer": filtered_answer,
            }
            records.append(record)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            del generated, input_tensor

    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    if not records:
        raise RuntimeError("No Dream BiCache requests completed.")

    b_values = [int(record["selected_b"]) for record in records]
    r_values = [int(record["prefix_ratio_r"]) for record in records]
    ratios = [float(record["equivalent_anchor_ratio"]) for record in records]
    nfe_values = [int(record["nfe"]) for record in records]
    refresh_values = [int(record["periodic_refresh_count"]) for record in records]
    b_hist = {str(value): count for value, count in sorted(collections.Counter(b_values).items())}
    r_hist = {str(value): count for value, count in sorted(collections.Counter(r_values).items())}
    refresh_hist = {
        str(value): count
        for value, count in sorted(collections.Counter(refresh_values).items())
    }

    policy_mismatches = []
    for record in records:
        expected_b = _policy_depth(policy, int(record["prefix_ratio_r"]))
        if expected_b != int(record["selected_b"]):
            policy_mismatches.append(
                {
                    "request_index": record["request_index"],
                    "r": record["prefix_ratio_r"],
                    "observed_b": record["selected_b"],
                    "expected_b": expected_b,
                }
            )

    threshold_checks = {
        "threshold": float(args.threshold),
        "threshold_applied_after_dream_shift": all(
            bool(record["audit"]["threshold_applied_after_shift"]) for record in records
        ),
        "dream_shift_alignment": all(
            record["audit"]["dream_shift_alignment"]
            == "torch.cat([logits[:, :1], logits[:, :-1]], dim=1)"
            for record in records
        ),
        "fixed_transfer_quota_active": False,
        "fixed_step_schedule_active": False,
        "selected_tokens": sum(int(record["audit"]["selected_tokens"]) for record in records),
        "expected_selected_tokens": len(records) * gen_length,
        "stopping_events": sum(int(record["audit"]["stopping_events"]) for record in records),
        "expected_stopping_events": len(records) * (gen_length // block_length),
    }
    threshold_checks["semantics_verified"] = (
        threshold_checks["threshold"] == THRESHOLD
        and threshold_checks["threshold_applied_after_dream_shift"]
        and threshold_checks["dream_shift_alignment"]
        and threshold_checks["selected_tokens"] == threshold_checks["expected_selected_tokens"]
        and threshold_checks["stopping_events"] == threshold_checks["expected_stopping_events"]
    )

    validation = {
        "all_requests_completed": len(records) == len(rows),
        "full_request_prefix_invariant": bool(prompt_payload["full_prefix_identical"]),
        "F_defined_by_separately_tokenized_prefix": (
            int(prompt_payload["F"])
            == int(prompt_payload["separately_tokenized_prefix_length"])
        ),
        "cache_prefix_source": "full_input_ids[:F]",
        "separately_tokenized_prefix_used_for_cache": False,
        "separately_tokenized_prefix_exact_match": bool(
            prompt_payload["separately_tokenized_prefix_exact_match"]
        ),
        "dynamic_r_to_b_lookup_active": bool(records) and not policy_mismatches,
        "shallow_prefix_kv_reuse": all(
            bool(record["audit"]["shallow_prefix_kv_reused"]) for record in records
        ),
        "boundary_hidden_state_restore": all(
            bool(record["audit"]["boundary_hidden_state_restored"]) for record in records
        ),
        "deep_prefix_recomputation": all(
            int(record["selected_b"]) >= layer_count
            or int(record["audit"]["deep_prefix_recomputation_events"]) >= gen_length // block_length
            for record in records
        ),
        "periodic_refresh_implementation_active": all(
            int(record["audit"]["intra_request_cache_update_interval"])
            == int(args.interval)
            for record in records
        ),
        "periodic_refresh_observed_count": sum(refresh_values),
        "dream_shifted_decoding_order": threshold_checks["dream_shift_alignment"],
        "threshold_0p9_semantics": threshold_checks["semantics_verified"],
        "policy_mismatches": policy_mismatches,
    }
    validation["all_valid"] = all(
        [
            validation["all_requests_completed"],
            validation["full_request_prefix_invariant"],
            validation["F_defined_by_separately_tokenized_prefix"],
            validation["cache_prefix_source"] == "full_input_ids[:F]",
            validation["separately_tokenized_prefix_used_for_cache"] is False,
            validation["dynamic_r_to_b_lookup_active"],
            validation["shallow_prefix_kv_reuse"],
            validation["boundary_hidden_state_restore"],
            validation["deep_prefix_recomputation"],
            validation["periodic_refresh_implementation_active"],
            validation["dream_shifted_decoding_order"],
            validation["threshold_0p9_semantics"],
        ]
    )
    if args.smoke:
        validation["periodic_refresh_observed_in_smoke"] = sum(refresh_values) > 0
        validation["all_valid"] = validation["all_valid"] and validation[
            "periodic_refresh_observed_in_smoke"
        ]

    metric_name = records[0]["metric"]
    scores = [float(record["score"]) for record in records]
    result = {
        "dataset": args.dataset,
        "shots": int(args.shots),
        "seed": int(args.seed),
        "model": args.model_path,
        "batch_size": 1,
        "requests": len(records),
        "accuracy": statistics.fmean(scores),
        "accuracy_percent": 100.0 * statistics.fmean(scores),
        "accuracy_metric": metric_name,
        "shared_prefix_length_F": int(prompt_payload["F"]),
        "separately_tokenized_prefix_length": int(
            prompt_payload["separately_tokenized_prefix_length"]
        ),
        "separately_tokenized_prefix_exact_match": bool(
            prompt_payload["separately_tokenized_prefix_exact_match"]
        ),
        "separately_tokenized_prefix_first_mismatch": prompt_payload[
            "separately_tokenized_prefix_first_mismatch"
        ],
        "separately_tokenized_prefix_difference_reason": prompt_payload[
            "separately_tokenized_prefix_difference_reason"
        ],
        "observed_prefix_ratio_r_range": [min(r_values), max(r_values)],
        "observed_prefix_ratio_r_histogram": r_hist,
        "selected_b_mean": statistics.fmean(b_values),
        "selected_b_min": min(b_values),
        "selected_b_max": max(b_values),
        "selected_b_histogram": b_hist,
        "equivalent_anchor_ratio_definition": f"({layer_count} - b_i) / {layer_count}",
        "equivalent_anchor_ratio_mean": statistics.fmean(ratios),
        "equivalent_anchor_ratio_median": statistics.median(ratios),
        "layer_count_L": layer_count,
        "average_nfe": statistics.fmean(nfe_values),
        "total_nfe": sum(nfe_values),
        "nfe_histogram": {
            str(value): count
            for value, count in sorted(collections.Counter(nfe_values).items())
        },
        "refresh_count_total": sum(refresh_values),
        "refresh_count_mean": statistics.fmean(refresh_values),
        "refresh_count_histogram": refresh_hist,
        "generation_length": gen_length,
        "steps": steps,
        "block_length": block_length,
        "temperature": float(args.temperature),
        "decoding_threshold": float(args.threshold),
        "intra_request_cache_update_interval": int(args.interval),
        "mask_id": int(args.mask_id),
        "prompt_source": builder.prompt_source,
        "sampled_fewshot_examples": builder.sampled_fewshot_examples,
        "babilong_exclusion": exclusion,
        "prompt_length_range": [
            int(prompt_payload["min_prompt_length"]),
            int(prompt_payload["max_prompt_length"]),
        ],
        "policy_metadata": policy_payload,
        "threshold_validation": threshold_checks,
        "validation": validation,
        "validation_status": "PASS" if validation["all_valid"] else "FAIL",
        "elapsed_seconds": elapsed,
        "requests_output": str(requests_output),
        "implementation_deviations": [
            "The Dream model implementation is added under the official BiCache checkout because the released artifact supports LLaDA only.",
            "Periodic refresh uses the official BiCache boundary restore and deep-prefix recomputation path; Dream's native dual-cache API is used for block updates.",
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    if not validation["all_valid"]:
        raise RuntimeError(f"Dream BiCache validation failed: {validation}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-root", type=Path, default=DEFAULT_OFFICIAL_ROOT)
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--model-path", default=MODEL_ID)
    parser.add_argument("--dataset", choices=("gsm8k", "mbpp", "babilong"), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shots", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--requests-output", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--gen-length", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--block-length", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--interval", type=int, default=INTER_REQUEST_REFRESH_INTERVAL)
    parser.add_argument("--cache-budget", type=int, default=5000)
    parser.add_argument("--mask-id", type=int, default=MASK_ID)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    _run(args)


if __name__ == "__main__":
    main()
