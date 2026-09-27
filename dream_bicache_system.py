#!/usr/bin/env python3
"""Dream BiCache + threshold Fast-dLLM system throughput.

This is the Dream counterpart of ``bicache_fast_dllm_system.py``.  It keeps
the official Dream BiCache policy, inter-request cache, boundary state,
periodic deep-prefix refresh, and threshold transfer loop, and measures the
same batch-one workload boundary as the LLaDA system runner: prompt
tokenization through generated token IDs synchronized to CPU.  Model loading,
policy loading, and warm-up are outside the timed interval.
"""

from __future__ import annotations

import collections
import argparse
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from dream_bicache_accuracy import (
    DEFAULT_OFFICIAL_ROOT,
    MASK_ID,
    MODEL_ID,
    TEMPERATURE,
    THRESHOLD,
    _build_full_request_ids,
    _import_official,
    _load_model_and_tokenizer,
    _load_requests,
    _make_prompt_builder,
)


REPO_ROOT = Path(__file__).resolve().parent
GEN_LENGTH = 256
BLOCK_LENGTH = 32
STEPS = 128
PROFILE_THRESHOLD = 0.97
INTER_REQUEST_REFRESH_INTERVAL = 16
CACHE_BUDGET = 5000

# Existing Dream prefix system measurements.  These are the matching
# no-cache Fast-dLLM and ACache reference runs, at BS=1, seed 0, and use the
# same prompt placement as this runner.
SYSTEM_BASELINES = {
    ("gsm8k", 1): {"fast_dllm": 129.4212393178846, "acache": 125.11658646841127},
    ("gsm8k", 2): {"fast_dllm": 120.21675833089509, "acache": 120.28201761823395},
    ("gsm8k", 4): {"fast_dllm": 103.320180444791, "acache": 116.55363918447985},
    ("mbpp", 1): {"fast_dllm": 194.59305631580176, "acache": 189.94025717386685},
    ("mbpp", 2): {"fast_dllm": 174.86622783166655, "acache": 183.26989945280863},
    ("mbpp", 4): {"fast_dllm": 136.91408835035173, "acache": 171.09821372719802},
}


def _load_policy(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    policy = {int(key): int(value) for key, value in payload["policy"].items()}
    if not policy or min(policy) != 0:
        raise RuntimeError("Dream BiCache policy must contain an r=0 entry.")
    if payload.get("threshold") != PROFILE_THRESHOLD:
        raise RuntimeError(
            "Dream BiCache policy was not profiled at the official cosine "
            f"threshold {PROFILE_THRESHOLD}."
        )
    return policy, payload


def _policy_depth(policy: dict[int, int], ratio: int):
    eligible = [key for key in policy if int(key) <= int(ratio)]
    return int(policy[max(eligible)]) if eligible else None


def _run(args):
    if args.dataset not in {"gsm8k", "mbpp"}:
        raise ValueError("Dream system throughput supports GSM8K and MBPP.")
    if args.shots not in {1, 2, 4}:
        raise ValueError("Dream system throughput supports 1, 2, or 4 shots.")
    if abs(float(args.threshold) - THRESHOLD) > 1e-12:
        raise ValueError("The validated Dream BiCache runner requires threshold=0.9.")
    if abs(float(args.temperature) - TEMPERATURE) > 1e-12:
        raise ValueError("The validated Dream workload requires temperature=0.0.")
    if args.gen_length <= 0 or args.gen_length % BLOCK_LENGTH != 0:
        raise ValueError("gen-length must be positive and divisible by block length.")
    if args.steps <= 0 or args.steps % (args.gen_length // BLOCK_LENGTH) != 0:
        raise ValueError(
            "steps must be divisible by the number of Dream Fast-dLLM blocks."
        )

    official_root = Path(args.official_root).resolve()
    FastdLLMDreamEngine, DreamModel = _import_official(official_root)
    policy, policy_metadata = _load_policy(Path(args.policy_path))
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("Dream BiCache workload requires CUDA; no CUDA device is visible.")

    model, tokenizer = _load_model_and_tokenizer(
        args.model_path,
        args.device,
        DreamModel,
    )
    builder = _make_prompt_builder(
        args.dataset,
        tokenizer,
        args.seed,
        args.gen_length,
        shots=args.shots,
    )
    rows = _load_requests(args.dataset)
    if args.limit is not None:
        rows = rows[: min(int(args.limit), len(rows))]
    if not rows:
        raise RuntimeError("No Dream workload requests were loaded.")

    # Build once before timing to audit the shared prefix and record prompt
    # lengths.  The exact same prompt construction is repeated inside the
    # timed tokenization interval below, matching the LLaDA system runner.
    prompt_payload = _build_full_request_ids(builder, args.dataset, rows)
    full_requests = prompt_payload.pop("full_input_ids")
    prefix_len = int(prompt_payload["F"])

    engine = FastdLLMDreamEngine(
        device=args.device,
        model=model,
        tokenizer=tokenizer,
        number_of_inter_request_caching_layer=policy,
        intra_request_cache_update_interval=INTER_REQUEST_REFRESH_INTERVAL,
        cache_budget=CACHE_BUDGET,
        show_speed=False,
        block_length=BLOCK_LENGTH,
        threshold=float(args.threshold),
    )

    warmup_started = time.perf_counter()
    if not args.skip_warmup:
        engine.warm_up(1000, args.warmup_steps)
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    warmup_seconds = time.perf_counter() - warmup_started

    started = time.time()
    tokenization_started = time.perf_counter()
    timed_prompts: list[list[int]] = []
    for row_index, row in enumerate(rows):
        request = SimpleNamespace(doc=dict(row), args=(), task_name=args.dataset)
        question = builder._extract_question_text(request)
        full_ids = [
            int(token) for token in builder._build_prefix_input_ids(question)
        ]
        if len(full_ids) < prefix_len:
            raise RuntimeError(
                f"Timed request {row_index} is shorter than F={prefix_len}."
            )
        if tuple(full_ids[:prefix_len]) != tuple(full_requests[row_index][:prefix_len]):
            raise RuntimeError(
                "Timed prompt construction diverged from the audited full request "
                f"at request {row_index}."
            )
        timed_prompts.append(full_ids)
    tokenization_seconds = time.perf_counter() - tokenization_started

    generation_started = time.perf_counter()
    for full_ids in timed_prompts:
        input_tensor = torch.tensor(
            full_ids,
            dtype=torch.long,
            device=args.device,
        )
        generated, _, _ = engine.generate_token_ids(
            input_tensor,
            prefix_len,
            steps=args.steps,
            gen_length=args.gen_length,
            mask_id=MASK_ID,
            temperature=float(args.temperature),
            block_length=BLOCK_LENGTH,
            threshold=float(args.threshold),
        )
        # Match the ACache/LLaDA timing boundary: generated IDs are copied to
        # CPU before the timer ends.
        _ = generated[-args.gen_length :].detach().cpu().tolist()
        del generated, input_tensor
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    generation_seconds = time.perf_counter() - generation_started
    ended = time.time()

    total_tokens = len(timed_prompts) * args.gen_length
    throughput = total_tokens / (ended - started)
    audits = list(engine.request_audits)
    if len(audits) != len(timed_prompts):
        raise RuntimeError(
            f"Expected {len(timed_prompts)} Dream request audits, got {len(audits)}."
        )

    r_values = [int(audit["prefix_ratio_r"]) for audit in audits]
    b_values = [int(audit["selected_b"]) for audit in audits]
    nfe_values = [int(audit["nfe"]) for audit in audits]
    lookup_counts = collections.Counter(
        (int(audit["prefix_ratio_r"]), int(audit["selected_b"]))
        for audit in audits
    )
    lookup_rows = [
        {"r": r, "b": b, "requests": count}
        for (r, b), count in sorted(lookup_counts.items())
    ]
    if not lookup_rows:
        raise RuntimeError("No Dream BiCache policy lookup was recorded.")

    expected_blocks = len(timed_prompts) * (args.gen_length // BLOCK_LENGTH)
    threshold_audit = {
        "threshold": float(args.threshold),
        "threshold_applied_after_dream_shift": all(
            bool(audit["threshold_applied_after_shift"]) for audit in audits
        ),
        "dream_shift_alignment": all(
            audit["dream_shift_alignment"]
            == "torch.cat([logits[:, :1], logits[:, :-1]], dim=1)"
            for audit in audits
        ),
        "fixed_transfer_quota_active": False,
        "fixed_step_schedule_active": False,
        "selected_tokens": sum(int(audit["selected_tokens"]) for audit in audits),
        "expected_selected_tokens": total_tokens,
        "stopping_events": sum(int(audit["stopping_events"]) for audit in audits),
        "expected_stopping_events": expected_blocks,
    }
    threshold_audit["semantics_verified"] = (
        threshold_audit["threshold"] == THRESHOLD
        and threshold_audit["threshold_applied_after_dream_shift"]
        and threshold_audit["dream_shift_alignment"]
        and threshold_audit["selected_tokens"] == total_tokens
        and threshold_audit["stopping_events"] == expected_blocks
    )
    if not threshold_audit["semantics_verified"]:
        raise RuntimeError(f"Dream threshold audit failed: {threshold_audit}")

    policy_mismatches = [
        {
            "request_index": index,
            "r": int(audit["prefix_ratio_r"]),
            "observed_b": int(audit["selected_b"]),
            "expected_b": _policy_depth(policy, int(audit["prefix_ratio_r"])),
        }
        for index, audit in enumerate(audits)
        if _policy_depth(policy, int(audit["prefix_ratio_r"]))
        != int(audit["selected_b"])
    ]
    if policy_mismatches:
        raise RuntimeError(f"Dream policy lookup mismatches: {policy_mismatches[:3]}")

    baseline = SYSTEM_BASELINES[(args.dataset, args.shots)]
    comparison = {
        "fast_dllm_throughput_generated_tokens_per_second": baseline["fast_dllm"],
        "acache_throughput_generated_tokens_per_second": baseline["acache"],
        "delta_vs_fast_dllm_generated_tokens_per_second": throughput - baseline["fast_dllm"],
        "delta_vs_acache_generated_tokens_per_second": throughput - baseline["acache"],
        "percent_vs_fast_dllm": 100.0 * (throughput / baseline["fast_dllm"] - 1.0),
        "percent_vs_acache": 100.0 * (throughput / baseline["acache"] - 1.0),
        "baseline_source": (
            "system_latest_20260907_results/throughput/dream/*_prefix_b1/*.log"
        ),
    }

    audit = {
        "policy_lookup_active": True,
        "policy_entries_used": lookup_rows,
        "policy_entry_count": len(lookup_rows),
        "engine_cache_lookups": int(engine.cache_lookups),
        "engine_cache_generation_events": int(engine.cache_generation_events),
        "engine_cached_tokens": int(engine.num_cache_tokens),
        "official_deep_prefix_recompute_events": int(engine.deep_prefix_recompute_events),
        "official_deep_prefix_refresh_updates": int(engine.periodic_refresh_events),
        "official_dual_cache_builds": int(engine.dual_cache_update_events),
        "official_deep_refresh_active": engine.periodic_refresh_events > 0,
        "official_shallow_prefix_kv_reuse": all(
            bool(audit["shallow_prefix_kv_reused"]) for audit in audits
        ),
        "official_boundary_hidden_state_restore": all(
            bool(audit["boundary_hidden_state_restored"]) for audit in audits
        ),
        "threshold_decoding_active": True,
        "threshold_decoding": threshold_audit,
        "policy_mismatches": policy_mismatches,
    }

    result = {
        "dataset": args.dataset,
        "shots": args.shots,
        "seed": args.seed,
        "batch_size": 1,
        "model": args.model_path,
        "node": os.environ.get("SLURMD_NODENAME", "unknown"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "unknown"),
        "requests": len(timed_prompts),
        "shared_prefix_length_F": prefix_len,
        "prefix_ratio_r_values": sorted(set(r_values)),
        "selected_shallow_layer_depth_b_values": sorted(set(b_values)),
        "full_request_prefix_invariant": bool(prompt_payload["full_prefix_identical"]),
        "prefix_cache_source": "full_input_ids[:F]",
        "separately_tokenized_prefix_used_for_cache": False,
        "prompt_length_min": int(prompt_payload["min_prompt_length"]),
        "prompt_length_max": int(prompt_payload["max_prompt_length"]),
        "generated_length": args.gen_length,
        "steps": args.steps,
        "block_length": BLOCK_LENGTH,
        "temperature": float(args.temperature),
        "mask_id": MASK_ID,
        "decoding_threshold": float(args.threshold),
        "decoding_policy": "Dream shifted-logit Nano-vDLLM threshold transfer with fallback stopping",
        "intra_request_cache_update_interval": INTER_REQUEST_REFRESH_INTERVAL,
        "profile_threshold": PROFILE_THRESHOLD,
        "cache_budget": CACHE_BUDGET,
        "throughput_generated_tokens_per_second": throughput,
        "total_seconds": ended - started,
        "tokenization_seconds": tokenization_seconds,
        "generation_seconds": generation_seconds,
        "warmup_seconds": warmup_seconds,
        "total_nfe": sum(nfe_values),
        "average_nfe": sum(nfe_values) / len(nfe_values),
        "nfe_histogram": {
            str(value): count
            for value, count in sorted(collections.Counter(nfe_values).items())
        },
        "policy_path": str(Path(args.policy_path).resolve()),
        "policy_metadata": policy_metadata,
        "comparison": comparison,
        "threshold_validation": threshold_audit,
        "audit": audit,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("DREAM_BICACHE_WORKLOAD_RESULT=" + json.dumps(result, sort_keys=True), flush=True)
    print(f"Dream BiCache tokens per second: {throughput}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-root", type=Path, default=DEFAULT_OFFICIAL_ROOT)
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--model-path", default=MODEL_ID)
    parser.add_argument("--dataset", choices=("gsm8k", "mbpp"), required=True)
    parser.add_argument("--shots", type=int, choices=(1, 2, 4), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--gen-length", type=int, default=GEN_LENGTH)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--skip-warmup", action="store_true")
    _run(parser.parse_args())


if __name__ == "__main__":
    main()
