#!/usr/bin/env python3
"""Profile Dream's BiCache shallow-layer policy with the official algorithm.

The official BiCache profiler is written as a serial loop over 101 prefix
ratios.  This entry point keeps its tokenization, prefix/full KV comparison,
cosine calculation, and threshold policy intact, but distributes independent
ratio buckets across Slurm workers.  Workers write only the official
per-layer mean cosine values; the aggregate step applies
``ProfilerBase.set_number_of_inter_request_caching_layer`` exactly once.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch


DEFAULT_OFFICIAL_ROOT = Path(__file__).resolve().parent / "third_party" / "BiCache"
DEFAULT_DREAM_RATIO_IDS = Path(__file__).resolve().parent / "policies/dream_ratio_ordered_WildChat_ids.npy"
MODEL_ID = "Dream-org/Dream-v0-Instruct-7B"
PROFILE_DATASET = "allenai/WildChat-4.8M"
PROFILE_SAMPLE_COUNT = 500
PROFILE_THRESHOLD = 0.97
PROFILE_MAX_SEQUENCE_LENGTH = 32768
OFFICIAL_COMMIT = "82a8d0122807daf58ceacd6a121201b8355f46c1"


def _official_import(root: Path):
    root = root.resolve()
    sys.path.insert(0, str(root))
    from bicache import DreamProfiler  # type: ignore
    from model.dream import DreamModel  # type: ignore
    return DreamProfiler, DreamModel


def _load_model_and_tokenizer(root: Path, model_path: str, device: str):
    DreamProfiler, DreamModel = _official_import(root)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = DreamModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to(device).eval()
    return DreamProfiler, model, tokenizer


def _ratio_mean_for_worker(args, root: Path) -> dict[str, list[float]]:
    DreamProfiler, model, tokenizer = _load_model_and_tokenizer(
        root, args.model_path, args.device
    )
    ids_path = Path(args.ratio_ids or DEFAULT_DREAM_RATIO_IDS)
    ids = np.load(ids_path, allow_pickle=True)
    if len(ids) != 101:
        raise RuntimeError(f"Expected 101 official ratio buckets, got {len(ids)}")

    from datasets import load_dataset

    dataset = load_dataset(PROFILE_DATASET, split="train")
    profiler = DreamProfiler(args.device, model, tokenizer)
    ratio_means: dict[str, list[float]] = {}
    started = time.perf_counter()

    for ratio in range(int(args.worker_rank), len(ids), int(args.worker_count)):
        candidates = ids[ratio]
        if len(candidates) < PROFILE_SAMPLE_COUNT:
            raise RuntimeError(
                f"Official ratio bucket r={ratio} has {len(candidates)} entries; "
                f"need {PROFILE_SAMPLE_COUNT}."
            )

        measurements = []
        for sample_id, turn_count in candidates:
            if len(measurements) >= PROFILE_SAMPLE_COUNT:
                break
            conversation = dataset["conversation"][int(sample_id)][: int(turn_count) + 1]
            sequence = [
                {"role": item["role"], "content": item["content"]}
                for item in conversation
            ]
            prefix_ids, user_prompt_ids = profiler.tokenize(sequence)
            prefix_length = len(prefix_ids)
            sequence_length = prefix_length + len(user_prompt_ids)
            calculated_ratio = round(prefix_length / sequence_length * 100)
            if calculated_ratio != ratio:
                raise AssertionError(
                    f"Official ratio assertion failed: bucket={ratio}, "
                    f"calculated={calculated_ratio}, sample={sample_id}, turn={turn_count}"
                )
            if (
                args.max_sequence_length is not None
                and sequence_length > int(args.max_sequence_length)
            ):
                continue

            kv_prefix, kv_full_prefix = profiler.generate_kvs(prefix_ids, user_prompt_ids)
            similarity = profiler.calculate_cosine_similarity(kv_prefix, kv_full_prefix)
            measurements.append(similarity.detach().cpu())
            del kv_prefix, kv_full_prefix, similarity

        if len(measurements) < PROFILE_SAMPLE_COUNT:
            raise RuntimeError(
                f"Only {len(measurements)} eligible samples for r={ratio} at "
                f"max_length={args.max_sequence_length}; need {PROFILE_SAMPLE_COUNT}."
            )

        ratio_means[str(ratio)] = torch.mean(torch.stack(measurements), dim=0).tolist()
        print(
            f"DREAM_BICACHE_PROFILE ratio={ratio} samples={len(measurements)} "
            f"worker={args.worker_rank}/{args.worker_count}",
            flush=True,
        )
        del measurements
        gc.collect()
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    payload = {
        "ratio_means": ratio_means,
        "worker_rank": int(args.worker_rank),
        "worker_count": int(args.worker_count),
        "official_commit": OFFICIAL_COMMIT,
        "model": args.model_path,
        "profile_dataset": PROFILE_DATASET,
        "ratio_ids": str(ids_path.resolve()),
        "num_profiling_data_per_ratio": PROFILE_SAMPLE_COUNT,
        "threshold": PROFILE_THRESHOLD,
        "max_length_profiling_data": args.max_sequence_length,
        "profile_node": os.environ.get("SLURMD_NODENAME", "unknown"),
        "profile_qos": os.environ.get("SLURM_JOB_QOS", "unknown"),
        "profile_job_id": os.environ.get("SLURM_JOB_ID", "unknown"),
        "profile_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID", "unknown"),
        "profile_seconds": time.perf_counter() - started,
        "profiler": "official bicache.DreamProfiler",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"DREAM_BICACHE_PROFILE_PART={output}", flush=True)
    return payload


def _aggregate(args, root: Path) -> dict:
    DreamProfiler, _ = _official_import(root)
    part_dir = Path(args.part_dir)
    ratio_means: dict[str, list[float]] = {}
    parts = []
    for rank in range(int(args.worker_count)):
        path = part_dir / f"worker_{rank}.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing Dream profiler worker output: {path}")
        part = json.loads(path.read_text(encoding="utf-8"))
        parts.append(part)
        for key, values in part.get("ratio_means", {}).items():
            if key in ratio_means:
                raise RuntimeError(f"Duplicate ratio {key} in Dream profile parts")
            ratio_means[key] = values

    expected = [str(ratio) for ratio in range(101)]
    if sorted(ratio_means, key=int) != expected:
        missing = sorted(set(expected) - set(ratio_means), key=int)
        extra = sorted(set(ratio_means) - set(expected), key=int)
        raise RuntimeError(f"Dream profile did not cover every ratio: missing={missing}, extra={extra}")

    similarity = [torch.tensor(ratio_means[str(ratio)]) for ratio in range(101)]
    profiler = object.__new__(DreamProfiler)
    policy = profiler.set_number_of_inter_request_caching_layer(
        similarity,
        PROFILE_THRESHOLD,
    )
    if not policy or min(policy) > 0:
        raise RuntimeError(f"Official Dream policy has no usable r=0 floor: {policy}")

    output = Path(args.output)
    payload = {
        "official_commit": OFFICIAL_COMMIT,
        "model": args.model_path,
        "profile_dataset": PROFILE_DATASET,
        "ratio_ids": str(Path(args.ratio_ids or DEFAULT_DREAM_RATIO_IDS).resolve()),
        "num_profiling_data_per_ratio": PROFILE_SAMPLE_COUNT,
        "threshold": PROFILE_THRESHOLD,
        "max_length_profiling_data": args.max_sequence_length,
        "profiler": "official bicache.DreamProfiler",
        "profile_parallel_workers": int(args.worker_count),
        "profile_worker_nodes": sorted({part.get("profile_node", "unknown") for part in parts}),
        "profile_part_dir": str(part_dir.resolve()),
        "profile_seconds": max(float(part.get("profile_seconds", 0.0)) for part in parts),
        "policy": {str(int(key)): int(value) for key, value in sorted(policy.items())},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("DREAM_BICACHE_POLICY=" + json.dumps(payload["policy"], sort_keys=True), flush=True)
    print(f"DREAM_BICACHE_POLICY_OUTPUT={output}", flush=True)
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("worker", "aggregate"), required=True)
    parser.add_argument("--official-root", type=Path, default=DEFAULT_OFFICIAL_ROOT)
    parser.add_argument("--model-path", default=MODEL_ID)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ratio-ids", default=str(DEFAULT_DREAM_RATIO_IDS))
    parser.add_argument("--max-sequence-length", type=int, default=PROFILE_MAX_SEQUENCE_LENGTH)
    parser.add_argument("--worker-rank", type=int, default=None)
    parser.add_argument("--worker-count", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--part-dir", default=None)
    args = parser.parse_args()

    if args.mode == "worker":
        if args.worker_rank is None:
            parser.error("--worker-rank is required in worker mode")
        _ratio_mean_for_worker(args, args.official_root)
    else:
        if args.part_dir is None:
            parser.error("--part-dir is required in aggregate mode")
        _aggregate(args, args.official_root)


if __name__ == "__main__":
    main()
