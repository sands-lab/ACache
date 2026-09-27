#!/usr/bin/env python3
"""BiCache + threshold Fast-dLLM on the ACache LLaDA prefix workload.

The GSM8K/MBPP builder follows the validated Nano-vDLLM system path.  The
BABILong builder follows the existing ``llada/eval_ACache.py`` path through
``acache_eval_shared.ACacheEvalHarnessMixin``.  The official BiCache engine is
used for cache lookup and denoising; this adapter only supplies the
already-tokenized ACache request so that the engine uses
``full_input_ids[:F]`` as its inter-request prefix.

The ``profile`` mode runs the official WildChat profiler and writes its
``r -> b(r)`` table.  The ``run`` mode consumes that table and executes one
configuration at batch size one.  Runtime records include the policy lookup,
the official periodic refresh events, the Nano-vDLLM threshold-transfer
audit, NFE accounting, and the throughput timer.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parent
NANO_ROOT = REPO_ROOT / "nano-vdllm"
DEFAULT_OFFICIAL_ROOT = Path(__file__).resolve().parent / "third_party" / "BiCache"
MODEL_ID = "GSAI-ML/LLaDA-8B-Instruct"

GEN_LENGTH = 256
BLOCK_LENGTH = 32
MASK_ID = 126336
TEMPERATURE = 0.0
DECODING_THRESHOLD = 0.9
REMASKING = "low_confidence"
INTRA_REQUEST_CACHE_UPDATE_INTERVAL = 16
PROFILE_DATASET = "allenai/WildChat-4.8M"
PROFILE_SAMPLE_COUNT = 500
PROFILE_THRESHOLD = 0.97
# The upstream artifact uses 200000 by default. The official README exposes
# this as a GPU-memory tuning parameter; the launcher may lower it for the
# A100-SXM4-40GB nodes without changing the profiler or its r -> b policy.
PROFILE_MAX_SEQUENCE_LENGTH = int(os.environ.get("BICACHE_PROFILE_MAX_SEQUENCE_LENGTH", "200000"))
CACHE_BUDGET = 5000
SEED = 0

# Values from the existing system-results workbook.  They are included in
# each new result for a direct same-configuration comparison; the old
# fixed-step BiCache results are deliberately absent.
SYSTEM_BASELINES = {
    ("gsm8k", 1): {"fast_dllm": 103.6944691547364, "acache": 99.3527431406719},
    ("gsm8k", 2): {"fast_dllm": 94.36583395313097, "acache": 98.44115960472348},
    ("gsm8k", 4): {"fast_dllm": 84.77016918859904, "acache": 92.09949171842172},
    ("mbpp", 1): {"fast_dllm": 209.4338485313253, "acache": 212.9133004073965},
    ("mbpp", 2): {"fast_dllm": 167.9832986280038, "acache": 194.3399599807184},
    ("mbpp", 4): {"fast_dllm": 122.9398755237329, "acache": 155.3683269926975},
}


def _import_official(official_root: Path):
    """Import local prompt code and then the exact official BiCache checkout.

    ``nano-vdllm/eval_llada.py`` imports a local package named ``model``.
    Remove that package from the module cache before importing the official
    ``model`` package; the prompt-builder methods used below do not instantiate
    the local model.
    """

    sys.path.insert(0, str(NANO_ROOT))
    from eval_llada import LLaDAEvalHarness  # type: ignore
    from utils import set_seed  # type: ignore

    for name in list(sys.modules):
        if name == "model" or name.startswith("model."):
            del sys.modules[name]

    sys.path.insert(0, str(official_root))
    from bicache import FastdLLMLLaDAEngine, LLaDAProfiler  # type: ignore
    from model import LLaDAModelLM  # type: ignore

    return LLaDAEvalHarness, set_seed, FastdLLMLLaDAEngine, LLaDAProfiler, LLaDAModelLM


def _load_model_and_tokenizer(model_path: str, device: str, LLaDAModelLM):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = LLaDAModelLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
    ).to(device).eval()
    return model, tokenizer


def _elide_profile_only_dense_zero_bias(model):
    """Keep the official bidirectional attention semantics without its O(T^2) zero tensor.

    The official LLaDA model materializes an all-zero ``[1, 1, T, T]`` bias
    for an unmasked bidirectional forward.  That tensor is unnecessary, but
    it exhausts a 40 GB A100 before the official profiler can process the
    long WildChat samples.  Returning ``None`` selects the same unmasked
    attention mathematically and lets PyTorch use its memory-efficient
    attention kernel.  This adapter is installed only on the temporary
    profiler model; workload execution remains the official model path.
    """

    def no_dense_zero_bias(_seq_len, _device):
        return None

    model.model.get_bidirectional_attention_bias = no_dense_zero_bias
    return model


def _make_prompt_builder(
    *,
    dataset_name: str,
    shots: int,
    tokenizer,
    LLaDAEvalHarness,
    set_seed,
    generation_length: int = GEN_LENGTH,
    seed: int = SEED,
):
    """Construct the prompt state with the current ACache code path."""

    if dataset_name == "babilong":
        # Keep BABILong on the existing ACache prompt path.  This deliberately
        # does not call the Nano-vDLLM BABILong prompt builder: the wording
        # changes both the token sequence and F.
        from acache_eval_shared import (
            ACacheEvalHarnessMixin,
            set_seed as acache_set_seed,
        )

        class ACacheBABILongPrefixPromptBuilder(ACacheEvalHarnessMixin):
            def _build_prefix_input_ids(self, question):
                input_ids, _, _, _, _ = self._build_input_ids_with_affix(
                    question,
                    self.default_affix_state,
                )
                return input_ids

        builder = ACacheBABILongPrefixPromptBuilder.__new__(
            ACacheBABILongPrefixPromptBuilder
        )
        builder.seed = int(seed)
        acache_set_seed(int(seed))
        builder.fewshot_num_examples = int(shots)
        builder.shared_prefix_text = ""
        builder.shared_prefix_extra_text = ""
        builder.shared_prefix_role = "system"
        builder.shared_prefix_use_chat_template = True
        builder.affix_type = "prefix"
        builder.gen_length = int(generation_length)
        builder.mask_id = MASK_ID
        builder.tokenizer = tokenizer
        builder.fewshot_dataset_path = "RMT-team/babilong-1k-samples"
        builder.fewshot_dataset_name = "0k"
        builder.fewshot_split = "qa1"
        builder.fewshot_question_key = "question"
        builder.fewshot_answer_key = "target"
        builder.prompt_style = "babilong_qa1"

        prefix_messages, sampled_examples = builder._build_prefix_fewshot_messages(
            state_name="default"
        )
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
        builder.prefix_affix_token_ids = prefix_affix_token_ids
        builder.prefix_affix_len = len(prefix_affix_token_ids)
        builder.prompt_source = (
            "llada/eval_ACache.py via "
            "acache_eval_shared.ACacheEvalHarnessMixin"
        )
        return builder

    builder = LLaDAEvalHarness.__new__(LLaDAEvalHarness)
    builder.seed = int(seed)
    set_seed(int(seed))
    builder.fewshot_num_examples = int(shots)
    builder.shared_prefix_text = ""
    builder.shared_prefix_extra_text = ""
    builder.shared_prefix_use_chat_template = True
    builder.sampled_fewshot_examples = []
    builder.tokenizer = tokenizer

    if dataset_name == "gsm8k":
        builder.fewshot_dataset_path = "gsm8k"
        builder.fewshot_dataset_name = "main"
        builder.fewshot_split = "train"
        builder.fewshot_question_key = "question"
        builder.fewshot_answer_key = "answer"
        builder.prompt_style = "gsm8k"
    elif dataset_name == "mbpp":
        builder.fewshot_dataset_path = "google-research-datasets/mbpp"
        builder.fewshot_dataset_name = "full"
        builder.fewshot_split = "prompt"
        builder.fewshot_question_key = "text"
        builder.fewshot_answer_key = "code"
        builder.prompt_style = "mbpp"
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    builder.prefix_fewshot_messages = builder._build_prefix_fewshot_messages()
    builder.prefix_affix_token_ids = builder._build_prefix_affix_token_ids()
    builder.prefix_affix_len = len(builder.prefix_affix_token_ids)
    return builder


def _load_requests(dataset_name: str):
    from datasets import load_dataset

    if dataset_name == "gsm8k":
        return load_dataset("gsm8k", "main", split="test")
    if dataset_name == "mbpp":
        return load_dataset("google-research-datasets/mbpp", "full", split="test")
    if dataset_name == "babilong":
        return load_dataset("RMT-team/babilong-1k-samples", "0k", split="qa1")
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def _build_full_request_ids(builder, dataset_name: str, rows: Iterable[dict[str, Any]]):
    """Build the exact ACache full prompt IDs and verify the shared span."""

    F = int(builder.prefix_affix_len)
    prompts: list[list[int]] = []
    shared_prefix: tuple[int, ...] | None = None
    min_len = None
    max_len = None
    for request_index, row in enumerate(rows):
        req = SimpleNamespace(doc=dict(row), args=(), task_name=dataset_name)
        question = builder._extract_question_text(req)
        full_ids = [int(token) for token in builder._build_prefix_input_ids(question)]
        if len(full_ids) < F:
            raise RuntimeError(
                f"request {request_index} has {len(full_ids)} prompt tokens, below F={F}"
            )
        current_prefix = tuple(full_ids[:F])
        if shared_prefix is None:
            shared_prefix = current_prefix
        elif current_prefix != shared_prefix:
            mismatch = next(
                i for i, (left, right) in enumerate(zip(shared_prefix, current_prefix)) if left != right
            )
            raise RuntimeError(
                "ACache prefix invariant failed: request-dependent token inside "
                f"full_input_ids[:F] at request {request_index}, position {mismatch}"
            )
        prompts.append(full_ids)
        min_len = len(full_ids) if min_len is None else min(min_len, len(full_ids))
        max_len = len(full_ids) if max_len is None else max(max_len, len(full_ids))

    if shared_prefix is None:
        raise RuntimeError("No workload requests were loaded")
    return prompts, F, min_len, max_len


def _policy_payload(policy: dict[int, int], **metadata):
    return {
        **metadata,
        "policy": {str(int(key)): int(value) for key, value in sorted(policy.items())},
    }


def _load_policy(path: Path) -> tuple[dict[int, int], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    policy = {int(key): int(value) for key, value in payload["policy"].items()}
    if not policy:
        raise ValueError(f"Empty BiCache policy in {path}")
    if min(policy) > 0:
        raise ValueError(
            f"Policy starts at r={min(policy)}; official lookup cannot serve small ratios"
        )
    return policy, payload


def _nano_add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Copy Nano-vDLLM's current ``utils.add_gumbel_noise`` semantics."""

    if temperature == 0:
        return logits
    noise = torch.rand_like(logits, dtype=logits.dtype)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def _nano_threshold_transfer_index(
    logits: torch.Tensor,
    temperature: float,
    remasking: str,
    mask_index: torch.Tensor,
    x: torch.Tensor,
    threshold: float,
    audit: dict[str, int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Port ``ModelRunner.get_transfer_index`` from the Nano-vDLLM tree.

    The source method is intentionally reproduced here instead of wrapping
    BiCache's fixed-quota helper.  In particular, the fallback is conditional
    on a row having no token at or above the threshold, and selects that row's
    maximum-confidence masked position.
    """

    logits_with_noise = _nano_add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)

    if remasking == "low_confidence":
        flat_logits = logits.reshape(-1, logits.shape[-1])
        flat_x0 = x0.reshape(-1)
        flat_probs = torch.empty(
            flat_x0.shape,
            dtype=torch.float32,
            device=logits.device,
        )
        chunk_rows = 64
        for start in range(0, flat_logits.shape[0], chunk_rows):
            end = min(start + chunk_rows, flat_logits.shape[0])
            logits_fp32 = flat_logits[start:end].to(torch.float32)
            chosen_logits = torch.gather(
                logits_fp32,
                dim=-1,
                index=flat_x0[start:end].unsqueeze(-1),
            ).squeeze(-1)
            flat_probs[start:end] = torch.exp(
                chosen_logits - torch.logsumexp(logits_fp32, dim=-1)
            )
        x0_p = flat_probs.view_as(x0)
    elif remasking == "random":
        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    else:
        raise NotImplementedError(remasking)

    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, -np.inf)

    # Exact Nano-vDLLM threshold selection and row fallback.
    transfer_index = (confidence >= threshold) & mask_index
    needs_transfer = (mask_index.any(dim=1)) & (~transfer_index.any(dim=1))
    max_conf_indices = torch.argmax(confidence, dim=1)
    row_indices = torch.where(needs_transfer)[0]
    transfer_index[row_indices, max_conf_indices[row_indices]] = True

    if audit is not None:
        threshold_only = (confidence >= threshold) & mask_index
        audit["selection_calls"] += int(logits.shape[0])
        audit["threshold_selected_tokens"] += int(threshold_only.sum().item())
        audit["fallback_selected_tokens"] += int(row_indices.numel())
        audit["fallback_selection_calls"] += int(row_indices.numel())
        audit["selected_tokens"] += int(transfer_index.sum().item())
        audit["masked_tokens_seen"] += int(mask_index.sum().item())
        if torch.any(transfer_index & ~mask_index):
            raise AssertionError("Nano threshold transfer selected an unmasked position")
        if torch.any(threshold_only & ~transfer_index):
            raise AssertionError("Nano threshold transfer dropped an eligible position")
        if torch.any(needs_transfer & threshold_only.any(dim=1)):
            raise AssertionError("Nano threshold fallback was used despite an eligible position")

    return x0, transfer_index


class AuditedFullTokenFastdLLMEngine:
    """Official BiCache execution with Nano-vDLLM threshold decoding.

    The cache construction, shallow-prefix lookup, boundary hidden-state
    restoration, dual-cache build, and periodic deep-prefix refresh follow the
    released Fast-dLLM engine. Only its fixed transfer-quota loop is replaced
    with the existing Nano-vDLLM threshold transfer/stopping rule.
    """

    def __init__(
        self,
        FastdLLMLLaDAEngine,
        *,
        device: str,
        model,
        tokenizer,
        policy: dict[int, int],
        cache_budget: int,
        block_length: int,
        interval: int,
    ):
        self._base_class = FastdLLMLLaDAEngine
        self._engine = FastdLLMLLaDAEngine(
            device=device,
            model=model,
            tokenizer=tokenizer,
            number_of_inter_request_caching_layer=policy,
            intra_request_cache_update_interval=interval,
            cache_budget=cache_budget,
            show_speed=True,
            block_length=block_length,
        )
        self.interval = int(interval)
        self.policy = policy
        self.lookup_counts: collections.Counter[tuple[int, int]] = collections.Counter()
        self.dual_cache_builds = 0
        self.inter_request_cache_lookups = 0
        self.deep_prefix_recompute_events = 0
        self.boundary_hidden_state_restore_events = 0
        self.inter_request_resets = 0
        self.deep_prefix_refresh_updates = 0
        self.deep_prefix_refresh_layers = 0
        self.deep_prefix_refresh_layer_counts: collections.Counter[int] = collections.Counter()
        self.threshold_audit: dict[str, int] = {
            "selection_calls": 0,
            "threshold_selected_tokens": 0,
            "fallback_selected_tokens": 0,
            "fallback_selection_calls": 0,
            "selected_tokens": 0,
            "masked_tokens_seen": 0,
            "stopping_events": 0,
        }
        self.total_nfe = 0
        self.cache_forward_nfe = 0
        self.decode_forward_nfe = 0
        self.total_decode_iterations = 0
        self.request_nfe: list[int] = []
        self.request_decode_iterations: list[int] = []
        # Request-level audit records are consumed by the accuracy harness.
        # They do not affect the official cache or decoding path.
        self.request_records: list[dict[str, int]] = []

    def __getattr__(self, name):
        return getattr(self._engine, name)

    def warm_up(self, input_length: int, n_step: int):
        return self._engine.warm_up(input_length, n_step)

    def _record_refresh(self, kvo):
        layer_count = len(kvo)
        self.deep_prefix_refresh_updates += 1
        self.deep_prefix_refresh_layers += layer_count
        self.deep_prefix_refresh_layer_counts[layer_count] += 1
        if kvo and kvo[-1][2] is not None:
            self.boundary_hidden_state_restore_events += 1

    def generate_from_full_ids(
        self,
        full_input_ids: list[int],
        prefix_len: int,
        *,
        gen_length: int,
        mask_id: int,
        temperature: float,
        threshold: float = DECODING_THRESHOLD,
        remasking: str = REMASKING,
    ) -> torch.Tensor:
        """Run official generation using an explicit full-token request."""

        engine = self._engine
        input_ids = list(full_input_ids)
        prefix_len = int(prefix_len)
        prefix_ratio = engine.get_prefix_ratio(len(input_ids) + gen_length, prefix_len)
        selected_b = engine.get_number_of_inter_request_caching_layer(prefix_ratio)
        self.lookup_counts[(int(prefix_ratio), int(selected_b))] += 1

        if prefix_len == 0:
            raise NotImplementedError

        inter_request_cache = engine.get_inter_request_cache(
            input_ids[:prefix_len], prefix_ratio
        )
        self.inter_request_cache_lookups += 1
        prefix_cache, prefix_hidden_state = inter_request_cache
        input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
        x = torch.full(
            (input_tensor.shape[0], input_tensor.shape[1] + gen_length),
            mask_id,
            dtype=torch.long,
        ).to(engine.device)
        x[:, : input_tensor.shape[1]] = input_tensor.clone()

        assert gen_length % engine.block_length == 0
        num_blocks = gen_length // engine.block_length

        if engine.show_speed:
            start_time = time.perf_counter()
        with torch.inference_mode():
            total_steps = 0
            request_nfe = 0
            request_decode_iterations = 0
            request_refresh_markers = 0
            refresh_updates_before = self.deep_prefix_refresh_updates
            for num_block in range(num_blocks):
                s = input_tensor.shape[1] + num_block * engine.block_length
                e = s + engine.block_length

                block_iterations = 0
                while True:
                    refresh_marker = total_steps % engine.intra_request_cache_update_interval == 0
                    if refresh_marker:
                        request_refresh_markers += 1
                        self.inter_request_resets += 1
                        prefix_cache, prefix_hidden_state = inter_request_cache
                    if block_iterations == 0:
                        dual_cache, kvo = engine.generate_dual_cache(
                            x,
                            s,
                            e,
                            prefix_len,
                            prefix_cache,
                            prefix_hidden_state,
                        )
                        self.dual_cache_builds += 1
                        self.deep_prefix_recompute_events += 1
                        self.cache_forward_nfe += 1
                        self.total_nfe += 1
                        request_nfe += 1
                        if refresh_marker:
                            prefix_cache = [
                                (k[:, :, :prefix_len, :], v[:, :, :prefix_len, :])
                                for k, v, _ in kvo
                            ]
                            prefix_hidden_state = (
                                kvo[-1][2][:, :prefix_len, :]
                                if kvo[-1][2] is not None
                                else prefix_hidden_state
                            )
                            self._record_refresh(kvo)

                    mask_index = x[:, s:e] == mask_id
                    logits, kvo = engine.model(
                        x[:, s:e],
                        prefix_len=prefix_len,
                        prefix_cache=prefix_cache,
                        prefix_hidden_state=prefix_hidden_state,
                        dual_cache=dual_cache,
                    )
                    logits = logits.logits[:, prefix_len:]
                    self.decode_forward_nfe += 1
                    self.total_nfe += 1
                    request_nfe += 1

                    if refresh_marker:
                        prefix_cache = [
                            (k[:, :, :prefix_len, :], v[:, :, :prefix_len, :])
                            for k, v, _ in kvo
                        ]
                        prefix_hidden_state = (
                            kvo[-1][2][:, :prefix_len, :]
                            if kvo[-1][2] is not None
                            else prefix_hidden_state
                        )
                        self._record_refresh(kvo)

                    x0, transfer_index = _nano_threshold_transfer_index(
                        logits,
                        temperature,
                        remasking,
                        mask_index,
                        x[:, s:e],
                        threshold,
                        audit=self.threshold_audit,
                    )
                    x[:, s:e][transfer_index] = x0[transfer_index]

                    if total_steps == 0 and engine.show_speed:
                        engine.ttft = time.perf_counter() - start_time
                    total_steps += 1
                    block_iterations += 1
                    request_decode_iterations += 1
                    self.total_decode_iterations += 1
                    if not torch.any(x[:, s:e] == mask_id):
                        self.threshold_audit["stopping_events"] += 1
                        break

            self.request_nfe.append(request_nfe)
            self.request_decode_iterations.append(request_decode_iterations)
            self.request_records.append(
                {
                    "prefix_ratio_r": int(prefix_ratio),
                    "selected_b": int(selected_b),
                    "request_nfe": int(request_nfe),
                    "decode_iterations": int(request_decode_iterations),
                    "official_refresh_markers": int(request_refresh_markers),
                    "official_deep_prefix_refresh_updates": int(
                        self.deep_prefix_refresh_updates - refresh_updates_before
                    ),
                }
            )
        return x[0][prefix_len:]


def _run_profile(args, official_root: Path):
    _, _, _, LLaDAProfiler, LLaDAModelLM = _import_official(official_root)
    device = args.device
    if not torch.cuda.is_available() and device.startswith("cuda"):
        raise RuntimeError("BiCache profiling requires CUDA; no CUDA device is visible")

    model, tokenizer = _load_model_and_tokenizer(args.model_path, device, LLaDAModelLM)
    _elide_profile_only_dense_zero_bias(model)
    ids_path = Path(args.ratio_ids or official_root / "ratio_ordered_WildChat_ids.npy")
    ids = np.load(ids_path, allow_pickle=True)
    profiler = LLaDAProfiler(device, model, tokenizer)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    started = time.perf_counter()
    policy = profiler.profile(
        dataset_name=PROFILE_DATASET,
        ids=ids,
        max_sequence_length=PROFILE_MAX_SEQUENCE_LENGTH,
        num_profiling_data_per_ratio=PROFILE_SAMPLE_COUNT,
        threshold=PROFILE_THRESHOLD,
    )
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    payload = _policy_payload(
        policy,
        official_commit=args.official_commit,
        model=args.model_path,
        profile_dataset=PROFILE_DATASET,
        ratio_ids=str(ids_path),
        num_profiling_data_per_ratio=PROFILE_SAMPLE_COUNT,
        threshold=PROFILE_THRESHOLD,
        max_length_profiling_data=PROFILE_MAX_SEQUENCE_LENGTH,
        profile_memory_adapter="elided official all-zero bidirectional attention bias for 40GB A100 profiling",
        profile_seconds=elapsed,
    )
    output = Path(args.policy_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("BICACHE_PROFILE_POLICY=" + json.dumps(payload["policy"], sort_keys=True))
    print(f"BICACHE_PROFILE_SECONDS={elapsed:.6f}")
    print(f"BICACHE_PROFILE_OUTPUT={output}")


def _run_profile_worker(args, official_root: Path):
    """Run the official profiler's per-ratio measurements on one GPU.

    The upstream ``profile`` method is a serial loop over ratios.  This keeps
    its tokenization, sample filtering, KV generation, cosine similarity, and
    sample averaging byte-for-byte in the same order, while distributing
    independent ratios across workers.  The official threshold-to-depth
    policy is applied once by ``_run_profile_aggregate``.
    """

    _, _, _, LLaDAProfiler, LLaDAModelLM = _import_official(official_root)
    device = args.device
    if not torch.cuda.is_available() and device.startswith("cuda"):
        raise RuntimeError("BiCache profiling requires CUDA; no CUDA device is visible")

    model, tokenizer = _load_model_and_tokenizer(args.model_path, device, LLaDAModelLM)
    _elide_profile_only_dense_zero_bias(model)
    ids_path = Path(args.ratio_ids or official_root / "ratio_ordered_WildChat_ids.npy")
    ids = np.load(ids_path, allow_pickle=True)

    from datasets import load_dataset

    dataset = load_dataset(PROFILE_DATASET, split="train")
    profiler = LLaDAProfiler(device, model, tokenizer)
    started = time.perf_counter()
    ratio_means = {}
    for r in range(args.profile_worker_rank, len(ids), args.profile_worker_count):
        candidates = ids[r]
        if len(candidates) < PROFILE_SAMPLE_COUNT:
            raise ValueError(f"Not enough profiling data for r={r}")

        measurements = []
        for sample_id, turn_count in candidates:
            if len(measurements) == PROFILE_SAMPLE_COUNT:
                break
            conversation = dataset["conversation"][int(sample_id)][: int(turn_count) + 1]
            sequence = [{"role": c["role"], "content": c["content"]} for c in conversation]

            prefix_ids, user_prompt_ids = profiler.tokenize(sequence)
            prefix_len = len(prefix_ids)
            seq_len = prefix_len + len(user_prompt_ids)
            prefix_ratio = round(prefix_len / seq_len * 100)
            assert prefix_ratio == r
            if PROFILE_MAX_SEQUENCE_LENGTH is not None and seq_len > PROFILE_MAX_SEQUENCE_LENGTH:
                continue

            kv1, kv2 = profiler.generate_kvs(prefix_ids, user_prompt_ids)
            measurements.append(profiler.calculate_cosine_similarity(kv1, kv2))
            del kv1, kv2

        if len(measurements) < PROFILE_SAMPLE_COUNT:
            raise ValueError(
                f"Only {len(measurements)} eligible samples for r={r}; "
                f"need {PROFILE_SAMPLE_COUNT} at max length {PROFILE_MAX_SEQUENCE_LENGTH}"
            )
        ratio_means[str(r)] = torch.mean(torch.stack(measurements), dim=0).detach().cpu().tolist()
        print(
            f"BICACHE_PROFILE_RATIO worker={args.profile_worker_rank} r={r} "
            f"samples={len(measurements)}",
            flush=True,
        )

    if device.startswith("cuda"):
        torch.cuda.synchronize()
    output = Path(args.profile_part_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "ratio_means": ratio_means,
        "worker_rank": args.profile_worker_rank,
        "worker_count": args.profile_worker_count,
        "official_commit": args.official_commit,
        "model": args.model_path,
        "profile_dataset": PROFILE_DATASET,
        "ratio_ids": str(ids_path),
        "num_profiling_data_per_ratio": PROFILE_SAMPLE_COUNT,
        "threshold": PROFILE_THRESHOLD,
        "max_length_profiling_data": PROFILE_MAX_SEQUENCE_LENGTH,
        "profile_memory_adapter": "elided official all-zero bidirectional attention bias for 40GB A100 profiling",
        "profile_node": os.environ.get("SLURMD_NODENAME", "unknown"),
        "profile_seconds": time.perf_counter() - started,
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"BICACHE_PROFILE_PART_OUTPUT={output}", flush=True)


def _run_profile_aggregate(args, official_root: Path):
    """Apply the official profiler threshold policy to parallel measurements."""

    _, _, _, LLaDAProfiler, _ = _import_official(official_root)
    part_dir = Path(args.profile_part_dir)
    parts = []
    for rank in range(args.profile_worker_count):
        path = part_dir / f"worker_{rank}.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing profiler worker output: {path}")
        parts.append(json.loads(path.read_text(encoding="utf-8")))

    ratio_means = {}
    for part in parts:
        if part.get("official_commit") != args.official_commit:
            raise ValueError("Profiler worker commit mismatch")
        if part.get("threshold") != PROFILE_THRESHOLD:
            raise ValueError("Profiler worker threshold mismatch")
        if part.get("num_profiling_data_per_ratio") != PROFILE_SAMPLE_COUNT:
            raise ValueError("Profiler worker sample-count mismatch")
        for ratio, values in part["ratio_means"].items():
            if ratio in ratio_means:
                raise ValueError(f"Duplicate profiler ratio: {ratio}")
            ratio_means[ratio] = values

    if sorted(map(int, ratio_means)) != list(range(101)):
        raise ValueError("Profiler workers did not cover every ratio from 0 through 100")

    similarity = [torch.tensor(ratio_means[str(r)]) for r in range(101)]
    profiler = object.__new__(LLaDAProfiler)
    policy = profiler.set_number_of_inter_request_caching_layer(similarity, PROFILE_THRESHOLD)
    payload = _policy_payload(
        policy,
        official_commit=args.official_commit,
        model=args.model_path,
        profile_dataset=PROFILE_DATASET,
        ratio_ids=str(Path(args.ratio_ids or official_root / "ratio_ordered_WildChat_ids.npy")),
        num_profiling_data_per_ratio=PROFILE_SAMPLE_COUNT,
        threshold=PROFILE_THRESHOLD,
        max_length_profiling_data=PROFILE_MAX_SEQUENCE_LENGTH,
        profile_memory_adapter="elided official all-zero bidirectional attention bias for 40GB A100 profiling",
        profile_parallel_workers=args.profile_worker_count,
        profile_worker_nodes=sorted({part.get("profile_node", "unknown") for part in parts}),
        profile_part_dir=str(part_dir.resolve()),
        profile_seconds=max(float(part.get("profile_seconds", 0.0)) for part in parts),
    )
    output = Path(args.policy_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("BICACHE_PROFILE_POLICY=" + json.dumps(payload["policy"], sort_keys=True), flush=True)
    print(f"BICACHE_PROFILE_OUTPUT={output}", flush=True)


def _run_workload(args, official_root: Path):
    (
        LLaDAEvalHarness,
        set_seed,
        FastdLLMLLaDAEngine,
        _,
        LLaDAModelLM,
    ) = _import_official(official_root)
    policy, policy_metadata = _load_policy(Path(args.policy_path))
    device = args.device
    if not torch.cuda.is_available() and device.startswith("cuda"):
        raise RuntimeError("BiCache workload requires CUDA; no CUDA device is visible")

    model, tokenizer = _load_model_and_tokenizer(args.model_path, device, LLaDAModelLM)
    builder = _make_prompt_builder(
        dataset_name=args.dataset,
        shots=args.shots,
        tokenizer=tokenizer,
        LLaDAEvalHarness=LLaDAEvalHarness,
        set_seed=set_seed,
    )
    rows = _load_requests(args.dataset)
    if args.limit is not None:
        rows = rows.select(range(min(int(args.limit), len(rows))))
    prompts, F, min_prompt_len, max_prompt_len = _build_full_request_ids(
        builder, args.dataset, rows
    )

    engine = AuditedFullTokenFastdLLMEngine(
        FastdLLMLLaDAEngine,
        device=device,
        model=model,
        tokenizer=tokenizer,
        policy=policy,
        cache_budget=CACHE_BUDGET,
        block_length=BLOCK_LENGTH,
        interval=INTRA_REQUEST_CACHE_UPDATE_INTERVAL,
    )
    warmup_started = time.perf_counter()
    if not args.skip_warmup:
        engine.warm_up(1000, args.warmup_steps)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    warmup_seconds = time.perf_counter() - warmup_started

    # Match nano-vdllm/eval_llada.py: time prompt tokenization and threshold
    # generation, then count the fixed 256 generated tokens. Model loading,
    # prompt-prefix construction, policy profiling, and warm-up are outside
    # this timer.
    started = time.time()
    tokenization_started = time.perf_counter()
    timed_prompts: list[list[int]] = []
    for row in rows:
        req = SimpleNamespace(doc=dict(row), args=(), task_name=args.dataset)
        question = builder._extract_question_text(req)
        full_ids = [int(token) for token in builder._build_prefix_input_ids(question)]
        if len(full_ids) < F or tuple(full_ids[:F]) != tuple(prompts[len(timed_prompts)][:F]):
            raise RuntimeError("Timed prompt construction diverged from the audited ACache prompt")
        timed_prompts.append(full_ids)
    tokenization_seconds = time.perf_counter() - tokenization_started

    generation_started = time.perf_counter()
    for full_ids in timed_prompts:
        generated = engine.generate_from_full_ids(
            full_ids,
            F,
            gen_length=args.gen_length,
            mask_id=MASK_ID,
            temperature=TEMPERATURE,
            threshold=DECODING_THRESHOLD,
            remasking=REMASKING,
        )
        # The ACache runner returns CPU token IDs before its timing ends. Force
        # the same completion/synchronization boundary for this CUDA engine.
        _ = generated[-args.gen_length :].detach().cpu().tolist()
        del generated
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    generation_seconds = time.perf_counter() - generation_started
    ended = time.time()

    total_tokens = len(timed_prompts) * args.gen_length
    throughput = total_tokens / (ended - started)
    lookup_rows = [
        {"r": r, "b": b, "requests": count}
        for (r, b), count in sorted(engine.lookup_counts.items())
    ]
    if not lookup_rows:
        raise RuntimeError("No BiCache policy lookup was recorded")
    observed_r = sorted({row["r"] for row in lookup_rows})
    observed_b = sorted({row["b"] for row in lookup_rows})
    expected_generated_tokens = len(timed_prompts) * args.gen_length
    threshold_audit = {
        **engine.threshold_audit,
        "threshold": DECODING_THRESHOLD,
        "remasking": REMASKING,
        "fixed_transfer_quota_active": False,
        "fixed_step_schedule_active": False,
        "threshold_stop_active": True,
        "threshold_semantics_verified": (
            engine.threshold_audit["selection_calls"] > 0
            and engine.threshold_audit["selected_tokens"] == expected_generated_tokens
            and engine.threshold_audit["selected_tokens"]
            == engine.threshold_audit["threshold_selected_tokens"]
            + engine.threshold_audit["fallback_selected_tokens"]
            and engine.threshold_audit["stopping_events"]
            == len(timed_prompts) * (args.gen_length // BLOCK_LENGTH)
        ),
    }
    if not threshold_audit["threshold_semantics_verified"]:
        raise RuntimeError(
            "Threshold audit failed: the run did not satisfy Nano-vDLLM "
            "selection/stopping semantics."
        )
    nfe_histogram = {
        str(nfe): count
        for nfe, count in sorted(collections.Counter(engine.request_nfe).items())
    }
    decode_iteration_histogram = {
        str(count): frequency
        for count, frequency in sorted(
            collections.Counter(engine.request_decode_iterations).items()
        )
    }
    baseline = SYSTEM_BASELINES[(args.dataset, args.shots)]
    comparison = {
        "fast_dllm_throughput_generated_tokens_per_second": baseline["fast_dllm"],
        "acache_throughput_generated_tokens_per_second": baseline["acache"],
        "delta_vs_fast_dllm_generated_tokens_per_second": (
            throughput - baseline["fast_dllm"]
        ),
        "delta_vs_acache_generated_tokens_per_second": throughput - baseline["acache"],
        "percent_vs_fast_dllm": 100.0 * (throughput / baseline["fast_dllm"] - 1.0),
        "percent_vs_acache": 100.0 * (throughput / baseline["acache"] - 1.0),
        "baseline_source": "system_results_latest.xlsx",
    }
    audit = {
        "policy_lookup_active": bool(lookup_rows),
        "policy_entries_used": lookup_rows,
        "policy_entry_count": len(lookup_rows),
        "engine_cache_misses": int(engine.cache_misses),
        "engine_cached_tokens": int(engine.num_cache_tokens),
        "official_inter_request_cache_lookups": int(engine.inter_request_cache_lookups),
        "official_interval": INTRA_REQUEST_CACHE_UPDATE_INTERVAL,
        "official_inter_request_reset_events": int(engine.inter_request_resets),
        "official_dual_cache_builds": int(engine.dual_cache_builds),
        "official_deep_prefix_recompute_events": int(engine.deep_prefix_recompute_events),
        "official_deep_prefix_refresh_updates": int(engine.deep_prefix_refresh_updates),
        "official_deep_prefix_refresh_layers": int(engine.deep_prefix_refresh_layers),
        "official_boundary_hidden_state_restore_events": int(
            engine.boundary_hidden_state_restore_events
        ),
        "refresh_layer_count_histogram": {
            str(k): int(v)
            for k, v in sorted(engine.deep_prefix_refresh_layer_counts.items())
        },
        "official_deep_refresh_active": engine.deep_prefix_refresh_updates > 0,
        "official_shallow_prefix_kv_reuse": engine.inter_request_cache_lookups == len(timed_prompts),
        "official_boundary_hidden_state_restore": engine.boundary_hidden_state_restore_events > 0,
        "official_deep_prefix_recomputation": engine.deep_prefix_recompute_events > 0,
        "threshold_decoding_active": True,
        "threshold_decoding": threshold_audit,
    }
    result = {
        "dataset": args.dataset,
        "shots": args.shots,
        "batch_size": 1,
        "model": args.model_path,
        "requests": len(timed_prompts),
        "shared_prefix_length_F": F,
        "prefix_ratio_r_values": observed_r,
        "selected_shallow_layer_depth_b_values": observed_b,
        "full_request_prefix_invariant": True,
        "prefix_cache_source": "full_input_ids[:F]",
        "separately_tokenized_prefix_used_for_cache": False,
        "prompt_length_min": min_prompt_len,
        "prompt_length_max": max_prompt_len,
        "generated_length": args.gen_length,
        "block_length": BLOCK_LENGTH,
        "temperature": TEMPERATURE,
        "mask_id": MASK_ID,
        "remasking": REMASKING,
        "decoding_threshold": DECODING_THRESHOLD,
        "decoding_policy": "Nano-vDLLM threshold transfer with fallback stopping",
        "intra_request_cache_update_interval": INTRA_REQUEST_CACHE_UPDATE_INTERVAL,
        "profile_threshold": PROFILE_THRESHOLD,
        "cache_budget": CACHE_BUDGET,
        "throughput_generated_tokens_per_second": throughput,
        "total_seconds": ended - started,
        "tokenization_seconds": tokenization_seconds,
        "generation_seconds": generation_seconds,
        "warmup_seconds": warmup_seconds,
        "total_nfe": int(engine.total_nfe),
        "average_nfe": engine.total_nfe / len(timed_prompts),
        "cache_forward_nfe": int(engine.cache_forward_nfe),
        "decode_forward_nfe": int(engine.decode_forward_nfe),
        "total_decode_iterations": int(engine.total_decode_iterations),
        "average_decode_iterations": engine.total_decode_iterations / len(timed_prompts),
        "nfe_histogram": nfe_histogram,
        "decode_iteration_histogram": decode_iteration_histogram,
        "fewshot_indices": [x["index"] for x in builder.sampled_fewshot_examples],
        "official_commit": args.official_commit,
        "policy_path": str(Path(args.policy_path).resolve()),
        "policy_metadata": policy_metadata,
        "comparison": comparison,
        "decoding_note": (
            "BiCache cache construction and official periodic refresh are preserved; "
            "the released fixed top-k transfer loop is replaced by the ported "
            "Nano-vDLLM threshold=0.9 transfer and block-stopping logic."
        ),
        "audit": audit,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("BICACHE_WORKLOAD_RESULT=" + json.dumps(result, sort_keys=True))
    print(f"Tokens per second: {throughput}")
    print("BICACHE_AUDIT=" + json.dumps(audit, sort_keys=True))


def _parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("profile", "profile-worker", "profile-aggregate", "run"), required=True)
    parser.add_argument("--official-root", default=os.environ.get("BICACHE_OFFICIAL_ROOT", str(DEFAULT_OFFICIAL_ROOT)))
    parser.add_argument("--official-commit", default="82a8d0122807daf58ceacd6a121201b8355f46c1")
    parser.add_argument("--model-path", default=MODEL_ID)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ratio-ids")
    parser.add_argument("--policy-output", default="bicache_fast_dllm_system_results/profile_policy.json")
    parser.add_argument("--policy-path")
    parser.add_argument("--output", default="bicache_fast_dllm_system_results/result.json")
    parser.add_argument("--dataset", choices=("gsm8k", "mbpp"))
    parser.add_argument("--shots", type=int, choices=(1, 2, 4))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--gen-length", type=int, default=GEN_LENGTH)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--skip-warmup", action="store_true")
    parser.add_argument("--profile-worker-rank", type=int)
    parser.add_argument("--profile-worker-count", type=int, default=1)
    parser.add_argument("--profile-part-output")
    parser.add_argument("--profile-part-dir")
    return parser


def main():
    args = _parser().parse_args()
    official_root = Path(args.official_root).resolve()
    if not (official_root / "bicache" / "bicache_fast_dllm_engines.py").exists():
        raise FileNotFoundError(f"Official BiCache checkout not found: {official_root}")
    if args.mode == "profile":
        _run_profile(args, official_root)
        return
    if args.mode == "profile-worker":
        if args.profile_worker_rank is None or args.profile_part_output is None:
            raise ValueError("profile-worker requires --profile-worker-rank and --profile-part-output")
        if not 0 <= args.profile_worker_rank < args.profile_worker_count:
            raise ValueError("invalid profiler worker rank")
        _run_profile_worker(args, official_root)
        return
    if args.mode == "profile-aggregate":
        if args.profile_part_dir is None:
            raise ValueError("profile-aggregate requires --profile-part-dir")
        _run_profile_aggregate(args, official_root)
        return
    if args.policy_path is None or args.dataset is None or args.shots is None:
        raise ValueError("run mode requires --policy-path, --dataset, and --shots")
    if args.gen_length <= 0 or args.gen_length % BLOCK_LENGTH != 0:
        raise ValueError("gen-length must be positive and divisible by block length")
    _run_workload(args, official_root)


if __name__ == "__main__":
    main()
