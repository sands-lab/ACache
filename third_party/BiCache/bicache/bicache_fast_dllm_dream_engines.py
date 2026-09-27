"""Dream extension of the official BiCache + Fast-dLLM engines.

The LLaDA implementation in this repository exposes BiCache through
``prefix_cache``/``prefix_hidden_state`` and combines it with its own dual
cache loop.  Dream has a Hugging Face style ``(key, value)`` cache and a
different shifted-logit generation order, so the adapter lives in this
module.  The cache policy, boundary restore, shallow prefix reuse, deep
prefix recomputation, and refresh interval remain the official BiCache
operations; only the model-facing tensor layout is Dream-specific.
"""

from __future__ import annotations

from array import array
from collections import OrderedDict
from dataclasses import dataclass
import time
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from .bicache_engines import EngineBase


KV = Tuple[torch.Tensor, torch.Tensor]


@dataclass
class DreamInterRequestCache:
    """Prefix-only tensors retained by one official BiCache cache entry."""

    key_values: List[KV]
    boundary_hidden_states: Dict[int, torch.Tensor]
    prefix_len: int


class DreamEngineBase(EngineBase):
    """Shared cache management and Dream prompt/token helpers."""

    def __init__(
        self,
        device: str,
        model,
        tokenizer,
        number_of_inter_request_caching_layer: Dict[int, int],
        intra_request_cache_update_interval: Optional[int] = 16,
        cache_budget: Optional[int] = 5000,
        show_speed: bool = False,
    ):
        super().__init__(
            device=device,
            model=model,
            tokenizer=tokenizer,
            number_of_inter_request_caching_layer=number_of_inter_request_caching_layer,
            intra_request_cache_update_interval=intra_request_cache_update_interval,
            cache_budget=cache_budget,
            show_speed=show_speed,
        )
        self.inter_request_caches: Dict[int, DreamInterRequestCache] = {}
        self.cache_access_history = OrderedDict()
        self.cache_generation_events = 0
        self.cache_lookups = 0
        self.deep_prefix_recompute_events = 0
        self.dual_cache_update_events = 0
        self.periodic_refresh_events = 0
        self.last_audit: Optional[dict] = None
        self.request_audits: List[dict] = []

    def check_memory_capacity(self, prefix_len: int) -> bool:
        if self.cache_budget is None:
            return True
        return prefix_len <= self.cache_budget - self.num_cache_tokens

    def evict_prefix_cache(self):
        if not self.cache_access_history:
            raise RuntimeError("prefix length is longer than cache budget")
        key = self.cache_access_history.popitem(last=False)[0]
        entry = self.inter_request_caches.pop(key)
        self.num_cache_tokens -= entry.prefix_len

    def _assistant_suffix_ids(self) -> List[int]:
        return list(
            self.tokenizer(
                "<|im_start|>assistant\n",
                add_special_tokens=False,
            )["input_ids"]
        )

    def tokenize(self, sequence: List[dict]) -> Tuple[List[int], int]:
        if not sequence or sequence[-1].get("role") != "user":
            raise ValueError("Dream BiCache expects a chat sequence ending in a user message.")
        full_ids = list(
            self.tokenizer.apply_chat_template(
                sequence,
                add_generation_prompt=True,
                tokenize=True,
            )
        )
        if len(sequence) == 1:
            return full_ids, 0
        prefix_with_generation_prompt = list(
            self.tokenizer.apply_chat_template(
                sequence[:-1],
                add_generation_prompt=True,
                tokenize=True,
            )
        )
        suffix_ids = self._assistant_suffix_ids()
        if not suffix_ids or prefix_with_generation_prompt[-len(suffix_ids) :] != suffix_ids:
            raise ValueError("Unexpected Dream assistant generation prompt tokenization.")
        prefix_ids = prefix_with_generation_prompt[: -len(suffix_ids)]
        if full_ids[: len(prefix_ids)] != prefix_ids:
            raise ValueError("Dream BiCache prompt prefix is not a true full-request prefix.")
        return full_ids, len(prefix_ids)

    def get_inter_request_cache(
        self,
        prefix_ids: Sequence[int],
        prefix_ratio: int,
    ) -> Tuple[List[KV], torch.Tensor, int]:
        prefix_ids = [int(value) for value in prefix_ids]
        key = self.get_prefix_hash(prefix_ids)
        number_of_caching_layers = self.get_number_of_inter_request_caching_layer(prefix_ratio)
        if number_of_caching_layers <= 0:
            raise ValueError(f"Dream BiCache policy selected b={number_of_caching_layers} for r={prefix_ratio}.")

        self.cache_lookups += 1
        if key not in self.inter_request_caches:
            if self.show_speed:
                self.cache_misses += 1
                start_time = time.perf_counter()
            while not self.check_memory_capacity(len(prefix_ids)):
                self.evict_prefix_cache()
            self.generate_inter_request_cache(key, prefix_ids)
            self.num_cache_tokens += len(prefix_ids)
            self.cache_generation_events += 1
            if self.show_speed:
                self.cache_miss_overhead += time.perf_counter() - start_time

        if key in self.cache_access_history:
            self.cache_access_history.move_to_end(key)
        else:
            self.cache_access_history[key] = None

        entry = self.inter_request_caches[key]
        if number_of_caching_layers not in entry.boundary_hidden_states:
            raise RuntimeError(
                "Dream BiCache cache entry does not contain the requested boundary state "
                f"for b={number_of_caching_layers}."
            )
        return (
            entry.key_values[:number_of_caching_layers],
            entry.boundary_hidden_states[number_of_caching_layers],
            number_of_caching_layers,
        )

    def generate_inter_request_cache(self, key: int, prefix_ids: List[int]) -> None:
        prefix_tensor = torch.tensor(
            prefix_ids,
            dtype=torch.long,
            device=self.device,
        ).unsqueeze(0)
        with torch.inference_mode():
            output = self.model(
                prefix_tensor,
                use_cache=True,
                output_hidden_states=True,
                num_logits_to_keep=1,
                return_dict=True,
            )

        key_values = [
            (key_tensor.detach(), value_tensor.detach())
            for key_tensor, value_tensor in output.past_key_values
        ]
        hidden_states = output.hidden_states
        boundary_hidden_states: Dict[int, torch.Tensor] = {}
        layer_count = len(key_values)
        for depth in range(1, layer_count + 1):
            # Dream's hidden_states[d] is the input to layer d, i.e. the
            # output after exactly d shallow layers.  For d=L, use the final
            # normalized state because there is no deep layer to resume.
            state = hidden_states[-1] if depth == layer_count else hidden_states[depth]
            boundary_hidden_states[depth] = state.detach()

        self.inter_request_caches[key] = DreamInterRequestCache(
            key_values=key_values,
            boundary_hidden_states=boundary_hidden_states,
            prefix_len=len(prefix_ids),
        )
        del output, prefix_tensor

    def warm_up(self, input_length: int, n_step: int):
        # Keep the official EngineBase hook.  Warm-up is deliberately tiny in
        # the accuracy runner and never changes the request-level cache policy.
        for _ in range(int(n_step)):
            input_ids = torch.randint(
                low=0,
                high=self.tokenizer.vocab_size,
                size=(1, int(input_length)),
                device=self.device,
                dtype=torch.long,
            )
            with torch.inference_mode():
                self.model(input_ids)


class FastdLLMDreamEngine(DreamEngineBase):
    """Official BiCache + Dream Fast-dLLM threshold decoder, batch size one."""

    def __init__(
        self,
        device: str,
        model,
        tokenizer,
        number_of_inter_request_caching_layer: Dict[int, int],
        intra_request_cache_update_interval: Optional[int] = 16,
        cache_budget: Optional[int] = 5000,
        show_speed: bool = False,
        block_length: int = 32,
        threshold: float = 0.9,
    ):
        super().__init__(
            device=device,
            model=model,
            tokenizer=tokenizer,
            number_of_inter_request_caching_layer=number_of_inter_request_caching_layer,
            intra_request_cache_update_interval=intra_request_cache_update_interval,
            cache_budget=cache_budget,
            show_speed=show_speed,
        )
        self.block_length = int(block_length)
        self.threshold = float(threshold)

    @staticmethod
    def align_logits_for_dream(logits: torch.Tensor) -> torch.Tensor:
        """Align predecessor logits before any threshold transfer decision."""

        if logits.shape[1] == 0:
            return logits
        return torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

    @staticmethod
    def _sample_tokens(
        logits: torch.Tensor,
        temperature: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            try:
                tokens = torch.distributions.Categorical(probs=probs).sample()
                confidence = torch.gather(probs, -1, tokens.unsqueeze(-1)).squeeze(-1)
            except RuntimeError:
                confidence, tokens = probs.max(dim=-1)
        else:
            probs = torch.softmax(logits, dim=-1)
            confidence, tokens = probs.max(dim=-1)
        return confidence, tokens

    def _full_bicache_forward(
        self,
        x: torch.Tensor,
        prefix_len: int,
        prefix_cache: Sequence[KV],
        prefix_hidden_state: torch.Tensor,
    ):
        total_len = x.shape[1]
        suffix = x[:, prefix_len:]
        suffix_positions = torch.arange(
            prefix_len,
            total_len,
            device=x.device,
            dtype=torch.long,
        ).unsqueeze(0)
        cache_positions = torch.arange(total_len, device=x.device, dtype=torch.long).unsqueeze(0)
        output = self.model(
            suffix,
            prefix_len=prefix_len,
            prefix_cache=prefix_cache,
            prefix_hidden_state=prefix_hidden_state,
            position_ids=suffix_positions,
            cache_position_ids=cache_positions,
            use_cache=True,
            return_dict=True,
        )
        self.deep_prefix_recompute_events += int(bool(output.bicache_deep_prefix_recomputed))
        return output

    def _dual_cache_forward(
        self,
        x: torch.Tensor,
        start: int,
        end: int,
        past_key_values,
    ):
        total_len = x.shape[1]
        replace_position = torch.zeros(
            (1, total_len),
            dtype=torch.bool,
            device=x.device,
        )
        replace_position[:, start:end] = True
        output = self.model(
            x[:, start:end],
            past_key_values=past_key_values,
            position_ids=torch.arange(start, end, device=x.device, dtype=torch.long).unsqueeze(0),
            cache_position_ids=torch.arange(total_len, device=x.device, dtype=torch.long).unsqueeze(0),
            replace_position=replace_position,
            dual_cache=True,
            use_cache=True,
            return_dict=True,
        )
        self.dual_cache_update_events += 1
        return output

    @staticmethod
    def _block_logits(output, start: int, end: int, prefix_len: int) -> torch.Tensor:
        offset = int(getattr(output, "bicache_output_offset", 0))
        if offset:
            start -= offset
            end -= offset
        if start < 0 or end > output.logits.shape[1]:
            raise RuntimeError(
                f"Dream BiCache output/logit alignment is invalid: block=({start},{end}), "
                f"logits_len={output.logits.shape[1]}, offset={offset}, prefix_len={prefix_len}."
            )
        return output.logits[:, start:end]

    def _refresh_active_prefix(
        self,
        output,
        prefix_len: int,
        number_of_caching_layers: int,
        expected_boundary: torch.Tensor,
    ) -> Tuple[List[KV], torch.Tensor, float]:
        key_values = [
            (key_tensor[:, :prefix_len], value_tensor[:, :prefix_len])
            for key_tensor, value_tensor in output.past_key_values[:number_of_caching_layers]
        ]
        boundary = output.bicache_boundary_hidden_state[:, :prefix_len]
        boundary_error = float((boundary - expected_boundary).abs().max().item())
        return key_values, boundary.detach(), boundary_error

    def generate_token_ids(
        self,
        full_input_ids: torch.Tensor,
        prefix_len: int,
        *,
        steps: int = 128,
        gen_length: int = 256,
        mask_id: int,
        temperature: float = 0.0,
        block_length: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> Tuple[torch.Tensor, int, dict]:
        if full_input_ids.dim() == 1:
            full_input_ids = full_input_ids.unsqueeze(0)
        if full_input_ids.shape[0] != 1:
            raise ValueError("FastdLLMDreamEngine is intentionally batch-size one.")
        if prefix_len <= 0 or prefix_len >= full_input_ids.shape[1]:
            raise ValueError("Dream BiCache requires a non-empty shared prefix shorter than the request.")

        block_length = self.block_length if block_length is None else int(block_length)
        threshold = self.threshold if threshold is None else float(threshold)
        input_len = int(full_input_ids.shape[1])
        total_len = input_len + int(gen_length)
        if gen_length % block_length != 0:
            raise ValueError("gen_length must be divisible by block_length for Dream Fast-dLLM.")
        num_blocks = gen_length // block_length
        if steps % num_blocks != 0:
            raise ValueError("steps must be divisible by the number of Dream Fast-dLLM blocks.")

        x = torch.full(
            (1, total_len),
            int(mask_id),
            dtype=torch.long,
            device=full_input_ids.device,
        )
        x[:, :input_len] = full_input_ids.to(device=x.device, dtype=torch.long)
        prefix_ratio = self.get_prefix_ratio(total_len, int(prefix_len))
        prefix_cache, prefix_hidden_state, number_of_caching_layers = self.get_inter_request_cache(
            x[0, :prefix_len].tolist(), prefix_ratio
        )
        dual_updates_before = self.dual_cache_update_events
        deep_recomputations_before = self.deep_prefix_recompute_events
        periodic_refreshes_before = self.periodic_refresh_events

        audit = {
            "prefix_len": int(prefix_len),
            "total_sequence_length": int(total_len),
            "prefix_ratio_r": int(prefix_ratio),
            "selected_b": int(number_of_caching_layers),
            "layer_count": int(getattr(self.model.config, "num_hidden_layers", 32)),
            "threshold": float(threshold),
            "block_length": int(block_length),
            "steps": int(steps),
            "temperature": float(temperature),
            "intra_request_cache_update_interval": int(self.intra_request_cache_update_interval),
            "prefix_cache_source": "full_input_ids[:F]",
            "dream_shift_alignment": "torch.cat([logits[:, :1], logits[:, :-1]], dim=1)",
            "threshold_applied_after_shift": True,
            "shallow_prefix_kv_reused": False,
            "boundary_hidden_state_restored": False,
            "deep_prefix_recomputation_events": 0,
            "dual_cache_update_events": 0,
            "periodic_refresh_events": 0,
            "refresh_boundary_max_abs_error": 0.0,
            "refresh_markers": 0,
            "threshold_selection_calls": 0,
            "threshold_candidate_tokens": 0,
            "forced_first_tokens": 0,
            "selected_tokens": 0,
            "stopping_events": 0,
            "nfe": 0,
        }

        active_prefix_cache = prefix_cache
        active_boundary = prefix_hidden_state
        inter_request_prefix_cache = prefix_cache
        inter_request_boundary = prefix_hidden_state
        total_steps = 0
        nfe = 0
        start_time = time.perf_counter() if self.show_speed else None

        with torch.inference_mode():
            for block_index in range(num_blocks):
                block_start = input_len + block_index * block_length
                block_end = block_start + block_length

                # The released BiCache engine resets to the inter-request
                # cache at every official refresh marker, including the
                # first block initialization, and replaces it with the
                # freshly recomputed shallow prefix after the forward pass.
                block_refresh_marker = (
                    total_steps % self.intra_request_cache_update_interval == 0
                )
                if block_refresh_marker:
                    active_prefix_cache = inter_request_prefix_cache
                    active_boundary = inter_request_boundary

                # Official Fast-dLLM initializes every block from the current
                # complete request cache.  This call is also the BiCache
                # shallow-KV/boundary/deep-prefix path.
                output = self._full_bicache_forward(
                    x,
                    prefix_len,
                    active_prefix_cache,
                    active_boundary,
                )
                nfe += 1
                audit["nfe"] = nfe
                audit["deep_prefix_recomputation_events"] += int(
                    bool(output.bicache_deep_prefix_recomputed)
                )
                audit["shallow_prefix_kv_reused"] = audit["shallow_prefix_kv_reused"] or bool(
                    output.bicache_shallow_layer_count == number_of_caching_layers
                )
                restored_boundary = output.bicache_boundary_hidden_state[:, :prefix_len]
                boundary_error = float((restored_boundary - active_boundary).abs().max().item())
                audit["refresh_boundary_max_abs_error"] = max(
                    audit["refresh_boundary_max_abs_error"], boundary_error
                )
                audit["boundary_hidden_state_restored"] = audit["boundary_hidden_state_restored"] or boundary_error < 1e-3
                past_key_values = output.past_key_values

                if block_refresh_marker:
                    audit["refresh_markers"] += 1
                    audit["periodic_refresh_events"] += 1
                    self.periodic_refresh_events += 1
                    active_prefix_cache, active_boundary, boundary_error = self._refresh_active_prefix(
                        output,
                        prefix_len,
                        number_of_caching_layers,
                        inter_request_boundary,
                    )
                    audit["refresh_boundary_max_abs_error"] = max(
                        audit["refresh_boundary_max_abs_error"], boundary_error
                    )

                aligned_logits = self.align_logits_for_dream(output.logits)
                output_offset = int(getattr(output, "bicache_output_offset", 0))
                relative_block_start = block_start - output_offset
                relative_block_end = block_end - output_offset
                if relative_block_start < 0 or relative_block_end > aligned_logits.shape[1]:
                    raise RuntimeError(
                        "Dream BiCache initial shifted-logit block is outside the returned "
                        f"logits: absolute=({block_start},{block_end}), offset={output_offset}, "
                        f"returned={aligned_logits.shape[1]}."
                    )
                aligned_block_logits = aligned_logits[:, relative_block_start:relative_block_end]
                _, first_tokens = self._sample_tokens(aligned_block_logits, temperature)
                x[:, block_start] = first_tokens[:, 0]
                # Dream's native block initialization transfers the first
                # token before entering the threshold loop.  Count it in the
                # request audit so selected_tokens covers the whole generated
                # span while threshold_candidate_tokens remains limited to
                # the post-shift threshold decisions.
                audit["forced_first_tokens"] += 1
                audit["selected_tokens"] += 1

                while bool((x[:, block_start:block_end] == int(mask_id)).any()):
                    if total_steps > 0 and total_steps % self.intra_request_cache_update_interval == 0:
                        audit["refresh_markers"] += 1
                        active_prefix_cache = inter_request_prefix_cache
                        active_boundary = inter_request_boundary
                        output = self._full_bicache_forward(
                            x,
                            prefix_len,
                            active_prefix_cache,
                            active_boundary,
                        )
                        nfe += 1
                        audit["nfe"] = nfe
                        audit["periodic_refresh_events"] += 1
                        self.periodic_refresh_events += 1
                        audit["deep_prefix_recomputation_events"] += int(
                            bool(output.bicache_deep_prefix_recomputed)
                        )
                        past_key_values = output.past_key_values
                        aligned_output_logits = self.align_logits_for_dream(output.logits)
                        if output.bicache_output_offset:
                            logits_blk = self.align_logits_for_dream(
                                self._block_logits(output, block_start, block_end, prefix_len)
                            )
                        else:
                            logits_blk = aligned_output_logits[:, block_start:block_end]
                        active_prefix_cache, active_boundary, boundary_error = self._refresh_active_prefix(
                            output,
                            prefix_len,
                            number_of_caching_layers,
                            active_boundary,
                        )
                        audit["refresh_boundary_max_abs_error"] = max(
                            audit["refresh_boundary_max_abs_error"], boundary_error
                        )
                    else:
                        output = self._dual_cache_forward(
                            x,
                            block_start,
                            block_end,
                            past_key_values,
                        )
                        nfe += 1
                        audit["nfe"] = nfe
                        logits_blk = self.align_logits_for_dream(output.logits)
                        past_key_values = output.past_key_values

                    mask_index = x[:, block_start:block_end] == int(mask_id)
                    if not bool(mask_index.any()):
                        break
                    mask_logits = logits_blk[mask_index]
                    confidence, x0 = self._sample_tokens(mask_logits, temperature)
                    x_candidate = torch.full_like(
                        x[:, block_start:block_end],
                        int(mask_id),
                    )
                    full_confidence = torch.full(
                        x[:, block_start:block_end].shape,
                        -torch.inf,
                        device=x.device,
                        dtype=logits_blk.dtype,
                    )
                    x_candidate[mask_index] = x0
                    full_confidence[mask_index] = confidence
                    current_transfer_tokens = int(mask_index.sum().item())
                    selected_confidence, selected_index = torch.topk(
                        full_confidence,
                        current_transfer_tokens,
                        dim=-1,
                    )
                    transfer_index = torch.zeros_like(x_candidate, dtype=torch.bool)
                    transfer_index[:, selected_index[0, 0]] = True
                    if current_transfer_tokens > 1:
                        transfer_index[0, selected_index[0, 1:]] = (
                            selected_confidence[0, 1:] >= threshold
                        )
                    audit["threshold_selection_calls"] += 1
                    audit["threshold_candidate_tokens"] += int(
                        (selected_confidence[0, 1:] >= threshold).sum().item()
                    )
                    audit["forced_first_tokens"] += 1
                    audit["selected_tokens"] += int(transfer_index.sum().item())
                    x[:, block_start:block_end][transfer_index] = x_candidate[transfer_index]
                    total_steps += 1

                audit["stopping_events"] += 1

                # Retain native Dream's block initialization/update order for
                # the next block; no LLaDA token indexing is used here.

        audit["nfe"] = int(nfe)
        audit["deep_prefix_recomputation_events"] = int(audit["deep_prefix_recomputation_events"])
        audit["dual_cache_update_events"] = int(
            self.dual_cache_update_events - dual_updates_before
        )
        audit["periodic_refresh_events"] = int(audit["periodic_refresh_events"])
        audit["engine_counter_delta"] = {
            "deep_prefix_recompute_events": int(
                self.deep_prefix_recompute_events - deep_recomputations_before
            ),
            "dual_cache_update_events": int(
                self.dual_cache_update_events - dual_updates_before
            ),
            "periodic_refresh_events": int(
                self.periodic_refresh_events - periodic_refreshes_before
            ),
        }
        audit["refresh_active"] = audit["periodic_refresh_events"] > 0
        audit["threshold_semantics_validated"] = (
            audit["threshold_applied_after_shift"] and abs(threshold - 0.9) < 1e-12
        )
        if self.show_speed and start_time is not None:
            audit["elapsed_seconds"] = time.perf_counter() - start_time
        self.last_audit = audit
        self.request_audits.append(audit)
        return x[0], int(nfe), audit

    def generate(
        self,
        sequence: List[dict],
        steps: int = 128,
        gen_length: int = 256,
        mask_id: int = 151666,
        temperature: float = 0.0,
    ) -> torch.Tensor:
        input_ids, prefix_len = self.tokenize(sequence)
        input_tensor = torch.tensor(input_ids, dtype=torch.long, device=self.device)
        generated, _, _ = self.generate_token_ids(
            input_tensor,
            prefix_len,
            steps=steps,
            gen_length=gen_length,
            mask_id=mask_id,
            temperature=temperature,
        )
        return generated[input_tensor.shape[0] if input_tensor.dim() > 1 else len(input_ids) :]
