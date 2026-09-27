"""HF, batch-one dKV-Cache+ACache evaluator for LLaDA.

This intentionally uses the ordinary LLaDA Hugging Face model and ACache
prompt harness.  It mirrors the dKV refresh policy used by the Nano runner:
refresh at steps 0/1 and every ``dkv_cache_interval`` thereafter, otherwise
query the one-step-delayed unresolved set.  The static non-Anchor affix KV is
kept in the normal Hugging Face ``past_key_values`` tensor.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.registry import register_model

from eval_ACache import LLaDAAnchorEvalHarness
from generate import add_gumbel_noise
from generate_ACache import (
    ceil_to_int,
    compute_attention_importance_cross_affix,
    select_anchor_tokens,
)
from acache_eval_shared import prepare_cli_args_for_custom_fewshot


KVCache = Sequence[Tuple[torch.Tensor, torch.Tensor]]


def _dkv_transfer_schedule(block_length: int, steps_per_block: int) -> list[int]:
    if block_length <= 0 or steps_per_block <= 0:
        raise ValueError("dKV block length and steps per block must be positive.")
    base, remainder = divmod(block_length, steps_per_block)
    return [base + int(step < remainder) for step in range(steps_per_block)]


def _full_cache_with_static_affix(
    affix_cache: KVCache,
    *,
    total_length: int,
    affix_start: int,
    affix_end: int,
) -> tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
    full_cache = []
    for affix_k, affix_v in affix_cache:
        batch_size, num_heads, affix_length, head_dim = affix_k.shape
        if batch_size != 1:
            raise ValueError("HF dKV+ACache evaluator supports batch_size=1 only.")
        if affix_length != affix_end - affix_start:
            raise ValueError("Precomputed affix KV length does not match the affix span.")
        layer_k = torch.zeros(
            batch_size,
            num_heads,
            total_length,
            head_dim,
            dtype=affix_k.dtype,
            device=affix_k.device,
        )
        layer_v = torch.zeros(
            batch_size,
            num_heads,
            total_length,
            head_dim,
            dtype=affix_v.dtype,
            device=affix_v.device,
        )
        layer_k[:, :, affix_start:affix_end, :] = affix_k
        layer_v[:, :, affix_start:affix_end, :] = affix_v
        full_cache.append((layer_k, layer_v))
    return tuple(full_cache)


@torch.inference_mode()
def generate_with_dkv_anchor_attention(
    model,
    prompt: torch.Tensor,
    *,
    steps: int,
    gen_length: int,
    block_length: int,
    temperature: float,
    remasking: str,
    mask_id: int,
    affix_start: int,
    affix_end: Optional[int],
    generation_start: Optional[int],
    anchor_ratio: float,
    selection_mode: str,
    drop_non_anchor: bool,
    dkv_cache_interval: int,
    precomputed_affix_cache: Optional[KVCache],
) -> tuple[torch.Tensor, int]:
    """Run dKV decoding using HF ``past_key_values`` and an ACache affix.

    The normal HF ACache evaluator is intentionally single-request.  Keeping
    that invariant avoids padding/layout changes and makes the quality path
    directly comparable to the existing ACache accuracy runs.
    """
    if prompt.shape[0] != 1:
        raise ValueError("HF dKV+ACache evaluator supports batch_size=1 only.")
    if drop_non_anchor:
        raise ValueError("HF dKV+ACache temporary evaluator does not support drop_non_anchor=True.")
    if precomputed_affix_cache is None or not precomputed_affix_cache:
        raise ValueError("dKV+ACache requires a precomputed affix KV cache.")
    if gen_length <= 0 or block_length <= 0 or gen_length % block_length:
        raise ValueError("gen_length must be a positive multiple of block_length.")
    if steps <= 0 or dkv_cache_interval <= 0:
        raise ValueError("dKV steps and cache interval must be positive.")

    batch_size, prompt_length = prompt.shape
    num_blocks = gen_length // block_length
    if steps % num_blocks:
        raise ValueError("dKV steps must be divisible by the number of generation blocks.")
    steps_per_block = steps // num_blocks
    transfer_schedule = _dkv_transfer_schedule(block_length, steps_per_block)

    if generation_start is None:
        x = torch.full(
            (batch_size, prompt_length + gen_length),
            mask_id,
            dtype=torch.long,
            device=model.device,
        )
        x[:, :prompt_length] = prompt
        generation_start = prompt_length
    else:
        generation_start = int(generation_start)
        generation_end = generation_start + gen_length
        if generation_start < 0 or generation_end > prompt_length:
            raise ValueError("Invalid in-place generation span for dKV+ACache.")
        x = prompt.clone().to(model.device)
        x[:, generation_start:generation_end] = mask_id

    total_length = int(x.shape[1])
    generation_end = generation_start + gen_length
    if affix_end is None:
        affix_end = prompt_length
    affix_start = int(affix_start)
    affix_end = int(affix_end)
    if not (0 <= affix_start <= affix_end <= total_length):
        raise ValueError("Invalid affix span for dKV+ACache.")
    if generation_start < affix_end and generation_end > affix_start:
        raise ValueError("The generation and affix spans must not overlap.")

    affix_length = affix_end - affix_start
    cached_affix_length = int(precomputed_affix_cache[0][0].shape[2])
    if cached_affix_length != affix_length:
        raise ValueError(
            f"Precomputed affix KV length {cached_affix_length} does not match affix length {affix_length}."
        )

    nfe = 0
    num_anchor = min(ceil_to_int(float(anchor_ratio) * affix_length), affix_length)
    if num_anchor > 0:
        importance = compute_attention_importance_cross_affix(
            model,
            x,
            precomputed_affix_cache,
            mask_id=mask_id,
            affix_start=affix_start,
        )
        anchor_positions = select_anchor_tokens(
            importance,
            num_anchor,
            selection_mode,
        )[0].add(affix_start)
        nfe += 1
    else:
        anchor_positions = torch.empty(0, dtype=torch.long, device=x.device)

    # Dynamic positions are recomputed at every dKV refresh.  All remaining
    # affix positions preserve their standalone precomputed KV cache.
    recompute_mask = torch.ones(batch_size, total_length, dtype=torch.bool, device=x.device)
    recompute_mask[:, affix_start:affix_end] = False
    if anchor_positions.numel():
        recompute_mask[0, anchor_positions] = True
    recompute_positions = recompute_mask[0].nonzero(as_tuple=True)[0]
    if not recompute_positions.numel():
        raise RuntimeError("dKV+ACache requires at least one dynamic query position.")

    past_key_values = _full_cache_with_static_affix(
        precomputed_affix_cache,
        total_length=total_length,
        affix_start=affix_start,
        affix_end=affix_end,
    )

    def query_positions(positions: list[int] | torch.Tensor):
        nonlocal past_key_values, nfe
        position_ids = torch.as_tensor(positions, dtype=torch.long, device=x.device)
        if position_ids.ndim != 1 or position_ids.numel() == 0:
            raise RuntimeError("dKV query positions must be a non-empty one-dimensional list.")
        query_input = x.index_select(1, position_ids)
        replace_position = torch.zeros_like(recompute_mask)
        replace_position[:, position_ids] = True
        output = model(
            query_input,
            past_key_values=past_key_values,
            use_cache=True,
            replace_position=replace_position,
            position_ids=position_ids,
        )
        past_key_values = output.past_key_values
        nfe += 1
        return output.logits, position_ids

    for block_idx in range(num_blocks):
        block_start = generation_start + block_idx * block_length
        block_end = block_start + block_length
        previous_unresolved: Optional[list[int]] = None
        delayed_unresolved: Optional[list[int]] = None

        for step, quota in enumerate(transfer_schedule):
            refresh = step <= 1 or step % dkv_cache_interval == 0
            if refresh:
                query_set = recompute_positions
            else:
                if delayed_unresolved is None:
                    raise RuntimeError("dKV compact decoding has no delayed unresolved-token state.")
                query_set = delayed_unresolved

            logits, queried_positions = query_positions(query_set)
            query_index = {int(pos): index for index, pos in enumerate(queried_positions.tolist())}
            current_mask_positions = [
                position
                for position in range(block_start, block_end)
                if int(x[0, position]) == int(mask_id)
            ]
            if not current_mask_positions:
                break
            try:
                current_query_rows = torch.tensor(
                    [query_index[position] for position in current_mask_positions],
                    dtype=torch.long,
                    device=x.device,
                )
            except KeyError as exc:
                raise RuntimeError("dKV delayed query omitted a currently masked token.") from exc

            masked_logits = logits.index_select(1, current_query_rows)
            proposals = torch.argmax(add_gumbel_noise(masked_logits, temperature), dim=-1)
            if remasking == "low_confidence":
                probabilities = torch.softmax(masked_logits.to(torch.float64), dim=-1)
                confidence = torch.gather(
                    probabilities,
                    dim=-1,
                    index=proposals.unsqueeze(-1),
                ).squeeze(-1)
            elif remasking == "random":
                confidence = torch.rand(
                    proposals.shape,
                    dtype=torch.float64,
                    device=proposals.device,
                )
            else:
                raise NotImplementedError(remasking)

            selected_count = min(int(quota), len(current_mask_positions))
            if selected_count:
                selected = torch.topk(confidence[0], k=selected_count).indices.tolist()
                for local_index in selected:
                    x[0, current_mask_positions[local_index]] = proposals[0, local_index]

            current_unresolved = [
                position
                for position in range(block_start, generation_end)
                if int(x[0, position]) == int(mask_id)
            ]
            delayed_unresolved = previous_unresolved
            previous_unresolved = current_unresolved

        if bool((x[:, block_start:block_end] == mask_id).any()):
            raise RuntimeError("dKV transfer schedule did not finish the current block.")

    return x, nfe


@register_model("llada_dkv_acache_hf")
class LLaDADKVACacheHFEvalHarness(LLaDAAnchorEvalHarness):
    """The standard HF ACache harness with dKV decoding instead of dual cache."""

    def __init__(self, dkv_steps=0, dkv_cache_interval=8, **kwargs):
        super().__init__(**kwargs)
        self.dkv_steps = int(dkv_steps) or int(self.steps)
        self.dkv_cache_interval = int(dkv_cache_interval)
        if self.dkv_steps <= 0 or self.dkv_cache_interval <= 0:
            raise ValueError("dkv_steps and dkv_cache_interval must be positive.")

    def _generate_with_affix_cache(
        self,
        input_ids: torch.Tensor,
        affix_start: int,
        affix_end: int,
        generation_start: int,
        affix_state,
    ):
        return generate_with_dkv_anchor_attention(
            self.model,
            input_ids,
            steps=self.dkv_steps,
            gen_length=self.gen_length,
            block_length=self.block_length,
            temperature=0.0,
            remasking=self.remasking,
            mask_id=self.mask_id,
            affix_start=affix_start,
            affix_end=affix_end,
            generation_start=generation_start if self.affix_type == "suffix" else None,
            anchor_ratio=self.anchor_ratio,
            selection_mode=self.selection_mode,
            drop_non_anchor=self.drop_non_anchor,
            dkv_cache_interval=self.dkv_cache_interval,
            precomputed_affix_cache=affix_state["precomputed_affix_cache"],
        )


if __name__ == "__main__":
    cli_evaluate(prepare_cli_args_for_custom_fewshot())
