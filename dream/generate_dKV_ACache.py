"""dKV-Cache-Decode with ACache for Dream's Hugging Face model.

Dream dKV uses a shifted cache-position mask (``shift_type='un'``), not the
LLaDA delayed-unresolved-token schedule.  A masked token at position ``i`` is
predicted from the raw logit at ``i - 1``.  The shifted mask consequently makes
that predecessor a query on the next iteration.  This module implements that
contract on top of Dream's full-slot ``dual_cache`` representation: frozen
non-Anchor affix slots remain in the cache, while dynamic slots are refreshed
or queried according to Dream's dKV mask.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch

from generate_ACache import (
    _align_logits_for_dream,
    add_gumbel_noise,
    ceil_to_int,
    compute_attention_importance_cross_affix,
    select_anchor_tokens,
)


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
    """Build Dream's full-length dual cache with only the affix initialized."""
    full_cache = []
    for affix_k, affix_v in affix_cache:
        if affix_k.ndim != 3 or affix_v.ndim != 3:
            raise ValueError("Dream affix caches must have shape [batch, sequence, kv_hidden].")
        batch_size, affix_length, kv_hidden = affix_k.shape
        if batch_size != 1:
            raise ValueError("HF dKV+ACache evaluation supports batch_size=1 only.")
        if affix_v.shape != (batch_size, affix_length, kv_hidden):
            raise ValueError("Dream affix key/value cache shapes do not match.")
        if affix_length != affix_end - affix_start:
            raise ValueError("Precomputed affix KV length does not match the affix span.")

        layer_k = torch.zeros(
            batch_size,
            total_length,
            kv_hidden,
            dtype=affix_k.dtype,
            device=affix_k.device,
        )
        layer_v = torch.zeros(
            batch_size,
            total_length,
            kv_hidden,
            dtype=affix_v.dtype,
            device=affix_v.device,
        )
        layer_k[:, affix_start:affix_end, :] = affix_k
        layer_v[:, affix_start:affix_end, :] = affix_v
        full_cache.append((layer_k, layer_v))
    return tuple(full_cache)


def _dream_shifted_cache_positions(x: torch.Tensor, mask_id: int) -> torch.Tensor:
    """Return Dream dKV's ``shift_type='un'`` cache-position mask.

    The cache bit at ``i`` is the resolved-state bit at ``i + 1`` (with the
    upstream cyclic boundary convention).  Therefore, if token ``i + 1`` is
    still masked, raw logits at ``i`` are recomputed on the following step.
    """
    resolved = x.ne(mask_id)
    return torch.cat((resolved[:, 1:], resolved[:, :1]), dim=1)


def _scatter_then_align_dream_logits(
    logits: torch.Tensor,
    query_positions: torch.Tensor,
    *,
    total_length: int,
) -> torch.Tensor:
    """Restore sparse raw logits to absolute positions before Dream's shift.

    dKV forwards only a subset of sequence positions.  Applying Dream's
    previous-token shift directly to that compact tensor shifts *query rows*,
    which is incorrect whenever the positions are non-contiguous.  Upstream
    Dream dKV scatters first and shifts second; retain that exact ordering.
    """
    if logits.ndim != 3:
        raise ValueError("Dream logits must have shape [batch, query_length, vocab].")
    if query_positions.ndim != 1 or query_positions.numel() != logits.shape[1]:
        raise ValueError("Dream dKV logits and query positions have incompatible shapes.")
    if query_positions.numel() == 0:
        raise ValueError("Dream dKV requires at least one query position.")
    if int(query_positions.min()) < 0 or int(query_positions.max()) >= total_length:
        raise ValueError("Dream dKV query position is outside the logical sequence.")
    if query_positions.unique().numel() != query_positions.numel():
        raise ValueError("Dream dKV query positions must be unique.")

    full_logits = logits.new_full(
        (logits.shape[0], total_length, logits.shape[-1]),
        -torch.inf,
    )
    full_logits.index_copy_(1, query_positions, logits)
    return _align_logits_for_dream(full_logits)


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
    """Decode a single Dream request with dKV refreshes and frozen affix KV.

    ``drop_non_anchor`` remains intentionally unsupported.  Unlike LLaDA,
    Dream's dKV compact steps query the complement of the previous shifted
    cache mask.  Frozen affix positions are normally excluded from updates but
    are queried read-only when one is needed to produce a masked successor's
    logit.
    """
    if prompt.ndim != 2 or prompt.shape[0] != 1:
        raise ValueError("HF dKV+ACache evaluation supports batch_size=1 only.")
    if drop_non_anchor:
        raise ValueError("HF dKV+ACache does not support drop_non_anchor=True.")
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
            device=prompt.device,
        )
        x[:, :prompt_length] = prompt
        generation_start = prompt_length
    else:
        generation_start = int(generation_start)
        generation_end = generation_start + gen_length
        if generation_start < 0 or generation_end > prompt_length:
            raise ValueError("Invalid in-place generation span for dKV+ACache.")
        x = prompt.clone()
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
    if precomputed_affix_cache is None or not precomputed_affix_cache:
        raise ValueError("dKV+ACache requires a precomputed affix KV cache.")

    affix_length = affix_end - affix_start
    cached_affix_length = int(precomputed_affix_cache[0][0].shape[1])
    if cached_affix_length != affix_length:
        raise ValueError(
            f"Precomputed affix KV length {cached_affix_length} does not match affix length {affix_length}."
        )

    nfe = 0
    num_anchor = min(ceil_to_int(float(anchor_ratio) * affix_length), affix_length)
    if num_anchor >= affix_length and affix_length > 0:
        anchor_positions = torch.arange(affix_start, affix_end, dtype=torch.long, device=x.device)
    elif num_anchor > 0:
        importance = compute_attention_importance_cross_affix(
            model,
            x,
            precomputed_affix_cache,
            mask_id=mask_id,
            affix_start=affix_start,
        )
        anchor_positions = select_anchor_tokens(importance, num_anchor, selection_mode)[0].add(affix_start)
        nfe += 1
    else:
        anchor_positions = torch.empty(0, dtype=torch.long, device=x.device)

    # Dynamic slots consist of all non-affix positions plus the selected
    # Anchor positions.  The remaining affix KV stays frozen in the full cache.
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
    cache_position_ids = torch.arange(total_length, dtype=torch.long, device=x.device).unsqueeze(0)
    frozen_affix_mask = ~recompute_mask

    def query_positions(
        positions: torch.Tensor,
        update_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Run sparse Dream queries and return absolute-position aligned logits.

        ``positions`` may contain a frozen affix predecessor needed solely for
        its query/logit.  ``update_positions`` is restricted to dynamic slots,
        so read-only queries never overwrite frozen ACache KV.
        """
        nonlocal past_key_values, nfe
        position_ids = torch.as_tensor(positions, dtype=torch.long, device=x.device)
        update_position_ids = torch.as_tensor(update_positions, dtype=torch.long, device=x.device)
        if position_ids.ndim != 1 or position_ids.numel() == 0:
            raise RuntimeError("dKV query positions must be a non-empty one-dimensional list.")
        if update_position_ids.ndim != 1:
            raise RuntimeError("dKV update positions must be one-dimensional.")
        if update_position_ids.numel() and not bool(torch.isin(update_position_ids, position_ids).all()):
            raise RuntimeError("dKV cache-update positions must be included in the query positions.")
        query_input = x.index_select(1, position_ids)
        replace_position = torch.zeros_like(recompute_mask)
        if update_position_ids.numel():
            replace_position[:, update_position_ids] = True
        output = model(
            query_input,
            past_key_values=past_key_values,
            use_cache=True,
            dual_cache=True,
            replace_position=replace_position,
            position_ids=position_ids.unsqueeze(0),
            cache_position_ids=cache_position_ids,
        )
        past_key_values = output.past_key_values
        nfe += 1
        return _scatter_then_align_dream_logits(
            output.logits,
            position_ids,
            total_length=total_length,
        )

    for block_idx in range(num_blocks):
        block_start = generation_start + block_idx * block_length
        block_end = block_start + block_length
        previous_cache_positions: Optional[torch.Tensor] = None

        for step, quota in enumerate(transfer_schedule):
            current_mask_positions = torch.arange(
                block_start,
                block_end,
                dtype=torch.long,
                device=x.device,
            )
            current_mask_positions = current_mask_positions[
                x[0, current_mask_positions].eq(mask_id)
            ]
            if not current_mask_positions.numel():
                break

            # Match upstream Dream dKV: a decoded-cache refresh occurs at the
            # first step and then at each cache interval.  The static cache
            # need not be cleared because every dynamic slot is overwritten on
            # a refresh; frozen affix slots intentionally remain intact.
            refresh = step % dkv_cache_interval == 0
            if refresh:
                dynamic_query_mask = recompute_mask[0].clone()
            else:
                if previous_cache_positions is None:
                    raise RuntimeError("Dream dKV compact decoding has no prior cache-position state.")
                dynamic_query_mask = recompute_mask[0] & ~previous_cache_positions[0]

            # A masked position p consumes the raw logit at p - 1 after
            # Dream's shift.  Include every predecessor even when it is a
            # frozen affix token; frozen predecessors are read-only queries.
            predecessor_positions = (current_mask_positions - 1).remainder(total_length)
            query_mask = dynamic_query_mask.clone()
            query_mask[predecessor_positions] = True
            query_set = query_mask.nonzero(as_tuple=True)[0]
            update_set = dynamic_query_mask.nonzero(as_tuple=True)[0]

            logits = query_positions(query_set, update_set)
            masked_logits = logits.index_select(1, current_mask_positions)
            if not bool(torch.isfinite(masked_logits).any(dim=-1).all()):
                raise RuntimeError("Dream dKV did not compute logits for every current mask position.")
            proposals = torch.argmax(add_gumbel_noise(masked_logits, temperature), dim=-1)
            if remasking == "low_confidence":
                probabilities = torch.softmax(masked_logits.to(torch.float64), dim=-1)
                confidence = torch.gather(
                    probabilities,
                    dim=-1,
                    index=proposals.unsqueeze(-1),
                ).squeeze(-1)
            elif remasking == "random":
                confidence = torch.rand(proposals.shape, dtype=torch.float64, device=proposals.device)
            else:
                raise NotImplementedError(remasking)

            selected_count = min(int(quota), int(current_mask_positions.numel()))
            if selected_count:
                selected = torch.topk(confidence[0], k=selected_count).indices.tolist()
                for local_index in selected:
                    x[0, current_mask_positions[local_index]] = proposals[0, local_index]

            previous_cache_positions = _dream_shifted_cache_positions(x, mask_id)
            # Frozen ACache slots are persistent cache entries regardless of
            # the dKV mask.  They may still be read-only query positions when
            # needed for a masked successor's logits.
            previous_cache_positions |= frozen_affix_mask

        if bool((x[:, block_start:block_end] == mask_id).any()):
            raise RuntimeError("dKV transfer schedule did not finish the current block.")

    return x, nfe
