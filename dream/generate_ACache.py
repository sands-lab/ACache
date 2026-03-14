"""
ACache Implementation

This module implements ACache for Diffusion LLMs. The key idea is:
1. For an affix (prefix/suffix/infix), select K "Anchor" tokens based on attention importance
2. Freeze non-Anchor tokens' KV cache in the affix
3. Only recompute K Anchor tokens' KV cache at block boundaries

This reduces computation while maintaining generation quality.
"""

import argparse
import inspect
import types
from typing import Optional, Tuple, List

import numpy as np
import torch
import torch.nn.functional as F
from model.generation_utils_block import sample_tokens
from model.modeling_dream import apply_rotary_pos_emb, repeat_kv


def _coerce_bool(value, default: bool, arg_name: str) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)

    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"{arg_name} must be a boolean-like value, got {value!r}.")


def ceil_to_int(value: float) -> int:
    integer = int(value)
    if value == integer:
        return integer
    return integer + 1


def _align_logits_for_dream(logits: torch.Tensor) -> torch.Tensor:
    """Dream predicts token i from hidden state i; generation expects shifted logits."""
    return torch.cat([logits[:, :1], logits[:, :-1]], dim=1)


def add_gumbel_noise(logits, temperature):
    """
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves
    perplexity score but reduces generation quality. Thus, we use float64.
    """
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(block_mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    """
    block_mask_index: (B, L) bool – which positions are masked in the current block
    returns: (B, steps) int – how many tokens to transfer at each step per batch item
    """
    device = block_mask_index.device
    dtype = torch.long

    total = block_mask_index.sum(dim=1)  # (B,)
    base = torch.div(total, steps, rounding_mode="floor")  # (B,)
    rem = total - base * steps  # (B,)

    num_transfer_tokens = base.unsqueeze(1).expand(-1, steps).to(dtype)  # (B, steps)
    cols = torch.arange(steps, device=device).unsqueeze(0)  # (1, steps)
    add_mask = cols < rem.unsqueeze(1)  # (B, steps)
    return num_transfer_tokens + add_mask.to(dtype)


def get_transfer_index(
    logits: torch.Tensor,
    temperature: float,
    remasking: str,
    mask_index: torch.Tensor,
    x: torch.Tensor,
    num_transfer_tokens,
    threshold: float = None,
):
    """
    Returns:
        x0: (B, L) long — proposed tokens
        transfer_index: (B, L) bool — which positions to update this step
    """
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)

    if remasking == "low_confidence":
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
    elif remasking == "random":
        x0_p = torch.rand(x0.shape, device=x0.device, dtype=torch.float64)
    else:
        raise NotImplementedError(remasking)

    x0 = torch.where(mask_index, x0, x)

    neg_inf = torch.tensor(torch.finfo(x0_p.dtype).min, device=x0_p.device, dtype=x0_p.dtype)
    confidence = torch.where(mask_index, x0_p, neg_inf)

    if threshold is not None:
        transfer_index = mask_index & (confidence >= threshold)
        max_conf_indices = torch.argmax(confidence, dim=1, keepdim=True)
        force_mask = torch.zeros_like(transfer_index).scatter_(1, max_conf_indices, True)
        transfer_index = (transfer_index | force_mask) & mask_index
        return x0, transfer_index

    if num_transfer_tokens is None:
        raise ValueError("num_transfer_tokens must be a tensor when threshold is None.")

    if num_transfer_tokens.dim() == 2 and num_transfer_tokens.size(1) == 1:
        num_transfer_tokens = num_transfer_tokens.squeeze(1)
    num_transfer_tokens = num_transfer_tokens.to(dtype=torch.long, device=confidence.device)
    num_transfer_tokens = torch.clamp(num_transfer_tokens, min=0)

    _, idx = torch.sort(confidence, dim=1, descending=True)
    bsz, seqlen = confidence.shape
    cols = torch.arange(seqlen, device=confidence.device).unsqueeze(0).expand(bsz, seqlen)
    k_expanded = num_transfer_tokens.unsqueeze(1).expand(bsz, seqlen)
    select_sorted = cols < k_expanded

    transfer_int = torch.zeros(bsz, seqlen, device=confidence.device, dtype=torch.int8)
    transfer_int = transfer_int.scatter(1, idx, select_sorted.to(torch.int8))
    transfer_index = transfer_int.bool() & mask_index
    return x0, transfer_index


def get_transfer_index_dynamic(logits, temperature, remasking, mask_index, x, num_transfer_tokens, factor=1):
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)
    if remasking == "low_confidence":
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
    elif remasking == "random":
        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    else:
        raise NotImplementedError(remasking)

    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, -np.inf)

    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    num_transfer_tokens = mask_index.sum(dim=1, keepdim=True)

    for j in range(confidence.shape[0]):
        num_tokens = int(num_transfer_tokens[j].item())
        if num_tokens == 0:
            continue

        ns = list(range(1, num_tokens + 1))
        es = [factor / (n + 1) for n in ns]
        threshs = [1 - e for e in es]

        threshs[0] = -1
        sorted_confidence = torch.sort(confidence[j][mask_index[j]], dim=-1, descending=True)[0]
        assert len(sorted_confidence) == len(threshs)
        for top_i in range(len(threshs)):
            if sorted_confidence[top_i] < threshs[top_i]:
                break

        if top_i == 0 or top_i == len(threshs) - 1:
            top_i += 1

        _, select_index = torch.topk(confidence[j], k=top_i)
        transfer_index[j, select_index] = True

    return x0, transfer_index


def compute_attention_importance_cross_affix(
    model,
    full_input_ids: torch.Tensor,
    affix_cache: Tuple[Tuple[torch.Tensor, torch.Tensor], ...],
    mask_id: int,
    affix_start: int = 0,
):
    """
    Compute affix-token importance using affix-keyed attention.

    Queries come from the sample's mask-token positions while keys are restricted to
    affix tokens. This keeps affix cache reusable while making Anchor selection
    sample-dependent.
    """
    bsz, seqlen = full_input_ids.shape
    affix_len = affix_cache[0][0].shape[1] if len(affix_cache) > 0 else 0
    mask_query_mask = full_input_ids == mask_id
    query_position_ids = torch.arange(seqlen, device=full_input_ids.device, dtype=torch.long).unsqueeze(0)
    query_position_ids = query_position_ids.expand(bsz, -1)
    affix_position_ids = torch.arange(
        affix_start,
        affix_start + affix_len,
        device=full_input_ids.device,
        dtype=torch.long,
    ).unsqueeze(0).expand(bsz, -1)

    with torch.no_grad():
        if affix_len <= 0:
            return torch.zeros(bsz, 0, device=full_input_ids.device)
        if seqlen <= 0:
            return torch.zeros(bsz, affix_len, device=full_input_ids.device)
        if not mask_query_mask.any():
            return torch.zeros(bsz, affix_len, device=full_input_ids.device)

        importance_sum = torch.zeros(bsz, affix_len, device=full_input_ids.device, dtype=torch.float32)
        layers = model.model.layers
        original_forwards = []

        def make_patched_forward(original_forward, accepted_params, layer_idx):
            def patched_forward(
                self,
                hidden_states,
                attention_mask=None,
                position_ids=None,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
                cache_position=None,
                position_embeddings=None,
                replace_position=None,
                dual_cache=False,
                **extra_kwargs,
            ):
                local_bsz, local_len, _ = hidden_states.size()

                query_states = self.q_proj(hidden_states)
                query_states = query_states.view(
                    local_bsz,
                    local_len,
                    self.num_heads,
                    self.head_dim,
                ).transpose(1, 2)

                affix_k = affix_cache[layer_idx][0]
                affix_k = affix_k.view(
                    local_bsz,
                    affix_len,
                    self.num_key_value_heads,
                    self.head_dim,
                ).transpose(1, 2)

                query_cos, query_sin = self.rotary_emb(query_states, query_position_ids[:local_bsz])
                q_rot, _ = apply_rotary_pos_emb(query_states, query_states, query_cos, query_sin)

                affix_cos, affix_sin = self.rotary_emb(affix_k, affix_position_ids[:local_bsz])
                _, affix_k_rot = apply_rotary_pos_emb(affix_k, affix_k, affix_cos, affix_sin)
                affix_k_rot = repeat_kv(affix_k_rot, self.num_key_value_groups)

                scaling = self.head_dim ** -0.5
                attention_scores = torch.matmul(q_rot, affix_k_rot.transpose(2, 3)) * scaling
                attention_weights = torch.softmax(attention_scores, dim=-1)

                mask_weights = mask_query_mask[:local_bsz, :local_len][:, None, :, None].to(attention_weights.dtype)
                token_importance = (attention_weights * mask_weights).sum(dim=-2).sum(dim=-2)
                if token_importance.dtype != torch.float32:
                    token_importance = token_importance.float()
                importance_sum[:local_bsz].add_(token_importance)

                call_kwargs = {
                    "attention_mask": attention_mask,
                    "position_ids": position_ids,
                    "past_key_value": past_key_value,
                    "output_attentions": output_attentions,
                    "use_cache": use_cache,
                    "cache_position": cache_position,
                    "position_embeddings": position_embeddings,
                    "replace_position": replace_position,
                    "dual_cache": dual_cache,
                }
                call_kwargs.update(extra_kwargs)
                filtered_kwargs = {k: v for k, v in call_kwargs.items() if k in accepted_params}
                return original_forward(hidden_states, **filtered_kwargs)

            return patched_forward

        for layer_idx, decoder_layer in enumerate(layers):
            attention = decoder_layer.self_attn
            original_forward = attention.forward
            signature = inspect.signature(original_forward)
            accepted_params = {name for name in signature.parameters if name != "self"}
            original_forwards.append((attention, original_forward))
            attention.forward = types.MethodType(
                make_patched_forward(original_forward, accepted_params, layer_idx),
                attention,
            )

        try:
            model.model.forward(
                input_ids=full_input_ids,
                use_cache=False,
                output_hidden_states=False,
            )
        finally:
            for attention, original_forward in original_forwards:
                attention.forward = original_forward

    return importance_sum


def select_anchor_tokens(
    importance_scores: torch.Tensor,
    k: int,
    selection_mode: str = 'top',
) -> torch.Tensor:
    """
    Select top-K or bottom-K Anchor tokens based on importance scores.

    Args:
        importance_scores: (B, affix_length) importance scores
        k: Number of Anchor tokens to select
        selection_mode: 'top' for highest importance, 'bottom' for lowest importance

    Returns:
        anchor_indices: (B, K) indices of Anchor tokens within the affix
    """
    B, affix_length = importance_scores.shape
    k = min(k, affix_length)

    # Select top-K or bottom-K indices
    if selection_mode == 'bottom':
        _, anchor_indices = torch.topk(importance_scores, k, dim=-1, largest=False)  # (B, K)
    else:
        _, anchor_indices = torch.topk(importance_scores, k, dim=-1, largest=True)  # (B, K)

    # Sort indices to maintain order
    anchor_indices = anchor_indices.sort(dim=-1).values

    return anchor_indices


def gather_tokens_for_recompute(
    x: torch.Tensor,
    anchor_global_indices: torch.Tensor,
    block_start: int,
    block_end: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Gather tokens that need to be recomputed (Anchor tokens + block tokens).

    Args:
        x: Full sequence (B, L)
        anchor_global_indices: Global indices of Anchor tokens (B, K)
        block_start: Start index of current block
        block_end: End index of current block

    Returns:
        gathered_tokens: Tokens to be recomputed (B, K + block_length)
        replace_position: Boolean mask indicating which positions to replace (B, L)
    """
    B, L = x.shape
    K = anchor_global_indices.shape[1]
    block_length = block_end - block_start

    # Create replace_position mask
    replace_position = torch.zeros(B, L, dtype=torch.bool, device=x.device)

    # Mark Anchor positions
    for b in range(B):
        replace_position[b, anchor_global_indices[b]] = True

    # Mark block positions
    replace_position[:, block_start:block_end] = True

    # Gather tokens in order
    # We need tokens at Anchor positions first, then block positions
    gathered = []
    for b in range(B):
        anchor_tokens = x[b, anchor_global_indices[b]]  # (K,)
        block_tokens = x[b, block_start:block_end]  # (block_length,)
        gathered.append(torch.cat([anchor_tokens, block_tokens]))

    gathered_tokens = torch.stack(gathered, dim=0)  # (B, K + block_length)

    return gathered_tokens, replace_position


@torch.no_grad()
def generate_with_anchor_attention(
    model,
    prompt: torch.Tensor,
    steps: int = 128,
    gen_length: int = 128,
    block_length: int = 128,
    temperature: float = 0.,
    remasking: str = 'low_confidence',
    mask_id: int = 126336,
    threshold: Optional[float] = None,
    factor: Optional[float] = None,
    affix_start: int = 0,
    affix_end: Optional[int] = None,
    generation_start: Optional[int] = None,
    anchor_ratio: float = 0.1,
    selection_mode: str = 'top',
    drop_non_anchor: bool = False,
    precomputed_affix_cache: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
) -> Tuple[torch.Tensor, int]:
    """
    Generate with ACache - recompute Anchor tokens + block, freeze non-Anchor affix.

    The algorithm:
    1. Compute full KV cache from entire sequence (including [MASK] tokens)
    2. Select K Anchor tokens from affix based on importance
    3. For subsequent blocks:
       a. Recompute Anchor tokens' KV cache (so they see current generation state)
       b. Recompute block tokens' KV cache
       c. Non-Anchor affix tokens keep frozen KV cache

    Args:
        model: The Dream model.
        prompt: Input prompt tensor (B, prompt_length)
        steps: Number of generation steps
        gen_length: Length of generated sequence
        block_length: Block size for semi-autoregressive generation
        temperature: Sampling temperature
        remasking: Remasking strategy
        mask_id: Mask token ID
        threshold: Confidence threshold for parallel decoding
        factor: Dynamic threshold factor
        affix_start: Start position of affix in prompt
        affix_end: End position of affix (exclusive), defaults to prompt length
        generation_start: Optional start position of generated span. If None,
            generation happens on an appended tail of length `gen_length`.
            If set, generation happens in-place on
            [generation_start, generation_start + gen_length) inside `prompt`.
        anchor_ratio: Ratio of affix tokens selected as Anchor tokens. K is ceil(anchor_ratio * affix_len).
        selection_mode: 'top' for highest importance, 'bottom' for lowest importance
        drop_non_anchor: If True, run the request directly on the compact
            kept-position sequence (Anchor tokens + non-affix tokens), so
            non-Anchor affix tokens are absent from block-0 logits onward.
        precomputed_affix_cache: Precomputed KV cache for affix tokens.
            Must be provided by caller using the same affix tokens and absolute
            affix positions as `affix_start:affix_end`.

    Returns:
        generated: Generated sequence (B, prompt_length + gen_length)
        nfe: Number of forward evaluations
    """
    B = prompt.shape[0]
    assert B == 1, "generate_with_anchor_attention currently supports batch_size=1."
    Lp = int(prompt.shape[1])

    drop_non_anchor = _coerce_bool(drop_non_anchor, default=False, arg_name="drop_non_anchor")

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    if generation_start is None:
        # Standard completion: append generation masks to prompt tail.
        x = torch.full((B, Lp + gen_length), mask_id, dtype=torch.long, device=model.device)
        x[:, :Lp] = prompt
        generation_start = Lp
        total_len = Lp + gen_length
    else:
        # Infill: generate in-place inside prompt while keeping right-context tokens fixed.
        generation_start = int(generation_start)
        if generation_start < 0:
            raise ValueError("generation_start must be non-negative.")
        generation_end = generation_start + gen_length
        if generation_end > Lp:
            raise ValueError("generation_start + gen_length must be <= prompt length for infill mode.")
        x = prompt.clone().to(model.device)
        x[:, generation_start:generation_end] = mask_id
        total_len = Lp

    generation_end = generation_start + gen_length

    if affix_end is None:
        affix_end = Lp
    if affix_end > total_len:
        raise ValueError("affix_end must be <= sequence length.")
    if not (affix_start <= affix_end):
        raise ValueError("affix_start must be <= affix_end.")
    if generation_start < affix_end and generation_end > affix_start:
        raise ValueError(
            "Generation span must not overlap affix span. "
            f"Got generation=[{generation_start}, {generation_end}), affix=[{affix_start}, {affix_end})."
        )

    nfe = 0

    # ACache Algorithm:
    # 1. Compute affix KV cache from affix tokens ALONE (reusable across samples)
    # 2. Select K Anchor tokens from affix based on importance
    # 3. Build full KV cache: affix positions from affix_cache, Anchor + non-affix from full x
    # 4. For cache recomputation (between blocks): recompute Anchor + generation region, freeze non-Anchor affix
    # 5. For block refinement (within-block): only process block tokens

    # Step 1: Consume precomputed affix KV cache.
    affix_cache = precomputed_affix_cache
    assert affix_cache is not None, "precomputed_affix_cache must be provided for ACache generation."
    assert len(affix_cache) > 0, "precomputed_affix_cache must contain at least one transformer layer."
    assert len(affix_cache[0]) == 2, "Each precomputed_affix_cache layer must be a (key, value) tuple."
    expected_affix_len = affix_end - affix_start
    cached_affix_len = affix_cache[0][0].shape[1]
    if cached_affix_len != expected_affix_len:
        raise ValueError(
            f"precomputed_affix_cache length {cached_affix_len} does not match expected affix length {expected_affix_len}."
        )

    # Step 2: Compute Anchor tokens from affix based on importance.
    # Use full-sequence mask tokens as queries and affix_cache keys so
    # each layer's mask queries are conditioned on current full-sequence context,
    # while affix cache values remain unchanged/reusable.
    affix_len = affix_end - affix_start
    k = min(ceil_to_int(anchor_ratio * affix_len), affix_len)
    if k >= affix_len and affix_len > 0:
        anchor_global_indices = torch.arange(
            affix_start,
            affix_end,
            device=x.device,
            dtype=torch.long,
        ).unsqueeze(0).expand(B, -1)
    elif k > 0 and affix_len > 0:
        importance = compute_attention_importance_cross_affix(
            model,
            x,
            affix_cache,
            mask_id=mask_id,
            affix_start=affix_start,
        )
        anchor_local_indices = select_anchor_tokens(importance, k, selection_mode)  # (B, K)
        anchor_global_indices = anchor_local_indices + affix_start  # Convert to global indices
        nfe += 1
    else:
        anchor_global_indices = torch.zeros(B, 0, dtype=torch.long, device=x.device)

    K = anchor_global_indices.shape[1]
    # Build a recompute mask: recompute everything except non-Anchor affix tokens.
    recompute_mask = torch.ones(B, total_len, dtype=torch.bool, device=x.device)
    if affix_end > affix_start:
        recompute_mask[:, affix_start:affix_end] = False
    if K > 0:
        batch_indices = torch.arange(B, device=x.device).unsqueeze(1)
        recompute_mask[batch_indices, anchor_global_indices] = True

    recompute_positions = recompute_mask[0].nonzero(as_tuple=True)[0]
    full_to_recompute = torch.full((total_len,), -1, dtype=torch.long, device=x.device)
    full_to_recompute[recompute_positions] = torch.arange(
        recompute_positions.shape[0], device=x.device, dtype=torch.long
    )
    init_position_ids = recompute_positions

    def block_compact_start(block_start: int, block_end: int) -> int:
        return int(full_to_recompute[block_start].item())

    init_forward_input = x[:, recompute_positions]
    if drop_non_anchor:
        # Strict ablation: initialize directly on the compact kept-position sequence,
        # so block-0 logits never depend on dropped non-Anchor affix tokens.
        out_init = model(
            init_forward_input,
            use_cache=True,
            position_ids=init_position_ids,
        )
        out_init_logits = _align_logits_for_dream(out_init.logits)
        past_key_values = out_init.past_key_values
        pruned_cache_position_ids = recompute_positions
        compact_recompute_replace_position = torch.ones(
            B, recompute_positions.shape[0], dtype=torch.bool, device=x.device
        )
    else:
        # Step 3: Build full KV cache
        # - Start with affix_cache for affix positions
        # - Process full x to get non-affix KV and recompute Anchor KV
        num_layers = len(affix_cache)
        sample_k, sample_v = affix_cache[0]
        kv_hidden = sample_k.shape[2]

        # Create full KV cache, initialize with zeros
        past_key_values = []
        for layer_idx in range(num_layers):
            layer_k = torch.zeros(B, total_len, kv_hidden, dtype=sample_k.dtype, device=sample_k.device)
            layer_v = torch.zeros(B, total_len, kv_hidden, dtype=sample_v.dtype, device=sample_v.device)

            # Copy affix cache
            affix_k, affix_v = affix_cache[layer_idx]
            layer_k[:, affix_start:affix_end, :] = affix_k
            layer_v[:, affix_start:affix_end, :] = affix_v

            past_key_values.append((layer_k, layer_v))
        past_key_values = tuple(past_key_values)

        # Process full x to compute non-affix KV and recompute selected positions.
        init_recompute_position = recompute_mask
        out_init = model(
            init_forward_input,
            past_key_values=past_key_values,
            use_cache=True,
            replace_position=init_recompute_position,
            position_ids=init_position_ids,
            dual_cache=True,
        )
        out_init_logits = _align_logits_for_dream(out_init.logits)
        past_key_values = out_init.past_key_values
        pruned_cache_position_ids = None
        compact_recompute_replace_position = None

    nfe += 1

    for nb in range(num_blocks):
        s = generation_start + nb * block_length
        e = s + block_length

        # Build replace_position for block-only refinement.
        if drop_non_anchor:
            block_replace_position = torch.zeros(B, recompute_positions.shape[0], dtype=torch.bool, device=x.device)
            block_start_compact = block_compact_start(s, e)
            block_replace_position[:, block_start_compact:block_start_compact + block_length] = True
        else:
            block_replace_position = torch.zeros(B, total_len, dtype=torch.bool, device=x.device)
            block_replace_position[:, s:e] = True

        if nb == 0:
            # First block: use initial forward pass results from out_init
            init_block_start = block_compact_start(s, e)
            logits_blk = out_init_logits[:, init_block_start:init_block_start + block_length]
        else:
            # Subsequent blocks: recompute Anchor tokens + generation region, keep non-Anchor affix frozen
            recompute_replace_position = compact_recompute_replace_position if drop_non_anchor else recompute_mask
            recompute_forward_input = x[:, recompute_positions]

            out_recompute = model(
                recompute_forward_input,
                past_key_values=past_key_values,
                use_cache=True,
                replace_position=recompute_replace_position,
                position_ids=init_position_ids,
                cache_position_ids=pruned_cache_position_ids,
                dual_cache=True,
            )
            past_key_values = out_recompute.past_key_values
            recompute_block_start = block_compact_start(s, e)
            out_recompute_logits = _align_logits_for_dream(out_recompute.logits)
            logits_blk = out_recompute_logits[:, recompute_block_start:recompute_block_start + block_length]
            nfe += 1

        # Match Dream dual-cache exactly: after the block-level recompute, commit
        # only the first token in the current block before cached refinement.
        _, block_seed_tokens = sample_tokens(logits_blk, temperature=temperature, top_p=None, top_k=None)
        x[:, s] = block_seed_tokens[:, 0]

        if factor is None and threshold is None and steps_per_block > 1:
            remaining_mask_index = (x[:, s:e] == mask_id)
            num_transfer_tokens = get_num_transfer_tokens(remaining_mask_index, steps_per_block - 1)
        else:
            num_transfer_tokens = None

        # Semi-autoregressive refinement within block (block-only, same as baseline)
        for i in range(1, steps_per_block):
            if (x[:, s:e] == mask_id).sum() == 0:
                break

            out_step = model(
                x[:, s:e],
                past_key_values=past_key_values,
                use_cache=True,
                replace_position=block_replace_position,
                position_ids=torch.arange(
                    s,
                    e,
                    device=x.device,
                    dtype=torch.long,
                ),
                cache_position_ids=pruned_cache_position_ids,
                dual_cache=True,
            )
            past_key_values = out_step.past_key_values
            logits_blk = _align_logits_for_dream(out_step.logits)
            nfe += 1

            mask_blk = (x[:, s:e] == mask_id)

            if factor is None:
                quota_i = None if threshold is not None else num_transfer_tokens[:, i - 1]
                x0_blk, transfer_idx_blk = get_transfer_index(
                    logits_blk, temperature, remasking, mask_blk, x[:, s:e], quota_i, threshold
                )
            else:
                x0_blk, transfer_idx_blk = get_transfer_index_dynamic(
                    logits_blk, temperature, remasking, mask_blk, x[:, s:e], None, factor
                )

            blk_old = x[:, s:e]
            blk_new = torch.where(transfer_idx_blk, x0_blk, blk_old)
            x = torch.cat([x[:, :s], blk_new, x[:, e:]], dim=1)

    return x, nfe


def parse_args():
    def parse_bool(value: str) -> bool:
        text = str(value).strip().lower()
        if text in {"1", "true", "yes", "y", "on"}:
            return True
        if text in {"0", "false", "no", "n", "off"}:
            return False
        raise argparse.ArgumentTypeError(f"Expected a boolean value, got: {value!r}")

    parser = argparse.ArgumentParser(description="Test ACache generation.")
    parser.add_argument(
        "--selection_mode",
        type=str,
        default="top",
        choices=["top", "bottom"],
        help="Anchor token selection strategy when anchor_ratio > 0.",
    )
    parser.add_argument(
        "--drop_non_anchor",
        type=parse_bool,
        default=False,
        help=(
            "If true, initialize and decode on the compact kept-position sequence so "
            "non-Anchor affix tokens are absent from block-0 logits onward. "
            "If false, keep them in cache."
        ),
    )
    return parser.parse_args()
