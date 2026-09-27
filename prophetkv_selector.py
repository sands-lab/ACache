"""Selector-only ProphetKV adaptation (arXiv:2602.02579, sections 4.2–4.3).

Q is the uncached prompt, never the generation span. Cached affix K/V are
borrowed from ACache, not regenerated. Native bidirectional attention propagates
the query states; affix-only attention probabilities determine anchor scores.
"""

import json
import math
import os

import torch


def query_positions(prompt, affix_start, affix_end, generation_start, gen_length, mask_id):
    if prompt.ndim != 2 or prompt.shape[0] != 1:
        raise ValueError("ProphetKV sidecar requires HF batch size 1")
    length = prompt.shape[1]
    if not 0 <= affix_start < affix_end <= length:
        raise ValueError("Invalid affix span")
    positions = torch.arange(length, device=prompt.device)
    keep = (positions < affix_start) | (positions >= affix_end)
    if generation_start is not None:
        end = generation_start + gen_length
        if not 0 <= generation_start < end <= length:
            raise ValueError("Invalid infill generation span")
        if max(generation_start, affix_start) < min(end, affix_end):
            raise ValueError("Affix overlaps generation")
        keep &= (positions < generation_start) | (positions >= end)
    result = positions[keep]
    if not result.numel() or (prompt[:, result] == mask_id).any():
        raise ValueError("Uncached prompt must be nonempty and contain no MASK tokens")
    return result


def explicit_query_positions(
    prompt,
    positions,
    affix_start,
    affix_end,
    generation_start,
    gen_length,
    mask_id,
):
    """Validate a caller-supplied, sorted subset of prompt positions.

    The content-only ProphetKV variant uses this entry point.  Keeping the
    validation here means it shares the baseline's disjointness and MASK
    safeguards without changing how the baseline derives its all-private-Q
    positions.
    """
    if prompt.ndim != 2 or prompt.shape[0] != 1:
        raise ValueError("ProphetKV sidecar requires HF batch size 1")
    length = prompt.shape[1]
    if not 0 <= affix_start < affix_end <= length:
        raise ValueError("Invalid affix span")
    if generation_start is not None:
        end = generation_start + gen_length
        if not 0 <= generation_start < end <= length:
            raise ValueError("Invalid infill generation span")
        if max(generation_start, affix_start) < min(end, affix_end):
            raise ValueError("Affix overlaps generation")

    result = torch.as_tensor(positions, dtype=torch.long, device=prompt.device)
    if result.ndim != 1 or not result.numel():
        raise ValueError("Explicit ProphetKV query positions must be a nonempty 1-D sequence")
    if result.min() < 0 or result.max() >= length:
        raise ValueError("Explicit ProphetKV query position is outside the prompt")
    if result.unique().numel() != result.numel():
        raise ValueError("Explicit ProphetKV query positions must be unique")
    if not torch.equal(result, result.sort().values):
        raise ValueError("Explicit ProphetKV query positions must be sorted")
    if ((result >= affix_start) & (result < affix_end)).any():
        raise ValueError("Explicit query positions overlap the affix")
    if generation_start is not None:
        end = generation_start + gen_length
        if ((result >= generation_start) & (result < end)).any():
            raise ValueError("Explicit query positions overlap generation MASK slots")
    if (prompt[:, result] == mask_id).any():
        raise ValueError("Explicit query positions contain MASK tokens")
    return result


def attention_mean(q, k, chunk_size=128):
    """Mean over query heads and tokens, FP32 softmax over affix keys only."""
    groups = q.shape[1] // k.shape[1]
    if q.shape[1] != k.shape[1] * groups:
        raise ValueError("Invalid grouped-query attention head counts")
    k = k.repeat_interleave(groups, dim=1).float()
    result = torch.zeros(q.shape[0], k.shape[-2], device=q.device, dtype=torch.float32)
    for part in q.split(chunk_size, dim=-2):
        weights = torch.softmax(part.float() @ k.transpose(-2, -1) / math.sqrt(q.shape[-1]), dim=-1)
        result.add_(weights.sum(dim=(1, 2)))
    return result / (q.shape[1] * q.shape[-2])


@torch.no_grad()
def compute_scores(model, full_input_ids, affix_cache, positions, affix_start, kind):
    axis = 2 if kind == "llada" else 1
    affix_len = affix_cache[0][0].shape[axis]
    affix_positions = torch.arange(affix_start, affix_start + affix_len, device=positions.device)
    compact_positions = torch.cat((positions, affix_positions)).sort().values
    if compact_positions.unique().numel() != compact_positions.numel():
        raise ValueError("Query and affix positions overlap")
    affix_slots = torch.searchsorted(compact_positions, affix_positions)
    query_slots = torch.searchsorted(compact_positions, positions)
    # Only the temporary copies are written by native attention. Exclude MASK
    # slots completely, while retaining original absolute RoPE coordinates.
    temporary_cache = []
    for pair in affix_cache:
        copied = []
        for source in pair:
            shape = list(source.shape)
            shape[axis] = compact_positions.numel()
            target = source.new_zeros(shape)
            target.index_copy_(axis, affix_slots, source)
            copied.append(target)
        temporary_cache.append(tuple(copied))
    replace = torch.zeros(1, compact_positions.numel(), dtype=torch.bool, device=positions.device)
    replace[:, query_slots] = True
    scores = torch.zeros(1, affix_len, dtype=torch.float32, device=positions.device)
    layers = model.model.transformer.blocks if kind == "llada" else model.model.layers
    if len(layers) != len(affix_cache):
        raise ValueError("Cache and model layer counts differ")
    visited = []

    def make_hook(index):
        def hook(module, args, kwargs):
            hidden = args[0] if args else kwargs.get("hidden_states", kwargs.get("x"))
            if kind == "llada":
                normalized = module.attn_norm(hidden)
                if hasattr(module, "att_proj"):
                    q = module.att_proj(normalized).split(module.fused_dims, dim=-1)[0]
                else:
                    q = module.q_proj(normalized)
                if module.q_norm is not None and module.k_norm is not None:
                    q = module.q_norm(q).to(hidden.dtype)
                q = q.view(1, positions.numel(), module.config.n_heads, -1).transpose(1, 2)
                k = affix_cache[index][0]
                if module.config.rope:
                    q, k = module.rotary_emb(q, k, position_ids=positions,
                                            key_position_ids=affix_positions, rotate_key_full=True)
            else:
                from model.modeling_dream import apply_rotary_pos_emb_with_cos_sin
                q = module.q_proj(hidden).view(1, positions.numel(), module.num_heads, module.head_dim).transpose(1, 2)
                k = affix_cache[index][0].view(1, affix_len, module.num_key_value_heads, module.head_dim).transpose(1, 2)
                qc, qs = module.rotary_emb(q, positions[None, :])
                kc, ks = module.rotary_emb(k, affix_positions[None, :])
                q, k = apply_rotary_pos_emb_with_cos_sin(q, k, qc, qs, kc, ks)
            scores.add_(attention_mean(q, k))
            visited.append(index)
        return hook

    handles = []
    try:
        for index, layer in enumerate(layers):
            module = layer if kind == "llada" else layer.self_attn
            handles.append(module.register_forward_pre_hook(make_hook(index), with_kwargs=True))
        kwargs = dict(input_ids=full_input_ids[:, positions], past_key_values=tuple(temporary_cache),
                      use_cache=False, output_hidden_states=False, replace_position=replace,
                      position_ids=positions[None, :], cache_position_ids=compact_positions[None, :])
        if kind == "llada":
            kwargs["last_logits_only"] = True
        else:
            kwargs["dual_cache"] = True
        model.model.forward(**kwargs)
    finally:
        for handle in handles:
            handle.remove()
    if visited != list(range(len(layers))) or not torch.isfinite(scores).all():
        raise RuntimeError("Incomplete or nonfinite ProphetKV layer scores")
    return scores / len(layers)


@torch.no_grad()
def generate(
    acache,
    kind,
    model,
    prompt,
    query_positions_override=None,
    query_mode="all_private_q",
    **kwargs,
):
    start = kwargs.get("affix_start", 0)
    cache = kwargs.get("precomputed_affix_cache")
    if not cache:
        raise ValueError("ProphetKV requires ACache's precomputed affix cache")
    length = cache[0][0].shape[2 if kind == "llada" else 1]
    end = kwargs.get("affix_end", start + length)
    if end - start != length:
        raise ValueError("Affix span/cache length mismatch")
    ratio = float(kwargs.get("anchor_ratio", 0.1))
    if not 0 <= ratio <= 1 or kwargs.get("selection_mode", "top") != "top":
        raise ValueError("ProphetKV requires a ratio in [0,1] and global top selection")
    gen_start = kwargs.get("generation_start")
    gen_length = kwargs.get("gen_length", 128)
    mask_id = kwargs.get("mask_id", 126336)
    if query_positions_override is None:
        positions = query_positions(prompt, start, end, gen_start, gen_length, mask_id)
    else:
        positions = explicit_query_positions(
            prompt,
            query_positions_override,
            start,
            end,
            gen_start,
            gen_length,
            mask_id,
        )
    expected = math.ceil(ratio * length)
    old_scorer = acache.compute_attention_importance_cross_affix
    old_select = acache.select_anchor_tokens
    audit = dict(model=kind, affix_start=start, affix_end=end, affix_len=length,
                 query_count=positions.numel(), query_min=positions[0].item(), query_max=positions[-1].item(),
                 generation_start=prompt.shape[1] if gen_start is None else gen_start,
                 generation_length=gen_length, anchor_ratio=ratio, expected_k=expected,
                 query_affix_disjoint=True, query_generation_disjoint=True, query_mask_free=True,
                 query_mode=str(query_mode))

    def scorer(model_arg, full_input_ids, affix_cache, mask_id, affix_start=0):
        if affix_cache is not cache or affix_start != start:
            raise ValueError("ACache changed the selector's reused cache/span")
        snapshot = None
        if os.environ.get("PROPHETKV_VERIFY_CACHE") == "1":
            snapshot = tuple(tuple(t.clone() for t in pair) for pair in affix_cache)
        scores = compute_scores(model_arg, full_input_ids, affix_cache, positions, start, kind)
        if snapshot is not None:
            if not all(torch.equal(t, saved) for pair, orig in zip(affix_cache, snapshot)
                       for t, saved in zip(pair, orig)):
                raise RuntimeError("ProphetKV mutated the reused ACache cache")
            audit["original_cache_unchanged"] = True
        return scores

    def select(scores, k, selection_mode="top"):
        if k != expected:
            raise ValueError(f"ACache selected budget {k}, expected ceil budget {expected}")
        indices = old_select(scores, k, selection_mode=selection_mode)
        if indices.numel() != expected or indices.unique().numel() != expected:
            raise RuntimeError("Incorrect anchor count")
        if expected and (indices.min() < 0 or indices.max() >= length):
            raise RuntimeError("Anchor out of affix range")
        audit.update(selected_count=indices.numel(), selected_affix_indices=indices.tolist())
        return indices

    acache.compute_attention_importance_cross_affix = scorer
    acache.select_anchor_tokens = select
    try:
        result = acache.generate_with_anchor_attention(model, prompt, **kwargs)
    finally:
        acache.compute_attention_importance_cross_affix = old_scorer
        acache.select_anchor_tokens = old_select
    if "selected_count" not in audit:
        if expected not in (0, length):
            raise RuntimeError("ACache unexpectedly bypassed ProphetKV selection")
        audit.update(selected_count=expected, selected_affix_indices=list(range(expected)))
    if os.environ.get("PROPHETKV_VERIFY_CACHE") == "1":
        gs = audit["generation_start"]
        if (result[0][:, gs:gs + gen_length] == mask_id).any():
            raise RuntimeError("Smoke generation left unfilled MASK tokens")
        audit["query_positions"] = positions.tolist()
        audit["query_token_ids"] = prompt[:, positions].tolist()
        audit["generation_mask_free"] = True
    audit["generation_succeeded"] = True
    path = os.environ.get("PROPHETKV_AUDIT_PATH")
    if path:
        with open(path, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(audit) + "\n")
    return result
