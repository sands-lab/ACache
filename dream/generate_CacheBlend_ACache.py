"""
CacheBlend-style Anchor selection for Dream ACache.

This sidecar module intentionally does not modify ``generate_ACache.py``.  It
keeps ACache's generation and cache-replacement path unchanged, but replaces the
one-shot Anchor selector with a CacheBlend-inspired high-KV-deviation scorer.
"""

from typing import Optional, Tuple

import torch

import generate_ACache as acache


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


def _normalize_selection_mode(selection_mode: str) -> str:
    text = str(selection_mode or "top").strip().lower()
    if text in {"top", "cacheblend_top", "hkvd_top"}:
        return "top"
    if text in {"bottom", "cacheblend_bottom", "hkvd_bottom"}:
        return "bottom"
    raise ValueError(
        "selection_mode must be one of: top, bottom, cacheblend_top, "
        f"cacheblend_bottom, hkvd_top, hkvd_bottom. Got: {selection_mode!r}."
    )


def _token_deviation(
    full_tensor: torch.Tensor,
    cached_tensor: torch.Tensor,
    metric: str,
    eps: float,
) -> torch.Tensor:
    full_tensor = full_tensor.float()
    cached_tensor = cached_tensor.float()
    diff = full_tensor - cached_tensor

    if metric == "l2":
        return torch.linalg.vector_norm(diff, ord=2, dim=-1)
    if metric == "l1":
        return diff.abs().sum(dim=-1)
    if metric == "relative_l2":
        denom = torch.linalg.vector_norm(cached_tensor, ord=2, dim=-1).clamp_min(eps)
        return torch.linalg.vector_norm(diff, ord=2, dim=-1) / denom

    raise ValueError(
        f"Unsupported cacheblend_score_metric={metric!r}. "
        "Expected one of: l2, l1, relative_l2."
    )


@torch.no_grad()
def compute_cacheblend_kv_deviation_scores(
    model,
    full_input_ids: torch.Tensor,
    affix_cache: Tuple[Tuple[torch.Tensor, torch.Tensor], ...],
    mask_id: Optional[int] = None,
    affix_start: int = 0,
    affix_end: Optional[int] = None,
    score_metric: str = "l2",
    include_values: bool = True,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Score affix tokens by KV deviation, following the CacheBlend HKVD idea.

    The score is computed once before generation:

        deviation_i = sum_layers ||KV_full_context_i - KV_affix_only_i||

    ``mask_id`` is accepted only for signature compatibility with ACache's
    attention-importance selector.
    """
    del mask_id

    if not affix_cache:
        return torch.zeros(full_input_ids.shape[0], 0, device=full_input_ids.device)
    if len(affix_cache[0]) != 2:
        raise ValueError("Each affix_cache layer must be a (key, value) tuple.")

    affix_len = affix_cache[0][0].shape[1]
    if affix_end is None:
        affix_end = affix_start + affix_len
    expected_affix_len = affix_end - affix_start
    if expected_affix_len != affix_len:
        raise ValueError(
            f"affix span length {expected_affix_len} does not match "
            f"affix_cache length {affix_len}."
        )
    if affix_len <= 0:
        return torch.zeros(full_input_ids.shape[0], 0, device=full_input_ids.device)

    metric = str(score_metric).strip().lower()
    include_values = _coerce_bool(include_values, default=True, arg_name="include_values")

    full_out = model(full_input_ids, use_cache=True)
    full_cache = full_out.past_key_values
    if full_cache is None:
        raise ValueError("Model did not return past_key_values for full-context scoring.")
    if len(full_cache) != len(affix_cache):
        raise ValueError(
            f"Layer count mismatch: full_cache has {len(full_cache)} layers, "
            f"affix_cache has {len(affix_cache)} layers."
        )

    scores = torch.zeros(
        full_input_ids.shape[0],
        affix_len,
        dtype=torch.float32,
        device=full_input_ids.device,
    )
    for layer_idx, ((full_k, full_v), (cached_k, cached_v)) in enumerate(zip(full_cache, affix_cache)):
        if full_k.dim() != 3 or full_v.dim() != 3 or cached_k.dim() != 3 or cached_v.dim() != 3:
            raise ValueError(
                "CacheBlend-style Dream scorer expects raw Dream cache tensors "
                f"with shape (batch, seq, hidden); got layer {layer_idx} shapes "
                f"K_full={tuple(full_k.shape)}, V_full={tuple(full_v.shape)}, "
                f"K_cached={tuple(cached_k.shape)}, V_cached={tuple(cached_v.shape)}."
            )

        full_k_affix = full_k[:, affix_start:affix_end, :]
        full_v_affix = full_v[:, affix_start:affix_end, :]
        if full_k_affix.shape != cached_k.shape or full_v_affix.shape != cached_v.shape:
            raise ValueError(
                f"Cache shape mismatch at layer {layer_idx}: "
                f"K_full_affix={tuple(full_k_affix.shape)}, K_cached={tuple(cached_k.shape)}, "
                f"V_full_affix={tuple(full_v_affix.shape)}, V_cached={tuple(cached_v.shape)}."
            )

        scores.add_(_token_deviation(full_k_affix, cached_k, metric, eps))
        if include_values:
            scores.add_(_token_deviation(full_v_affix, cached_v, metric, eps))

    return scores


@torch.no_grad()
def generate_with_cacheblend_anchor_attention(
    model,
    prompt: torch.Tensor,
    steps: int = 128,
    gen_length: int = 128,
    block_length: int = 128,
    temperature: float = 0.0,
    remasking: str = "low_confidence",
    mask_id: int = 126336,
    threshold: Optional[float] = None,
    factor: Optional[float] = None,
    affix_start: int = 0,
    affix_end: Optional[int] = None,
    generation_start: Optional[int] = None,
    anchor_ratio: float = 0.1,
    selection_mode: str = "top",
    drop_non_anchor: bool = False,
    precomputed_affix_cache: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
    cacheblend_score_metric: str = "l2",
    cacheblend_include_values: bool = True,
) -> Tuple[torch.Tensor, int]:
    """
    Run ACache with a one-shot CacheBlend-style HKVD Anchor selector.

    The generation path is delegated to ``generate_ACache.generate_with_anchor_attention``.
    Only the selector is replaced, and the original selector is restored even if
    generation raises.
    """
    normalized_selection_mode = _normalize_selection_mode(selection_mode)
    include_values = _coerce_bool(
        cacheblend_include_values,
        default=True,
        arg_name="cacheblend_include_values",
    )

    original_selector = acache.compute_attention_importance_cross_affix

    def cacheblend_selector(model_arg, full_input_ids, affix_cache, mask_id, affix_start=0):
        local_affix_len = affix_cache[0][0].shape[1] if affix_cache else 0
        return compute_cacheblend_kv_deviation_scores(
            model_arg,
            full_input_ids,
            affix_cache,
            mask_id=mask_id,
            affix_start=affix_start,
            affix_end=affix_start + local_affix_len,
            score_metric=cacheblend_score_metric,
            include_values=include_values,
        )

    acache.compute_attention_importance_cross_affix = cacheblend_selector
    try:
        return acache.generate_with_anchor_attention(
            model,
            prompt,
            steps=steps,
            gen_length=gen_length,
            block_length=block_length,
            temperature=temperature,
            remasking=remasking,
            mask_id=mask_id,
            threshold=threshold,
            factor=factor,
            affix_start=affix_start,
            affix_end=affix_end,
            generation_start=generation_start,
            anchor_ratio=anchor_ratio,
            selection_mode=normalized_selection_mode,
            drop_non_anchor=drop_non_anchor,
            precomputed_affix_cache=precomputed_affix_cache,
        )
    finally:
        acache.compute_attention_importance_cross_affix = original_selector
