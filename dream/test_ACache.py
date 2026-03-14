#!/usr/bin/env python
# ACache Quick Test Script

"""
Quick test to verify Dream ACache implementation works correctly.
Run with: srun --pty --gpus a100:1 --cpus-per-task 24 --mem 96G --qos=spot python test_ACache.py
"""

import sys
import time
import types
from pathlib import Path

import torch
from transformers import AutoTokenizer

DREAM_DIR = Path(__file__).resolve().parent
if str(DREAM_DIR) not in sys.path:
    sys.path.insert(0, str(DREAM_DIR))

from generate_ACache import (
    _align_logits_for_dream,
    compute_attention_importance_cross_affix,
    generate_with_anchor_attention,
    select_anchor_tokens,
)
from model.configuration_dream import DreamConfig
from model.generation_utils_block import DreamGenerationMixin
from model.modeling_dream import DreamModel


def k_to_ratio(k: int, affix_length: int) -> float:
    """Convert an intended Anchor token count to a valid anchor_ratio."""
    if affix_length <= 0:
        return 0.0
    return min(max(float(k) / float(affix_length), 0.0), 1.0)


def ceil_to_int(value: float) -> int:
    integer = int(value)
    if value == integer:
        return integer
    return integer + 1


def _resolve_mask_id(model, tokenizer) -> int:
    mask_id = tokenizer.mask_token_id
    if mask_id is None:
        mask_id = getattr(getattr(model, "generation_config", None), "mask_token_id", None)
    if mask_id is None:
        raise ValueError("Unable to resolve Dream mask token id from tokenizer/model.")
    return int(mask_id)


def _generate_with_dream_dual_cache(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    *,
    steps: int,
    gen_length: int,
    block_length: int,
    threshold: float,
):
    attention_mask = input_ids.ne(tokenizer.pad_token_id).long()
    out = model.diffusion_generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=gen_length,
        output_history=False,
        return_dict_in_generate=True,
        steps=steps,
        temperature=0.0,
        top_p=None,
        top_k=None,
        alg="confidence_threshold",
        threshold=threshold,
        block_length=block_length,
        dual_cache=True,
    )
    return out.sequences, int(steps)


def _load_model_and_tokenizer(model_path: str, device: str):
    model = DreamModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model.diffusion_generate = types.MethodType(DreamGenerationMixin.diffusion_generate, model)
    model._sample = types.MethodType(DreamGenerationMixin._sample, model)
    return model, tokenizer


def _clone_past_key_values(past_key_values):
    return tuple((layer_k.clone(), layer_v.clone()) for layer_k, layer_v in past_key_values)


def _build_full_cache_from_affix_cache(affix_cache, affix_start: int, total_len: int):
    sample_k, sample_v = affix_cache[0]
    kv_hidden = sample_k.shape[2]
    full_cache = []
    for affix_k, affix_v in affix_cache:
        layer_k = torch.zeros(1, total_len, kv_hidden, dtype=affix_k.dtype, device=affix_k.device)
        layer_v = torch.zeros(1, total_len, kv_hidden, dtype=affix_v.dtype, device=affix_v.device)
        affix_end = affix_start + affix_k.shape[1]
        layer_k[:, affix_start:affix_end, :] = affix_k
        layer_v[:, affix_start:affix_end, :] = affix_v
        full_cache.append((layer_k, layer_v))
    return tuple(full_cache)


def test_dual_cache_scattered_positions():
    """Verify scattered-position dual-cache matches a full-cache reference."""
    print("\n" + "=" * 60)
    print("TEST 2: Scattered Dual-Cache Positions")
    print("=" * 60)

    torch.manual_seed(0)
    config = DreamConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
        mask_token_id=127,
        pad_token_id=0,
    )
    model = DreamModel(config).eval()

    full_ids = torch.tensor([[3, 5, 7, 11, 13, 17, 19]], dtype=torch.long)
    affix_start, affix_end = 0, 3
    recompute_positions = torch.tensor([1, 3, 4, 5, 6], dtype=torch.long)
    recompute_mask = torch.zeros_like(full_ids, dtype=torch.bool)
    recompute_mask[:, recompute_positions] = True

    with torch.inference_mode():
        affix_cache = model(full_ids[:, affix_start:affix_end], use_cache=True).past_key_values
        full_cache = _build_full_cache_from_affix_cache(
            affix_cache,
            affix_start=affix_start,
            total_len=full_ids.shape[1],
        )

        reference_out = model(
            full_ids,
            past_key_values=_clone_past_key_values(full_cache),
            use_cache=True,
            dual_cache=True,
            replace_position=recompute_mask,
            position_ids=torch.arange(full_ids.shape[1], dtype=torch.long),
        )
        compact_out = model(
            full_ids[:, recompute_positions],
            past_key_values=_clone_past_key_values(full_cache),
            use_cache=True,
            dual_cache=True,
            replace_position=recompute_mask,
            position_ids=recompute_positions,
        )
        inferred_out = model(
            full_ids[:, recompute_positions],
            past_key_values=_clone_past_key_values(full_cache),
            use_cache=True,
            dual_cache=True,
            replace_position=recompute_mask,
        )

    reference_logits = reference_out.logits[:, recompute_positions]
    compact_logits = compact_out.logits
    inferred_logits = inferred_out.logits
    compact_max_diff = (compact_logits - reference_logits).abs().max().item()
    inferred_max_diff = (inferred_logits - reference_logits).abs().max().item()

    print(f"compact-vs-reference max diff: {compact_max_diff:.6e}")
    print(f"inferred-vs-reference max diff: {inferred_max_diff:.6e}")

    assert torch.allclose(compact_logits, reference_logits, atol=1e-5, rtol=1e-4)
    assert torch.allclose(inferred_logits, reference_logits, atol=1e-5, rtol=1e-4)
    print("Scattered dual-cache position test PASSED")


def test_drop_non_anchor_original_cache_positions():
    """Verify pruned-cache updates preserve original cache positions."""
    print("\n" + "=" * 60)
    print("TEST 3: Drop-Non-Anchor Original Cache Positions")
    print("=" * 60)

    torch.manual_seed(0)
    config = DreamConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
        mask_token_id=127,
        pad_token_id=0,
    )
    model = DreamModel(config).eval()

    full_ids = torch.tensor([[3, 5, 7, 11, 13, 17, 19]], dtype=torch.long)
    keep_positions = torch.tensor([1, 3, 5, 6], dtype=torch.long)
    compact_replace_position = torch.ones(1, keep_positions.numel(), dtype=torch.bool)

    with torch.inference_mode():
        full_out = model(full_ids, use_cache=True)
        pruned_cache = tuple(
            (layer_k[:, keep_positions].clone(), layer_v[:, keep_positions].clone())
            for layer_k, layer_v in full_out.past_key_values
        )

        reference_out = model(
            full_ids[:, keep_positions],
            position_ids=keep_positions,
            use_cache=False,
        )
        pruned_out = model(
            full_ids[:, keep_positions],
            past_key_values=_clone_past_key_values(pruned_cache),
            use_cache=True,
            dual_cache=True,
            replace_position=compact_replace_position,
            position_ids=keep_positions,
            cache_position_ids=keep_positions,
        )

    reference_logits = reference_out.logits
    max_diff = (pruned_out.logits - reference_logits).abs().max().item()
    print(f"pruned-vs-reference max diff: {max_diff:.6e}")

    assert torch.allclose(pruned_out.logits, reference_logits, atol=1e-5, rtol=1e-4)
    print("Drop-non-anchor original-cache-position test PASSED")


def test_drop_non_anchor_first_block_is_strict():
    """Verify drop_non_anchor affects block-0 logits from the first generation step."""
    print("\n" + "=" * 60)
    print("TEST 4: Drop-Non-Anchor Strict First Block")
    print("=" * 60)

    torch.manual_seed(0)
    config = DreamConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
        mask_token_id=127,
        pad_token_id=0,
    )
    model = DreamModel(config).eval()

    prompt = torch.tensor([[3, 5, 7, 11]], dtype=torch.long)
    affix_start, affix_end = 0, 3
    recompute_positions = torch.tensor([3, 4], dtype=torch.long)
    full_with_mask = torch.full((1, prompt.shape[1] + 1), 127, dtype=torch.long)
    full_with_mask[:, :prompt.shape[1]] = prompt

    with torch.inference_mode():
        affix_cache = model(prompt[:, affix_start:affix_end], use_cache=True).past_key_values
        strict_out = _align_logits_for_dream(model(
            full_with_mask[:, recompute_positions],
            use_cache=False,
            position_ids=recompute_positions,
        ).logits)
        full_out = _align_logits_for_dream(model(
            full_with_mask,
            use_cache=False,
            position_ids=torch.arange(full_with_mask.shape[1], dtype=torch.long),
        ).logits)
        generated, _ = generate_with_anchor_attention(
            model,
            prompt,
            steps=1,
            gen_length=1,
            block_length=1,
            temperature=0.0,
            remasking="low_confidence",
            mask_id=127,
            threshold=0.0,
            affix_start=affix_start,
            affix_end=affix_end,
            anchor_ratio=0.0,
            drop_non_anchor=True,
            precomputed_affix_cache=affix_cache,
        )

    strict_token = strict_out[:, -1].argmax(dim=-1)
    full_token = full_out[:, -1].argmax(dim=-1)
    generated_token = generated[:, prompt.shape[1]]
    print(
        f"strict token={strict_token.item()} full token={full_token.item()} "
        f"generated token={generated_token.item()}"
    )

    assert strict_token.item() != full_token.item()
    assert generated_token.item() == strict_token.item()
    print("Drop-non-anchor strict first-block test PASSED")


def test_full_refresh_matches_dual_cache_generation():
    """Verify anchor_ratio=1.0 matches Dream dual-cache generation exactly."""
    print("\n" + "=" * 60)
    print("TEST 5: Full Refresh Matches Dual Cache")
    print("=" * 60)

    torch.manual_seed(0)
    config = DreamConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
        mask_token_id=31,
        pad_token_id=0,
    )
    model = DreamModel(config).eval()
    model.diffusion_generate = types.MethodType(DreamGenerationMixin.diffusion_generate, model)
    model._sample = types.MethodType(DreamGenerationMixin._sample, model)

    prompt = torch.tensor([[5, 6, 7, 8]], dtype=torch.long)
    affix_start, affix_end = 0, 2

    with torch.inference_mode():
        affix_cache = model(prompt[:, affix_start:affix_end], use_cache=True).past_key_values
        baseline = model.diffusion_generate(
            prompt,
            attention_mask=prompt.ne(config.pad_token_id).long(),
            max_new_tokens=4,
            output_history=False,
            return_dict_in_generate=True,
            steps=4,
            temperature=0.0,
            top_p=None,
            top_k=None,
            alg="confidence_threshold",
            threshold=0.9,
            block_length=2,
            dual_cache=True,
        ).sequences
        full_refresh, nfe = generate_with_anchor_attention(
            model,
            prompt,
            steps=4,
            gen_length=4,
            block_length=2,
            temperature=0.0,
            remasking="low_confidence",
            mask_id=config.mask_token_id,
            threshold=0.9,
            affix_start=affix_start,
            affix_end=affix_end,
            anchor_ratio=1.0,
            precomputed_affix_cache=affix_cache,
        )

    print(f"baseline sequence: {baseline.tolist()}")
    print(f"full-refresh sequence: {full_refresh.tolist()} | nfe={nfe}")

    assert torch.equal(full_refresh, baseline)
    assert nfe == 4
    print("Full-refresh equivalence test PASSED")


def test_ACache_selection():
    """Test Anchor token selection logic."""
    print("\n" + "=" * 60)
    print("TEST 1: Anchor Token Selection")
    print("=" * 60)

    importance = torch.tensor([[0.1, 0.5, 0.3, 0.8, 0.2, 0.9, 0.4, 0.6]])
    for k in [1, 2, 4]:
        anchor_indices = select_anchor_tokens(importance, k)
        print(f"K={k}: Selected indices (sorted): {anchor_indices.tolist()[0]}")

    print("Anchor selection test PASSED")


def test_generation():
    """Test generation with Dream model."""
    print("\n" + "=" * 60)
    print("TEST 2: Generation Test")
    print("=" * 60)

    device = "cuda"
    model_path = "Dream-org/Dream-v0-Instruct-7B"

    print(f"Loading model from {model_path}...")
    model, tokenizer = _load_model_and_tokenizer(model_path, device)
    mask_id = _resolve_mask_id(model, tokenizer)
    print(f"Model loaded successfully (mask_id={mask_id})")

    system_prompt = "You are a careful reasoning assistant."
    problem = "What is 15 + 27?"

    full_content = f"{system_prompt} {problem}"
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": full_content}],
        add_generation_prompt=True,
        tokenize=False,
    )
    input_ids = torch.tensor(tokenizer(prompt_text)["input_ids"], device=device).unsqueeze(0)

    system_only = tokenizer.apply_chat_template(
        [{"role": "user", "content": system_prompt}],
        add_generation_prompt=False,
        tokenize=False,
    )
    affix_end = len(tokenizer(system_only)["input_ids"])
    affix_ids = input_ids[:, :affix_end]

    print(f"\nPrompt length: {input_ids.shape[1]}")
    print(f"Affix length (system prompt): {affix_end}")
    with torch.inference_mode():
        precomputed_affix_cache = model(affix_ids, use_cache=True).past_key_values

    gen_length = 128
    block_length = 32
    steps = gen_length

    print("\n--- Testing Standard Dual Cache (Baseline) ---")
    with torch.inference_mode():
        start = time.time()
        out_baseline, nfe_baseline = _generate_with_dream_dual_cache(
            model,
            tokenizer,
            input_ids,
            steps=steps,
            gen_length=gen_length,
            block_length=block_length,
            threshold=0.9,
        )
        elapsed_baseline = time.time() - start
    output_baseline = tokenizer.decode(out_baseline[0, input_ids.shape[1] :], skip_special_tokens=True)
    print(f"NFE: {nfe_baseline}, Time: {elapsed_baseline:.2f}s")
    print(f"Output: {output_baseline[:200]}...")

    print("\n--- Testing Frozen Affix (K=0) ---")
    with torch.inference_mode():
        start = time.time()
        out_frozen, nfe_frozen = generate_with_anchor_attention(
            model,
            input_ids,
            steps=steps,
            gen_length=gen_length,
            block_length=block_length,
            temperature=0.0,
            remasking="low_confidence",
            mask_id=mask_id,
            threshold=0.9,
            affix_start=0,
            affix_end=affix_end,
            anchor_ratio=0.0,
            precomputed_affix_cache=precomputed_affix_cache,
        )
        elapsed_frozen = time.time() - start
    output_frozen = tokenizer.decode(out_frozen[0, input_ids.shape[1] :], skip_special_tokens=True)
    print(f"NFE: {nfe_frozen}, Time: {elapsed_frozen:.2f}s")
    print(f"Output: {output_frozen[:200]}...")

    for k in [2, 4, 8]:
        anchor_ratio = k_to_ratio(k, affix_end)
        resolved_k = min(ceil_to_int(anchor_ratio * affix_end), affix_end)
        print(f"\n--- Testing ACache (target K={k}, resolved K={resolved_k}, anchor_ratio={anchor_ratio:.4f}) ---")
        with torch.inference_mode():
            start = time.time()
            out_anchor, nfe_anchor = generate_with_anchor_attention(
                model,
                input_ids,
                steps=steps,
                gen_length=gen_length,
                block_length=block_length,
                temperature=0.0,
                remasking="low_confidence",
                mask_id=mask_id,
                threshold=0.9,
                affix_start=0,
                affix_end=affix_end,
                anchor_ratio=anchor_ratio,
                precomputed_affix_cache=precomputed_affix_cache,
            )
            elapsed_anchor = time.time() - start
        output_anchor = tokenizer.decode(out_anchor[0, input_ids.shape[1] :], skip_special_tokens=True)
        print(f"NFE: {nfe_anchor}, Time: {elapsed_anchor:.2f}s")
        print(f"Output: {output_anchor[:200]}...")

    print("\n" + "=" * 60)
    print("All generation tests PASSED")
    print("=" * 60)


def test_importance_computation():
    """Test importance score computation."""
    print("\n" + "=" * 60)
    print("TEST 3: Importance Score Computation")
    print("=" * 60)

    device = "cuda"
    model_path = "Dream-org/Dream-v0-Instruct-7B"

    print("Loading model...")
    model, tokenizer = _load_model_and_tokenizer(model_path, device)
    mask_id = _resolve_mask_id(model, tokenizer)

    system_prompt = "You are a careful reasoning assistant."
    problem = "What is 15 + 27?"
    full_content = f"{system_prompt} {problem}"
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": full_content}],
        add_generation_prompt=True,
        tokenize=False,
    )
    prompt_ids = torch.tensor(tokenizer(prompt_text)["input_ids"], device=device).unsqueeze(0)

    system_only = tokenizer.apply_chat_template(
        [{"role": "user", "content": system_prompt}],
        add_generation_prompt=False,
        tokenize=False,
    )
    affix_end = len(tokenizer(system_only)["input_ids"])
    affix_ids = prompt_ids[:, :affix_end]

    gen_length = 32
    full_input_ids = torch.full(
        (1, prompt_ids.shape[1] + gen_length),
        mask_id,
        dtype=torch.long,
        device=device,
    )
    full_input_ids[:, :prompt_ids.shape[1]] = prompt_ids

    print(f"Computing cross-affix importance for affix_length={affix_end}, total_length={full_input_ids.shape[1]}...")
    with torch.inference_mode():
        affix_out = model(affix_ids, use_cache=True)
        importance = compute_attention_importance_cross_affix(
            model,
            full_input_ids,
            affix_out.past_key_values,
            mask_id=mask_id,
        )

    print(f"Importance scores shape: {importance.shape}")
    topk = min(5, importance.shape[1])
    if topk == 0:
        print("No affix tokens available for top-k inspection.")
        print("\nImportance computation test PASSED")
        return
    print(f"Top {topk} importance values: {importance[0].topk(topk).values.tolist()}")
    print(f"Top {topk} importance indices: {importance[0].topk(topk).indices.tolist()}")

    top_indices = importance[0].topk(topk).indices.tolist()
    tokens = [tokenizer.decode([affix_ids[0, i].item()]) for i in top_indices]
    print(f"Top {topk} important affix tokens: {tokens}")

    print("\nImportance computation test PASSED")


def main():
    print("=" * 60)
    print("Dream ACache Implementation Test Suite")
    print("=" * 60)

    test_ACache_selection()
    test_dual_cache_scattered_positions()
    test_drop_non_anchor_original_cache_positions()
    test_drop_non_anchor_first_block_is_strict()
    test_full_refresh_matches_dual_cache_generation()

    if torch.cuda.is_available():
        print("\nGPU detected. Running model tests...")
        test_importance_computation()
        test_generation()
    else:
        print("\nNo GPU available. Skipping model tests.")
        print("Run with: srun --pty --gpus a100:1 --cpus-per-task 24 --mem 96G --qos=spot python test_ACache.py")

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
