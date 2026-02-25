#!/usr/bin/env python
# ACache Quick Test Script

"""
Quick test to verify ACache implementation works correctly.
Run with: srun --pty --gpus a100:1 --cpus-per-task 24 --mem 96G --qos=spot python test_ACache.py
"""

import torch
import sys
import time
from pathlib import Path

LLADA_DIR = Path(__file__).resolve().parent
if str(LLADA_DIR) not in sys.path:
    sys.path.insert(0, str(LLADA_DIR))

from model.configuration_llada import LLaDAConfig
from model.modeling_llada import LLaDAModelLM
from transformers import AutoTokenizer, AutoConfig
from generate import generate_with_dual_cache
from generate_ACache import (
    generate_with_anchor_attention,
    compute_attention_importance_cross_affix,
    select_anchor_tokens
)


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


def _clone_past_key_values(past_key_values):
    return tuple((layer_k.clone(), layer_v.clone()) for layer_k, layer_v in past_key_values)


def test_drop_non_anchor_original_cache_positions():
    """Verify pruned-cache updates preserve original cache positions."""
    print("\n" + "=" * 60)
    print("TEST 2: Drop-Non-Anchor Original Cache Positions")
    print("=" * 60)

    torch.manual_seed(0)
    config = LLaDAConfig(
        vocab_size=128,
        embedding_size=128,
        d_model=32,
        n_heads=4,
        n_kv_heads=4,
        n_layers=2,
        mlp_hidden_size=64,
        activation_type="silu",
        block_type="llama",
        rope=True,
        flash_attention=False,
        attention_dropout=0.0,
        residual_dropout=0.0,
        embedding_dropout=0.0,
        eos_token_id=0,
        pad_token_id=0,
        mask_token_id=127,
        init_device="cpu",
    )
    model = LLaDAModelLM(config).eval()

    full_ids = torch.tensor([[3, 5, 7, 11, 13, 17, 19]], dtype=torch.long)
    keep_positions = torch.tensor([1, 3, 5, 6], dtype=torch.long)
    compact_replace_position = torch.ones(1, keep_positions.numel(), dtype=torch.bool)

    with torch.inference_mode():
        full_out = model(full_ids, use_cache=True)
        pruned_cache = tuple(
            (layer_k[:, :, keep_positions].clone(), layer_v[:, :, keep_positions].clone())
            for layer_k, layer_v in full_out.past_key_values
        )

        reference_out = model(
            full_ids[:, keep_positions],
            use_cache=False,
            position_ids=keep_positions,
        )
        pruned_out = model(
            full_ids[:, keep_positions],
            past_key_values=_clone_past_key_values(pruned_cache),
            use_cache=True,
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
    print("TEST 3: Drop-Non-Anchor Strict First Block")
    print("=" * 60)

    torch.manual_seed(70)
    config = LLaDAConfig(
        vocab_size=128,
        embedding_size=128,
        d_model=32,
        n_heads=4,
        n_kv_heads=4,
        n_layers=2,
        mlp_hidden_size=64,
        activation_type="silu",
        block_type="llama",
        rope=True,
        flash_attention=False,
        attention_dropout=0.0,
        residual_dropout=0.0,
        embedding_dropout=0.0,
        eos_token_id=0,
        pad_token_id=0,
        mask_token_id=127,
        init_device="cpu",
    )
    model = LLaDAModelLM(config).eval()

    prompt = torch.tensor([[3, 5, 7, 11]], dtype=torch.long)
    affix_start, affix_end = 0, 3
    recompute_positions = torch.tensor([3, 4], dtype=torch.long)
    full_with_mask = torch.full((1, prompt.shape[1] + 1), 127, dtype=torch.long)
    full_with_mask[:, :prompt.shape[1]] = prompt

    with torch.inference_mode():
        affix_cache = model(prompt[:, affix_start:affix_end], use_cache=True).past_key_values
        strict_out = model(
            full_with_mask[:, recompute_positions],
            use_cache=False,
            position_ids=recompute_positions,
        )
        full_out = model(full_with_mask, use_cache=False)
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

    strict_token = strict_out.logits[:, -1].argmax(dim=-1)
    full_token = full_out.logits[:, -1].argmax(dim=-1)
    generated_token = generated[:, prompt.shape[1]]
    print(
        f"strict token={strict_token.item()} full token={full_token.item()} "
        f"generated token={generated_token.item()}"
    )

    assert strict_token.item() != full_token.item()
    assert generated_token.item() == strict_token.item()
    print("Drop-non-anchor strict first-block test PASSED")


def test_ACache_selection():
    """Test Anchor token selection logic."""
    print("\n" + "=" * 60)
    print("TEST 1: Anchor Token Selection")
    print("=" * 60)

    # Create mock importance scores
    importance = torch.tensor([[0.1, 0.5, 0.3, 0.8, 0.2, 0.9, 0.4, 0.6]])  # (1, 8)

    for k in [1, 2, 4]:
        anchor_indices = select_anchor_tokens(importance, k)
        print(f"K={k}: Selected indices (sorted): {anchor_indices.tolist()[0]}")

    print("Anchor selection test PASSED")


def test_generation():
    """Test generation with model."""
    print("\n" + "=" * 60)
    print("TEST 2: Generation Test")
    print("=" * 60)

    device = 'cuda'
    model_path = 'GSAI-ML/LLaDA-8B-Instruct'

    print(f"Loading model from {model_path}...")
    config = AutoConfig.from_pretrained(model_path)
    config.flash_attention = True

    model = LLaDAModelLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        config=config
    ).to(device).eval()

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    print("Model loaded successfully")

    # Test prompts
    system_prompt = "You are a careful reasoning assistant."
    problem = "What is 15 + 27?"

    full_content = system_prompt + " " + problem
    m = [{"role": "user", "content": full_content}]
    prompt_text = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
    input_ids = torch.tensor(tokenizer(prompt_text)['input_ids']).to(device).unsqueeze(0)

    # Determine affix boundary for this prefix test case.
    system_only = tokenizer.apply_chat_template([{"role": "user", "content": system_prompt}], add_generation_prompt=False, tokenize=False)
    system_tokens = tokenizer(system_only)['input_ids']
    affix_end = len(system_tokens)
    affix_ids = input_ids[:, :affix_end]

    print(f"\nPrompt length: {input_ids.shape[1]}")
    print(f"Affix length (system prompt): {affix_end}")
    affix_position_ids = torch.arange(0, affix_end, device=device, dtype=torch.long)
    with torch.inference_mode():
        precomputed_affix_cache = model(
            affix_ids,
            use_cache=True,
            position_ids=affix_position_ids,
        ).past_key_values

    gen_length = 128
    block_length = 32
    steps = gen_length

    # Test 1: Standard dual cache (baseline)
    print("\n--- Testing Standard Dual Cache (Baseline) ---")
    with torch.inference_mode():
        start = time.time()
        out_baseline, nfe_baseline = generate_with_dual_cache(
            model, input_ids,
            steps=steps, gen_length=gen_length, block_length=block_length,
            temperature=0., remasking='low_confidence',
            threshold=0.9
        )
        elapsed_baseline = time.time() - start
    output_baseline = tokenizer.decode(out_baseline[0, input_ids.shape[1]:], skip_special_tokens=True)
    print(f"NFE: {nfe_baseline}, Time: {elapsed_baseline:.2f}s")
    print(f"Output: {output_baseline[:200]}...")

    # Test 2: Frozen affix (K=0)
    print("\n--- Testing Frozen Affix (K=0) ---")
    with torch.inference_mode():
        start = time.time()
        out_frozen, nfe_frozen = generate_with_anchor_attention(
            model, input_ids,
            steps=steps, gen_length=gen_length, block_length=block_length,
            temperature=0., remasking='low_confidence',
            threshold=0.9,
            affix_start=0, affix_end=affix_end,
            anchor_ratio=0.0,
            precomputed_affix_cache=precomputed_affix_cache,
        )
        elapsed_frozen = time.time() - start
    output_frozen = tokenizer.decode(out_frozen[0, input_ids.shape[1]:], skip_special_tokens=True)
    print(f"NFE: {nfe_frozen}, Time: {elapsed_frozen:.2f}s")
    print(f"Output: {output_frozen[:200]}...")

    # Test 3: ACache with different K values
    for k in [2, 4, 8]:
        anchor_ratio = k_to_ratio(k, affix_end)
        resolved_k = min(ceil_to_int(anchor_ratio * affix_end), affix_end)
        print(f"\n--- Testing ACache (target K={k}, resolved K={resolved_k}, anchor_ratio={anchor_ratio:.4f}) ---")
        with torch.inference_mode():
            start = time.time()
            out_anchor, nfe_anchor = generate_with_anchor_attention(
                model, input_ids,
                steps=steps, gen_length=gen_length, block_length=block_length,
                temperature=0., remasking='low_confidence',
                threshold=0.9,
                affix_start=0, affix_end=affix_end,
                anchor_ratio=anchor_ratio,
                precomputed_affix_cache=precomputed_affix_cache,
            )
            elapsed_anchor = time.time() - start
        output_anchor = tokenizer.decode(out_anchor[0, input_ids.shape[1]:], skip_special_tokens=True)
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

    device = 'cuda'
    model_path = 'GSAI-ML/LLaDA-8B-Instruct'

    print(f"Loading model...")
    config = AutoConfig.from_pretrained(model_path)
    config.flash_attention = True

    model = LLaDAModelLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        config=config
    ).to(device).eval()

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Build a small prefix-affix prompt.
    system_prompt = "You are a careful reasoning assistant."
    problem = "What is 15 + 27?"
    full_content = system_prompt + " " + problem
    m = [{"role": "user", "content": full_content}]
    prompt_text = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
    prompt_ids = torch.tensor(tokenizer(prompt_text)['input_ids']).to(device).unsqueeze(0)

    # Determine affix boundary for this prefix test case.
    system_only = tokenizer.apply_chat_template(
        [{"role": "user", "content": system_prompt}],
        add_generation_prompt=False,
        tokenize=False
    )
    affix_end = len(tokenizer(system_only)['input_ids'])
    affix_ids = prompt_ids[:, :affix_end]

    # Cross-affix importance expects non-affix [MASK] tokens as queries.
    gen_length = 32
    mask_id = 126336
    full_input_ids = torch.full(
        (1, prompt_ids.shape[1] + gen_length),
        mask_id,
        dtype=torch.long,
        device=device
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

    # Decode top affix tokens.
    top_indices = importance[0].topk(topk).indices.tolist()
    tokens = [tokenizer.decode([affix_ids[0, i].item()]) for i in top_indices]
    print(f"Top {topk} important affix tokens: {tokens}")

    print("\nImportance computation test PASSED")


def main():
    print("=" * 60)
    print("ACache Implementation Test Suite")
    print("=" * 60)

    # Run tests
    test_ACache_selection()
    test_drop_non_anchor_original_cache_positions()
    test_drop_non_anchor_first_block_is_strict()

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


if __name__ == '__main__':
    main()
