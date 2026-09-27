import pytest
import torch

from model.acache_attention import (
    acache_attention,
    baseline_decode_attention,
    baseline_ragged_attention,
    materialize_relocated_affix,
)


CUDA_AVAILABLE = torch.cuda.is_available()
pytestmark = pytest.mark.skipif(
    not CUDA_AVAILABLE,
    reason="CUDA is required for Triton attention kernel tests",
)


@pytest.fixture
def cuda_device():
    torch.manual_seed(0)
    return torch.device("cuda")


@pytest.fixture
def flash_attn_ops():
    try:
        from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
    except Exception as exc:
        pytest.skip(f"flash_attn is required for attention parity tests: {exc}")
    return flash_attn_varlen_func, flash_attn_with_kvcache


def _build_block_tables(lengths: list[int], block_size: int, device: torch.device):
    blocks_per_seq = [(length + block_size - 1) // block_size for length in lengths]
    total_blocks = sum(blocks_per_seq)
    permuted_ids = torch.randperm(total_blocks, device=device, dtype=torch.int32)
    max_blocks = max(blocks_per_seq)
    block_tables = torch.full(
        (len(lengths), max_blocks),
        -1,
        dtype=torch.int32,
        device=device,
    )

    offset = 0
    for i, num_blocks in enumerate(blocks_per_seq):
        block_tables[i, :num_blocks] = permuted_ids[offset:offset + num_blocks]
        offset += num_blocks

    return block_tables, total_blocks


def _scatter_dense_to_paged(
    k_dense: torch.Tensor,
    v_dense: torch.Tensor,
    lengths: list[int],
    block_tables: torch.Tensor,
    block_size: int,
    num_blocks: int,
):
    num_kv_heads = k_dense.shape[1]
    head_dim = k_dense.shape[2]
    device = k_dense.device
    k_cache = torch.zeros(
        num_blocks,
        block_size,
        num_kv_heads,
        head_dim,
        device=device,
        dtype=k_dense.dtype,
    )
    v_cache = torch.zeros_like(k_cache)

    offset = 0
    for batch_idx, length in enumerate(lengths):
        for pos in range(length):
            block_idx = pos // block_size
            block_id = int(block_tables[batch_idx, block_idx].item())
            pos_in_block = pos % block_size
            k_cache[block_id, pos_in_block] = k_dense[offset + pos]
            v_cache[block_id, pos_in_block] = v_dense[offset + pos]
        offset += length

    return k_cache.contiguous(), v_cache.contiguous()


def _dense_attention_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
):
    original_dtype = q.dtype
    kv_group_num = num_heads // num_kv_heads
    q = q.float()
    k = k.float()
    v = v.float()
    out = torch.empty_like(q)
    scale = q.shape[-1] ** -0.5

    for head_idx in range(num_heads):
        kv_head = head_idx // kv_group_num
        scores = torch.matmul(q[:, head_idx], k[:, kv_head].transpose(0, 1)) * scale
        probs = torch.softmax(scores, dim=-1)
        out[:, head_idx] = torch.matmul(probs, v[:, kv_head])

    return out.to(dtype=original_dtype)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    half = x.shape[-1] // 2
    rotated_half = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
    return (x * cos[:, None, :] + rotated_half * sin[:, None, :]).to(x.dtype)


def _rope_table(length: int, head_dim: int, device: torch.device, dtype: torch.dtype):
    inv_freq = 1.0 / (
        10000 ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )
    freqs = torch.outer(torch.arange(length, device=device, dtype=torch.float32), inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def test_materialized_affix_matches_key_side_relocation(cuda_device):
    num_layers = 2
    num_slots = 24
    num_kv_heads = 2
    head_dim = 64
    affix_len = 7
    affix_shift = 5
    dst_slot = 12
    dtype = torch.bfloat16
    k_cache = torch.randn(
        num_layers,
        num_slots,
        num_kv_heads,
        head_dim,
        device=cuda_device,
        dtype=dtype,
    )
    v_cache = torch.randn_like(k_cache)
    source_k = k_cache[:, :affix_len].clone()
    source_v = v_cache[:, :affix_len].clone()
    cos, sin = _rope_table(num_slots, head_dim, cuda_device, torch.float32)

    materialize_relocated_affix(
        k_cache,
        v_cache,
        cos,
        sin,
        dst_slot,
        affix_len,
        affix_shift,
    )
    expected_k = _apply_rope(
        source_k.reshape(-1, num_kv_heads, head_dim),
        cos[affix_shift].expand(num_layers * affix_len, -1),
        sin[affix_shift].expand(num_layers * affix_len, -1),
    ).view_as(source_k)

    torch.testing.assert_close(k_cache[:, dst_slot:dst_slot + affix_len], expected_k, rtol=0, atol=0)
    torch.testing.assert_close(v_cache[:, dst_slot:dst_slot + affix_len], source_v, rtol=0, atol=0)


def test_materialized_affix_attention_is_bitwise_equal_to_key_side_relocation(
    cuda_device,
):
    num_layers = 1
    num_slots = 32
    num_heads = 4
    num_kv_heads = 2
    head_dim = 64
    q_len = 5
    kv_len = 13
    affix_start = 7
    affix_len = 3
    dst_slot = 24
    dtype = torch.bfloat16
    q = torch.randn(q_len, num_heads, head_dim, device=cuda_device, dtype=dtype)
    k_cache = torch.randn(
        num_layers,
        num_slots,
        num_kv_heads,
        head_dim,
        device=cuda_device,
        dtype=dtype,
    )
    v_cache = torch.randn_like(k_cache)
    cos, sin = _rope_table(num_slots, head_dim, cuda_device, torch.float32)
    key_side_slots = list(range(affix_len, affix_len + kv_len))
    for local_pos, logical_pos in enumerate(range(affix_start, affix_start + affix_len)):
        key_side_slots[logical_pos] = local_pos
    cu_q = torch.tensor([0, q_len], device=cuda_device, dtype=torch.int32)
    cu_k = torch.tensor([0, kv_len], device=cuda_device, dtype=torch.int32)
    affix_shifts = torch.tensor([affix_start], device=cuda_device, dtype=torch.int32)

    key_side = acache_attention(
        q,
        k_cache[0].view(4, 8, num_kv_heads, head_dim),
        v_cache[0].view(4, 8, num_kv_heads, head_dim),
        cu_q,
        cu_k,
        torch.tensor(key_side_slots, device=cuda_device, dtype=torch.int32),
        shared_prefix_len=affix_len,
        affix_shifts=affix_shifts,
        rope_cos=cos,
        rope_sin=sin,
    )

    materialize_relocated_affix(
        k_cache,
        v_cache,
        cos,
        sin,
        dst_slot,
        affix_len,
        affix_start,
    )
    materialized_slots = list(key_side_slots)
    for local_pos, logical_pos in enumerate(range(affix_start, affix_start + affix_len)):
        materialized_slots[logical_pos] = dst_slot + local_pos
    materialized = acache_attention(
        q,
        k_cache[0].view(4, 8, num_kv_heads, head_dim),
        v_cache[0].view(4, 8, num_kv_heads, head_dim),
        cu_q,
        cu_k,
        torch.tensor(materialized_slots, device=cuda_device, dtype=torch.int32),
    )

    torch.testing.assert_close(materialized, key_side, rtol=0, atol=0)


def test_large_head_dim_acache_matches_dense_reference(cuda_device):
    num_blocks = 8
    block_size = 8
    num_heads = 4
    num_kv_heads = 2
    head_dim = 160
    q_lens = [3, 2]
    kv_lens = [7, 5]

    q = torch.randn(
        sum(q_lens),
        num_heads,
        head_dim,
        device=cuda_device,
        dtype=torch.float16,
    ).contiguous()
    k_cache = torch.randn(
        num_blocks,
        block_size,
        num_kv_heads,
        head_dim,
        device=cuda_device,
        dtype=torch.float16,
    )
    v_cache = torch.randn_like(k_cache)
    flat_k = k_cache.view(-1, num_kv_heads, head_dim)
    flat_v = v_cache.view(-1, num_kv_heads, head_dim)

    maps = []
    for kv_len in kv_lens:
        maps.append(
            torch.randperm(
                num_blocks * block_size,
                device=cuda_device,
                dtype=torch.int32,
            )[:kv_len]
        )
    read_slot_mapping = torch.cat(maps, dim=0)
    cu_q = torch.tensor([0, q_lens[0], sum(q_lens)], device=cuda_device, dtype=torch.int32)
    cu_k = torch.tensor([0, kv_lens[0], sum(kv_lens)], device=cuda_device, dtype=torch.int32)

    out = acache_attention(q, k_cache, v_cache, cu_q, cu_k, read_slot_mapping)

    expected = []
    for batch_idx in range(len(q_lens)):
        q_seq = q[cu_q[batch_idx]:cu_q[batch_idx + 1]]
        slots = read_slot_mapping[cu_k[batch_idx]:cu_k[batch_idx + 1]].long()
        k_seq = flat_k[slots]
        v_seq = flat_v[slots]
        expected.append(
            _dense_attention_reference(q_seq, k_seq, v_seq, num_heads, num_kv_heads)
        )
    expected = torch.cat(expected, dim=0)

    torch.testing.assert_close(out.float(), expected.float(), rtol=5e-2, atol=5e-2)


def test_acache_relocates_shared_post_rope_keys(cuda_device):
    num_blocks = 2
    block_size = 8
    num_heads = 4
    num_kv_heads = 2
    head_dim = 64
    q_len = 3
    kv_len = 7
    affix_start = 3
    shared_len = 2
    dtype = torch.float16

    q_single = torch.randn(q_len, num_heads, head_dim, device=cuda_device, dtype=dtype)
    q = torch.cat((q_single, q_single), dim=0)
    raw_k = torch.randn(kv_len, num_kv_heads, head_dim, device=cuda_device, dtype=dtype)
    values = torch.randn_like(raw_k)
    k_cache = torch.zeros(
        num_blocks, block_size, num_kv_heads, head_dim, device=cuda_device, dtype=dtype
    )
    v_cache = torch.zeros_like(k_cache)
    flat_k = k_cache.view(-1, num_kv_heads, head_dim)
    flat_v = v_cache.view(-1, num_kv_heads, head_dim)
    cos, sin = _rope_table(kv_len, head_dim, cuda_device, dtype)
    read_slot_mapping = []
    for logical_pos in range(kv_len):
        if affix_start <= logical_pos < affix_start + shared_len:
            source_pos = logical_pos - affix_start
            slot = source_pos
        else:
            source_pos = logical_pos
            slot = shared_len + logical_pos
        flat_k[slot] = _apply_rope(
            raw_k[logical_pos:logical_pos + 1],
            cos[source_pos:source_pos + 1],
            sin[source_pos:source_pos + 1],
        )[0]
        flat_v[slot] = values[logical_pos]
        read_slot_mapping.append(slot)
    packed_mapping = [
        *[slot for slot in read_slot_mapping if slot < shared_len],
        *[slot for slot in read_slot_mapping if slot >= shared_len],
    ]
    read_slot_mapping = torch.tensor(
        packed_mapping * 2, device=cuda_device, dtype=torch.int32
    )
    cu_q = torch.tensor([0, q_len, 2 * q_len], device=cuda_device, dtype=torch.int32)
    cu_k = torch.tensor([0, kv_len, 2 * kv_len], device=cuda_device, dtype=torch.int32)
    shared_kv_lens = torch.tensor(
        [shared_len, shared_len], device=cuda_device, dtype=torch.int32
    )
    affix_shifts = torch.tensor(
        [affix_start, affix_start], device=cuda_device, dtype=torch.int32
    )

    out = acache_attention(
        q,
        k_cache,
        v_cache,
        cu_q,
        cu_k,
        read_slot_mapping,
        shared_prefix_len=shared_len,
        shared_kv_lens=shared_kv_lens,
        affix_shifts=affix_shifts,
        rope_cos=cos,
        rope_sin=sin,
    )

    expected_k = _apply_rope(raw_k, cos[:kv_len], sin[:kv_len])
    expected_single = _dense_attention_reference(
        q_single, expected_k, values, num_heads, num_kv_heads
    )
    expected = torch.cat((expected_single, expected_single), dim=0)

    torch.testing.assert_close(out.float(), expected.float(), rtol=5e-2, atol=5e-2)


def test_baseline_ragged_matches_flash_attn_varlen(cuda_device, flash_attn_ops):
    flash_attn_varlen_func, _ = flash_attn_ops
    block_size = 8
    num_heads = 4
    num_kv_heads = 2
    head_dim = 64
    dtype = torch.float16
    scale = head_dim ** -0.5
    lengths = [7, 5]
    total_tokens = sum(lengths)

    q = torch.randn(
        total_tokens,
        num_heads,
        head_dim,
        device=cuda_device,
        dtype=dtype,
    ).contiguous()
    k = torch.randn(
        total_tokens,
        num_kv_heads,
        head_dim,
        device=cuda_device,
        dtype=dtype,
    ).contiguous()
    v = torch.randn_like(k)

    block_tables, num_blocks = _build_block_tables(lengths, block_size, cuda_device)
    k_cache, v_cache = _scatter_dense_to_paged(
        k,
        v,
        lengths,
        block_tables,
        block_size,
        num_blocks,
    )
    cu = torch.tensor([0, lengths[0], total_tokens], device=cuda_device, dtype=torch.int32)

    out_custom = baseline_ragged_attention(q, k_cache, v_cache, cu, cu, block_tables, scale)
    out_ref = flash_attn_varlen_func(
        q,
        k,
        v,
        cu,
        cu,
        max(lengths),
        max(lengths),
        softmax_scale=scale,
        causal=False,
    )

    torch.testing.assert_close(out_custom.float(), out_ref.float(), rtol=5e-2, atol=5e-2)


def test_baseline_decode_matches_flash_attn_with_kvcache(cuda_device, flash_attn_ops):
    _, flash_attn_with_kvcache = flash_attn_ops
    # flash_attn_with_kvcache requires paged KV block size to be divisible by 256.
    block_size = 256
    num_heads = 4
    num_kv_heads = 2
    head_dim = 64
    dtype = torch.float16
    scale = head_dim ** -0.5
    lengths = [300, 270]
    query_len = 4

    q = torch.randn(
        len(lengths),
        query_len,
        num_heads,
        head_dim,
        device=cuda_device,
        dtype=dtype,
    ).contiguous()
    k = torch.randn(
        sum(lengths),
        num_kv_heads,
        head_dim,
        device=cuda_device,
        dtype=dtype,
    ).contiguous()
    v = torch.randn_like(k)

    block_tables, num_blocks = _build_block_tables(lengths, block_size, cuda_device)
    k_cache, v_cache = _scatter_dense_to_paged(
        k,
        v,
        lengths,
        block_tables,
        block_size,
        num_blocks,
    )
    seqlens = torch.tensor(lengths, device=cuda_device, dtype=torch.int32)

    out_custom = baseline_decode_attention(q, k_cache, v_cache, seqlens, block_tables, scale)
    out_ref = flash_attn_with_kvcache(
        q,
        k_cache,
        v_cache,
        cache_seqlens=seqlens,
        block_table=block_tables,
        softmax_scale=scale,
        causal=False,
    )

    torch.testing.assert_close(out_custom.float(), out_ref.float(), rtol=5e-2, atol=5e-2)


def test_acache_matches_flash_attn_on_effective_kv(cuda_device, flash_attn_ops):
    flash_attn_varlen_func, _ = flash_attn_ops
    block_size = 8
    num_heads = 4
    num_kv_heads = 2
    head_dim = 64
    dtype = torch.float16
    scale = head_dim ** -0.5
    q_lens = [5, 3]
    kv_lens = [9, 6]
    num_blocks = 8

    q = torch.randn(
        sum(q_lens),
        num_heads,
        head_dim,
        device=cuda_device,
        dtype=dtype,
    ).contiguous()
    k_cache = torch.randn(
        num_blocks,
        block_size,
        num_kv_heads,
        head_dim,
        device=cuda_device,
        dtype=dtype,
    )
    v_cache = torch.randn_like(k_cache)

    flat_k = k_cache.view(-1, num_kv_heads, head_dim)
    flat_v = v_cache.view(-1, num_kv_heads, head_dim)
    maps = []
    for kv_len in kv_lens:
        maps.append(
            torch.randperm(
                num_blocks * block_size,
                device=cuda_device,
                dtype=torch.int32,
            )[:kv_len]
        )
    read_slot_mapping = torch.cat(maps, dim=0)

    cu_q = torch.tensor([0, q_lens[0], sum(q_lens)], device=cuda_device, dtype=torch.int32)
    cu_k = torch.tensor([0, kv_lens[0], sum(kv_lens)], device=cuda_device, dtype=torch.int32)

    eff_k = []
    eff_v = []
    for batch_idx in range(len(q_lens)):
        slots = read_slot_mapping[cu_k[batch_idx]:cu_k[batch_idx + 1]].long()
        eff_k.append(flat_k[slots])
        eff_v.append(flat_v[slots])
    eff_k = torch.cat(eff_k, dim=0).contiguous()
    eff_v = torch.cat(eff_v, dim=0).contiguous()

    out_custom = acache_attention(q, k_cache, v_cache, cu_q, cu_k, read_slot_mapping, scale)
    out_ref = flash_attn_varlen_func(
        q,
        eff_k,
        eff_v,
        cu_q,
        cu_k,
        max(q_lens),
        max(kv_lens),
        softmax_scale=scale,
        causal=False,
    )

    torch.testing.assert_close(out_custom.float(), out_ref.float(), rtol=5e-2, atol=5e-2)
