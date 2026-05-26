import triton
import triton.language as tl
import torch


@triton.jit
def _baseline_ragged_attention_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    block_table_ptr,
    cu_seqlens_q_ptr,
    cu_seqlens_k_ptr,
    out_ptr,
    sm_scale,
    stride_qt,
    stride_qh,
    stride_qd,
    stride_kb,
    stride_ks,
    stride_kh,
    stride_kd,
    stride_vb,
    stride_vs,
    stride_vh,
    stride_vd,
    stride_bt,
    stride_ot,
    stride_oh,
    stride_od,
    kv_group_num: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    q_tile_id = tl.program_id(0)
    cur_batch = tl.program_id(1)
    cur_head = tl.program_id(2)

    q_start = tl.load(cu_seqlens_q_ptr + cur_batch)
    q_end = tl.load(cu_seqlens_q_ptr + cur_batch + 1)
    q_len = q_end - q_start
    if q_tile_id * BLOCK_M >= q_len:
        return
    kv_len = tl.load(cu_seqlens_k_ptr + cur_batch + 1) - tl.load(cu_seqlens_k_ptr + cur_batch)
    if kv_len <= 0:
        return

    offs_m = q_tile_id * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    q_mask = offs_m < q_len
    d_mask = offs_d < HEAD_DIM
    q_tokens = q_start + offs_m
    kv_head = cur_head // kv_group_num

    q_ptrs = q_ptr + q_tokens[:, None] * stride_qt + cur_head * stride_qh + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=q_mask[:, None] & d_mask[None, :], other=0.0)

    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    start_n = 0
    while start_n < kv_len:
        offs_n = start_n + tl.arange(0, BLOCK_N)
        k_mask = offs_n < kv_len
        block_indices = offs_n // BLOCK_SIZE
        pos_in_block = offs_n % BLOCK_SIZE
        block_ids = tl.load(block_table_ptr + cur_batch * stride_bt + block_indices, mask=k_mask, other=0)

        k_ptrs = (
            k_ptr
            + block_ids[None, :] * stride_kb
            + pos_in_block[None, :] * stride_ks
            + kv_head * stride_kh
            + offs_d[:, None] * stride_kd
        )
        v_ptrs = (
            v_ptr
            + block_ids[:, None] * stride_vb
            + pos_in_block[:, None] * stride_vs
            + kv_head * stride_vh
            + offs_d[None, :] * stride_vd
        )

        k = tl.load(k_ptrs, mask=d_mask[:, None] & k_mask[None, :], other=0.0)
        v = tl.load(v_ptrs, mask=k_mask[:, None] & d_mask[None, :], other=0.0)

        qk = tl.dot(q, k)
        qk = qk * sm_scale
        qk = tl.where(k_mask[None, :], qk, -float("inf"))

        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp(qk - m_i_new[:, None])
        alpha = tl.exp(m_i - m_i_new)
        acc = acc * alpha[:, None]
        acc += tl.dot(p.to(v.dtype), v)
        l_i = alpha * l_i + tl.sum(p, 1)
        m_i = m_i_new
        start_n += BLOCK_N

    l_i = tl.where(l_i > 0, l_i, 1.0)
    acc = acc / l_i[:, None]

    out_ptrs = out_ptr + q_tokens[:, None] * stride_ot + cur_head * stride_oh + offs_d[None, :] * stride_od
    tl.store(out_ptrs, acc, mask=q_mask[:, None] & d_mask[None, :])


@triton.jit
def _baseline_decode_attention_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    seqlens_ptr,
    block_table_ptr,
    out_ptr,
    sm_scale,
    stride_qb,
    stride_qt,
    stride_qh,
    stride_qd,
    stride_kb,
    stride_ks,
    stride_kh,
    stride_kd,
    stride_vb,
    stride_vs,
    stride_vh,
    stride_vd,
    stride_bt,
    stride_ob,
    stride_ot,
    stride_oh,
    stride_od,
    kv_group_num: tl.constexpr,
    QUERY_LEN: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    q_tile_id = tl.program_id(0)
    cur_batch = tl.program_id(1)
    cur_head = tl.program_id(2)

    kv_len = tl.load(seqlens_ptr + cur_batch)
    if kv_len <= 0 or q_tile_id * BLOCK_M >= QUERY_LEN:
        return

    offs_m = q_tile_id * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    q_mask = offs_m < QUERY_LEN
    d_mask = offs_d < HEAD_DIM
    kv_head = cur_head // kv_group_num

    q_ptrs = q_ptr + cur_batch * stride_qb + offs_m[:, None] * stride_qt + cur_head * stride_qh + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=q_mask[:, None] & d_mask[None, :], other=0.0)

    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    start_n = 0
    while start_n < kv_len:
        offs_n = start_n + tl.arange(0, BLOCK_N)
        k_mask = offs_n < kv_len
        block_indices = offs_n // BLOCK_SIZE
        pos_in_block = offs_n % BLOCK_SIZE
        block_ids = tl.load(block_table_ptr + cur_batch * stride_bt + block_indices, mask=k_mask, other=0)

        k_ptrs = (
            k_ptr
            + block_ids[None, :] * stride_kb
            + pos_in_block[None, :] * stride_ks
            + kv_head * stride_kh
            + offs_d[:, None] * stride_kd
        )
        v_ptrs = (
            v_ptr
            + block_ids[:, None] * stride_vb
            + pos_in_block[:, None] * stride_vs
            + kv_head * stride_vh
            + offs_d[None, :] * stride_vd
        )

        k = tl.load(k_ptrs, mask=d_mask[:, None] & k_mask[None, :], other=0.0)
        v = tl.load(v_ptrs, mask=k_mask[:, None] & d_mask[None, :], other=0.0)

        qk = tl.dot(q, k)
        qk = qk * sm_scale
        qk = tl.where(k_mask[None, :], qk, -float("inf"))

        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp(qk - m_i_new[:, None])
        alpha = tl.exp(m_i - m_i_new)
        acc = acc * alpha[:, None]
        acc += tl.dot(p.to(v.dtype), v)
        l_i = alpha * l_i + tl.sum(p, 1)
        m_i = m_i_new
        start_n += BLOCK_N

    l_i = tl.where(l_i > 0, l_i, 1.0)
    acc = acc / l_i[:, None]

    out_ptrs = out_ptr + cur_batch * stride_ob + offs_m[:, None] * stride_ot + cur_head * stride_oh + offs_d[None, :] * stride_od
    tl.store(out_ptrs, acc, mask=q_mask[:, None] & d_mask[None, :])


@triton.jit
def _acache_attention_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    read_slot_map_ptr,
    cu_seqlens_q_ptr,
    cu_seqlens_k_ptr,
    out_ptr,
    sm_scale,
    stride_qt,
    stride_qh,
    stride_qd,
    stride_ks,
    stride_kh,
    stride_kd,
    stride_vs,
    stride_vh,
    stride_vd,
    stride_ot,
    stride_oh,
    stride_od,
    kv_group_num: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    q_tile_id = tl.program_id(0)
    cur_batch = tl.program_id(1)
    cur_head = tl.program_id(2)

    q_start = tl.load(cu_seqlens_q_ptr + cur_batch)
    q_end = tl.load(cu_seqlens_q_ptr + cur_batch + 1)
    q_len = q_end - q_start
    if q_tile_id * BLOCK_M >= q_len:
        return
    kv_start = tl.load(cu_seqlens_k_ptr + cur_batch)
    kv_end = tl.load(cu_seqlens_k_ptr + cur_batch + 1)
    kv_len = kv_end - kv_start
    if kv_len <= 0:
        return

    offs_m = q_tile_id * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    q_mask = offs_m < q_len
    d_mask = offs_d < HEAD_DIM
    q_tokens = q_start + offs_m
    kv_head = cur_head // kv_group_num

    q_ptrs = q_ptr + q_tokens[:, None] * stride_qt + cur_head * stride_qh + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=q_mask[:, None] & d_mask[None, :], other=0.0)

    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    start_n = 0
    while start_n < kv_len:
        offs_n = start_n + tl.arange(0, BLOCK_N)
        k_mask = offs_n < kv_len
        slots = tl.load(read_slot_map_ptr + kv_start + offs_n, mask=k_mask, other=0)

        k_ptrs = k_ptr + slots[None, :] * stride_ks + kv_head * stride_kh + offs_d[:, None] * stride_kd
        v_ptrs = v_ptr + slots[:, None] * stride_vs + kv_head * stride_vh + offs_d[None, :] * stride_vd

        k = tl.load(k_ptrs, mask=d_mask[:, None] & k_mask[None, :], other=0.0)
        v = tl.load(v_ptrs, mask=k_mask[:, None] & d_mask[None, :], other=0.0)

        qk = tl.dot(q, k)
        qk = qk * sm_scale
        qk = tl.where(k_mask[None, :], qk, -float("inf"))

        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp(qk - m_i_new[:, None])
        alpha = tl.exp(m_i - m_i_new)
        acc = acc * alpha[:, None]
        acc += tl.dot(p.to(v.dtype), v)
        l_i = alpha * l_i + tl.sum(p, 1)
        m_i = m_i_new
        start_n += BLOCK_N

    l_i = tl.where(l_i > 0, l_i, 1.0)
    acc = acc / l_i[:, None]

    out_ptrs = out_ptr + q_tokens[:, None] * stride_ot + cur_head * stride_oh + offs_d[None, :] * stride_od
    tl.store(out_ptrs, acc, mask=q_mask[:, None] & d_mask[None, :])


def _get_launch_config(head_dim: int):
    block_m = 16
    block_n = 64 if head_dim <= 128 else 32
    block_d = triton.next_power_of_2(head_dim)
    num_warps = 4 if head_dim <= 128 else 8
    return block_m, block_n, block_d, num_warps


def baseline_ragged_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    block_tables: torch.Tensor,
    softmax_scale: float | None = None,
):
    assert q.dim() == 3
    assert block_tables is not None
    assert q.is_cuda and k_cache.is_cuda and v_cache.is_cuda and block_tables.is_cuda
    head_dim = q.shape[-1]
    num_heads = q.shape[1]
    num_kv_heads = k_cache.shape[2]
    kv_group_num = num_heads // num_kv_heads
    batch = cu_seqlens_q.numel() - 1
    max_q = int((cu_seqlens_q[1:] - cu_seqlens_q[:-1]).max().item())
    if softmax_scale is None:
        softmax_scale = head_dim ** -0.5

    block_m, block_n, block_d, num_warps = _get_launch_config(head_dim)
    out = torch.empty_like(q)
    grid = (triton.cdiv(max_q, block_m), batch, num_heads)
    _baseline_ragged_attention_kernel[grid](
        q,
        k_cache,
        v_cache,
        block_tables,
        cu_seqlens_q,
        cu_seqlens_k,
        out,
        softmax_scale,
        stride_qt=q.stride(0),
        stride_qh=q.stride(1),
        stride_qd=q.stride(2),
        stride_kb=k_cache.stride(0),
        stride_ks=k_cache.stride(1),
        stride_kh=k_cache.stride(2),
        stride_kd=k_cache.stride(3),
        stride_vb=v_cache.stride(0),
        stride_vs=v_cache.stride(1),
        stride_vh=v_cache.stride(2),
        stride_vd=v_cache.stride(3),
        stride_bt=block_tables.stride(0),
        stride_ot=out.stride(0),
        stride_oh=out.stride(1),
        stride_od=out.stride(2),
        kv_group_num=kv_group_num,
        BLOCK_SIZE=k_cache.shape[1],
        HEAD_DIM=head_dim,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        num_warps=num_warps,
        num_stages=2,
    )
    return out


def baseline_decode_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    seqlens: torch.Tensor,
    block_tables: torch.Tensor,
    softmax_scale: float | None = None,
):
    assert q.dim() == 4
    assert block_tables is not None
    assert q.is_cuda and k_cache.is_cuda and v_cache.is_cuda and block_tables.is_cuda
    batch, query_len, num_heads, head_dim = q.shape
    num_kv_heads = k_cache.shape[2]
    kv_group_num = num_heads // num_kv_heads
    if softmax_scale is None:
        softmax_scale = head_dim ** -0.5

    block_m, block_n, block_d, num_warps = _get_launch_config(head_dim)
    out = torch.empty_like(q)
    grid = (triton.cdiv(query_len, block_m), batch, num_heads)
    _baseline_decode_attention_kernel[grid](
        q,
        k_cache,
        v_cache,
        seqlens,
        block_tables,
        out,
        softmax_scale,
        stride_qb=q.stride(0),
        stride_qt=q.stride(1),
        stride_qh=q.stride(2),
        stride_qd=q.stride(3),
        stride_kb=k_cache.stride(0),
        stride_ks=k_cache.stride(1),
        stride_kh=k_cache.stride(2),
        stride_kd=k_cache.stride(3),
        stride_vb=v_cache.stride(0),
        stride_vs=v_cache.stride(1),
        stride_vh=v_cache.stride(2),
        stride_vd=v_cache.stride(3),
        stride_bt=block_tables.stride(0),
        stride_ob=out.stride(0),
        stride_ot=out.stride(1),
        stride_oh=out.stride(2),
        stride_od=out.stride(3),
        kv_group_num=kv_group_num,
        QUERY_LEN=query_len,
        BLOCK_SIZE=k_cache.shape[1],
        HEAD_DIM=head_dim,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        num_warps=num_warps,
        num_stages=2,
    )
    return out


def acache_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    read_slot_mapping: torch.Tensor,
    softmax_scale: float | None = None,
):
    assert q.dim() == 3
    assert q.is_cuda and k_cache.is_cuda and v_cache.is_cuda
    assert cu_seqlens_q.is_cuda and cu_seqlens_k.is_cuda and read_slot_mapping.is_cuda
    assert q.stride(-1) == 1
    assert k_cache.stride(-1) == 1 and v_cache.stride(-1) == 1

    total_q, num_heads, head_dim = q.shape
    num_kv_heads = k_cache.shape[2]
    kv_group_num = num_heads // num_kv_heads
    batch = cu_seqlens_q.numel() - 1
    max_q = int((cu_seqlens_q[1:] - cu_seqlens_q[:-1]).max().item())
    if softmax_scale is None:
        softmax_scale = head_dim ** -0.5

    block_m, block_n, block_d, num_warps = _get_launch_config(head_dim)

    out = torch.empty_like(q)
    grid = (triton.cdiv(max_q, block_m), batch, num_heads)
    _acache_attention_kernel[grid](
        q,
        k_cache,
        v_cache,
        read_slot_mapping,
        cu_seqlens_q,
        cu_seqlens_k,
        out,
        softmax_scale,
        stride_qt=q.stride(0),
        stride_qh=q.stride(1),
        stride_qd=q.stride(2),
        stride_ks=k_cache.stride(1),
        stride_kh=k_cache.stride(2),
        stride_kd=k_cache.stride(3),
        stride_vs=v_cache.stride(1),
        stride_vh=v_cache.stride(2),
        stride_vd=v_cache.stride(3),
        stride_ot=out.stride(0),
        stride_oh=out.stride(1),
        stride_od=out.stride(2),
        kv_group_num=kv_group_num,
        HEAD_DIM=head_dim,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        num_warps=num_warps,
        num_stages=2,
    )
    return out
