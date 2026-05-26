import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from model.acache_attention import acache_attention, baseline_decode_attention, baseline_ragged_attention
from utils import get_context


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        num_kv_heads,
        scale=None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads
        self.scale = scale
        self.use_reference_attention = False
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        assert k_cache.numel() and v_cache.numel()
        reshape_output = False
        if k.dim() == 4: # for batched decoding
            bs, seqlen = k.shape[:2]
            k = k.view(bs * seqlen, self.num_kv_heads, self.head_dim)
            v = v.view(bs * seqlen, self.num_kv_heads, self.head_dim)
        store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.use_acache:
            if q.dim() == 4:
                q = q.view(bs * seqlen, self.num_heads, self.head_dim)
                reshape_output = True
            o = acache_attention(
                q,
                k_cache,
                v_cache,
                context.cu_seqlens_q,
                context.cu_seqlens_k,
                context.read_slot_mapping,
                self.scale,
            )
            if reshape_output:
                o = o.view(bs, seqlen, self.num_heads, self.head_dim)
        elif context.is_caching:
            if self.use_reference_attention:
                o = flash_attn_varlen_func(q, k, v,
                                            max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                            max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                            softmax_scale=self.scale)
            else:
                o = baseline_ragged_attention(
                    q,
                    k_cache,
                    v_cache,
                    context.cu_seqlens_q,
                    context.cu_seqlens_k,
                    context.block_tables,
                    self.scale,
                )
        else:
            if self.use_reference_attention:
                o = flash_attn_with_kvcache(q, k_cache, v_cache,
                                            cache_seqlens=context.seqlens,
                                            softmax_scale=self.scale, block_table=context.block_tables)
            else:
                o = baseline_decode_attention(
                    q,
                    k_cache,
                    v_cache,
                    context.seqlens,
                    context.block_tables,
                    self.scale,
                )
        return o
