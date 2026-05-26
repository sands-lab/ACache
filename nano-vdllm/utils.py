from dataclasses import dataclass
import torch
import random
import numpy as np


@dataclass
class Context:
    is_caching: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    seqlens: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None
    use_acache: bool = False
    read_slot_mapping: torch.Tensor | None = None

_CONTEXT = Context()

def get_context():
    return _CONTEXT

def set_context(
    is_caching,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    seqlens=None,
    max_seqlen_q=0,
    max_seqlen_k=0,
    slot_mapping=None,
    block_tables=None,
    use_acache=False,
    read_slot_mapping=None,
):
    global _CONTEXT
    _CONTEXT = Context(
        is_caching,
        cu_seqlens_q,
        cu_seqlens_k,
        seqlens,
        max_seqlen_q,
        max_seqlen_k,
        slot_mapping,
        block_tables,
        use_acache,
        read_slot_mapping,
    )

def reset_context():
    global _CONTEXT
    _CONTEXT = Context()


def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    # logits = logits.to(torch.float64)
    # noise = torch.rand_like(logits, dtype=torch.float64)
    noise = torch.rand_like(logits, dtype=logits.dtype)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
