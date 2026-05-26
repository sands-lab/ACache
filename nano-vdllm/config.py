from dataclasses import dataclass
from transformers import AutoConfig


@dataclass
class Config:
    hf_config: AutoConfig = None
    mask_id: int = 126336
    recompute_batch_size: int = 4
    anchor_selection_batch_size: int = 1
    max_num_seqs: int = 0
    gen_length: int = 1024
    block_length: int = 1024
    remasking: str = 'low_confidence'  # 'low_confidence', 'random', 'none'
    threshold: float = 0.9
    cache_block_size: int = 256
    max_num_batched_tokens: int = 16384
    gpu_memory_utilization: float = 0.9
    num_kvcache_blocks: int = -1
    temperature: float = 0.
    enable_acache: bool = False
    anchor_ratio: float = 0.05
    selection_mode: str = 'top'
    use_reference_attention: bool = False
    acache_memory_reserve_gb: float = 4.0
    profile_timing: bool = False
    model_type: str = 'llada'
