import gc
import time
import types
from contextlib import contextmanager
import torch
import numpy as np
from tqdm import tqdm
from utils import add_gumbel_noise, get_context, reset_context, set_context
from sequence import Sequence
from block_manager import BlockManager
from model.acache_attention import materialize_relocated_affix

class ModelRunner:

    def __init__(self, model, config):
        self.config = config
        self.model = model
        self.base_model = model.module if hasattr(model, "module") else model
        self.device = model.device
        self.kv_cache = None
        self.kv_cache_block_bytes = 0
        self.block_manager = None
        self.shared_prefix_len = 0
        self.shared_prefix_blocks = 0
        self.acache_ready = False
        self.shared_prefix_token_ids: tuple[int, ...] = ()
        self._affix_relocation_enabled = False
        self._affix_relocation_max_position = 0
        self._relocated_affix_slot_offset = None
        self.profile_timing = bool(getattr(config, "profile_timing", False))
        self.last_timing = {}
        self._active_timing = None
        self.allocate_kv_cache()

    def _is_dream_model(self):
        return str(getattr(self.config, "model_type", "")).lower() == "dream"

    def _uses_decoder_layers(self):
        return str(getattr(self.config, "model_type", "")).lower() in {
            "dream",
        }

    def _forward_model(self, input_ids, position_ids=None, compute_logits=True):
        if self._uses_decoder_layers():
            return self.base_model(
                input_ids=input_ids,
                position_ids=position_ids,
                compute_logits=compute_logits,
            )
        return self.base_model.model.forward(
            input_ids=input_ids,
            position_ids=position_ids,
            compute_logits=compute_logits,
        )

    def _dream_decode_logits(self, logits):
        if not self._is_dream_model():
            return logits
        if logits.dim() != 3:
            return logits
        return torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

    def _dream_shift_keep_indices(self, keep_indices, row_starts):
        row_starts = row_starts.to(device=keep_indices.device, dtype=keep_indices.dtype)
        min_indices = row_starts.unsqueeze(1)
        return torch.maximum(keep_indices, min_indices + 1) - 1

    def _sample_dream_seed_tokens(self, logits):
        temperature = float(self.config.temperature)
        if temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            try:
                x0 = torch.distributions.Categorical(probs=probs).sample()
            except Exception:
                x0 = probs.argmax(dim=-1)
        else:
            x0 = logits.argmax(dim=-1)
        return x0

    def _profile_timing_enabled(self):
        return self.profile_timing

    def _new_timing(self, mode: str, num_prompts: int):
        if not self.profile_timing:
            return None
        return {
            "mode": mode,
            "num_prompts": int(num_prompts),
        }

    def _start_timing(self):
        if not self._profile_timing_enabled():
            return None
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return time.perf_counter()

    def _finish_timing(self, timing: dict | None, key: str, start):
        if start is None or timing is None:
            return
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        timing[key] = timing.get(key, 0.0) + (time.perf_counter() - start)

    def _add_timing_count(self, timing: dict | None, key: str, value=1):
        if timing is None:
            return
        timing[key] = timing.get(key, 0) + int(value)

    def _record_kv_cache_usage(self, timing: dict | None):
        if timing is None or self.block_manager is None:
            return
        cache_block_size = int(self.config.cache_block_size)
        used_blocks = len(self.block_manager.used_block_ids)
        capacity_blocks = len(self.block_manager.blocks)
        used_bytes = used_blocks * self.kv_cache_block_bytes
        capacity_bytes = capacity_blocks * self.kv_cache_block_bytes
        timing["kv_cache_block_size"] = cache_block_size
        timing["kv_cache_block_bytes"] = int(self.kv_cache_block_bytes)
        timing["kv_cache_slot_bytes"] = int(self.kv_cache_block_bytes // cache_block_size)
        timing["kv_cache_peak_used_blocks"] = max(timing.get("kv_cache_peak_used_blocks", 0), used_blocks)
        timing["kv_cache_peak_used_slots"] = max(
            timing.get("kv_cache_peak_used_slots", 0),
            used_blocks * cache_block_size,
        )
        timing["kv_cache_peak_used_bytes"] = max(timing.get("kv_cache_peak_used_bytes", 0), used_bytes)
        timing["kv_cache_peak_used_gb"] = timing["kv_cache_peak_used_bytes"] / (1024 ** 3)
        timing["kv_cache_peak_capacity_blocks"] = max(
            timing.get("kv_cache_peak_capacity_blocks", 0),
            capacity_blocks,
        )
        timing["kv_cache_peak_capacity_slots"] = max(
            timing.get("kv_cache_peak_capacity_slots", 0),
            capacity_blocks * cache_block_size,
        )
        timing["kv_cache_peak_capacity_bytes"] = max(
            timing.get("kv_cache_peak_capacity_bytes", 0),
            capacity_bytes,
        )
        timing["kv_cache_peak_capacity_gb"] = timing["kv_cache_peak_capacity_bytes"] / (1024 ** 3)

    def _reset_acache_state(self):
        self.shared_prefix_len = 0
        self.shared_prefix_blocks = 0
        self.acache_ready = False
        self.shared_prefix_token_ids = ()
        self._affix_relocation_enabled = False
        self._affix_relocation_max_position = 0
        self._relocated_affix_slot_offset = None

    def _release_kv_cache(self):
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = None
                module.v_cache = None
        if self.kv_cache is not None:
            del self.kv_cache
            self.kv_cache = None
        self.block_manager = None
        self._reset_acache_state()
        if self.device.type == "cuda":
            gc.collect()
            torch.cuda.empty_cache()

    def allocate_kv_cache(self, num_blocks: int | None = None):
        config, hf_config = self.config, self.config.hf_config
        num_heads = getattr(hf_config, "n_heads", getattr(hf_config, "num_attention_heads", None))
        if num_heads is None:
            raise AttributeError("HF config must define n_heads or num_attention_heads.")
        num_kv_heads = getattr(hf_config, "effective_n_kv_heads", None)
        if num_kv_heads is None:
            num_kv_heads = getattr(hf_config, "n_kv_heads", None)
        if num_kv_heads is None:
            num_kv_heads = getattr(hf_config, "num_key_value_heads", None)
        if num_kv_heads is None:
            num_kv_heads = num_heads
        hidden_size = getattr(hf_config, "d_model", getattr(hf_config, "hidden_size", None))
        if hidden_size is None:
            raise AttributeError("HF config must define d_model or hidden_size.")
        num_layers = getattr(hf_config, "n_layers", getattr(hf_config, "num_hidden_layers", None))
        if num_layers is None:
            raise AttributeError("HF config must define n_layers or num_hidden_layers.")
        head_dim = hidden_size // num_heads
        cache_block_size = config.cache_block_size

        block_bytes = 2 * num_layers * cache_block_size * num_kv_heads * head_dim * 2 # bfloat16
        self.kv_cache_block_bytes = block_bytes

        if num_blocks is None and config.num_kvcache_blocks < 0:
            free, total = torch.cuda.mem_get_info()
            used = total - free
            peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
            current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
            reserve_bytes = 0
            if config.enable_acache:
                reserve_bytes = int(max(float(config.acache_memory_reserve_gb), 0.0) * (1024 ** 3))
            available_bytes = total * config.gpu_memory_utilization - used - peak + current - reserve_bytes
            num_blocks = int(available_bytes // block_bytes)
        elif num_blocks is None:
            num_blocks = config.num_kvcache_blocks
        config.num_kvcache_blocks = int(num_blocks)
        assert config.num_kvcache_blocks >= 1, 'Not enough GPU memory.'

        if self.kv_cache is not None:
            self._release_kv_cache()
        self.kv_cache = torch.empty(2, num_layers, config.num_kvcache_blocks,
                                    cache_block_size, num_kv_heads, head_dim,
                                    dtype=torch.bfloat16, device=self.device)
        self._reset_acache_state()
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                module.use_reference_attention = self.config.use_reference_attention
                module.shared_prefix_len = 0
                layer_id += 1
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.cache_block_size)

    def _llada_blocks(self):
        inner_model = self.base_model.model
        if hasattr(inner_model, "transformer") and hasattr(inner_model.transformer, "blocks"):
            return list(inner_model.transformer.blocks)
        if hasattr(inner_model, "layers"):
            return list(inner_model.layers)
        raise AttributeError("Unable to locate transformer blocks/layers on model.")

    def _shared_affix_k(self, layer_idx: int):
        return self.kv_cache[0, layer_idx].view(-1, self.kv_cache.shape[-2], self.kv_cache.shape[-1])[:self.shared_prefix_len]

    def _configure_affix_relocation(self, seqs: list[Sequence]):
        uses_rope = bool(getattr(self.config.hf_config, "rope", True))
        relocation_required = uses_rope and any(seq.affix_start != 0 for seq in seqs)
        max_position = max((len(seq) for seq in seqs), default=0)
        if not relocation_required and not self._affix_relocation_enabled:
            return
        if (
            relocation_required
            and self._affix_relocation_enabled
            and self._affix_relocation_max_position >= max_position
        ):
            return

        for block in self._llada_blocks():
            if self._uses_decoder_layers():
                attention = block.self_attn.attn
                rotary = block.self_attn.rotary_emb
            else:
                attention = block.attn
                rotary = block.rotary_emb

            attention.shared_prefix_len = self.shared_prefix_len if relocation_required else 0
            if not relocation_required:
                continue
            if attention.relocation_rope_cos.shape[0] >= max_position:
                continue

            positions = torch.arange(max_position, device=self.device, dtype=torch.long)
            if self._uses_decoder_layers():
                sample = torch.empty(
                    max_position,
                    attention.num_kv_heads,
                    attention.head_dim,
                    device=self.device,
                    dtype=self.kv_cache.dtype,
                )
                cos, sin = rotary(sample, positions)
            else:
                sin, cos = rotary.get_rotary_embedding(max_position, self.device)
            attention.relocation_rope_cos = cos
            attention.relocation_rope_sin = sin
        self._affix_relocation_enabled = relocation_required
        self._affix_relocation_max_position = max_position if relocation_required else 0

    def _attention_read_layout(self, seq: Sequence, use_query_relocation: bool):
        if not use_query_relocation or seq.affix_start == 0:
            return list(seq.read_slot_map), 0
        shared_start = seq.shared_slot_offset
        shared_end = shared_start + self.shared_prefix_len
        shared_slots = [slot for slot in seq.read_slot_map if shared_start <= slot < shared_end]
        private_slots = [slot for slot in seq.read_slot_map if not (shared_start <= slot < shared_end)]
        return shared_slots + private_slots, len(shared_slots)

    def _use_query_side_affix_relocation(self, relocation_materialized: bool):
        return (
            self._affix_relocation_enabled
            and not relocation_materialized
            and self.config.max_num_seqs != 1
        )

    def _reserve_single_sequence_relocation_workspace(
        self, seq: Sequence, *, force: bool = False
    ):
        if (self.config.max_num_seqs != 1 and not force) or seq.affix_start == 0:
            return False
        if self._relocated_affix_slot_offset is not None:
            if self.block_manager.can_allocate(seq):
                return True
            self.block_manager.release_prefix_tail(self.shared_prefix_blocks)
            self._relocated_affix_slot_offset = None
            return False
        required_blocks = self.shared_prefix_blocks
        if len(self.block_manager.free_block_ids) < required_blocks + seq.num_cache_blocks:
            return False
        workspace_end = self.shared_prefix_blocks + required_blocks
        workspace_ids = range(self.shared_prefix_blocks, workspace_end)
        if any(not self.block_manager.blocks[block_id].is_free for block_id in workspace_ids):
            return False
        self.block_manager.reserve_prefix(workspace_end)
        self._relocated_affix_slot_offset = self.shared_prefix_blocks * self.config.cache_block_size
        self._record_kv_cache_usage(self._active_timing)
        return True

    def _materialize_single_sequence_affix(self, seq: Sequence):
        if self._relocated_affix_slot_offset is None or seq.affix_start == 0:
            seq.relocated_affix_materialized = False
            return
        self._configure_affix_relocation([seq])
        timing_start = self._start_timing()
        num_layers = self.kv_cache.shape[1]
        num_slots = self.kv_cache.shape[2] * self.kv_cache.shape[3]
        k_cache = self.kv_cache[0].view(num_layers, num_slots, *self.kv_cache.shape[-2:])
        v_cache = self.kv_cache[1].view_as(k_cache)
        attention = self._llada_blocks()[0]
        attention = attention.self_attn.attn if self._uses_decoder_layers() else attention.attn
        materialize_relocated_affix(
            k_cache,
            v_cache,
            attention.relocation_rope_cos,
            attention.relocation_rope_sin,
            self._relocated_affix_slot_offset,
            seq.affix_len,
            seq.affix_start,
        )
        shared_start = seq.shared_slot_offset
        shared_end = shared_start + seq.affix_len
        for local_pos, logical_pos in enumerate(range(seq.affix_start, seq.affix_end)):
            if shared_start <= seq.read_slot_map[logical_pos] < shared_end:
                seq.read_slot_map[logical_pos] = self._relocated_affix_slot_offset + local_pos
        seq.relocated_affix_materialized = True
        self._finish_timing(self._active_timing, "affix_materialize", timing_start)

    @contextmanager
    def _temporary_reference_attention(self):
        attention_modules = [module for module in self.model.modules() if hasattr(module, "use_reference_attention")]
        original_flags = [module.use_reference_attention for module in attention_modules]
        try:
            for module in attention_modules:
                module.use_reference_attention = True
            yield
        finally:
            for module, original_flag in zip(attention_modules, original_flags):
                module.use_reference_attention = original_flag

    @contextmanager
    def _temporary_dense_probe_attention(self):
        attention_modules = [
            module
            for module in self.model.modules()
            if hasattr(module, "num_heads") and hasattr(module, "num_kv_heads") and hasattr(module, "head_dim")
        ]
        original_forwards = [
            (module, "forward" in module.__dict__, module.__dict__.get("forward"))
            for module in attention_modules
        ]

        def make_dense_forward(original_forward):
            def dense_forward(module, q, k, v):
                context = get_context()
                scale = module.scale

                def expand_kv_heads(x):
                    if x.shape[-2] == module.num_heads:
                        return x
                    num_groups = module.num_heads // x.shape[-2]
                    if num_groups > 1:
                        return x.repeat_interleave(num_groups, dim=-2)
                    return x

                if q.dim() == 3:
                    cu_seqlens = context.cu_seqlens_q
                    if cu_seqlens is None:
                        raise RuntimeError("Dense probe attention requires cu_seqlens for ragged inputs.")
                    outputs = []
                    for seq_idx in range(cu_seqlens.numel() - 1):
                        start = int(cu_seqlens[seq_idx].item())
                        end = int(cu_seqlens[seq_idx + 1].item())
                        q_seq = q[start:end].transpose(0, 1).unsqueeze(0)
                        k_seq = expand_kv_heads(k[start:end].transpose(0, 1).unsqueeze(0))
                        v_seq = expand_kv_heads(v[start:end].transpose(0, 1).unsqueeze(0))
                        o_seq = torch.nn.functional.scaled_dot_product_attention(
                            q_seq,
                            k_seq,
                            v_seq,
                            dropout_p=0.0,
                            scale=scale,
                        )
                        outputs.append(o_seq.squeeze(0).transpose(0, 1))
                    return torch.cat(outputs, dim=0)

                if q.dim() == 4:
                    q = q.transpose(1, 2)
                    k = expand_kv_heads(k.transpose(1, 2))
                    v = expand_kv_heads(v.transpose(1, 2))
                    o = torch.nn.functional.scaled_dot_product_attention(
                        q,
                        k,
                        v,
                        dropout_p=0.0,
                        scale=scale,
                    )
                    return o.transpose(1, 2)

                return original_forward(q, k, v)

            return dense_forward

        try:
            for module, _, _ in original_forwards:
                original_forward = module.forward
                module.forward = types.MethodType(make_dense_forward(original_forward), module)
            yield
        finally:
            for module, had_instance_attr, original_forward in reversed(original_forwards):
                if had_instance_attr:
                    module.forward = original_forward
                else:
                    delattr(module, "forward")

    @contextmanager
    def _temporary_uncompiled_modules(self):
        patched_methods = []
        try:
            for module in self.model.modules():
                forward = getattr(module, "forward", None)
                if forward is not None and hasattr(forward, "__wrapped__"):
                    patched_methods.append((module, "forward", "forward" in module.__dict__, module.__dict__.get("forward")))
                    module.forward = types.MethodType(forward.__wrapped__, module)
                get_rotary_embedding = getattr(module, "get_rotary_embedding", None)
                if get_rotary_embedding is not None and hasattr(get_rotary_embedding, "__wrapped__"):
                    patched_methods.append((
                        module,
                        "get_rotary_embedding",
                        "get_rotary_embedding" in module.__dict__,
                        module.__dict__.get("get_rotary_embedding"),
                    ))
                    module.get_rotary_embedding = types.MethodType(get_rotary_embedding.__wrapped__, module)
            yield
        finally:
            for module, attr_name, had_instance_attr, original_method in reversed(patched_methods):
                if had_instance_attr:
                    setattr(module, attr_name, original_method)
                else:
                    delattr(module, attr_name)

    @contextmanager
    def _temporary_llada_logit_rows(self, row_indices: torch.Tensor):
        """Apply LLaDA's output head only to selected ragged-token rows."""
        llada_model = self.base_model.model
        if llada_model.config.weight_tying:
            raise NotImplementedError("Selective dKV logits require LLaDA's untied output head.")
        output_head = llada_model.transformer.ff_out
        original_forward = output_head.forward

        def selected_forward(module, hidden_states):
            return original_forward(hidden_states.index_select(0, row_indices))

        output_head.forward = types.MethodType(selected_forward, output_head)
        try:
            yield
        finally:
            output_head.forward = original_forward

    def _resolve_declared_prefix_tokens(self, seqs: list[Sequence]) -> tuple[int, ...]:
        if not seqs:
            return ()
        affix_len = seqs[0].prompt_affix_len
        if affix_len <= 0:
            return ()
        affix_token_ids = tuple(seqs[0].prompt_affix_token_ids)
        for seq in seqs[1:]:
            if seq.prompt_affix_len != affix_len:
                raise ValueError(
                    "ACache requires the same explicit affix length for every sequence in the batch."
                )
            if tuple(seq.prompt_affix_token_ids) != affix_token_ids:
                raise ValueError(
                    "ACache requires identical explicit affix tokens for every sequence in the batch."
                )
        return affix_token_ids

    def _precompute_shared_prefix(self, prefix_tokens: list[int]):
        prefix_len = len(prefix_tokens)
        if prefix_len == 0:
            return
        prefix_token_ids = tuple(prefix_tokens)
        if self.acache_ready and self.shared_prefix_token_ids == prefix_token_ids:
            return
        cache_block_size = self.config.cache_block_size
        shared_blocks = (prefix_len + cache_block_size - 1) // cache_block_size
        self.block_manager.reserve_prefix(shared_blocks)
        self._record_kv_cache_usage(self._active_timing)

        timing = self._active_timing
        prepare_start = self._start_timing()
        input_ids = torch.tensor(prefix_tokens, dtype=torch.int64, pin_memory=True).to(self.device, non_blocking=True)
        positions = torch.arange(prefix_len, dtype=torch.int64, device=self.device)
        cu_seqlens = torch.tensor([0, prefix_len], dtype=torch.int32, pin_memory=True).to(self.device, non_blocking=True)
        slot_mapping = torch.arange(prefix_len, dtype=torch.int32, device=self.device)
        self._finish_timing(timing, "shared_prefix_prepare", prepare_start)
        retry_oom = None
        set_context(True, cu_seqlens, cu_seqlens, None, prefix_len, prefix_len, slot_mapping, None)
        try:
            with self._temporary_reference_attention():
                forward_start = self._start_timing()
                self._forward_model(
                    input_ids=input_ids,
                    position_ids=positions,
                    compute_logits=False,
                )
                self._finish_timing(timing, "shared_prefix_forward", forward_start)
                self._add_timing_count(timing, "shared_prefix_forward_calls", 1)
        except torch.OutOfMemoryError as oom:
            retry_oom = torch.OutOfMemoryError(str(oom))
        finally:
            reset_context()
            if retry_oom is not None:
                input_ids = None
                positions = None
                cu_seqlens = None
                slot_mapping = None
                gc.collect()
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
        if retry_oom is not None:
            raise retry_oom

        self.shared_prefix_len = prefix_len
        self.shared_prefix_blocks = shared_blocks
        self.shared_prefix_token_ids = prefix_token_ids
        self.acache_ready = True

    def _compute_anchor_positions(self, seq: Sequence):
        return self._compute_anchor_positions_batch([seq])[0]

    def _finalize_anchor_selection_probe_layout(self, seq: Sequence, shared_slot_offset: int = 0):
        # Reserve anchor slots, but probe only dynamic request tokens against shared affix KV.
        private_slots = []
        for i in range(seq.num_cache_blocks):
            start = seq.block_table[i] * seq.cache_block_size
            if i != seq.num_cache_blocks - 1:
                end = start + seq.cache_block_size
            else:
                end = start + seq.last_cache_block_num_tokens
            private_slots.extend(range(start, end))
        assert len(private_slots) == seq.num_private_slots

        dynamic_slots = private_slots[seq.num_anchor_tokens:]
        assert len(dynamic_slots) == seq.num_tokens - seq.affix_len
        read_slot_map = [-1] * seq.num_tokens
        for local_pos, pos in enumerate(range(seq.affix_start, seq.affix_end)):
            read_slot_map[pos] = shared_slot_offset + local_pos
        for pos, slot in zip(seq.non_affix_positions, dynamic_slots):
            read_slot_map[pos] = slot

        seq.shared_slot_offset = shared_slot_offset
        seq.recompute_positions = seq.non_affix_positions
        seq.recompute_slot_mapping = dynamic_slots
        seq.read_slot_map = read_slot_map

    def _minimum_kv_cache_blocks_for_acache(self, seqs: list[Sequence], prefix_len: int):
        shared_blocks = (prefix_len + self.config.cache_block_size - 1) // self.config.cache_block_size
        if not seqs:
            return max(shared_blocks, 1)
        max_private_blocks = max(seq.num_cache_blocks for seq in seqs)
        # Keep enough KV blocks for the shared prefix plus at least one full sequence.
        return max(shared_blocks + max_private_blocks, 1)

    def _selection_kv_cache_blocks(self, seqs: list[Sequence], prefix_len: int):
        shared_blocks = (prefix_len + self.config.cache_block_size - 1) // self.config.cache_block_size
        if not seqs:
            return max(shared_blocks, 1)

        # Anchor selection remains per request to preserve the reference ACache procedure.
        selection_batch_size = 1
        max_private_blocks = 0
        for i in range(0, len(seqs), selection_batch_size):
            batch_private_blocks = sum(seq.num_cache_blocks for seq in seqs[i:i + selection_batch_size])
            max_private_blocks = max(max_private_blocks, batch_private_blocks)
        return max(shared_blocks + max_private_blocks, 1)

    def _shrink_kv_cache_for_acache(
        self,
        seqs: list[Sequence],
        prefix_len: int,
        minimum_blocks: int | None = None,
    ):
        if self.device.type != "cuda" or self.kv_cache_block_bytes <= 0:
            return False
        current_blocks = int(self.config.num_kvcache_blocks)
        if minimum_blocks is None:
            minimum_blocks = self._minimum_kv_cache_blocks_for_acache(seqs, prefix_len)
        if current_blocks <= minimum_blocks:
            return False

        step_blocks = min(4, current_blocks - minimum_blocks)
        new_blocks = max(minimum_blocks, current_blocks - step_blocks)
        if new_blocks >= current_blocks:
            return False

        print(
            f"[acache] anchor selection OOM, shrinking KV cache blocks from "
            f"{current_blocks} to {new_blocks} and retrying."
        )
        self.allocate_kv_cache(num_blocks=new_blocks)
        return True

    def _ensure_shared_prefix(
        self,
        prefix_token_ids: tuple[int, ...],
        seqs: list[Sequence],
        minimum_blocks: int | None = None,
    ):
        prefix_len = len(prefix_token_ids)
        while True:
            retry_oom = None
            try:
                self._precompute_shared_prefix(list(prefix_token_ids))
                return
            except torch.OutOfMemoryError as oom:
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                retry_oom = torch.OutOfMemoryError(str(oom))
            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            if not self._shrink_kv_cache_for_acache(seqs, prefix_len, minimum_blocks=minimum_blocks):
                raise retry_oom

    def _compute_anchor_positions_batch(self, seqs: list[Sequence]):
        anchor_positions_by_seq = [[] for _ in seqs]
        active_entries = [
            (seq_idx, seq)
            for seq_idx, seq in enumerate(seqs)
            if seq.acache_enabled and seq.num_anchor_tokens > 0
        ]
        if not active_entries:
            return anchor_positions_by_seq

        active_indices = [seq_idx for seq_idx, _ in active_entries]
        active_seqs = [seq for _, seq in active_entries]
        affix_len = active_seqs[0].affix_len
        if affix_len <= 0:
            return anchor_positions_by_seq
        for seq in active_seqs[1:]:
            if seq.affix_len != affix_len:
                raise ValueError("Batched Anchor Selection requires a consistent explicit prefix affix length.")

        timing = self._active_timing
        prepare_start = self._start_timing()
        saved_layouts = []
        temporarily_allocated = []
        original_forwards = []
        selection_relocation_workspace = False
        retry_oom = None
        input_ids = None
        position_ids = None
        mask_query_mask = None
        mask_query_positions_by_seq = None
        importance_sum = torch.zeros(len(active_seqs), affix_len, device=self.device, dtype=torch.float32)

        try:
            for seq in active_seqs:
                saved_layouts.append((
                    seq,
                    list(seq.block_table),
                    list(seq.anchor_positions),
                    list(seq.recompute_positions),
                    list(seq.recompute_slot_mapping),
                    list(seq.read_slot_map),
                    seq.shared_slot_offset,
                    seq.relocated_affix_materialized,
                ))
                if not seq.block_table:
                    if len(active_seqs) == 1:
                        selection_relocation_workspace = (
                            self._reserve_single_sequence_relocation_workspace(
                                seq, force=True
                            )
                        )
                    if not self.block_manager.can_allocate(seq):
                        raise torch.OutOfMemoryError("Not enough KV cache blocks for anchor-selection probe.")
                    self.block_manager.allocate(seq)
                    self._record_kv_cache_usage(timing)
                    temporarily_allocated.append(seq)
                seq.anchor_positions = []
                self._finalize_anchor_selection_probe_layout(seq)
                if selection_relocation_workspace:
                    self._materialize_single_sequence_affix(seq)

            input_ids, position_ids, _ = self.prepare_caching_acache(
                active_seqs,
                use_query_relocation=False,
            )
            context = get_context()
            seq_spans = []
            for seq_idx in range(len(active_seqs)):
                start = int(context.cu_seqlens_q[seq_idx].item())
                end = int(context.cu_seqlens_q[seq_idx + 1].item())
                seq_spans.append((start, end))
            mask_query_mask = (input_ids == self.config.mask_id)
            mask_query_positions_by_seq = [
                torch.where(mask_query_mask[start:end])[0]
                for start, end in seq_spans
            ]
            probe_max_pos = max(len(seq) for seq in active_seqs)
            relocation_positions_by_seq = [
                torch.full(
                    (seq.affix_len,),
                    seq.affix_start,
                    device=self.device,
                    dtype=torch.long,
                )
                for seq in active_seqs
            ]
            self._add_timing_count(timing, "anchor_selection_query_tokens", int(input_ids.numel()))
            self._add_timing_count(timing, "anchor_selection_kv_tokens", sum(len(seq) for seq in active_seqs))
            self._finish_timing(timing, "anchor_selection_prepare", prepare_start)

            def make_patched_attention_forward(original_forward, layer_idx, block):
                def patched_forward(attention, q, k, v):
                    q_heads = q.reshape(-1, attention.num_heads, attention.head_dim).permute(1, 0, 2)
                    cached_affix_k = self._shared_affix_k(layer_idx)
                    for batch_idx, (start, end) in enumerate(seq_spans):
                        mask_query_positions = mask_query_positions_by_seq[batch_idx]
                        if mask_query_positions.numel() == 0:
                            continue
                        seq = active_seqs[batch_idx]
                        relocation_positions = relocation_positions_by_seq[batch_idx]
                        if self._uses_decoder_layers():
                            rotary = block.self_attn.rotary_emb
                            affix_cos, affix_sin = rotary(cached_affix_k, relocation_positions)
                            head_dim = attention.head_dim
                            rotated_half = torch.cat(
                                (-cached_affix_k[..., head_dim // 2:], cached_affix_k[..., :head_dim // 2]),
                                dim=-1,
                            )
                            affix_k = (
                                cached_affix_k * affix_cos.unsqueeze(-2)
                                + rotated_half * affix_sin.unsqueeze(-2)
                            )
                        elif block.config.rope:
                            _, affix_k = block.rotary_emb(
                                cached_affix_k,
                                cached_affix_k,
                                position_ids=relocation_positions,
                                max_pos=max(probe_max_pos, seq.affix_start + 1),
                            )
                        else:
                            affix_k = cached_affix_k
                        affix_k = affix_k.permute(1, 0, 2)
                        num_groups = attention.num_heads // affix_k.shape[0]
                        if num_groups > 1:
                            affix_k = affix_k.repeat_interleave(num_groups, dim=0)
                        seq_q = q_heads[:, start:end, :][:, mask_query_positions, :]
                        scale = attention.scale if attention.scale is not None else attention.head_dim ** -0.5
                        attn_scores = torch.matmul(seq_q, affix_k.transpose(-2, -1)) * scale
                        attn_weights = torch.softmax(attn_scores, dim=-1)
                        token_importance = attn_weights.sum(dim=-2).sum(dim=-2)
                        importance_sum[batch_idx].add_(token_importance.float())
                    return original_forward(q, k, v)

                return patched_forward

            for layer_idx, block in enumerate(self._llada_blocks()):
                attention = block.self_attn.attn if self._uses_decoder_layers() else block.attn
                original_forward = attention.forward
                original_forwards.append((attention, original_forward))
                attention.forward = types.MethodType(
                    make_patched_attention_forward(original_forward, layer_idx, block),
                    attention,
                )
            forward_start = self._start_timing()
            with self._temporary_uncompiled_modules():
                self._forward_model(
                    input_ids=input_ids,
                    position_ids=position_ids,
                    compute_logits=False,
                )
            self._finish_timing(timing, "anchor_selection_forward", forward_start)
            self._add_timing_count(timing, "anchor_selection_forward_calls", 1)
            self._add_timing_count(timing, "anchor_selection_sequences", len(active_seqs))
        except torch.OutOfMemoryError as oom:
            retry_oom = torch.OutOfMemoryError(str(oom))
        finally:
            for block, original_forward in original_forwards:
                block.forward = original_forward
            reset_context()
            for seq in reversed(temporarily_allocated):
                if seq.block_table:
                    self.block_manager.deallocate(seq)
            for (
                seq,
                block_table,
                anchor_positions,
                recompute_positions,
                recompute_slot_mapping,
                read_slot_map,
                shared_slot_offset,
                relocated_affix_materialized,
            ) in saved_layouts:
                seq.block_table = block_table
                seq.anchor_positions = anchor_positions
                seq.recompute_positions = recompute_positions
                seq.recompute_slot_mapping = recompute_slot_mapping
                seq.read_slot_map = read_slot_map
                seq.shared_slot_offset = shared_slot_offset
                seq.relocated_affix_materialized = relocated_affix_materialized
            if selection_relocation_workspace:
                self.block_manager.release_prefix_tail(self.shared_prefix_blocks)
                self._relocated_affix_slot_offset = None
            if retry_oom is not None:
                input_ids = None
                position_ids = None
                mask_query_mask = None
                mask_query_positions_by_seq = None
                gc.collect()
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
        if retry_oom is not None:
            raise retry_oom

        topk_start = self._start_timing()
        for batch_idx, seq in enumerate(active_seqs):
            k = seq.num_anchor_tokens
            if self.config.selection_mode == 'bottom':
                _, anchor_indices = torch.topk(importance_sum[batch_idx], k, largest=False)
            else:
                _, anchor_indices = torch.topk(importance_sum[batch_idx], k, largest=True)
            anchor_positions_by_seq[active_indices[batch_idx]] = (
                anchor_indices.sort().values.add(seq.affix_start).tolist()
            )
        self._finish_timing(timing, "anchor_selection_topk", topk_start)

        return anchor_positions_by_seq

    def _prepare_acache(self, seqs: list[Sequence]):
        prefix_token_ids = self._resolve_declared_prefix_tokens(seqs)
        if not prefix_token_ids:
            return ()

        prefix_len = len(prefix_token_ids)
        for seq in seqs:
            seq.enable_acache(
                prefix_len,
                self.config.anchor_ratio,
                affix_start=seq.prompt_affix_start,
                affix_end=seq.prompt_affix_end,
            )
        return prefix_token_ids

    def _select_anchor_positions_for_sequences(
        self,
        seqs: list[Sequence],
        prefix_token_ids: tuple[int, ...],
        allow_kv_cache_resize: bool,
        minimum_blocks: int | None = None,
    ):
        if not seqs:
            return

        prefix_len = len(prefix_token_ids)
        default_batch_size = 1
        i = 0
        while i < len(seqs):
            remaining = len(seqs) - i
            batch_size = min(default_batch_size, remaining)
            while True:
                retry_oom = None
                try:
                    batch = seqs[i:i + batch_size]
                    batch_anchor_positions = self._compute_anchor_positions_batch(batch)
                    for seq, anchor_positions in zip(batch, batch_anchor_positions):
                        seq.set_anchor_positions(anchor_positions)
                    i += batch_size
                    break
                except torch.OutOfMemoryError as oom:
                    if self.device.type == "cuda":
                        torch.cuda.empty_cache()
                    if batch_size > 1:
                        batch_size = max(batch_size // 2, 1)
                        continue
                    retry_oom = torch.OutOfMemoryError(str(oom))
                gc.collect()
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                if not allow_kv_cache_resize or not self._shrink_kv_cache_for_acache(
                    seqs,
                    prefix_len,
                    minimum_blocks=minimum_blocks,
                ):
                    raise retry_oom
                self._ensure_shared_prefix(
                    prefix_token_ids,
                    seqs,
                    minimum_blocks=minimum_blocks,
                )

    def _prepare_admitted_sequences_for_acache(
        self,
        seqs: list[Sequence],
        prefix_token_ids: tuple[int, ...],
        allow_kv_cache_resize: bool,
    ):
        if not seqs:
            return

        if self.acache_ready and self.shared_prefix_token_ids != prefix_token_ids:
            raise ValueError("ACache prefix mode requires a consistent shared prefix across active requests.")

        prefix_len = len(prefix_token_ids)
        target_num_kvcache_blocks = int(self.config.num_kvcache_blocks)
        selection_minimum_blocks = min(
            target_num_kvcache_blocks,
            self._selection_kv_cache_blocks(seqs, prefix_len),
        )

        try:
            if self.acache_ready:
                if self.shared_prefix_token_ids != prefix_token_ids:
                    raise ValueError("ACache prefix mode requires a consistent shared prefix across active requests.")
            else:
                if allow_kv_cache_resize:
                    self._ensure_shared_prefix(
                        prefix_token_ids,
                        seqs,
                        minimum_blocks=selection_minimum_blocks,
                    )
                else:
                    self._precompute_shared_prefix(list(prefix_token_ids))

            self._select_anchor_positions_for_sequences(
                seqs,
                prefix_token_ids,
                allow_kv_cache_resize=allow_kv_cache_resize,
                minimum_blocks=selection_minimum_blocks,
            )
        except Exception:
            if allow_kv_cache_resize and self.config.num_kvcache_blocks != target_num_kvcache_blocks:
                self.allocate_kv_cache(num_blocks=target_num_kvcache_blocks)
            raise
        if allow_kv_cache_resize and self.config.num_kvcache_blocks != target_num_kvcache_blocks:
            self.allocate_kv_cache(num_blocks=target_num_kvcache_blocks)
            self._ensure_shared_prefix(prefix_token_ids, seqs)

    def _finalize_sequence_layout(self, seq: Sequence):
        seq.finalize_slot_mapping()
        self._materialize_single_sequence_affix(seq)

    def _collect_admissible_sequences(
        self,
        prompts: list[Sequence],
        next_prompt_idx: int,
        current_batch_size: int,
        max_num_seqs: int,
    ):
        admissible = []
        remaining_blocks = len(self.block_manager.free_block_ids)
        while next_prompt_idx < len(prompts) and current_batch_size + len(admissible) < max_num_seqs:
            seq = prompts[next_prompt_idx]
            if seq.num_cache_blocks > remaining_blocks:
                break
            admissible.append(seq)
            remaining_blocks -= seq.num_cache_blocks
            next_prompt_idx += 1
        return admissible, next_prompt_idx

    def _admit_sequences_with_acache(
        self,
        batched_prompts: list[Sequence],
        new_prompts: list[Sequence],
        prefix_token_ids: tuple[int, ...],
        allow_kv_cache_resize: bool,
    ):
        if not new_prompts:
            return 0
        self._prepare_admitted_sequences_for_acache(
            new_prompts,
            prefix_token_ids,
            allow_kv_cache_resize=allow_kv_cache_resize,
        )
        if new_prompts:
            self._reserve_single_sequence_relocation_workspace(new_prompts[0])
        admitted = 0
        for seq in new_prompts:
            if not self.block_manager.can_allocate(seq):
                break
            self.block_manager.allocate(seq)
            self._record_kv_cache_usage(self._active_timing)
            self._finalize_sequence_layout(seq)
            batched_prompts.append(seq)
            admitted += 1
        return admitted
    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_caching(self, seqs: list[Sequence], active_indices: list[int] | None = None):
        device = self.device
        cache_block_size = self.config.cache_block_size

        # If active_indices is provided, only process those sequences
        if active_indices is not None:
            processing_seqs = [seqs[i] for i in active_indices]
        else:
            processing_seqs = seqs

        block_tables = self.prepare_block_tables(processing_seqs)

        # Pre-calculate sizes
        num_seqs = len(processing_seqs)
        total_tokens = sum(len(seq) for seq in processing_seqs)

        # Pre-allocate lists/arrays
        input_ids = [0] * total_tokens
        positions = [0] * total_tokens
        cu_seqlens = [0] * (num_seqs + 1)
        slot_mapping = []
        generate_starts = [0] * num_seqs
        current_block_indices = [0] * num_seqs

        # Fill pre-allocated arrays
        input_idx = 0
        max_seqlen = 0

        for seq_idx, seq in enumerate(processing_seqs):
            seqlen = len(seq)

            input_ids[input_idx:input_idx + seqlen] = seq.token_ids
            positions[input_idx:input_idx + seqlen] = list(range(seqlen))
            generate_starts[seq_idx] = cu_seqlens[seq_idx] + seq.generation_start
            current_block_indices[seq_idx] = seq.current_block_idx
            cu_seqlens[seq_idx + 1] = cu_seqlens[seq_idx] + seqlen
            max_seqlen = max(seqlen, max_seqlen)

            for i in range(0, seq.num_cache_blocks):
                start = seq.block_table[i] * cache_block_size
                if i != seq.num_cache_blocks - 1:
                    end = start + cache_block_size
                else:
                    end = start + seq.last_cache_block_num_tokens
                slot_mapping.extend(list(range(start, end)))

            input_idx += seqlen

        # Convert to tensors
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).to(device, non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).to(device, non_blocking=True)
        cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32, pin_memory=True).to(device, non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).to(device, non_blocking=True)
        generate_starts = torch.tensor(generate_starts, dtype=torch.int32, pin_memory=True).to(device, non_blocking=True)
        current_block_indices = torch.tensor(current_block_indices, dtype=torch.int32, pin_memory=True).to(device, non_blocking=True)

        set_context(True, cu_seqlens, cu_seqlens, None, max_seqlen, max_seqlen, slot_mapping, block_tables)

        return input_ids, positions, generate_starts, current_block_indices

    def prepare_decoding(self, seqs: list[Sequence], active_indices: list[int] | None = None):
        device = self.device
        block_length = self.config.block_length
        cache_block_size = self.config.cache_block_size

        if active_indices is not None:
            processing_seqs = [seqs[i] for i in active_indices]
        else:
            processing_seqs = seqs

        num_seqs = len(processing_seqs)

        start_positions = [0] * num_seqs
        seqlens = [0] * num_seqs
        for i, seq in enumerate(processing_seqs):
            start_positions[i] = seq.generation_start + seq.current_block_idx * block_length
            seqlens[i] = len(seq)

        start_positions = torch.tensor(start_positions, dtype=torch.int64, pin_memory=True).to(device, non_blocking=True)
        seqlens = torch.tensor(seqlens, dtype=torch.int32, pin_memory=True).to(device, non_blocking=True)

        # Prepare block tables first
        block_tables = self.prepare_block_tables(processing_seqs)

        # Generate all positions at once
        positions = start_positions.unsqueeze(1) + torch.arange(block_length, device=device)

        # Vectorized slot mapping computation
        # Flatten positions for easier computation
        flat_positions = positions.flatten()
        cache_block_indices = flat_positions // cache_block_size
        positions_in_blocks = flat_positions % cache_block_size

        # Expand block_tables for gathering
        seq_indices = torch.arange(num_seqs, device=device).repeat_interleave(block_length)

        # Gather the correct block IDs from block_tables
        block_ids = block_tables[seq_indices, cache_block_indices]

        # Compute final slot mapping
        slot_mapping = block_ids * cache_block_size + positions_in_blocks

        set_context(False, None, None, seqlens, None, None, slot_mapping, block_tables)

        return positions

    def prepare_dkv_caching(
        self,
        seqs: list[Sequence],
        query_positions: list[list[int]],
    ):
        """Prepare ragged dKV queries while reading each request's full KV cache."""
        if len(seqs) != len(query_positions):
            raise ValueError("Each dKV sequence must have one query-position list.")

        device = self.device
        cache_block_size = self.config.cache_block_size
        block_tables = self.prepare_block_tables(seqs)
        total_queries = sum(len(positions) for positions in query_positions)
        if total_queries == 0:
            raise ValueError("dKV-Cache requires at least one query token per forward pass.")

        input_ids = [0] * total_queries
        positions = [0] * total_queries
        slot_mapping = [0] * total_queries
        cu_seqlens_q = [0] * (len(seqs) + 1)
        cu_seqlens_k = [0] * (len(seqs) + 1)
        query_offsets = [0] * len(seqs)
        max_q = 0
        max_k = 0
        offset = 0

        for row, (seq, row_positions) in enumerate(zip(seqs, query_positions)):
            if not row_positions:
                raise ValueError("Every active dKV request must contain a query token.")
            if row_positions != sorted(set(row_positions)):
                raise ValueError("dKV query positions must be sorted and unique.")
            if row_positions[0] < 0 or row_positions[-1] >= len(seq):
                raise ValueError("dKV query position lies outside the logical sequence.")

            query_offsets[row] = offset
            row_len = len(row_positions)
            input_ids[offset:offset + row_len] = [seq.token_ids[pos] for pos in row_positions]
            positions[offset:offset + row_len] = row_positions
            slot_mapping[offset:offset + row_len] = [
                seq.block_table[pos // cache_block_size] * cache_block_size
                + pos % cache_block_size
                for pos in row_positions
            ]
            offset += row_len
            cu_seqlens_q[row + 1] = offset
            cu_seqlens_k[row + 1] = cu_seqlens_k[row] + len(seq)
            max_q = max(max_q, row_len)
            max_k = max(max_k, len(seq))

        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).to(device, non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).to(device, non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).to(device, non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).to(device, non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).to(device, non_blocking=True)

        set_context(
            True,
            cu_seqlens_q,
            cu_seqlens_k,
            None,
            max_q,
            max_k,
            slot_mapping,
            block_tables,
        )
        return input_ids, positions, query_offsets

    def prepare_dkv_caching_acache(
        self,
        seqs: list[Sequence],
        query_positions: list[list[int]],
    ):
        """Prepare dKV queries against an ACache logical read-slot layout."""
        if len(seqs) != len(query_positions):
            raise ValueError("Each dKV sequence must have one query-position list.")

        self._configure_affix_relocation(seqs)
        device = self.device
        total_queries = sum(len(positions) for positions in query_positions)
        if total_queries == 0:
            raise ValueError("dKV-Cache requires at least one query token per forward pass.")

        input_ids = [0] * total_queries
        positions = [0] * total_queries
        slot_mapping = [0] * total_queries
        read_slot_mapping = []
        cu_seqlens_q = [0] * (len(seqs) + 1)
        cu_seqlens_k = [0] * (len(seqs) + 1)
        query_offsets = [0] * len(seqs)
        shared_kv_lens = [0] * len(seqs)
        affix_shifts = [0] * len(seqs)
        relocation_materialized = all(
            getattr(seq, "relocated_affix_materialized", False)
            for seq in seqs
        )
        use_query_relocation = self._use_query_side_affix_relocation(
            relocation_materialized
        )
        max_q = 0
        max_k = 0
        offset = 0

        for row, (seq, row_positions) in enumerate(zip(seqs, query_positions)):
            if not row_positions:
                raise ValueError("Every active dKV request must contain a query token.")
            if row_positions != sorted(set(row_positions)):
                raise ValueError("dKV query positions must be sorted and unique.")
            if row_positions[0] < 0 or row_positions[-1] >= len(seq):
                raise ValueError("dKV query position lies outside the logical sequence.")

            slot_by_position = dict(zip(seq.recompute_positions, seq.recompute_slot_mapping))
            try:
                # Dream may need a frozen predecessor's hidden state to predict
                # its masked successor. A -1 write slot keeps its cached KV intact.
                row_slots = [
                    slot_by_position.get(pos, -1) if self._is_dream_model()
                    else slot_by_position[pos] for pos in row_positions
                ]
            except KeyError as exc:
                raise ValueError("dKV+ACache cannot query a frozen non-anchor affix token.") from exc

            query_offsets[row] = offset
            row_len = len(row_positions)
            input_ids[offset:offset + row_len] = [seq.token_ids[pos] for pos in row_positions]
            positions[offset:offset + row_len] = row_positions
            slot_mapping[offset:offset + row_len] = row_slots
            attention_slots, shared_kv_len = self._attention_read_layout(seq, use_query_relocation)
            read_slot_mapping.extend(attention_slots)
            shared_kv_lens[row] = shared_kv_len
            affix_shifts[row] = seq.affix_start if self._affix_relocation_enabled else 0
            offset += row_len
            cu_seqlens_q[row + 1] = offset
            cu_seqlens_k[row + 1] = cu_seqlens_k[row] + len(seq)
            max_q = max(max_q, row_len)
            max_k = max(max_k, len(seq))

        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).to(device, non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).to(device, non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).to(device, non_blocking=True)
        read_slot_mapping = torch.tensor(
            read_slot_mapping, dtype=torch.int32, pin_memory=True
        ).to(device, non_blocking=True)
        cu_seqlens_q = torch.tensor(
            cu_seqlens_q, dtype=torch.int32, pin_memory=True
        ).to(device, non_blocking=True)
        cu_seqlens_k = torch.tensor(
            cu_seqlens_k, dtype=torch.int32, pin_memory=True
        ).to(device, non_blocking=True)
        shared_kv_lens = (
            torch.tensor(shared_kv_lens, dtype=torch.int32, pin_memory=True).to(
                device, non_blocking=True
            )
            if use_query_relocation
            else None
        )
        affix_shifts = (
            None
            if relocation_materialized
            else torch.tensor(affix_shifts, dtype=torch.int32, pin_memory=True).to(
                device, non_blocking=True
            )
        )

        set_context(
            True,
            cu_seqlens_q,
            cu_seqlens_k,
            None,
            max_q,
            max_k,
            slot_mapping,
            None,
            True,
            read_slot_mapping,
            shared_kv_lens,
            affix_shifts,
        )
        return input_ids, positions, query_offsets

    def prepare_caching_acache(
        self,
        seqs: list[Sequence],
        active_indices: list[int] | None = None,
        use_query_relocation: bool | None = None,
    ):
        device = self.device

        if active_indices is not None:
            processing_seqs = [seqs[i] for i in active_indices]
        else:
            processing_seqs = seqs

        self._configure_affix_relocation(processing_seqs)

        num_seqs = len(processing_seqs)
        total_queries = sum(len(seq.recompute_positions) for seq in processing_seqs)
        total_kv = sum(len(seq) for seq in processing_seqs)

        input_ids = [0] * total_queries
        positions = [0] * total_queries
        cu_seqlens_q = [0] * (num_seqs + 1)
        cu_seqlens_k = [0] * (num_seqs + 1)
        slot_mapping = []
        read_slot_mapping = []
        shared_kv_lens = [0] * num_seqs
        affix_shifts = [0] * num_seqs
        relocation_materialized = all(
            getattr(seq, "relocated_affix_materialized", False)
            for seq in processing_seqs
        )
        if use_query_relocation is None:
            use_query_relocation = self.config.max_num_seqs != 1
        use_query_relocation = (
            self._affix_relocation_enabled
            and not relocation_materialized
            and bool(use_query_relocation)
        )
        block_query_starts = [0] * num_seqs
        kv_seqlens = [0] * num_seqs

        input_idx = 0
        max_q = 0
        max_k = 0
        for seq_idx, seq in enumerate(processing_seqs):
            q_positions = seq.recompute_positions
            q_len = len(q_positions)
            k_len = len(seq)
            assert len(seq.recompute_slot_mapping) == q_len
            assert len(seq.read_slot_map) == k_len

            input_ids[input_idx:input_idx + q_len] = [seq.token_ids[pos] for pos in q_positions]
            positions[input_idx:input_idx + q_len] = q_positions
            slot_mapping.extend(seq.recompute_slot_mapping)
            attention_slots, shared_kv_len = self._attention_read_layout(seq, use_query_relocation)
            read_slot_mapping.extend(attention_slots)
            shared_kv_lens[seq_idx] = shared_kv_len
            affix_shifts[seq_idx] = seq.affix_start if self._affix_relocation_enabled else 0
            cu_seqlens_q[seq_idx + 1] = cu_seqlens_q[seq_idx] + q_len
            cu_seqlens_k[seq_idx + 1] = cu_seqlens_k[seq_idx] + k_len
            block_query_starts[seq_idx] = cu_seqlens_q[seq_idx] + seq.block_query_start()
            kv_seqlens[seq_idx] = k_len
            max_q = max(max_q, q_len)
            max_k = max(max_k, k_len)
            input_idx += q_len

        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).to(device, non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).to(device, non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).to(device, non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).to(device, non_blocking=True)
        kv_seqlens = torch.tensor(kv_seqlens, dtype=torch.int32, pin_memory=True).to(device, non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).to(device, non_blocking=True)
        read_slot_mapping = torch.tensor(read_slot_mapping, dtype=torch.int32, pin_memory=True).to(device, non_blocking=True)
        shared_kv_lens = (
            torch.tensor(shared_kv_lens, dtype=torch.int32, pin_memory=True).to(device, non_blocking=True)
            if use_query_relocation
            else None
        )
        affix_shifts = (
            None
            if relocation_materialized
            else torch.tensor(affix_shifts, dtype=torch.int32, pin_memory=True).to(device, non_blocking=True)
        )
        block_query_starts = torch.tensor(block_query_starts, dtype=torch.int32, pin_memory=True).to(device, non_blocking=True)

        set_context(
            True,
            cu_seqlens_q,
            cu_seqlens_k,
            kv_seqlens,
            max_q,
            max_k,
            slot_mapping,
            None,
            True,
            read_slot_mapping,
            shared_kv_lens,
            affix_shifts,
        )

        return input_ids, positions, block_query_starts

    def prepare_decoding_acache(self, seqs: list[Sequence], active_indices: list[int] | None = None):
        device = self.device
        block_length = self.config.block_length

        if active_indices is not None:
            processing_seqs = [seqs[i] for i in active_indices]
        else:
            processing_seqs = seqs

        self._configure_affix_relocation(processing_seqs)

        num_seqs = len(processing_seqs)
        start_positions = [0] * num_seqs
        kv_seqlens = [0] * num_seqs
        cu_seqlens_k = [0] * (num_seqs + 1)
        slot_mapping = []
        read_slot_mapping = []
        shared_kv_lens = [0] * num_seqs
        affix_shifts = [0] * num_seqs
        relocation_materialized = all(
            getattr(seq, "relocated_affix_materialized", False)
            for seq in processing_seqs
        )
        use_query_relocation = self._use_query_side_affix_relocation(
            relocation_materialized
        )
        max_k = 0

        for i, seq in enumerate(processing_seqs):
            start = seq.generation_start + seq.current_block_idx * block_length
            end = start + block_length
            start_positions[i] = start
            kv_seqlens[i] = len(seq)
            cu_seqlens_k[i + 1] = cu_seqlens_k[i] + len(seq)
            slot_mapping.extend(seq.read_slot_map[start:end])
            attention_slots, shared_kv_len = self._attention_read_layout(seq, use_query_relocation)
            read_slot_mapping.extend(attention_slots)
            shared_kv_lens[i] = shared_kv_len
            affix_shifts[i] = seq.affix_start if self._affix_relocation_enabled else 0
            max_k = max(max_k, len(seq))

        start_positions = torch.tensor(start_positions, dtype=torch.int64, pin_memory=True).to(device, non_blocking=True)
        positions = start_positions.unsqueeze(1) + torch.arange(block_length, device=device)
        cu_seqlens_q = torch.arange(0, (num_seqs + 1) * block_length, block_length, dtype=torch.int32, device=device)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).to(device, non_blocking=True)
        kv_seqlens = torch.tensor(kv_seqlens, dtype=torch.int32, pin_memory=True).to(device, non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).to(device, non_blocking=True)
        read_slot_mapping = torch.tensor(read_slot_mapping, dtype=torch.int32, pin_memory=True).to(device, non_blocking=True)
        shared_kv_lens = (
            torch.tensor(shared_kv_lens, dtype=torch.int32, pin_memory=True).to(device, non_blocking=True)
            if use_query_relocation
            else None
        )
        affix_shifts = (
            None
            if relocation_materialized
            else torch.tensor(affix_shifts, dtype=torch.int32, pin_memory=True).to(device, non_blocking=True)
        )

        set_context(
            False,
            cu_seqlens_q,
            cu_seqlens_k,
            kv_seqlens,
            block_length,
            max_k,
            slot_mapping,
            None,
            True,
            read_slot_mapping,
            shared_kv_lens,
            affix_shifts,
        )

        return positions

    @torch.inference_mode()
    def generate_with_dual_cache(self, prompts: list[Sequence]):
        device = self.device
        model = self.model
        mask_id, block_length, temperature, remasking, threshold = \
            self.config.mask_id, self.config.block_length, self.config.temperature, \
            self.config.remasking, self.config.threshold
        recompute_batch_size = self.config.recompute_batch_size
        assert recompute_batch_size > 0, "recompute_batch_size must be larger than 0"
        max_num_seqs = self.config.max_num_seqs if self.config.max_num_seqs > 0 else len(prompts)

        vocab_size = self.config.hf_config.vocab_size
        timing = self._new_timing("dual_cache", len(prompts))
        self.last_timing = timing or {}
        previous_active_timing = self._active_timing
        self._active_timing = timing
        total_start = self._start_timing()

        total_nfe = 0
        all_generated_parts = [None] * len(prompts)
        result_index_by_seq_id = {seq.seq_id: i for i, seq in enumerate(prompts)}

        # initialize the first batch
        next_prompt_idx = 0
        batched_prompts = []
        scheduler_start = self._start_timing()
        while (
            next_prompt_idx < len(prompts)
            and len(batched_prompts) < max_num_seqs
            and self.block_manager.can_allocate(prompts[next_prompt_idx])
        ):
            self.block_manager.allocate(prompts[next_prompt_idx])
            self._record_kv_cache_usage(timing)
            batched_prompts.append(prompts[next_prompt_idx])
            next_prompt_idx += 1
        self._finish_timing(timing, "scheduler", scheduler_start)

        batch_size = len(batched_prompts)
        if batch_size == 0 and prompts:
            raise RuntimeError("No requests could be admitted under the current KV-cache budget.")
        working_blocks = torch.full((batch_size, block_length), mask_id, dtype=torch.int64, device=device)

        block_offset = torch.arange(block_length, device=device)
        with tqdm(total=len(prompts), desc="Processing requests", smoothing=0) as pbar:
            # Calculate actual batch size for initial processing
            for i in range(0, len(batched_prompts), recompute_batch_size):
                j = min(i + recompute_batch_size, batch_size)
                # Create active indices for the slice
                prepare_start = self._start_timing()
                input_ids, positions, generate_starts, _ = self.prepare_caching(batched_prompts[i:j])
                self._finish_timing(timing, "cache_prepare", prepare_start)
                keep_indices = generate_starts.unsqueeze(1) + block_offset
                if self._is_dream_model():
                    row_starts = generate_starts - torch.tensor(
                        [seq.generation_start for seq in batched_prompts[i:j]],
                        dtype=torch.int64,
                        device=device,
                    )
                    keep_indices = self._dream_shift_keep_indices(keep_indices, row_starts)
                keep_indices = keep_indices.flatten()
                forward_start = self._start_timing()
                if self._uses_decoder_layers():
                    output = model(input_ids, position_ids=positions, logits_keep_indices=keep_indices)
                else:
                    output = model(input_ids, position_ids=positions)
                self._finish_timing(timing, "cache_forward", forward_start)
                self._add_timing_count(timing, "cache_forward_calls", 1)
                self._add_timing_count(timing, "cache_forward_nfe", j - i)
                # get the logits for the working_blocks
                update_start = self._start_timing()
                if self._uses_decoder_layers():
                    working_logits = output.logits.view(-1, block_length, vocab_size)
                else:
                    working_logits = output.logits[keep_indices].view(-1, block_length, vocab_size)

                if self._is_dream_model():
                    x0 = self._sample_dream_seed_tokens(working_logits)
                    update_rows_global = torch.arange(i, j, dtype=torch.int64, device=device)
                    working_blocks[update_rows_global, 0] = x0[:, 0]
                    del x0, working_logits, output
                else:
                    mask_index = (working_blocks[i:j] == mask_id)
                    x0, ti = self.get_transfer_index(working_logits, temperature, remasking, mask_index, working_blocks[i:j], threshold)
                    del working_logits, output
                    update_rows_local, update_cols = torch.where(ti)
                    update_rows_global = torch.arange(i, j, dtype=torch.int64, device=device)[update_rows_local]
                    working_blocks[update_rows_global, update_cols] = x0[ti]
                self._finish_timing(timing, "cache_update", update_start)

            total_nfe += batch_size

            old_active_indices = None
            cache_recomputed, batch_adjusted = False, False
            while batched_prompts:
                # gather the tokens to check for remaining [MASK]s
                block_has_mask = (working_blocks == mask_id).any(dim=1)

                # A block is finished if it has no [MASK]s
                finished_block_mask = ~block_has_mask

                # Enter cache recomputation as long as at least one sequence needs update.
                if finished_block_mask.any():
                    scheduler_start = self._start_timing()
                    update_indices = torch.where(finished_block_mask)[0]
                    finished_indices = []
                    # Put the working_blocks back into seqs
                    for idx in update_indices:
                        seq = batched_prompts[idx]
                        start = seq.generation_start + (seq.current_block_idx * block_length)
                        end = start + block_length
                        seq.token_ids[start:end] = working_blocks[idx].tolist()
                        seq.current_block_idx += 1
                        if seq.current_block_idx >= seq.num_blocks_to_generate:
                            self.block_manager.deallocate(seq)

                            finished_indices.append(idx.item())
                            all_generated_parts[result_index_by_seq_id[seq.seq_id]] = seq.generated_token_ids

                            pbar.update(1)

                    # Reset the working blocks for the updated sequences
                    working_blocks[update_indices] = mask_id

                    if finished_indices:
                        batch_adjusted = True

                        finished_set = set(finished_indices)
                        # Rebuild the list excluding finished sequences
                        batched_prompts = [prompt for i, prompt in enumerate(batched_prompts) if i not in finished_set]

                        while (
                            next_prompt_idx < len(prompts)
                            and len(batched_prompts) < max_num_seqs
                            and self.block_manager.can_allocate(prompts[next_prompt_idx])
                        ):
                            batched_prompts.append(prompts[next_prompt_idx])
                            self.block_manager.allocate(prompts[next_prompt_idx])
                            self._record_kv_cache_usage(timing)
                            next_prompt_idx += 1

                        finished_indices = torch.tensor(finished_indices, dtype=torch.int64, pin_memory=True).to(device, non_blocking=True)
                        unfinished_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
                        unfinished_mask[finished_indices] = False
                        unfinished_indices = torch.where(unfinished_mask)[0]
                        new_batch_size = len(batched_prompts)
                        if new_batch_size != batch_size:
                            new_working_blocks = torch.full((new_batch_size, block_length), mask_id, dtype=torch.int64, device=device)
                            # Copy unfinished working blocks to new tensor
                            if len(unfinished_indices) > 0:
                                new_working_blocks[:len(unfinished_indices)] = working_blocks[unfinished_indices]
                            # Update working_blocks to point to the new tensor
                            working_blocks = new_working_blocks
                            batch_size = new_batch_size
                        else:
                            if len(unfinished_indices) > 0:
                                working_blocks[:len(unfinished_indices)] = working_blocks[unfinished_indices]
                            working_blocks[len(unfinished_indices):] = mask_id
                    self._finish_timing(timing, "scheduler", scheduler_start)

                    block_all_mask = (working_blocks == mask_id).all(dim=1)
                    update_mask = block_all_mask
                    active_indices = torch.where(update_mask)[0]
                    if len(active_indices) > 0:
                        cache_recomputed = True
                        for i in range(0, len(active_indices), recompute_batch_size):
                            j = min(i + recompute_batch_size, len(active_indices))
                            prepare_start = self._start_timing()
                            input_ids, positions, generate_starts, current_block_indices = \
                                self.prepare_caching(batched_prompts, active_indices[i:j])
                            self._finish_timing(timing, "cache_prepare", prepare_start)
                            keep_indices = (
                                generate_starts.unsqueeze(1)
                                + block_length * current_block_indices.unsqueeze(1)
                                + block_offset
                            )
                            selected_seqs = [batched_prompts[int(idx)] for idx in active_indices[i:j].tolist()]
                            if self._is_dream_model():
                                row_starts = generate_starts - torch.tensor(
                                    [seq.generation_start for seq in selected_seqs],
                                    dtype=torch.int64,
                                    device=device,
                                )
                                keep_indices = self._dream_shift_keep_indices(keep_indices, row_starts)
                            keep_indices = keep_indices.flatten()
                            forward_start = self._start_timing()
                            if self._uses_decoder_layers():
                                output = model(input_ids, position_ids=positions, logits_keep_indices=keep_indices)
                            else:
                                output = model(input_ids, position_ids=positions)
                            self._finish_timing(timing, "cache_forward", forward_start)
                            self._add_timing_count(timing, "cache_forward_calls", 1)
                            self._add_timing_count(timing, "cache_forward_nfe", j - i)
                            # get the logits for the working_blocks
                            update_start = self._start_timing()
                            if self._uses_decoder_layers():
                                working_logits = output.logits.view(-1, block_length, vocab_size)
                            else:
                                working_logits = output.logits[keep_indices].view(-1, block_length, vocab_size)

                            if self._is_dream_model():
                                x0 = self._sample_dream_seed_tokens(working_logits)
                                working_blocks[active_indices[i:j], 0] = x0[:, 0]
                                del x0, working_logits, output
                            else:
                                mask_index = (working_blocks[active_indices[i:j]] == mask_id)
                                x0, ti = self.get_transfer_index(working_logits, temperature, remasking, mask_index,
                                                            working_blocks[active_indices[i:j]], threshold)
                                del working_logits, output
                                update_rows_local, update_cols = torch.where(ti)
                                update_rows_global = active_indices[i:j][update_rows_local]
                                working_blocks[update_rows_global, update_cols] = x0[ti]
                            self._finish_timing(timing, "cache_update", update_start)

                        total_nfe += len(active_indices)
                        continue

                if batch_adjusted:
                    batch_adjusted = False
                    block_has_mask = (working_blocks == mask_id).any(dim=1)
                active_indices = torch.where(block_has_mask)[0]
                if len(active_indices) > 0:
                    if old_active_indices is None or len(active_indices) < len(old_active_indices) or cache_recomputed:
                        cache_recomputed = False
                        old_active_indices = active_indices
                        prepare_start = self._start_timing()
                        decoding_positions = self.prepare_decoding(batched_prompts, active_indices)
                        self._finish_timing(timing, "decode_prepare", prepare_start)

                    forward_start = self._start_timing()
                    logits = model(working_blocks[active_indices], position_ids=decoding_positions).logits
                    logits = self._dream_decode_logits(logits)
                    self._finish_timing(timing, "decode_forward", forward_start)
                    self._add_timing_count(timing, "decode_forward_calls", 1)
                    self._add_timing_count(timing, "decode_forward_nfe", len(active_indices))
                    update_start = self._start_timing()
                    mask_index = (working_blocks[active_indices] == mask_id)
                    x0, transfer_index = self.get_transfer_index(
                        logits, temperature, remasking, mask_index, working_blocks[active_indices], threshold)
                    del logits

                    update_rows_local, update_cols = torch.where(transfer_index)
                    update_rows_global = active_indices[update_rows_local]
                    working_blocks[update_rows_global, update_cols] = x0[transfer_index]
                    self._finish_timing(timing, "decode_update", update_start)

                    total_nfe += len(active_indices)
        self._finish_timing(timing, "total", total_start)
        if timing is not None:
            timing["total_nfe"] = int(total_nfe)
        self._active_timing = previous_active_timing
        return all_generated_parts, total_nfe

    @staticmethod
    def _dkv_transfer_schedule(block_length: int, steps_per_block: int) -> list[int]:
        if block_length <= 0 or steps_per_block <= 0:
            raise ValueError("dKV block length and steps per block must be positive.")
        base, remainder = divmod(block_length, steps_per_block)
        return [base + (step < remainder) for step in range(steps_per_block)]

    @staticmethod
    def _dkv_query_positions(
        seq: Sequence,
        delayed_unresolved: list[int] | None,
        step: int,
        cache_interval: int,
        refresh_positions: list[int] | None = None,
    ) -> tuple[list[int], bool]:
        refresh = step <= 1 or step % cache_interval == 0
        if refresh:
            if refresh_positions is None:
                refresh_positions = list(range(len(seq)))
            return refresh_positions, True
        if delayed_unresolved is None:
            raise RuntimeError("dKV compact decoding requires the delayed unresolved-token state.")
        return delayed_unresolved, False

    def _dream_dkv_query_positions(self, seq, step, cache_interval, refresh_positions=None):
        """Dream shift_type='un': query predecessors of unresolved tokens."""
        dynamic = set(range(len(seq)) if refresh_positions is None else refresh_positions)
        refresh = step % cache_interval == 0
        predecessors = {
            (pos - 1) % len(seq)
            for pos, token in enumerate(seq.token_ids) if token == self.config.mask_id
        }
        queries = dynamic if refresh else dynamic & predecessors
        start = seq.generation_start + seq.current_block_idx * self.config.block_length
        # Absolute positions are essential: shifting packed query rows is wrong.
        queries |= {
            max(pos - 1, 0) for pos in range(start, start + self.config.block_length)
            if seq.token_ids[pos] == self.config.mask_id
        }
        return sorted(queries), refresh

    def _dkv_select_masked_tokens(
        self,
        logits: torch.Tensor,
        temperature: float,
        remasking: str,
        num_transfer_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Select a fixed quota from logits that correspond only to masked tokens."""
        if logits.dim() != 3:
            raise ValueError("dKV masked-token logits must have shape [batch, tokens, vocab].")
        batch_size, num_masked, _ = logits.shape
        if not 0 <= num_transfer_tokens <= num_masked:
            raise ValueError(
                f"Cannot transfer {num_transfer_tokens} tokens from {num_masked} masked positions."
            )

        logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
        x0 = torch.argmax(logits_with_noise, dim=-1)
        if num_transfer_tokens == 0:
            empty = torch.empty((batch_size, 0), dtype=torch.int64, device=logits.device)
            return x0, empty

        if remasking == "low_confidence":
            flat_logits = logits.reshape(-1, logits.shape[-1])
            flat_x0 = x0.reshape(-1)
            flat_confidence = torch.empty(
                flat_x0.shape,
                dtype=torch.float64,
                device=logits.device,
            )
            for start in range(0, flat_logits.shape[0], 64):
                end = min(start + 64, flat_logits.shape[0])
                logits_fp64 = flat_logits[start:end].to(torch.float64)
                chosen = torch.gather(
                    logits_fp64,
                    dim=-1,
                    index=flat_x0[start:end].unsqueeze(-1),
                ).squeeze(-1)
                flat_confidence[start:end] = chosen - torch.logsumexp(logits_fp64, dim=-1)
            confidence = flat_confidence.view_as(x0)
        elif remasking == "random":
            confidence = torch.rand((batch_size, num_masked), device=logits.device)
        else:
            raise NotImplementedError(remasking)

        selected = torch.topk(confidence, k=num_transfer_tokens, dim=-1).indices
        return x0, selected

    @torch.inference_mode()
    def generate_with_dkv_cache(self, prompts: list[Sequence]):
        """Generate with published dKV-Cache-Decode semantics on paged KV slots."""
        return self._generate_with_dkv_cache(prompts, use_acache=False)

    @torch.inference_mode()
    def generate_with_dkv_acache(self, prompts: list[Sequence]):
        """Generate with ACache affix reuse layered onto dKV-Cache-Decode."""
        return self._generate_with_dkv_cache(prompts, use_acache=True)

    def _generate_with_dkv_cache(self, prompts: list[Sequence], use_acache: bool):
        is_dream = self._is_dream_model()

        device = self.device
        model = self.model
        mask_id = self.config.mask_id
        block_length = self.config.block_length
        temperature = self.config.temperature
        remasking = self.config.remasking
        cache_interval = int(self.config.dkv_cache_interval)
        steps = int(self.config.dkv_steps) or int(self.config.gen_length)
        num_blocks = int(self.config.gen_length) // block_length
        if int(self.config.gen_length) % block_length != 0:
            raise ValueError("dKV generation length must be divisible by block length.")
        if steps % num_blocks != 0:
            raise ValueError("dKV denoising steps must be divisible by the number of blocks.")
        if cache_interval <= 0:
            raise ValueError("dKV cache interval must be positive.")
        steps_per_block = steps // num_blocks
        transfer_schedule = self._dkv_transfer_schedule(block_length, steps_per_block)
        max_num_seqs = self.config.max_num_seqs if self.config.max_num_seqs > 0 else len(prompts)

        timing = self._new_timing("dkv_acache" if use_acache else "dkv_cache", len(prompts))
        self.last_timing = timing or {}
        previous_active_timing = self._active_timing
        self._active_timing = timing
        total_start = self._start_timing()
        prefix_token_ids = ()
        if use_acache:
            metadata_start = self._start_timing()
            prefix_token_ids = self._prepare_acache(prompts)
            self._finish_timing(timing, "acache_metadata", metadata_start)
            if not prefix_token_ids:
                use_acache = False
        total_nfe = 0
        all_generated_parts = [None] * len(prompts)
        result_index_by_seq_id = {seq.seq_id: index for index, seq in enumerate(prompts)}
        next_prompt_idx = 0

        with tqdm(total=len(prompts), desc="Processing requests", smoothing=0) as pbar:
            while next_prompt_idx < len(prompts):
                scheduler_start = self._start_timing()
                batched_prompts = []
                if use_acache:
                    new_prompts, collected_until = self._collect_admissible_sequences(
                        prompts,
                        next_prompt_idx,
                        current_batch_size=0,
                        max_num_seqs=max_num_seqs,
                    )
                    admitted = self._admit_sequences_with_acache(
                        batched_prompts,
                        new_prompts,
                        prefix_token_ids,
                        allow_kv_cache_resize=True,
                    )
                    next_prompt_idx = collected_until - (len(new_prompts) - admitted)
                else:
                    while (
                        next_prompt_idx < len(prompts)
                        and len(batched_prompts) < max_num_seqs
                        and self.block_manager.can_allocate(prompts[next_prompt_idx])
                    ):
                        seq = prompts[next_prompt_idx]
                        self.block_manager.allocate(seq)
                        self._record_kv_cache_usage(timing)
                        batched_prompts.append(seq)
                        next_prompt_idx += 1
                self._finish_timing(timing, "scheduler", scheduler_start)
                if not batched_prompts:
                    raise RuntimeError("No dKV request could be admitted under the current KV-cache budget.")

                batch_size = len(batched_prompts)
                for block_idx in range(num_blocks):
                    for seq in batched_prompts:
                        seq.current_block_idx = block_idx
                    previous_unresolved = [None] * batch_size
                    delayed_unresolved = [None] * batch_size

                    for step, quota in enumerate(transfer_schedule):
                        query_positions = []
                        is_refresh = None
                        for row, seq in enumerate(batched_prompts):
                            if is_dream:
                                row_positions, row_refresh = self._dream_dkv_query_positions(
                                    seq, step, cache_interval,
                                    seq.recompute_positions if use_acache else None,
                                )
                            else:
                                row_positions, row_refresh = self._dkv_query_positions(
                                    seq, delayed_unresolved[row], step, cache_interval,
                                    seq.recompute_positions if use_acache else None,
                                )
                            query_positions.append(row_positions)
                            if is_refresh is None:
                                is_refresh = row_refresh
                            elif is_refresh != row_refresh:
                                raise RuntimeError("dKV batch rows disagree on refresh state.")

                        prepare_start = self._start_timing()
                        if use_acache:
                            input_ids, positions, query_offsets = self.prepare_dkv_caching_acache(
                                batched_prompts,
                                query_positions,
                            )
                        else:
                            input_ids, positions, query_offsets = self.prepare_dkv_caching(
                                batched_prompts,
                                query_positions,
                            )
                        self._finish_timing(timing, "cache_prepare", prepare_start)

                        current_mask_positions = []
                        logit_rows = []
                        for row, seq in enumerate(batched_prompts):
                            block_start = seq.generation_start + block_idx * block_length
                            block_end = block_start + block_length
                            row_mask_positions = [
                                pos
                                for pos in range(block_start, block_end)
                                if seq.token_ids[pos] == mask_id
                            ]
                            current_mask_positions.append(row_mask_positions)
                            query_index_by_position = {
                                pos: query_offsets[row] + local_index
                                for local_index, pos in enumerate(query_positions[row])
                            }
                            try:
                                logit_rows.extend(
                                    query_index_by_position[max(pos - 1, 0) if is_dream else pos]
                                    for pos in row_mask_positions
                                )
                            except KeyError as exc:
                                raise RuntimeError(
                                    "dKV delayed query omitted a currently masked token."
                                ) from exc

                        num_masked = len(current_mask_positions[0])
                        if any(len(row) != num_masked for row in current_mask_positions):
                            raise RuntimeError("dKV batch rows have different masked-token counts.")

                        row_indices = torch.tensor(logit_rows, dtype=torch.int64, device=device)
                        forward_before = (timing or {}).get("cache_forward", 0.0)
                        forward_start = self._start_timing()
                        if is_dream:
                            output = model(input_ids, position_ids=positions, logits_keep_indices=row_indices)
                        else:
                            with self._temporary_llada_logit_rows(row_indices):
                                output = model(input_ids, position_ids=positions)
                        self._finish_timing(timing, "cache_forward", forward_start)
                        forward_elapsed = (timing or {}).get("cache_forward", 0.0) - forward_before
                        forward_key = "dkv_refresh_forward" if is_refresh else "dkv_compact_forward"
                        call_key = f"{forward_key}_calls"
                        if timing is not None:
                            timing[forward_key] = timing.get(forward_key, 0.0) + forward_elapsed
                        self._add_timing_count(timing, call_key, 1)
                        self._add_timing_count(timing, "cache_forward_calls", 1)
                        self._add_timing_count(timing, "cache_forward_nfe", batch_size)
                        total_nfe += batch_size

                        update_start = self._start_timing()
                        masked_logits = output.logits.view(
                            batch_size,
                            num_masked,
                            -1,
                        )
                        x0, selected = self._dkv_select_masked_tokens(
                            masked_logits,
                            temperature,
                            remasking,
                            quota,
                        )
                        selected_rows = selected.tolist()
                        for row, seq in enumerate(batched_prompts):
                            for selected_index in selected_rows[row]:
                                token_position = current_mask_positions[row][selected_index]
                                seq.token_ids[token_position] = int(x0[row, selected_index].item())

                            block_start = seq.generation_start + block_idx * block_length
                            current_unresolved = [
                                pos
                                for pos in range(block_start, seq.generation_end)
                                if seq.token_ids[pos] == mask_id
                            ]
                            delayed_unresolved[row] = previous_unresolved[row]
                            previous_unresolved[row] = current_unresolved
                        del masked_logits, output
                        self._finish_timing(timing, "cache_update", update_start)

                    for seq in batched_prompts:
                        block_start = seq.generation_start + block_idx * block_length
                        block_end = block_start + block_length
                        if mask_id in seq.token_ids[block_start:block_end]:
                            raise RuntimeError("dKV transfer schedule did not finish the current block.")

                scheduler_start = self._start_timing()
                for seq in batched_prompts:
                    all_generated_parts[result_index_by_seq_id[seq.seq_id]] = seq.generated_token_ids
                    self.block_manager.deallocate(seq)
                    pbar.update(1)
                self._finish_timing(timing, "scheduler", scheduler_start)

        self._finish_timing(timing, "total", total_start)
        if timing is not None:
            timing["total_nfe"] = int(total_nfe)
        self._active_timing = previous_active_timing
        return all_generated_parts, total_nfe

    @torch.inference_mode()
    def generate_with_acache(self, prompts: list[Sequence]):
        timing = self._new_timing("acache", len(prompts))
        self.last_timing = timing or {}
        previous_active_timing = self._active_timing
        self._active_timing = timing
        total_start = self._start_timing()
        metadata_start = self._start_timing()
        prefix_token_ids = self._prepare_acache(prompts)
        self._finish_timing(timing, "acache_metadata", metadata_start)
        if not prefix_token_ids:
            self._finish_timing(timing, "total", total_start)
            self._active_timing = previous_active_timing
            return self.generate_with_dual_cache(prompts)

        device = self.device
        model = self.model
        mask_id, block_length, temperature, remasking, threshold = \
            self.config.mask_id, self.config.block_length, self.config.temperature, \
            self.config.remasking, self.config.threshold
        recompute_batch_size = self.config.recompute_batch_size
        assert recompute_batch_size > 0, "recompute_batch_size must be larger than 0"
        max_num_seqs = self.config.max_num_seqs if self.config.max_num_seqs > 0 else len(prompts)
        vocab_size = self.config.hf_config.vocab_size

        total_nfe = 0
        all_generated_parts = [None] * len(prompts)
        result_index_by_seq_id = {seq.seq_id: i for i, seq in enumerate(prompts)}

        next_prompt_idx = 0
        batched_prompts = []
        scheduler_start = self._start_timing()
        initial_prompts, next_prompt_idx = self._collect_admissible_sequences(
            prompts,
            next_prompt_idx,
            current_batch_size=0,
            max_num_seqs=max_num_seqs,
        )
        admitted_initial = self._admit_sequences_with_acache(
            batched_prompts,
            initial_prompts,
            prefix_token_ids,
            allow_kv_cache_resize=True,
        )
        next_prompt_idx -= len(initial_prompts) - admitted_initial
        self._finish_timing(timing, "scheduler", scheduler_start)

        batch_size = len(batched_prompts)
        if batch_size == 0 and prompts:
            raise RuntimeError("No requests could be admitted under the current ACache KV-cache budget.")
        working_blocks = torch.full((batch_size, block_length), mask_id, dtype=torch.int64, device=device)
        block_offset = torch.arange(block_length, device=device)

        with tqdm(total=len(prompts), desc="Processing requests", smoothing=0) as pbar:
            for i in range(0, len(batched_prompts), recompute_batch_size):
                j = min(i + recompute_batch_size, batch_size)
                prepare_start = self._start_timing()
                input_ids, positions, block_query_starts = self.prepare_caching_acache(batched_prompts[i:j])
                self._finish_timing(timing, "cache_prepare", prepare_start)
                keep_indices = block_query_starts.unsqueeze(1) + block_offset
                if self._is_dream_model():
                    q_starts = block_query_starts - torch.tensor(
                        [seq.block_query_start() for seq in batched_prompts[i:j]],
                        dtype=torch.int32,
                        device=device,
                    )
                    keep_indices = self._dream_shift_keep_indices(keep_indices, q_starts)
                keep_indices = keep_indices.flatten()
                forward_start = self._start_timing()
                if self._uses_decoder_layers():
                    output = model(input_ids, position_ids=positions, logits_keep_indices=keep_indices)
                else:
                    output = model(input_ids, position_ids=positions)
                self._finish_timing(timing, "cache_forward", forward_start)
                self._add_timing_count(timing, "cache_forward_calls", 1)
                self._add_timing_count(timing, "cache_forward_nfe", j - i)
                update_start = self._start_timing()
                if self._uses_decoder_layers():
                    working_logits = output.logits.view(-1, block_length, vocab_size)
                else:
                    working_logits = output.logits[keep_indices].view(-1, block_length, vocab_size)

                if self._is_dream_model():
                    x0 = self._sample_dream_seed_tokens(working_logits)
                    update_rows_global = torch.arange(i, j, dtype=torch.int64, device=device)
                    working_blocks[update_rows_global, 0] = x0[:, 0]
                    del x0, working_logits, output
                else:
                    mask_index = (working_blocks[i:j] == mask_id)
                    x0, ti = self.get_transfer_index(working_logits, temperature, remasking, mask_index, working_blocks[i:j], threshold)
                    del working_logits, output
                    update_rows_local, update_cols = torch.where(ti)
                    update_rows_global = torch.arange(i, j, dtype=torch.int64, device=device)[update_rows_local]
                    working_blocks[update_rows_global, update_cols] = x0[ti]
                self._finish_timing(timing, "cache_update", update_start)

            total_nfe += batch_size

            old_active_indices = None
            cache_recomputed, batch_adjusted = False, False
            while batched_prompts:
                block_has_mask = (working_blocks == mask_id).any(dim=1)
                finished_block_mask = ~block_has_mask

                if finished_block_mask.any():
                    scheduler_start = self._start_timing()
                    update_indices = torch.where(finished_block_mask)[0]
                    finished_indices = []
                    for idx in update_indices:
                        seq = batched_prompts[idx]
                        start = seq.generation_start + (seq.current_block_idx * block_length)
                        end = start + block_length
                        seq.token_ids[start:end] = working_blocks[idx].tolist()
                        seq.current_block_idx += 1
                        if seq.current_block_idx >= seq.num_blocks_to_generate:
                            self.block_manager.deallocate(seq)
                            finished_indices.append(idx.item())
                            all_generated_parts[result_index_by_seq_id[seq.seq_id]] = seq.generated_token_ids
                            pbar.update(1)

                    working_blocks[update_indices] = mask_id

                    if finished_indices:
                        batch_adjusted = True
                        finished_set = set(finished_indices)
                        batched_prompts = [prompt for i, prompt in enumerate(batched_prompts) if i not in finished_set]

                        allow_resize_for_new_admission = len(batched_prompts) == 0
                        new_prompts, next_prompt_idx = self._collect_admissible_sequences(
                            prompts,
                            next_prompt_idx,
                            current_batch_size=len(batched_prompts),
                            max_num_seqs=max_num_seqs,
                        )
                        admitted_new = self._admit_sequences_with_acache(
                            batched_prompts,
                            new_prompts,
                            prefix_token_ids,
                            allow_kv_cache_resize=allow_resize_for_new_admission,
                        )
                        next_prompt_idx -= len(new_prompts) - admitted_new

                        finished_indices = torch.tensor(finished_indices, dtype=torch.int64, pin_memory=True).to(device, non_blocking=True)
                        unfinished_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
                        unfinished_mask[finished_indices] = False
                        unfinished_indices = torch.where(unfinished_mask)[0]
                        new_batch_size = len(batched_prompts)
                        if new_batch_size != batch_size:
                            new_working_blocks = torch.full((new_batch_size, block_length), mask_id, dtype=torch.int64, device=device)
                            if len(unfinished_indices) > 0:
                                new_working_blocks[:len(unfinished_indices)] = working_blocks[unfinished_indices]
                            working_blocks = new_working_blocks
                            batch_size = new_batch_size
                        else:
                            if len(unfinished_indices) > 0:
                                working_blocks[:len(unfinished_indices)] = working_blocks[unfinished_indices]
                            working_blocks[len(unfinished_indices):] = mask_id
                    self._finish_timing(timing, "scheduler", scheduler_start)

                    block_all_mask = (working_blocks == mask_id).all(dim=1)
                    active_indices = torch.where(block_all_mask)[0]
                    if len(active_indices) > 0:
                        cache_recomputed = True
                        for i in range(0, len(active_indices), recompute_batch_size):
                            j = min(i + recompute_batch_size, len(active_indices))
                            prepare_start = self._start_timing()
                            input_ids, positions, block_query_starts = \
                                self.prepare_caching_acache(batched_prompts, active_indices[i:j])
                            self._finish_timing(timing, "cache_prepare", prepare_start)
                            keep_indices = block_query_starts.unsqueeze(1) + block_offset
                            selected_seqs = [batched_prompts[int(idx)] for idx in active_indices[i:j].tolist()]
                            if self._is_dream_model():
                                q_starts = block_query_starts - torch.tensor(
                                    [seq.block_query_start() for seq in selected_seqs],
                                    dtype=torch.int32,
                                    device=device,
                                )
                                keep_indices = self._dream_shift_keep_indices(keep_indices, q_starts)
                            keep_indices = keep_indices.flatten()
                            forward_start = self._start_timing()
                            if self._uses_decoder_layers():
                                output = model(input_ids, position_ids=positions, logits_keep_indices=keep_indices)
                            else:
                                output = model(input_ids, position_ids=positions)
                            self._finish_timing(timing, "cache_forward", forward_start)
                            self._add_timing_count(timing, "cache_forward_calls", 1)
                            self._add_timing_count(timing, "cache_forward_nfe", j - i)
                            update_start = self._start_timing()
                            if self._uses_decoder_layers():
                                working_logits = output.logits.view(-1, block_length, vocab_size)
                            else:
                                working_logits = output.logits[keep_indices].view(-1, block_length, vocab_size)

                            if self._is_dream_model():
                                x0 = self._sample_dream_seed_tokens(working_logits)
                                working_blocks[active_indices[i:j], 0] = x0[:, 0]
                                del x0, working_logits, output
                            else:
                                mask_index = (working_blocks[active_indices[i:j]] == mask_id)
                                x0, ti = self.get_transfer_index(working_logits, temperature, remasking, mask_index,
                                                            working_blocks[active_indices[i:j]], threshold)
                                del working_logits, output
                                update_rows_local, update_cols = torch.where(ti)
                                update_rows_global = active_indices[i:j][update_rows_local]
                                working_blocks[update_rows_global, update_cols] = x0[ti]
                            self._finish_timing(timing, "cache_update", update_start)

                        total_nfe += len(active_indices)
                        continue

                if batch_adjusted:
                    batch_adjusted = False
                    block_has_mask = (working_blocks == mask_id).any(dim=1)
                active_indices = torch.where(block_has_mask)[0]
                if len(active_indices) > 0:
                    if old_active_indices is None or len(active_indices) < len(old_active_indices) or cache_recomputed:
                        cache_recomputed = False
                        old_active_indices = active_indices
                        prepare_start = self._start_timing()
                        decoding_positions = self.prepare_decoding_acache(batched_prompts, active_indices)
                        self._finish_timing(timing, "decode_prepare", prepare_start)

                    forward_start = self._start_timing()
                    logits = model(working_blocks[active_indices], position_ids=decoding_positions).logits
                    logits = self._dream_decode_logits(logits)
                    self._finish_timing(timing, "decode_forward", forward_start)
                    self._add_timing_count(timing, "decode_forward_calls", 1)
                    self._add_timing_count(timing, "decode_forward_nfe", len(active_indices))
                    update_start = self._start_timing()
                    mask_index = (working_blocks[active_indices] == mask_id)
                    x0, transfer_index = self.get_transfer_index(
                        logits, temperature, remasking, mask_index, working_blocks[active_indices], threshold)
                    del logits

                    update_rows_local, update_cols = torch.where(transfer_index)
                    update_rows_global = active_indices[update_rows_local]
                    working_blocks[update_rows_global, update_cols] = x0[transfer_index]
                    self._finish_timing(timing, "decode_update", update_start)

                    total_nfe += len(active_indices)

        self._finish_timing(timing, "total", total_start)
        if timing is not None:
            timing["total_nfe"] = int(total_nfe)
        self._active_timing = previous_active_timing
        return all_generated_parts, total_nfe

    def get_transfer_index(self, logits, temperature, remasking, mask_index, x, threshold=0.9):
        logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
        x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

        if remasking == 'low_confidence':
            flat_logits = logits.reshape(-1, logits.shape[-1])
            flat_x0 = x0.reshape(-1)
            flat_probs = torch.empty(flat_x0.shape, dtype=torch.float32, device=logits.device)
            chunk_rows = 64
            for start in range(0, flat_logits.shape[0], chunk_rows):
                end = min(start + chunk_rows, flat_logits.shape[0])
                logits_fp32 = flat_logits[start:end].to(torch.float32)
                chosen_logits = torch.gather(logits_fp32, dim=-1, index=flat_x0[start:end].unsqueeze(-1)).squeeze(-1)
                flat_probs[start:end] = torch.exp(chosen_logits - torch.logsumexp(logits_fp32, dim=-1))
            x0_p = flat_probs.view_as(x0)
        elif remasking == 'random':
            x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
        else:
            raise NotImplementedError(remasking)

        x0 = torch.where(mask_index, x0, x)
        confidence = torch.where(mask_index, x0_p, -np.inf)

        # Threshold-based selection
        transfer_index = (confidence >= threshold) & mask_index

        # Ensure at least one token is transferred per sequence
        needs_transfer = (mask_index.any(dim=1)) & (~transfer_index.any(dim=1))

        # For those sequences, find the maximum confidence index
        max_conf_indices = torch.argmax(confidence, dim=1)
        # Use advanced indexing to set those positions to True
        row_indices = torch.where(needs_transfer)[0]
        transfer_index[row_indices, max_conf_indices[row_indices]] = True

        return x0, transfer_index
