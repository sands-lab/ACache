from copy import copy
from enum import Enum, auto
from itertools import count
import math


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    counter = count()

    def __init__(self, token_ids: list[int], block_length: int=32, gen_length: int=256,
                 cache_block_size: int=256, mask_id: int=126336, prompt_affix_len: int=0):
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)
        self.token_ids.extend([mask_id] * gen_length)

        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.prompt_affix_len = min(max(int(prompt_affix_len), 0), self.num_prompt_tokens)

        self.block_table = []
        self.gen_length = gen_length
        self.block_length = block_length
        self.cache_block_size = cache_block_size

        self.current_block_idx = 0
        self.num_blocks_to_generate = (self.gen_length + block_length - 1) // block_length

        self.acache_enabled = False
        self.affix_len = 0
        self.anchor_positions: list[int] = []
        self.num_anchor_tokens = 0
        self.recompute_positions: list[int] = list(range(self.num_tokens))
        self.recompute_slot_mapping: list[int] = []
        self.read_slot_map: list[int] = []
        self.shared_slot_offset = 0

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def generated_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_private_slots(self):
        if not self.acache_enabled:
            return self.num_tokens
        return self.num_anchor_tokens + (self.num_tokens - self.affix_len)

    @property
    def num_cache_blocks(self):
        return (self.num_private_slots + self.cache_block_size - 1) // self.cache_block_size

    @property
    def last_cache_block_num_tokens(self):
        return self.num_private_slots - (self.num_cache_blocks - 1) * self.cache_block_size

    def block(self, i):
        assert 0 <= i < self.num_cache_blocks
        return self.token_ids[i*self.cache_block_size: (i+1)*self.cache_block_size]

    def enable_acache(self, affix_len: int, anchor_ratio: float):
        self.affix_len = min(max(int(affix_len), 0), self.num_prompt_tokens)
        self.acache_enabled = self.affix_len > 0
        self.num_anchor_tokens = min(math.ceil(anchor_ratio * self.affix_len), self.affix_len)
        self.anchor_positions = []
        self.recompute_positions = list(range(self.num_tokens))
        self.recompute_slot_mapping = []
        self.read_slot_map = []

    def set_anchor_positions(self, anchor_positions: list[int]):
        anchor_positions = sorted(set(int(pos) for pos in anchor_positions))
        if not self.acache_enabled:
            self.anchor_positions = []
            self.num_anchor_tokens = 0
            self.recompute_positions = list(range(self.num_tokens))
            return
        for pos in anchor_positions:
            assert 0 <= pos < self.affix_len
        expected = min(self.num_anchor_tokens, self.affix_len)
        assert len(anchor_positions) == expected
        self.anchor_positions = anchor_positions
        self.num_anchor_tokens = len(anchor_positions)
        self.recompute_positions = self.anchor_positions + list(range(self.affix_len, self.num_tokens))

    def finalize_slot_mapping(self, shared_slot_offset: int = 0):
        self.shared_slot_offset = shared_slot_offset
        private_slots = []
        for i in range(self.num_cache_blocks):
            start = self.block_table[i] * self.cache_block_size
            if i != self.num_cache_blocks - 1:
                end = start + self.cache_block_size
            else:
                end = start + self.last_cache_block_num_tokens
            private_slots.extend(range(start, end))
        assert len(private_slots) == self.num_private_slots

        if not self.acache_enabled:
            self.recompute_positions = list(range(self.num_tokens))
            self.recompute_slot_mapping = private_slots
            self.read_slot_map = private_slots
            return

        anchor_slots = private_slots[:self.num_anchor_tokens]
        dynamic_slots = private_slots[self.num_anchor_tokens:]
        assert len(dynamic_slots) == self.num_tokens - self.affix_len
        anchor_slot_by_pos = {pos: slot for pos, slot in zip(self.anchor_positions, anchor_slots)}

        read_slot_map = [-1] * self.num_tokens
        for pos in range(self.affix_len):
            read_slot_map[pos] = anchor_slot_by_pos.get(pos, shared_slot_offset + pos)
        for pos in range(self.affix_len, self.num_tokens):
            read_slot_map[pos] = dynamic_slots[pos - self.affix_len]

        self.recompute_positions = self.anchor_positions + list(range(self.affix_len, self.num_tokens))
        self.recompute_slot_mapping = anchor_slots + dynamic_slots
        self.read_slot_map = read_slot_map

    def block_query_start(self):
        return self.num_anchor_tokens + (self.num_prompt_tokens + self.current_block_idx * self.block_length - self.affix_len)

    def __getstate__(self):
        return (self.num_tokens, self.num_prompt_tokens, self.block_table)

    def __setstate__(self, state):
        self.num_tokens, self.num_prompt_tokens, self.block_table, self.token_ids = state
