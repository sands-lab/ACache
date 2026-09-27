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

    def __init__(
        self,
        token_ids: list[int],
        block_length: int = 32,
        gen_length: int = 256,
        cache_block_size: int = 256,
        mask_id: int = 126336,
        prompt_affix_len: int = 0,
        prompt_affix_start: int | None = None,
        prompt_affix_end: int | None = None,
        generation_start: int | None = None,
    ):
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)
        self.num_prompt_tokens = len(token_ids)

        if generation_start is None:
            self.generation_start = self.num_prompt_tokens
            self.token_ids.extend([mask_id] * gen_length)
        else:
            self.generation_start = int(generation_start)
            generation_end = self.generation_start + int(gen_length)
            if not (0 <= self.generation_start <= generation_end <= self.num_prompt_tokens):
                raise ValueError(
                    "Infill generation span must be contained in the provided prompt."
                )
            self.token_ids[self.generation_start:generation_end] = [mask_id] * gen_length
        self.generation_end = self.generation_start + int(gen_length)
        self.num_tokens = len(self.token_ids)

        if prompt_affix_start is None:
            prompt_affix_start = 0
        if prompt_affix_end is None:
            prompt_affix_end = int(prompt_affix_start) + int(prompt_affix_len)
        self.prompt_affix_start = int(prompt_affix_start)
        self.prompt_affix_end = int(prompt_affix_end)
        if not (0 <= self.prompt_affix_start <= self.prompt_affix_end <= self.num_tokens):
            raise ValueError("Affix span must be contained in the logical sequence.")
        if (
            self.generation_start < self.prompt_affix_end
            and self.generation_end > self.prompt_affix_start
        ):
            raise ValueError("Generation and affix spans must not overlap.")
        self.prompt_affix_len = self.prompt_affix_end - self.prompt_affix_start

        self.block_table = []
        self.gen_length = gen_length
        self.block_length = block_length
        self.cache_block_size = cache_block_size

        self.current_block_idx = 0
        self.num_blocks_to_generate = (self.gen_length + block_length - 1) // block_length

        self.acache_enabled = False
        self.affix_start = self.prompt_affix_start
        self.affix_end = self.prompt_affix_end
        self.affix_len = 0
        self.anchor_positions: list[int] = []
        self.num_anchor_tokens = 0
        self.recompute_positions: list[int] = list(range(self.num_tokens))
        self.recompute_slot_mapping: list[int] = []
        self.read_slot_map: list[int] = []
        self.shared_slot_offset = 0
        self.relocated_affix_materialized = False

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
        return self.token_ids[self.generation_start:self.generation_end]

    @property
    def prompt_affix_token_ids(self):
        return self.token_ids[self.prompt_affix_start:self.prompt_affix_end]

    @property
    def non_affix_positions(self):
        return list(range(self.affix_start)) + list(range(self.affix_end, self.num_tokens))

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

    def enable_acache(
        self,
        affix_len: int | None = None,
        anchor_ratio: float = 0.0,
        *,
        affix_start: int | None = None,
        affix_end: int | None = None,
    ):
        if affix_start is None:
            affix_start = self.prompt_affix_start
        if affix_end is None:
            if affix_len is None:
                affix_end = self.prompt_affix_end
            else:
                affix_end = int(affix_start) + int(affix_len)
        self.affix_start = int(affix_start)
        self.affix_end = int(affix_end)
        if not (0 <= self.affix_start <= self.affix_end <= self.num_tokens):
            raise ValueError("ACache affix span must be contained in the logical sequence.")
        self.affix_len = self.affix_end - self.affix_start
        self.acache_enabled = self.affix_len > 0
        self.num_anchor_tokens = min(math.ceil(anchor_ratio * self.affix_len), self.affix_len)
        self.anchor_positions = []
        self.recompute_positions = list(range(self.num_tokens))
        self.recompute_slot_mapping = []
        self.read_slot_map = []
        self.relocated_affix_materialized = False

    def set_anchor_positions(self, anchor_positions: list[int]):
        anchor_positions = sorted(set(int(pos) for pos in anchor_positions))
        if not self.acache_enabled:
            self.anchor_positions = []
            self.num_anchor_tokens = 0
            self.recompute_positions = list(range(self.num_tokens))
            return
        for pos in anchor_positions:
            assert self.affix_start <= pos < self.affix_end
        expected = min(self.num_anchor_tokens, self.affix_len)
        assert len(anchor_positions) == expected
        self.anchor_positions = anchor_positions
        self.num_anchor_tokens = len(anchor_positions)
        self.recompute_positions = sorted(self.anchor_positions + self.non_affix_positions)

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
        for local_pos, pos in enumerate(range(self.affix_start, self.affix_end)):
            read_slot_map[pos] = anchor_slot_by_pos.get(pos, shared_slot_offset + local_pos)
        for pos, slot in zip(self.non_affix_positions, dynamic_slots):
            read_slot_map[pos] = slot

        slot_by_position = dict(zip(self.anchor_positions, anchor_slots))
        slot_by_position.update(zip(self.non_affix_positions, dynamic_slots))
        self.recompute_positions = sorted(slot_by_position)
        self.recompute_slot_mapping = [slot_by_position[pos] for pos in self.recompute_positions]
        self.read_slot_map = read_slot_map

    def block_query_start(self):
        block_start = self.generation_start + self.current_block_idx * self.block_length
        return self.recompute_positions.index(block_start)

    def __getstate__(self):
        return (self.num_tokens, self.num_prompt_tokens, self.block_table)

    def __setstate__(self, state):
        self.num_tokens, self.num_prompt_tokens, self.block_table, self.token_ids = state
