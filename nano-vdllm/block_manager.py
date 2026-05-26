from collections import deque

from sequence import Sequence


class Block:
    def __init__(self, block_id):
        self.block_id = block_id
        self.is_free = True


class BlockManager:
    def __init__(self, num_blocks: int, cache_block_size: int):
        self.cache_block_size = cache_block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()
        self.reserved_block_ids: set[int] = set()

    def _allocate_block(self, block_id: int) -> Block:
        block = self.blocks[block_id]
        assert block.is_free, f"Block {block_id} is not free but trying to allocate"
        block.is_free = False  # Mark block as in use
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return self.blocks[block_id]

    def _deallocate_block(self, block_id: int):
        block = self.blocks[block_id]
        assert not block.is_free, f"Block {block_id} is already free but trying to deallocate"
        block.is_free = True  # Mark block as free
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def reserve_prefix(self, num_blocks: int):
        for block_id in range(num_blocks):
            if block_id in self.reserved_block_ids:
                continue
            self._allocate_block(block_id)
            self.reserved_block_ids.add(block_id)

    def can_allocate(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= seq.num_cache_blocks

    def allocate(self, seq: Sequence):
        assert not seq.block_table
        for _ in range(seq.num_cache_blocks):
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            seq.block_table.append(block_id)

    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            self._deallocate_block(block_id)
        seq.block_table.clear()
