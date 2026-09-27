import pytest

from block_manager import BlockManager
from sequence import Sequence


def _allocate_fake_blocks(seq, first_block=2):
    seq.block_table = list(range(first_block, first_block + seq.num_cache_blocks))


@pytest.mark.parametrize(
    ("tokens", "affix_start", "affix_end", "generation_start"),
    [
        ([10, 11, 12], 0, 2, None),
        ([10, 20, 21, 11], 1, 3, None),
        ([10, 99, 99, 20, 21], 3, 5, 1),
    ],
)
def test_affix_layout_maps_shared_and_private_slots(
    tokens,
    affix_start,
    affix_end,
    generation_start,
):
    seq = Sequence(
        tokens,
        gen_length=2,
        block_length=2,
        cache_block_size=4,
        mask_id=99,
        prompt_affix_start=affix_start,
        prompt_affix_end=affix_end,
        generation_start=generation_start,
    )
    seq.enable_acache(anchor_ratio=0.5)
    anchor = affix_end - 1
    seq.set_anchor_positions([anchor])
    _allocate_fake_blocks(seq)
    seq.finalize_slot_mapping(shared_slot_offset=0)

    assert seq.recompute_positions == sorted([anchor] + seq.non_affix_positions)
    assert seq.read_slot_map[affix_start] == 0
    assert seq.read_slot_map[anchor] != anchor - affix_start
    assert all(slot >= 0 for slot in seq.read_slot_map)
    assert seq.block_query_start() == seq.recompute_positions.index(seq.generation_start)


def test_suffix_layout_generates_inside_prompt_and_preserves_right_context():
    seq = Sequence(
        [10, 50, 51, 20, 21],
        gen_length=2,
        block_length=2,
        mask_id=99,
        prompt_affix_start=3,
        prompt_affix_end=5,
        generation_start=1,
    )

    assert seq.token_ids == [10, 99, 99, 20, 21]
    assert seq.generated_token_ids == [99, 99]
    assert seq.prompt_affix_token_ids == [20, 21]


def test_generation_and_affix_spans_must_not_overlap():
    with pytest.raises(ValueError, match="must not overlap"):
        Sequence(
            [10, 11, 12, 13],
            gen_length=2,
            mask_id=99,
            prompt_affix_start=1,
            prompt_affix_end=3,
            generation_start=2,
        )


def test_relocation_workspace_blocks_can_be_released():
    manager = BlockManager(num_blocks=8, cache_block_size=4)
    manager.reserve_prefix(5)

    manager.release_prefix_tail(2)

    assert manager.reserved_block_ids == {0, 1}
    assert manager.used_block_ids == {0, 1}
    assert set(manager.free_block_ids) == set(range(2, 8))
