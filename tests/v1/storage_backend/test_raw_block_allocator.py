# SPDX-License-Identifier: Apache-2.0

# Future
from __future__ import annotations

# Third Party
import pytest

# First Party
from lmcache.v1.storage_backend.raw_block.allocator import (
    FixedSlotAllocator,
    SlotBuddyAllocator,
)


def test_fixed_slot_allocator_reuses_freed_slots():
    allocator = FixedSlotAllocator(
        data_base_offset=4096,
        slot_bytes=1024,
        max_slots=3,
    )

    first = allocator.allocate(512)
    second = allocator.allocate(512)
    third = allocator.allocate(512)

    assert [first.offset, second.offset, third.offset] == [4096, 5120, 6144]
    assert allocator.free_offset(second.offset) == 1024

    reused = allocator.allocate(512)
    assert reused.offset == second.offset
    assert reused.allocated_bytes == 1024


def test_fixed_slot_allocator_rejects_misaligned_offsets():
    allocator = FixedSlotAllocator(
        data_base_offset=4096,
        slot_bytes=1024,
        max_slots=3,
    )

    assert allocator.allocation_size(4097) == 0
    assert allocator.free_offset(4097) == 0
    with pytest.raises(ValueError, match="invalid fixed slot allocation"):
        allocator.mark_allocated(4097, 1024)


def test_slot_buddy_allocator_uses_power_of_two_classes():
    allocator = SlotBuddyAllocator(
        data_base_offset=4096,
        min_subslot_bytes=1024,
        slot_bytes=8192,
        managed_bytes=16 * 1024,
    )

    small = allocator.allocate(512)
    medium = allocator.allocate(2000)
    large = allocator.allocate(5000)

    assert small.allocated_bytes == 1024
    assert medium.allocated_bytes == 2048
    assert large.allocated_bytes == 8192
    assert allocator.allocation_size(medium.offset) == 2048
    assert allocator.allocated_bytes(0, 0) == 11 * 1024


def test_slot_buddy_allocator_merges_freed_buddies():
    allocator = SlotBuddyAllocator(
        data_base_offset=0,
        min_subslot_bytes=1024,
        slot_bytes=4096,
        managed_bytes=4096,
    )

    first = allocator.allocate(512)
    second = allocator.allocate(512)
    assert first.offset == 0
    assert second.offset == 1024

    assert allocator.free_offset(first.offset) == 1024
    assert allocator.free_offset(second.offset) == 1024

    merged = allocator.allocate(1500)
    assert merged.offset == 0
    assert merged.allocated_bytes == 2048


def test_slot_buddy_allocator_fails_when_fragmented_without_matching_block():
    allocator = SlotBuddyAllocator(
        data_base_offset=0,
        min_subslot_bytes=1024,
        slot_bytes=4096,
        managed_bytes=4096,
    )
    allocations = [allocator.allocate(512) for _ in range(4)]

    assert allocator.free_offset(allocations[0].offset) == 1024
    assert allocator.free_offset(allocations[2].offset) == 1024

    with pytest.raises(RuntimeError, match="No suitable allocation block"):
        allocator.allocate(1500)


def test_slot_buddy_allocator_checkpoint_state_records_free_blocks():
    allocator = SlotBuddyAllocator(
        data_base_offset=4096,
        min_subslot_bytes=1024,
        slot_bytes=4096,
        managed_bytes=8192,
    )
    allocation = allocator.allocate(512)
    assert allocator.free_offset(allocation.offset) == 1024

    state = allocator.checkpoint_state()

    assert state["min_subslot_bytes"] == 1024
    assert state["managed_bytes"] == 8192
    assert state["free_blocks"] == {"0": [], "1": [], "2": [4096, 8192]}


def test_slot_buddy_allocator_rejects_overlapping_checkpoint_partition():
    with pytest.raises(ValueError, match="partition overlaps"):
        SlotBuddyAllocator(
            data_base_offset=0,
            min_subslot_bytes=1024,
            slot_bytes=4096,
            managed_bytes=4096,
            free_blocks={2: {0}},
            allocated_blocks={0: 1024},
        )


def test_slot_buddy_allocator_rejects_incomplete_checkpoint_partition():
    with pytest.raises(ValueError, match="partition has gaps"):
        SlotBuddyAllocator(
            data_base_offset=0,
            min_subslot_bytes=1024,
            slot_bytes=4096,
            managed_bytes=4096,
            free_blocks={0: {0}, 1: {2048}},
            allocated_blocks={},
        )
