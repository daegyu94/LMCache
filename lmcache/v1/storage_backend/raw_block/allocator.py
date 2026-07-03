# SPDX-License-Identifier: Apache-2.0

# Future
from __future__ import annotations

# Standard
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RawBlockAllocation:
    """Allocated raw-block byte range."""

    offset: int
    allocated_bytes: int


class RawBlockAllocator(ABC):
    """Allocator interface used by RawBlockCore."""

    @abstractmethod
    def allocate(self, required_bytes: int) -> RawBlockAllocation:
        """Allocate a raw-block byte range for ``required_bytes``."""

    @abstractmethod
    def free_offset(self, offset: int) -> int:
        """Recycle the allocation starting at ``offset`` and return freed bytes."""

    @abstractmethod
    def mark_allocated(self, offset: int, allocated_bytes: int) -> None:
        """Record an already-allocated range during checkpoint recovery."""

    @abstractmethod
    def allocation_size(self, offset: int) -> int:
        """Return allocated bytes for an active allocation offset."""

    @abstractmethod
    def is_valid_allocation(self, offset: int, allocation_bytes: int) -> bool:
        """Return whether an allocation belongs to this allocator layout."""

    @abstractmethod
    def usable_capacity_bytes(self) -> int:
        """Return total allocator-managed bytes."""

    @abstractmethod
    def allocated_bytes(self, indexed_count: int, inflight_count: int) -> int:
        """Return allocated bytes for current indexed and in-flight counts."""

    @abstractmethod
    def checkpoint_state(self) -> dict[str, Any]:
        """Return allocator checkpoint fields."""


class FixedSlotAllocator(RawBlockAllocator):
    """Fixed-size raw-block slot allocator preserving fixed-slot semantics."""

    def __init__(
        self,
        *,
        data_base_offset: int,
        slot_bytes: int,
        max_slots: int,
        next_slot: int = 0,
        free_slots: list[int] | None = None,
    ) -> None:
        """Create a fixed-slot allocator.

        Args:
            data_base_offset: Byte offset where data slots begin.
            slot_bytes: Size of one fixed raw-block slot.
            max_slots: Total number of allocatable slots.
            next_slot: First never-allocated slot index.
            free_slots: Reusable slot indexes.
        """
        self.data_base_offset = int(data_base_offset)
        self.slot_bytes = int(slot_bytes)
        self.max_slots = int(max_slots)
        self.next_slot = int(next_slot)
        self.free_slots = list(free_slots or [])

    def allocate(self, required_bytes: int) -> RawBlockAllocation:
        """Allocate one fixed slot.

        Args:
            required_bytes: Required bytes for the caller's payload and header.

        Returns:
            Raw block allocation covering one fixed slot.

        Raises:
            RuntimeError: If the request cannot fit or no slot is available.
        """
        if int(required_bytes) > self.slot_bytes:
            raise RuntimeError("requested bytes exceed fixed slot size")
        if self.free_slots:
            slot = self.free_slots.pop()
            return RawBlockAllocation(
                offset=self.slot_to_offset(slot),
                allocated_bytes=self.slot_bytes,
            )
        if self.next_slot < self.max_slots:
            slot = self.next_slot
            self.next_slot += 1
            return RawBlockAllocation(
                offset=self.slot_to_offset(slot),
                allocated_bytes=self.slot_bytes,
            )
        raise RuntimeError("No free slots available")

    def free_offset(self, offset: int) -> int:
        """Recycle an allocation by offset.

        Args:
            offset: Byte offset previously returned by ``allocate``.

        Returns:
            Number of bytes made available for reuse.
        """
        if not self.is_valid_allocation(int(offset), self.slot_bytes):
            return 0
        slot = self.offset_to_slot(int(offset))
        self.free_slot(slot)
        return self.slot_bytes

    def mark_allocated(self, offset: int, allocated_bytes: int) -> None:
        """Validate an already-allocated fixed slot during checkpoint recovery."""
        if not self.is_valid_allocation(int(offset), int(allocated_bytes)):
            raise ValueError("invalid fixed slot allocation")

    def allocation_size(self, offset: int) -> int:
        """Return fixed allocation bytes for an offset."""
        if not self.is_valid_allocation(int(offset), self.slot_bytes):
            return 0
        return self.slot_bytes

    def is_valid_allocation(self, offset: int, allocation_bytes: int) -> bool:
        """Return whether an allocation belongs to this fixed-slot layout."""
        if int(allocation_bytes) != self.slot_bytes:
            return False
        rel = int(offset) - self.data_base_offset
        if rel < 0 or rel % self.slot_bytes != 0:
            return False
        slot = rel // self.slot_bytes
        return 0 <= slot < self.max_slots

    def free_slot(self, slot: int) -> None:
        """Recycle a slot index if it belongs to the managed range."""
        if slot < 0 or slot >= self.max_slots:
            return
        if slot in self.free_slots:
            return
        self.free_slots.append(slot)

    def slot_to_offset(self, slot: int) -> int:
        """Convert a data-slot index to its byte offset."""
        return self.data_base_offset + int(slot) * self.slot_bytes

    def offset_to_slot(self, offset: int) -> int:
        """Convert a data-slot byte offset to its slot index."""
        return (int(offset) - self.data_base_offset) // self.slot_bytes

    def allocated_bytes(self, indexed_count: int, inflight_count: int) -> int:
        """Return allocated bytes for current indexed and in-flight counts."""
        return (int(indexed_count) + int(inflight_count)) * self.slot_bytes

    def usable_capacity_bytes(self) -> int:
        """Return total allocator-managed bytes."""
        return self.max_slots * self.slot_bytes

    def checkpoint_state(self) -> dict[str, Any]:
        """Return fixed-slot allocator checkpoint fields."""
        return {
            "next_slot": self.next_slot,
            "free_slots": list(self.free_slots),
        }


class SlotBuddyAllocator(RawBlockAllocator):
    """Buddy allocator that splits fixed-size slots into raw-block byte ranges."""

    def __init__(
        self,
        *,
        data_base_offset: int,
        min_subslot_bytes: int,
        slot_bytes: int,
        managed_bytes: int,
        free_blocks: dict[int, set[int]] | None = None,
        allocated_blocks: dict[int, int] | None = None,
    ) -> None:
        """Create a buddy allocator.

        Args:
            data_base_offset: Byte offset where managed data begins.
            min_subslot_bytes: Smallest power-of-two subslot size.
            slot_bytes: Fixed slot size and largest allocation size.
            managed_bytes: Total managed data bytes.
            free_blocks: Optional free-list state keyed by buddy order.
            allocated_blocks: Optional active allocations keyed by offset.

        Raises:
            ValueError: If allocation sizes or checkpoint state are invalid.
        """
        self.data_base_offset = int(data_base_offset)
        self.min_subslot_bytes = int(min_subslot_bytes)
        self.slot_bytes = int(slot_bytes)
        self.managed_bytes = int(managed_bytes)
        self._validate_config()
        self.max_order = self.order_for_allocation_bytes(self.slot_bytes)
        self.free_blocks = self._copy_free_blocks(free_blocks)
        self._allocations: dict[int, int] = {}
        for offset, allocated_bytes in (allocated_blocks or {}).items():
            self.mark_allocated(int(offset), int(allocated_bytes))
        if free_blocks is not None or allocated_blocks is not None:
            self._validate_partition()

    def allocate(self, required_bytes: int) -> RawBlockAllocation:
        """Allocate the smallest power-of-two block that fits required bytes."""
        allocation_bytes = self.allocation_bytes_for_required(int(required_bytes))
        order = self.order_for_allocation_bytes(allocation_bytes)

        source_order = None
        for candidate in range(order, self.max_order + 1):
            if self.free_blocks.get(candidate):
                source_order = candidate
                break
        if source_order is None:
            raise RuntimeError("No suitable allocation block available")

        offset = min(self.free_blocks[source_order])
        self.free_blocks[source_order].remove(offset)
        while source_order > order:
            source_order -= 1
            buddy = offset + (self.min_subslot_bytes << source_order)
            self.free_blocks[source_order].add(buddy)

        self._allocations[offset] = allocation_bytes
        return RawBlockAllocation(offset=offset, allocated_bytes=allocation_bytes)

    def free_offset(self, offset: int) -> int:
        """Recycle an allocated block and merge free buddies when possible."""
        offset = int(offset)
        allocation_bytes = self._allocations.pop(offset, 0)
        if allocation_bytes <= 0:
            return 0
        freed_bytes = allocation_bytes

        order = self.order_for_allocation_bytes(allocation_bytes)
        rel = offset - self.data_base_offset
        while order < self.max_order:
            buddy_rel = rel ^ allocation_bytes
            buddy = self.data_base_offset + buddy_rel
            free_set = self.free_blocks.setdefault(order, set())
            if buddy not in free_set:
                break
            free_set.remove(buddy)
            rel = min(rel, buddy_rel)
            allocation_bytes *= 2
            order += 1

        self.free_blocks.setdefault(order, set()).add(self.data_base_offset + rel)
        return freed_bytes

    def mark_allocated(self, offset: int, allocated_bytes: int) -> None:
        """Record an active allocation from checkpoint metadata."""
        if not self.is_valid_allocation(offset, allocated_bytes):
            raise ValueError("invalid slot buddy allocation")
        if offset in self._allocations:
            raise ValueError("duplicate slot buddy allocation")
        self._allocations[int(offset)] = int(allocated_bytes)

    def allocation_size(self, offset: int) -> int:
        """Return allocated bytes for an active allocation offset."""
        return int(self._allocations.get(int(offset), 0))

    def allocated_bytes(self, indexed_count: int, inflight_count: int) -> int:
        """Return live allocated block bytes."""
        return sum(int(size) for size in self._allocations.values())

    def usable_capacity_bytes(self) -> int:
        """Return total allocator-managed bytes."""
        return self.managed_bytes

    def checkpoint_state(self) -> dict[str, Any]:
        """Return buddy allocator checkpoint fields."""
        return {
            "min_subslot_bytes": self.min_subslot_bytes,
            "managed_bytes": self.managed_bytes,
            "free_blocks": {
                str(order): sorted(offsets)
                for order, offsets in sorted(self.free_blocks.items())
            },
        }

    def allocation_bytes_for_required(self, required_bytes: int) -> int:
        """Return the smallest configured allocation class for required bytes."""
        allocation_bytes = self.min_subslot_bytes
        while allocation_bytes < required_bytes:
            allocation_bytes *= 2
        if allocation_bytes > self.slot_bytes:
            raise RuntimeError("No allocation size can satisfy requested bytes")
        return allocation_bytes

    def order_for_allocation_bytes(self, allocation_bytes: int) -> int:
        """Return buddy order for an allocation size."""
        if not _is_power_of_two(int(allocation_bytes)):
            raise ValueError("allocation_bytes must be a power of two")
        if (
            allocation_bytes < self.min_subslot_bytes
            or allocation_bytes > self.slot_bytes
        ):
            raise ValueError("allocation_bytes is outside configured range")
        return (int(allocation_bytes) // self.min_subslot_bytes).bit_length() - 1

    def is_valid_allocation(self, offset: int, allocation_bytes: int) -> bool:
        """Return whether an allocation belongs to this allocator layout."""
        if not _is_power_of_two(int(allocation_bytes)):
            return False
        if (
            allocation_bytes < self.min_subslot_bytes
            or allocation_bytes > self.slot_bytes
        ):
            return False
        if offset < self.data_base_offset:
            return False
        rel = int(offset) - self.data_base_offset
        if rel % int(allocation_bytes) != 0:
            return False
        return rel + int(allocation_bytes) <= self.managed_bytes

    def _validate_config(self) -> None:
        if self.min_subslot_bytes <= 0 or self.slot_bytes <= 0:
            raise ValueError("allocation sizes must be > 0")
        if not _is_power_of_two(self.min_subslot_bytes):
            raise ValueError("min_subslot_bytes must be a power of two")
        if not _is_power_of_two(self.slot_bytes):
            raise ValueError("slot_bytes must be a power of two")
        if self.min_subslot_bytes > self.slot_bytes:
            raise ValueError("min_subslot_bytes must be <= slot_bytes")
        if self.managed_bytes <= 0 or self.managed_bytes % self.slot_bytes != 0:
            raise ValueError("managed_bytes must be a positive slot_bytes multiple")

    def _copy_free_blocks(
        self,
        free_blocks: dict[int, set[int]] | None,
    ) -> dict[int, set[int]]:
        copied: dict[int, set[int]] = {
            order: set() for order in range(self.max_order + 1)
        }
        if free_blocks is None:
            for rel in range(0, self.managed_bytes, self.slot_bytes):
                copied[self.max_order].add(self.data_base_offset + rel)
            return copied

        for order, offsets in free_blocks.items():
            if int(order) < 0 or int(order) > self.max_order:
                raise ValueError("free block order out of range")
            allocation_bytes = self.min_subslot_bytes << int(order)
            for offset in offsets:
                if not self.is_valid_allocation(int(offset), allocation_bytes):
                    raise ValueError("invalid free block")
                copied[int(order)].add(int(offset))
        return copied

    def _validate_partition(self) -> None:
        ranges: list[tuple[int, int]] = []
        for offset, allocated_bytes in self._allocations.items():
            ranges.append((int(offset), int(offset) + int(allocated_bytes)))
        for order, offsets in self.free_blocks.items():
            allocation_bytes = self.min_subslot_bytes << int(order)
            for offset in offsets:
                ranges.append((int(offset), int(offset) + allocation_bytes))

        ranges.sort()
        previous_end = self.data_base_offset
        for start, end in ranges:
            if start < self.data_base_offset:
                raise ValueError("slot buddy partition starts before managed range")
            if end > self.data_base_offset + self.managed_bytes:
                raise ValueError("slot buddy partition exceeds managed range")
            if start < previous_end:
                raise ValueError("slot buddy partition overlaps")
            if start > previous_end:
                raise ValueError("slot buddy partition has gaps")
            previous_end = end
        if previous_end != self.data_base_offset + self.managed_bytes:
            raise ValueError("slot buddy partition does not cover managed range")


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0
