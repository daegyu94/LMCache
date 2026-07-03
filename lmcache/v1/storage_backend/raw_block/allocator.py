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
    def usable_capacity_bytes(self) -> int:
        """Return total allocator-managed bytes."""

    @abstractmethod
    def allocated_bytes(self, indexed_count: int, inflight_count: int) -> int:
        """Return allocated bytes for current indexed and in-flight counts."""

    @abstractmethod
    def checkpoint_state(self) -> dict[str, Any]:
        """Return allocator checkpoint fields."""


class FixedSlotAllocator(RawBlockAllocator):
    """Fixed-size raw-block slot allocator preserving v1 allocation semantics."""

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
        slot = self.offset_to_slot(int(offset))
        self.free_slot(slot)
        return self.slot_bytes

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
        """Return v1 fixed-slot allocator checkpoint fields."""
        return {
            "next_slot": self.next_slot,
            "free_slots": list(self.free_slots),
        }
