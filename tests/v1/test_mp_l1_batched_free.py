# SPDX-License-Identifier: Apache-2.0

# Standard
from concurrent.futures import ThreadPoolExecutor
import threading

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.memory_allocators.tensor_memory_allocator import TensorMemoryAllocator
from lmcache.v1.memory_management import AddressManager


def test_free_fragmented_ranges_and_reallocate() -> None:
    allocator = TensorMemoryAllocator(torch.empty(4096 * 64, dtype=torch.uint8))
    objs = allocator.batched_allocate(torch.Size([64]), torch.uint8, 64)
    assert objs is not None
    allocator.batched_free(objs[::2])
    assert allocator.total_allocated_size == 32 * 4096
    allocator.batched_free(objs[1::2])
    assert allocator.total_allocated_size == 0
    again = allocator.batched_allocate(torch.Size([64]), torch.uint8, 64)
    assert again is not None and len(again) == 64
    allocator.batched_free(again)


def test_concurrent_batch_free_and_allocate() -> None:
    manager = AddressManager(4096 * 512)
    active: set[int] = set()
    active_lock = threading.Lock()
    allocated_barrier = threading.Barrier(8)

    def worker() -> None:
        for _ in range(100):
            blocks = manager.batched_allocate(4096, 16)
            addresses = {address for address, _ in blocks}
            with active_lock:
                assert len(addresses) == len(blocks)
                assert active.isdisjoint(addresses)
                active.update(addresses)
            allocated_barrier.wait(timeout=10)
            with active_lock:
                active.difference_update(addresses)
            manager.batched_free(blocks[::2])
            manager.batched_free(blocks[1::2])

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(lambda _: worker(), range(8)))
    assert manager.get_free_size() == 4096 * 512
    assert len(manager.batched_allocate(4096, 512)) == 512


class RecordingAddressManager(AddressManager):
    """An extension that records release requests through its public hook."""

    def __init__(self, size: int) -> None:
        super().__init__(size)
        self.released: list[tuple[int, int]] = []
        self.reject_address: int | None = None

    def free(self, address: int, size: int) -> None:
        if address == self.reject_address:
            raise ValueError("release rejected by extension")
        self.released.append((address, size))
        super().free(address, size)


def test_batch_release_preserves_extension_dispatch() -> None:
    manager = RecordingAddressManager(4096 * 8)
    blocks = manager.batched_allocate(4096, 8)
    order = blocks[::2] + blocks[1::2]
    manager.batched_free(order)
    assert manager.released == order
    assert manager.check_consistency()
    assert manager.get_free_size() == manager.get_heap_size()


def test_extension_rejection_preserves_partial_release() -> None:
    manager = RecordingAddressManager(4096 * 8)
    blocks = manager.batched_allocate(4096, 8)
    manager.reject_address = blocks[2][0]
    with pytest.raises(ValueError, match="release rejected"):
        manager.batched_free(blocks)
    assert manager.released == blocks[:2]
    assert manager.total_allocated_size == 4096 * 6
    assert manager.check_consistency()
    manager.reject_address = None
    manager.batched_free(blocks[2:])
    assert manager.get_free_size() == manager.get_heap_size()


def test_release_reversed_unequal_ranges() -> None:
    manager = AddressManager(4096 * 12)
    blocks = [manager.allocate(4096 * count) for count in (1, 3, 2, 6)]
    manager.batched_free(list(reversed(blocks)))
    assert manager.check_consistency()
    assert manager.allocate(4096 * 12) == (0, 4096 * 12)
