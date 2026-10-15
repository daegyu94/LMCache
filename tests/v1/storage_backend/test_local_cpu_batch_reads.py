# SPDX-License-Identifier: Apache-2.0
# Standard
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
import threading

# Third Party
import pytest
import torch

# First Party
from lmcache.observability import LMCStatsMonitor
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_allocators.tensor_memory_allocator import TensorMemoryAllocator
from lmcache.v1.memory_management import TensorMemoryObj
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend


@pytest.fixture(params=["LRU", "LFU", "FIFO", "MRU"])
def batch_cache(
    request: pytest.FixtureRequest,
) -> Iterator[tuple[LocalCPUBackend, list[CacheEngineKey], list[TensorMemoryObj]]]:
    """Provide resident objects backed by a real CPU allocation pool."""
    config = LMCacheEngineConfig.from_defaults(
        local_cpu=True, cache_policy=request.param
    )
    allocator = TensorMemoryAllocator(torch.empty(3 * 4096, dtype=torch.uint8))
    backend = LocalCPUBackend(config, memory_allocator=allocator)
    keys = [CacheEngineKey("batch-test", 1, 0, i, torch.float32) for i in range(3)]
    objects = []
    for key in keys:
        obj = allocator.allocate(torch.Size([2, 1, 8, 1]), torch.float32)
        assert obj is not None
        backend.submit_put_task(key, obj)
        obj.ref_count_down()
        objects.append(obj)
    yield backend, keys, objects
    backend.clear()
    assert allocator.memcheck()
    LMCStatsMonitor.DestroyInstance()


def test_batched_reads_empty(
    batch_cache: tuple[LocalCPUBackend, list[CacheEngineKey], list[TensorMemoryObj]],
) -> None:
    backend, _, objects = batch_cache
    assert backend.batched_get_blocking([]) == []
    assert all(obj.get_ref_count() == 1 and not obj.is_pinned for obj in objects)


def test_batched_get_preserves_misses_order_and_duplicate_references(
    batch_cache: tuple[LocalCPUBackend, list[CacheEngineKey], list[TensorMemoryObj]],
) -> None:
    backend, keys, objects = batch_cache
    missing = CacheEngineKey("batch-test", 1, 0, 99, torch.float32)
    results = backend.batched_get_blocking(
        [missing, keys[2], keys[0], missing, keys[2]]
    )
    assert results == [None, objects[2], objects[0], None, objects[2]]
    assert [obj.get_ref_count() for obj in objects] == [2, 1, 3]
    for obj in results:
        if obj is not None:
            obj.ref_count_down()
    assert all(obj.get_ref_count() == 1 for obj in objects)


def test_batched_get_retains_objects_during_concurrent_remove(
    batch_cache: tuple[LocalCPUBackend, list[CacheEngineKey], list[TensorMemoryObj]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend, keys, objects = batch_cache
    acquiring_reference = threading.Event()
    finish_reference = threading.Event()
    removing = threading.Event()
    removed = threading.Event()
    ref_count_up = objects[0].ref_count_up

    def pause_reference() -> None:
        acquiring_reference.set()
        assert finish_reference.wait(timeout=5)
        ref_count_up()

    def remove_second() -> bool:
        removing.set()
        result = backend.remove(keys[1])
        removed.set()
        return result

    monkeypatch.setattr(objects[0], "ref_count_up", pause_reference)
    with ThreadPoolExecutor(max_workers=2) as executor:
        reading = executor.submit(backend.batched_get_blocking, keys[:2])
        assert acquiring_reference.wait(timeout=5)
        removal = executor.submit(remove_second)
        try:
            assert removing.wait(timeout=5)
            assert not removed.wait(timeout=0.05)
        finally:
            finish_reference.set()
        results = reading.result(timeout=5)
        assert removal.result(timeout=5)
    assert results == objects[:2]
    assert objects[1].is_valid()
    assert objects[1].get_ref_count() == 1
    for obj in results:
        assert obj is not None
        obj.ref_count_down()
    assert not objects[1].is_valid()
