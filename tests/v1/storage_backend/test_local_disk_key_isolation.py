# SPDX-License-Identifier: Apache-2.0
# Standard
from collections.abc import Iterator
from pathlib import Path
import asyncio
import threading

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_allocators.tensor_memory_allocator import TensorMemoryAllocator
from lmcache.v1.memory_management import MemoryFormat
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
from lmcache.v1.storage_backend.local_disk_backend import LocalDiskBackend

pytestmark = pytest.mark.no_shared_allocator


@pytest.fixture
def disk_backend(tmp_path: Path) -> Iterator[LocalDiskBackend]:
    """Provide a real disk backend with a small CPU pool and running I/O loop."""
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever)
    thread.start()
    config = LMCacheEngineConfig.from_defaults(
        local_disk=str(tmp_path), max_local_disk_size=0.01
    )
    allocator = TensorMemoryAllocator(torch.empty(32768, dtype=torch.uint8))
    cpu_backend = LocalCPUBackend(config, memory_allocator=allocator, dst_device="cpu")
    backend = LocalDiskBackend(config, loop, cpu_backend, dst_device="cpu")
    try:
        yield backend
    finally:
        backend.close()
        cpu_backend.close()
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
        loop.close()


def store_chunk(backend: LocalDiskBackend, key: CacheEngineKey, value: int) -> None:
    """Store a real chunk and wait for its public completion callback.

    Args:
        backend: Disk backend receiving the chunk.
        key: Complete model and request-tag identity.
        value: Value to fill into the chunk's tensor.
    """
    memory_obj = backend.get_allocator_backend().allocate(
        torch.Size([2, 1, 8, 1]), torch.float32, MemoryFormat.KV_2LTD
    )
    assert memory_obj is not None
    completed = threading.Event()
    try:
        assert memory_obj.tensor is not None
        memory_obj.tensor.fill_(value)
        backend.submit_put_task(key, memory_obj, lambda _: completed.set())
        assert completed.wait(timeout=5), "disk write did not complete"
    finally:
        memory_obj.ref_count_down()


@pytest.mark.parametrize(
    "first_model, second_model, first_job, second_job",
    [
        ("org/model", "org-model", "job", "job"),
        ("model", "model", "run/1", "run-1"),
        ("model", "model", "run/1", "run%2F1"),
        ("모델" * 100, "모델" * 100, "작업/1", "작업-1"),
    ],
    ids=["model-slash", "tag-slash", "tag-percent", "long-unicode"],
)
def test_disk_keys_preserve_model_and_tag_isolation(
    disk_backend: LocalDiskBackend,
    tmp_path: Path,
    first_model: str,
    second_model: str,
    first_job: str,
    second_job: str,
) -> None:
    """Distinct model or job identities retain their own bytes and lifecycles."""
    keys = [
        CacheEngineKey(
            model_name=model,
            world_size=1,
            worker_id=0,
            chunk_hash=1,
            dtype=torch.float32,
            request_configs={"lmcache.tag.job": job},
        )
        for model, job in [(first_model, first_job), (second_model, second_job)]
    ]
    assert keys[0] != keys[1]
    for key, value in zip(keys, [11, 22], strict=True):
        store_chunk(disk_backend, key, value)

    for key, value in zip(keys, [11, 22], strict=True):
        loaded = disk_backend.get_blocking(key)
        assert loaded is not None
        try:
            assert loaded.tensor is not None
            assert torch.all(loaded.tensor == value)
        finally:
            loaded.ref_count_down()

    files = list(tmp_path.glob("*.pt"))
    assert len(files) == 2
    assert all(len(path.name.encode("utf-8")) <= 255 for path in files)
    assert disk_backend.remove(keys[0])
    assert not disk_backend.contains(keys[0])
    loaded = disk_backend.get_blocking(keys[1])
    assert loaded is not None
    try:
        assert loaded.tensor is not None
        assert torch.all(loaded.tensor == 22)
    finally:
        loaded.ref_count_down()
    assert disk_backend.remove(keys[1])
    assert not list(tmp_path.iterdir())
