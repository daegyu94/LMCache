# SPDX-License-Identifier: Apache-2.0
"""Integration tests for the filesystem ObjectKey codec."""

# Standard
from pathlib import Path
from typing import Any
import select
import time

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.l2_adapters import create_l2_adapter
from lmcache.v1.distributed.l2_adapters.fs_l2_adapter import (
    FSL2AdapterConfig,
)
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObjMetadata,
    TensorMemoryObj,
)
from lmcache.v1.platform import consume_fd

pytest.importorskip("lmcache.lmcache_native")


def _memory_obj(data: torch.Tensor) -> TensorMemoryObj:
    """Wrap one contiguous byte tensor for filesystem adapter I/O."""
    metadata = MemoryObjMetadata(
        shape=data.shape,
        dtype=data.dtype,
        address=data.data_ptr(),
        phy_size=data.numel(),
        ref_count=1,
        fmt=MemoryFormat.BINARY,
    )
    return TensorMemoryObj(data, metadata, parent_allocator=None)


def _wait_for_store(adapter: Any, task_id: int) -> Any:
    """Wait for one filesystem adapter store result."""
    poller = select.poll()
    event_fd = adapter.get_store_event_fd()
    poller.register(event_fd, select.POLLIN)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        for _fd, _events in poller.poll(100):
            try:
                consume_fd(event_fd)
            except BlockingIOError:
                pass
        result = adapter.pop_completed_store_tasks().get(task_id)
        if result is not None:
            return result
    raise TimeoutError("native store did not complete")


def _wait_for_load(adapter: Any, task_id: int) -> Any:
    """Wait for one filesystem adapter load result."""
    poller = select.poll()
    event_fd = adapter.get_load_event_fd()
    poller.register(event_fd, select.POLLIN)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        for _fd, _events in poller.poll(100):
            try:
                consume_fd(event_fd)
            except BlockingIOError:
                pass
        result = adapter.query_load_result(task_id)
        if result is not None:
            return result
    raise TimeoutError("filesystem load did not complete")


def test_salted_object_group_key_round_trips_through_fs(
    tmp_path: Path,
) -> None:
    """The fs adapter stores and loads the current five-field key shape."""
    adapter = create_l2_adapter(
        FSL2AdapterConfig(
            base_path=str(tmp_path),
            use_odirect=False,
        )
    )
    key = ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(12345),
        model_name="benchmark/model",
        kv_rank=2,
        object_group_id=7,
        cache_salt="tenant-a",
    )
    source = torch.arange(4096, dtype=torch.int32).view(torch.uint8)
    destination = torch.zeros_like(source)
    try:
        store_result = _wait_for_store(
            adapter,
            adapter.submit_store_task([key], [_memory_obj(source)]),
        )
        assert store_result.is_successful()
        expected_name = (
            f"benchmark-SEP-model@0x{key.kv_rank:08x}"
            f"@{key.object_group_id:x}@{key.chunk_hash.hex()}@{key.cache_salt}.data"
        )
        assert (tmp_path / expected_name).is_file()

        load_result = _wait_for_load(
            adapter,
            adapter.submit_load_task([key], [_memory_obj(destination)]),
        )
        assert load_result.popcount() == 1
        assert torch.equal(destination, source)
    finally:
        adapter.close()
