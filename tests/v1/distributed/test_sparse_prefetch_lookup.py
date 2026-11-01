# SPDX-License-Identifier: Apache-2.0

# Standard
from unittest.mock import MagicMock

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import (
    GroupedObjectKeys,
    MemoryLayoutDesc,
    MemoryResidency,
    MemorySegment,
    MemoryTags,
    ObjectKey,
    PrefetchLockMode,
    PrefetchTaskSpec,
)
from lmcache.v1.distributed.config import L1ManagerConfig, L1MemoryManagerConfig
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.l2_adapters.fs_native_l2_adapter import (
    FSNativeL2AdapterConfig,
)
from lmcache.v1.distributed.l2_adapters.native_connector_l2_adapter import (
    NativeConnectorL2Adapter,
)
from lmcache.v1.distributed.storage_controllers.prefetch_controller import (
    PrefetchController,
)
from lmcache.v1.distributed.storage_controllers.prefetch_policy import (
    DefaultPrefetchPolicy,
)
from lmcache.v1.distributed.storage_controllers.utils import (
    L1ManagerDescriptor,
    L2AdapterDescriptor,
)
from tests.v1.distributed.test_native_connector_l2_adapter import MockNativeConnector


# These helpers use real L1 locking and allocation without requiring pinned GPU memory.
from lmcache.v1.distributed.l1_manager.base import BaseL1Manager
from lmcache.v1.distributed.memory_manager.l1_memory_manager import L1MemoryManager
from lmcache.v1.distributed.api import L1BackendType
from lmcache.v1.memory_allocators.tensor_memory_allocator import TensorMemoryAllocator
from lmcache.v1.memory_management import BytesBufferMemoryObj
from tests.v1.distributed.test_native_connector_l2_adapter import wait_for_event_fd


class RecordingClient(MockNativeConnector):
    def __init__(self) -> None:
        super().__init__()
        self.lookup_keys: list[list[str]] = []

    def submit_batch_exists(self, keys: list[str]) -> int:
        self.lookup_keys.append(list(keys))
        return super().submit_batch_exists(keys)


class RecordingAdapter(NativeConnectorL2Adapter):
    def __init__(self, native_client: object, memory_tags: MemoryTags) -> None:
        super().__init__(native_client, memory_tags)
        self.unlocked_keys: list[ObjectKey] = []

    def submit_unlock(self, keys: list[ObjectKey]) -> None:
        self.unlocked_keys.extend(keys)
        super().submit_unlock(keys)


class FullLookupAdapter(RecordingAdapter):
    def supports_sparse_lookup(self) -> bool:
        return False


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("fetching_policy", ["prefix", "full"])
def test_partial_l1_lookup_preserves_grid_and_locks(
    sparse: bool, fetching_policy: str
) -> None:
    client = RecordingClient()
    tags = MemoryTags(MemorySegment.STORAGE, MemoryResidency.LOCAL)
    adapter_cls = RecordingAdapter if sparse else FullLookupAdapter
    adapter = adapter_cls(client, tags)
    layout = MemoryLayoutDesc([torch.Size([8])], [torch.uint8])
    rows = [
        [
            ObjectKey(
                chunk_hash=bytes([i]), model_name="m", kv_rank=0, object_group_id=gid
            )
            for i in range(4)
        ]
        for gid in range(2)
    ]
    # The two rows have unequal missing subsets, without changing the public grid.
    held = {rows[0][0], rows[0][1], rows[1][0], rows[1][2]}
    manager = MagicMock()
    manager.get_memory_tags.return_value = MemoryTags(
        MemorySegment.DRAM, MemoryResidency.LOCAL
    )
    manager.get_memory_usage.return_value = (64, 4096 * 64)
    manager.get_staging_memory_usage.return_value = 0
    manager.reserve_read.side_effect = lambda keys, count: {
        key: (L1Error.SUCCESS if key in held else L1Error.KEY_NOT_EXIST, None)
        for key in keys
    }
    config = L1ManagerConfig(L1MemoryManagerConfig(4096 * 64, False))
    controller = PrefetchController(
        [manager],
        [L1ManagerDescriptor(0, config)],
        [adapter],
        [L2AdapterDescriptor(0, FSNativeL2AdapterConfig("/tmp"))],
        DefaultPrefetchPolicy(),
    )
    controller.start()
    try:
        spec = PrefetchTaskSpec(
            key_groups=[
                GroupedObjectKeys(keys=row, object_group_id=gid, layout_desc=layout)
                for gid, row in enumerate(rows)
            ],
            fetching_policy=fetching_policy,
            lock_mode=PrefetchLockMode.NO_LOCK,
        )
        ident = controller.submit_prefetch_request(spec)
        assert controller.wait_prefetch_result(ident, 5)
        result = controller.query_prefetch_result(ident)
        assert result is not None
        assert len(client.lookup_keys) == 1
        assert len(client.lookup_keys[0]) == (4 if sparse else 8)
        # Only the first column is a complete hit, in either fetching mode.
        assert [bitmap.get_indices_list() for bitmap in result.hit_cells] == [[0], [0]]
        assert (
            sum(len(call.args[0]) for call in manager.finish_read.call_args_list) == 4
        )
    finally:
        controller.stop()
        adapter.close()


class CpuMemory(L1MemoryManager):
    def __init__(self) -> None:
        self._allocator = TensorMemoryAllocator(
            torch.empty(4096 * 64, dtype=torch.uint8)
        )
        self._size_in_bytes = 4096 * 64
        self._align_bytes = 4096


class CpuL1(BaseL1Manager):
    @property
    def backend_type(self) -> L1BackendType:
        return L1BackendType.DRAM

    def get_memory_tags(self) -> MemoryTags:
        return MemoryTags(MemorySegment.DRAM, MemoryResidency.LOCAL)


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("fetching_policy", ["prefix", "full"])
@pytest.mark.parametrize("l2_missing", [False, True])
def test_sparse_lookup_loads_missing_cells_and_preserves_payload(
    sparse: bool, fetching_policy: str, l2_missing: bool
) -> None:
    client = RecordingClient()
    adapter_cls = RecordingAdapter if sparse else FullLookupAdapter
    adapter = adapter_cls(
        client, MemoryTags(MemorySegment.STORAGE, MemoryResidency.LOCAL)
    )
    layout = MemoryLayoutDesc([torch.Size([8])], [torch.uint8])
    rows = [
        [
            ObjectKey(
                chunk_hash=bytes([i]), model_name="load", kv_rank=0, object_group_id=gid
            )
            for i in range(4)
        ]
        for gid in range(2)
    ]
    keys = [key for row in rows for key in row]
    config = L1ManagerConfig(L1MemoryManagerConfig(4096 * 64, False))
    manager = CpuL1(config, CpuMemory())
    # Unequal missing positions in each group exercise sparse-to-grid scatter.
    existing = [rows[0][0], rows[0][1], rows[1][0], rows[1][2]]
    reserved = manager.reserve_write(existing, [False] * len(existing), layout)
    for key, (error, obj) in reserved.items():
        assert error == L1Error.SUCCESS
        obj.raw_data.fill_(keys.index(key) + 1)
    manager.finish_write(existing)
    sources = [BytesBufferMemoryObj(bytes([i + 1]) * 8) for i in range(8)]
    store_keys = keys[:-1] if l2_missing else keys
    store_sources = sources[:-1] if l2_missing else sources
    store = adapter.submit_store_task(store_keys, store_sources)
    assert wait_for_event_fd(adapter.get_store_event_fd())
    assert adapter.pop_completed_store_tasks()[store].is_successful()
    controller = PrefetchController(
        [manager],
        [L1ManagerDescriptor(0, config)],
        [adapter],
        [L2AdapterDescriptor(0, FSNativeL2AdapterConfig("/tmp"))],
        DefaultPrefetchPolicy(),
    )
    controller.start()
    try:
        spec = PrefetchTaskSpec(
            key_groups=[
                GroupedObjectKeys(keys=row, object_group_id=gid, layout_desc=layout)
                for gid, row in enumerate(rows)
            ],
            fetching_policy=fetching_policy,
            lock_mode=PrefetchLockMode.NO_LOCK,
        )
        ident = controller.submit_prefetch_request(spec)
        assert controller.wait_prefetch_result(ident, 5)
        result = controller.query_prefetch_result(ident)
        assert result is not None
        expected_count = 3 if l2_missing else 4
        assert [bitmap.get_indices_list() for bitmap in result.hit_cells] == [
            list(range(expected_count))
        ] * 2
        assert len(client.lookup_keys[0]) == (4 if sparse else 8)
        read_keys = [key for row in rows for key in row[:expected_count]]
        objects = manager.reserve_read(read_keys)
        for key in read_keys:
            i = keys.index(key)
            error, obj = objects[key]
            assert error == L1Error.SUCCESS
            assert bytes(obj.byte_array) == bytes([i + 1]) * 8
        manager.finish_read(read_keys)
        expected_unlocks = set(store_keys) - (set(existing) if sparse else set())
        assert len(adapter.unlocked_keys) == len(expected_unlocks)
        assert set(adapter.unlocked_keys) == expected_unlocks
    finally:
        controller.stop()
        adapter.close()
        manager.close()


def test_existing_custom_subclasses_keep_full_lookup() -> None:
    class CustomPolicy(DefaultPrefetchPolicy):
        def plan_load(self, *args, **kwargs):
            return super().plan_load(*args, **kwargs)

    class CustomAdapter(NativeConnectorL2Adapter):
        def submit_lookup_and_lock_task(self, *args, **kwargs):
            return super().submit_lookup_and_lock_task(*args, **kwargs)

    assert not CustomPolicy().prefers_l1_hits()
    # The capability only checks the public method, so no running client is needed.
    adapter = CustomAdapter.__new__(CustomAdapter)
    assert not adapter.supports_sparse_lookup()
