# SPDX-License-Identifier: Apache-2.0
"""Admission-race regressions using real L1 state and native filesystem I/O.

Only pinned host allocation is replaced with CPU tensors; reservations, TTL
locks, staging ownership, controller planning, and filesystem I/O are real.
"""

# Standard
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch
import time

# Third Party
import pytest
import torch

# First Party
from lmcache import device_ops
from lmcache.lmcache_fs import LMCacheFSClient
from lmcache.v1.distributed.api import (
    FetchingPolicy,
    GroupedObjectKeys,
    MemoryLayoutDesc,
    ObjectKey,
    PrefetchLockMode,
    PrefetchTaskSpec,
)
from lmcache.v1.distributed.config import L1ManagerConfig, L1MemoryManagerConfig
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.internal_api import L1OperationResult
from lmcache.v1.distributed.l1_manager import L1Manager
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
    PrefetchPlan,
)
from lmcache.v1.distributed.storage_controllers.utils import (
    L1ManagerDescriptor,
    L2AdapterDescriptor,
    MapState,
)
from lmcache.v1.memory_management import MemoryObj

LAYOUT = MemoryLayoutDesc([torch.Size([4096])], [torch.uint8])


@pytest.fixture
def l1_config() -> L1ManagerConfig:
    """Return a fixed-size CPU arena configuration."""
    return L1ManagerConfig(
        memory_config=L1MemoryManagerConfig(
            size_in_bytes=1 << 20, use_lazy=False, align_bytes=4096
        ),
        read_ttl_seconds=300,
        write_ttl_seconds=300,
    )


@pytest.fixture
def l1(
    l1_config: L1ManagerConfig, monkeypatch: pytest.MonkeyPatch
) -> Iterator[L1Manager]:
    """Run the real manager with CPU-owned buffers instead of GPU pinning."""
    buffers: dict[int, torch.Tensor] = {}

    def allocate(size: int, flags: int) -> int:
        tensor = torch.empty(size, dtype=torch.uint8)
        buffers[tensor.data_ptr()] = tensor
        return tensor.data_ptr()

    def free(pointer: int) -> None:
        buffers.pop(pointer)

    monkeypatch.setattr(device_ops, "alloc_pinned_ptr", allocate)
    monkeypatch.setattr(device_ops, "free_pinned_ptr", free)
    manager = L1Manager(l1_config)
    try:
        yield manager
    finally:
        manager.close()


@pytest.fixture
def adapter(tmp_path: Path) -> Iterator[NativeConnectorL2Adapter]:
    """Create a real native filesystem adapter."""
    instance = NativeConnectorL2Adapter(LMCacheFSClient(str(tmp_path), 2))
    try:
        yield instance
    finally:
        instance.close()


def make_groups(rows: int = 4, columns: int = 3) -> list[GroupedObjectKeys]:
    """Build rows spanning object groups and KV ranks."""
    return [
        GroupedObjectKeys(
            [
                ObjectKey(i.to_bytes(8, "big"), "model", row // 2, row % 2)
                for i in range(columns)
            ],
            row % 2,
            LAYOUT,
        )
        for row in range(rows)
    ]


def stage(l1: L1Manager, keys: list[ObjectKey]) -> list[MemoryObj]:
    """Stage distinguishable payloads for later admission or L2 seeding."""
    results = l1.reserve_write(keys, [False] * len(keys), LAYOUT)
    objects = []
    for key in keys:
        error, obj = results[key]
        assert error == L1Error.SUCCESS and obj is not None
        assert obj.tensor is not None
        obj.tensor.fill_(int.from_bytes(key.chunk_hash, "big") + 1)
        objects.append(obj)
    return objects


def seed_adapter(
    l1: L1Manager, adapter: NativeConnectorL2Adapter, keys: list[ObjectKey]
) -> None:
    """Publish complete FS objects, then discard the temporary L1 staging."""
    objects = stage(l1, keys)
    store_id = adapter.submit_store_task(keys, objects)
    deadline = time.monotonic() + 5
    results = adapter.pop_completed_store_tasks()
    while not results:
        assert time.monotonic() < deadline
        time.sleep(0.005)
        results = adapter.pop_completed_store_tasks()
    assert len(results) == 1
    assert results[store_id].is_successful()
    l1.finish_write_and_delete(keys)


def make_controller(
    l1: L1Manager,
    config: L1ManagerConfig,
    adapter: NativeConnectorL2Adapter,
    tmp_path: Path,
    policy: DefaultPrefetchPolicy | None = None,
) -> PrefetchController:
    """Construct the public controller with one filesystem affinity tier."""
    return PrefetchController(
        [l1],
        [L1ManagerDescriptor(0, config)],
        [adapter],
        [L2AdapterDescriptor(0, FSNativeL2AdapterConfig(str(tmp_path)))],
        policy or DefaultPrefetchPolicy(),
    )


def admit_after_lookup(
    l1: L1Manager, keys: list[ObjectKey], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Admit a racing writer immediately after the initial L1 read check."""
    reserve_read = l1.reserve_read
    pending = True

    def read(
        keys_to_read: list[ObjectKey], read_locks: int = 1
    ) -> dict[ObjectKey, L1OperationResult]:
        nonlocal pending
        result = reserve_read(keys_to_read, read_locks)
        if pending:
            pending = False
            l1.finish_write(keys)
        return result

    monkeypatch.setattr(l1, "reserve_read", read)


@pytest.mark.parametrize("fetching_policy", ["prefix", "full"])
@pytest.mark.parametrize("readers", [1, 3])
@pytest.mark.parametrize("lock_mode", list(PrefetchLockMode))
def test_mixed_resident_recovered_and_loaded_columns(
    l1: L1Manager,
    l1_config: L1ManagerConfig,
    adapter: NativeConnectorL2Adapter,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fetching_policy: FetchingPolicy,
    readers: int,
    lock_mode: PrefetchLockMode,
) -> None:
    """Mixed columns remain complete and recovered cells are counted as L1 hits."""
    groups = make_groups()
    keys = [key for group in groups for key in group.keys]
    seed_adapter(l1, adapter, keys)

    original = [groups[0].keys[0], groups[3].keys[1]]
    recovered = [groups[1].keys[0], groups[2].keys[1], groups[0].keys[2]]
    stage(l1, original)
    l1.finish_write(original)
    stage(l1, recovered)
    admit_after_lookup(l1, recovered, monkeypatch)
    controller = make_controller(l1, l1_config, adapter, tmp_path)
    controller.start()
    try:
        with (
            patch.object(
                adapter, "submit_load_task", wraps=adapter.submit_load_task
            ) as load,
            patch.object(
                adapter, "submit_unlock", wraps=adapter.submit_unlock
            ) as unlock,
        ):
            request_id = controller.submit_prefetch_request(
                PrefetchTaskSpec(
                    groups,
                    readers,
                    fetching_policy=fetching_policy,
                    lock_mode=lock_mode,
                )
            )
            assert controller.wait_prefetch_result(request_id, 5)
            result = controller.query_prefetch_result(request_id)
            assert result is not None
            assert [row.get_indices_list() for row in result.hit_cells] == [
                [0, 1, 2]
            ] * 4
            l1_hits = {
                group.keys[col]
                for group, row in zip(groups, result.l1_hit_cells, strict=True)
                for col in row.get_indices_list()
            }
            assert l1_hits == set(original + recovered)
            loaded = [key for call in load.call_args_list for key in call.args[0]]
            assert set(loaded) == set(keys) - l1_hits
            unlocked = [key for call in unlock.call_args_list for key in call.args[0]]
            assert Counter(unlocked) == Counter(keys)
            assert l1.get_staging_memory_usage() == 0
            assert controller.query_prefetch_result(request_id) is None
            if lock_mode == PrefetchLockMode.LOCK:
                assert result.l1_owners == dict.fromkeys(keys, l1.l1_manager_id)
                for _ in range(readers - 1):
                    l1.finish_read(keys)
                    assert all(not l1.is_key_evictable(key) for key in keys)
                l1.finish_read(keys)
            else:
                assert result.l1_owners == {}
            assert all(l1.is_key_evictable(key) for key in original + recovered)
            resident_count = (
                len(original + recovered)
                if lock_mode == PrefetchLockMode.LOCK
                else len(keys)
            )
            assert l1.get_memory_usage()[0] == resident_count * 4096
    finally:
        controller.stop()


def test_all_conflicts_recovered_without_loading(
    l1: L1Manager,
    l1_config: L1ManagerConfig,
    adapter: NativeConnectorL2Adapter,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An entirely recovered request replans without submitting any disk loads."""
    groups = make_groups(rows=1)
    keys = groups[0].keys
    seed_adapter(l1, adapter, keys)
    stage(l1, keys)
    admit_after_lookup(l1, keys, monkeypatch)
    controller = make_controller(l1, l1_config, adapter, tmp_path)
    controller.start()
    try:
        with patch.object(
            adapter, "submit_load_task", wraps=adapter.submit_load_task
        ) as load:
            request_id = controller.submit_prefetch_request(PrefetchTaskSpec(groups, 1))
            assert controller.wait_prefetch_result(request_id, 5)
            result = controller.query_prefetch_result(request_id)
            assert result is not None
            assert result.hit_cells[0].popcount() == 3
            assert result.l1_hit_cells[0].popcount() == 3
            assert result.l2_hit_cells[0].popcount() == 0
            load.assert_not_called()
            l1.finish_read(keys)
            assert all(l1.is_key_evictable(key) for key in keys)
    finally:
        controller.stop()


@pytest.mark.parametrize("kind", ["evicted", "staging"])
def test_conflict_without_readable_object_remains_a_miss(
    l1: L1Manager,
    l1_config: L1ManagerConfig,
    adapter: NativeConnectorL2Adapter,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    """Eviction and an unfinished same-tag writer must not become false hits."""
    groups = make_groups(rows=1)
    keys = groups[0].keys
    seed_adapter(l1, adapter, keys)
    if kind == "evicted":
        stage(l1, [keys[1]])
        admit_after_lookup(l1, [keys[1]], monkeypatch)
    reserve_write = l1.reserve_write
    writer_tags: list[str] = []

    def reserve(
        keys: list[ObjectKey],
        is_temporary: list[bool],
        layout_desc: MemoryLayoutDesc,
        tag: str = "",
    ) -> dict[ObjectKey, L1OperationResult]:
        if kind == "staging":
            reserve_write([keys[1]], [False], layout_desc, tag=tag)
            writer_tags.append(tag)
        results = reserve_write(keys, is_temporary, layout_desc, tag=tag)
        if kind == "evicted":
            assert l1.delete([keys[1]])[keys[1]] == L1Error.SUCCESS
        return results

    monkeypatch.setattr(l1, "reserve_write", reserve)
    controller = make_controller(l1, l1_config, adapter, tmp_path)
    controller.start()
    try:
        request_id = controller.submit_prefetch_request(PrefetchTaskSpec(groups, 1))
        assert controller.wait_prefetch_result(request_id, 5)
        result = controller.query_prefetch_result(request_id)
        assert result is not None
        assert result.hit_cells[0].get_indices_list() == [0]
        assert result.l1_hit_cells[0].popcount() == 0
        l1.finish_read(keys[:1])
        for tag in writer_tags:
            l1.finish_write_and_delete([keys[1]], tag=tag)
        assert l1.get_staging_memory_usage() == 0
        assert l1.get_memory_usage()[0] == 0
    finally:
        controller.stop()


class DropRecoveredPolicy(DefaultPrefetchPolicy):
    """A custom policy that stops using a request after resident hits appear."""

    def plan_load(
        self,
        key_groups: list[GroupedObjectKeys],
        l1_locked_keys: MapState,
        l2_locked_keys: MapState,
        l1_manager_descs: dict[int, L1ManagerDescriptor],
        l2_adapter_descs: dict[int, L2AdapterDescriptor],
        fetching_policy: FetchingPolicy,
    ) -> PrefetchPlan:
        """Keep initial L2 discovery, then decline the recovered L1 plan."""
        if l1_locked_keys.merge().popcount():
            return PrefetchPlan(MapState(), MapState())
        return super().plan_load(
            key_groups,
            l1_locked_keys,
            l2_locked_keys,
            l1_manager_descs,
            l2_adapter_descs,
            fetching_policy,
        )


@pytest.mark.parametrize("abort", [False, True])
def test_recovered_locks_released_on_policy_drop_or_abort(
    l1: L1Manager,
    l1_config: L1ManagerConfig,
    adapter: NativeConnectorL2Adapter,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    abort: bool,
) -> None:
    """Recovered read locks and unused staging are unwound on both exit paths."""
    groups = make_groups(rows=1)
    keys = groups[0].keys
    seed_adapter(l1, adapter, keys)
    stage(l1, [keys[1]])
    admit_after_lookup(l1, [keys[1]], monkeypatch)
    if abort:

        def fail_load(keys: list[ObjectKey], objs: list[MemoryObj]) -> int:
            raise RuntimeError("injected load submission failure")

        monkeypatch.setattr(adapter, "submit_load_task", fail_load)
    controller = make_controller(
        l1,
        l1_config,
        adapter,
        tmp_path,
        DefaultPrefetchPolicy() if abort else DropRecoveredPolicy(),
    )
    controller.start()
    try:
        request_id = controller.submit_prefetch_request(PrefetchTaskSpec(groups, 3))
        assert controller.wait_prefetch_result(request_id, 5)
        result = controller.query_prefetch_result(request_id)
        assert result is not None
        assert result.hit_cells[0].popcount() == 0
        assert l1.is_key_evictable(keys[1])
        assert l1.get_staging_memory_usage() == 0
        assert l1.get_memory_usage()[0] == 4096
        assert l1.delete([keys[1]])[keys[1]] == L1Error.SUCCESS
    finally:
        controller.stop()
