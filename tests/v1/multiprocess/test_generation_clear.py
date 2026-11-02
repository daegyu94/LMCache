# SPDX-License-Identifier: Apache-2.0
"""End-to-end tests for ``clear_mode="generation"`` on a real storage manager.

A real :class:`MPCacheServerContext` (L1 + mock L2 adapter) is driven through
the public ``LookupModule`` and ``ManagementModule`` handlers. Stores are
simulated by resolving a worker's object keys exactly as the STORE path does
(:meth:`MPCacheServerContext.resolve_obj_keys`) and committing them to L1, from
where the store controller copies them to L2.
"""

# Standard
from collections.abc import Callable, Iterator
import time

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.config import (
    EvictionConfig,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    StorageManagerConfig,
)
from lmcache.v1.distributed.l2_adapters.base import L2AdapterInterface
from lmcache.v1.distributed.l2_adapters.config import L2AdaptersConfig
from lmcache.v1.distributed.l2_adapters.mock_l2_adapter import (
    MockL2Adapter,
    MockL2AdapterConfig,
)
from lmcache.v1.distributed.storage_manager import l1_owners_of
from lmcache.v1.multiprocess.config import MPServerConfig
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.engine_context import MPCacheServerContext
from lmcache.v1.multiprocess.modules.lookup import LookupModule
from lmcache.v1.multiprocess.modules.management import ManagementModule

CHUNK_SIZE = 16
NUM_CHUNKS = 3
TOKENS = list(range(1000, 1000 + NUM_CHUNKS * CHUNK_SIZE))
MODEL = "m"
LAYOUT = MemoryLayoutDesc(
    shapes=[torch.Size([CHUNK_SIZE, 2, 64])], dtypes=[torch.bfloat16]
)


def _wait_for(predicate: Callable[[], bool], timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


def _storage_config() -> StorageManagerConfig:
    return StorageManagerConfig(
        l1_manager_config=L1ManagerConfig(
            memory_config=L1MemoryManagerConfig(
                size_in_bytes=32 * 1024 * 1024,
                use_lazy=False,
                init_size_in_bytes=32 * 1024 * 1024,
                align_bytes=0x1000,
            ),
            write_ttl_seconds=600,
            read_ttl_seconds=300,
        ),
        eviction_config=EvictionConfig(eviction_policy="LRU"),
        l2_adapter_config=L2AdaptersConfig(
            adapters=[MockL2AdapterConfig(max_size_gb=0.05, mock_bandwidth_gb=10.0)]
        ),
    )


class _Server:
    """The modules under test sharing one context."""

    def __init__(self, clear_mode: str) -> None:
        self.ctx = MPCacheServerContext(_storage_config(), chunk_size=CHUNK_SIZE)
        self.ctx.layout_desc_registry.register(MODEL, 1, LAYOUT)
        self.lookup = LookupModule(self.ctx)
        self.management = ManagementModule(self.ctx, clear_mode=clear_mode)  # type: ignore[arg-type]

    @property
    def l2(self) -> MockL2Adapter:
        adapter: L2AdapterInterface = self.ctx.storage_manager.l2_adapters()[0][1]
        assert isinstance(adapter, MockL2Adapter)
        return adapter

    def close(self) -> None:
        self.management.close()
        self.lookup.close()
        self.ctx.close()

    def resolve_store_keys(self, request_id: str) -> list[ObjectKey]:
        """Resolve worker 0's keys the way STORE does."""
        return self.ctx.resolve_obj_keys(_key(request_id, worker_id=0), [0])[0]

    def commit(self, keys: list[ObjectKey]) -> None:
        """Commit keys to L1 and wait until the store controller wrote L2."""
        sm = self.ctx.storage_manager
        reserved = sm.reserve_write(keys, LAYOUT)
        assert len(reserved) == len(keys)
        sm.finish_write(list(reserved), l1_owners_of(reserved))
        assert _wait_for(
            lambda: (
                all(self.l2.debug_has_key(k) for k in keys)
                and all(sm.l1_managers()[0].is_key_evictable(k) for k in keys)
            )
        )


def _key(request_id: str, worker_id: int | None) -> IPCCacheServerKey:
    return IPCCacheServerKey.from_token_ids(
        model_name=MODEL,
        world_size=1,
        worker_id=worker_id,
        token_ids=TOKENS,
        start=0,
        end=len(TOKENS),
        request_id=request_id,
    )


@pytest.fixture
def generation_server() -> Iterator[_Server]:
    server = _Server("generation")
    yield server
    server.close()


@pytest.fixture
def l1_server() -> Iterator[_Server]:
    server = _Server("l1")
    yield server
    server.close()


def _hit_chunks(server: _Server, request_id: str) -> int:
    key = _key(request_id, worker_id=None)
    server.lookup.lookup(key, 1)
    hits = server.lookup.wait_prefetch_status(request_id, 10.0)
    assert hits is not None
    server.lookup.free_lookup_locks(key, 1)
    server.lookup.end_session(request_id)
    return hits


def test_default_mode_keeps_hashes_and_serves_l2_after_clear(l1_server) -> None:
    keys = l1_server.resolve_store_keys("writer")
    l1_server.commit(keys)

    l1_server.management.clear()

    assert l1_server.ctx.cache_generation.current == 0
    assert "cache_generation" not in l1_server.management.report_status()
    # L2 is untouched by an L1 clear, so the old KV is promoted back.
    assert _hit_chunks(l1_server, "reader") == NUM_CHUNKS
    assert all(l1_server.l2.debug_has_key(k) for k in keys)


def test_generation_clear_makes_l1_and_l2_unreachable(generation_server) -> None:
    server = generation_server
    old_keys = server.resolve_store_keys("writer")
    server.commit(old_keys)
    server.lookup.end_session("writer")
    assert _hit_chunks(server, "warm") == NUM_CHUNKS

    server.management.clear()

    assert server.ctx.cache_generation.current == 1
    assert _hit_chunks(server, "reader") == 0
    # The old objects are garbage collected from L2 in the background.
    assert _wait_for(lambda: not any(server.l2.debug_has_key(k) for k in old_keys))
    status = server.management.report_status()["cache_generation"]
    assert status["generation"] == 1
    assert status["stale_l2_gc"]["deleted_keys"] >= len(old_keys)


def test_pre_bump_store_committing_after_bump_is_not_visible(
    generation_server,
) -> None:
    server = generation_server
    # The store's keys are resolved before the bump (KV computed under the
    # old weights) and the write lands only after it.
    in_flight_keys = server.resolve_store_keys("in-flight")
    server.management.clear()
    late_keys = server.resolve_store_keys("in-flight")
    assert late_keys == in_flight_keys
    server.commit(late_keys)

    assert _hit_chunks(server, "reader") == 0

    # The late commit is tracked anew and collected by the next clear.
    server.management.clear()
    assert _wait_for(lambda: not any(server.l2.debug_has_key(k) for k in late_keys))


def test_new_generation_data_round_trips(generation_server) -> None:
    server = generation_server
    server.management.clear()

    new_keys = server.resolve_store_keys("writer")
    server.commit(new_keys)
    server.lookup.end_session("writer")

    assert _hit_chunks(server, "reader") == NUM_CHUNKS


def test_pre_bump_session_stores_under_new_generation_after_relookup(
    generation_server,
) -> None:
    server = generation_server
    request_id = "preempted"
    pre_bump_keys = server.resolve_store_keys(request_id)

    server.management.clear()
    # The engine re-schedules the request after the weight update: its next
    # lookup runs under the new generation, so its later stores do too.
    key = _key(request_id, worker_id=None)
    server.lookup.lookup(key, 1)
    assert server.lookup.wait_prefetch_status(request_id, 10.0) == 0
    server.lookup.free_lookup_locks(key, 1)
    post_lookup_keys = server.resolve_store_keys(request_id)
    assert not set(post_lookup_keys) & set(pre_bump_keys)
    server.commit(post_lookup_keys)
    server.lookup.end_session(request_id)

    assert _hit_chunks(server, "reader") == NUM_CHUNKS


def test_generation_mode_rejects_blend() -> None:
    with pytest.raises(ValueError, match="generation"):
        MPServerConfig(clear_mode="generation", engine_type="blend")
    assert MPServerConfig().clear_mode == "l1"
