# SPDX-License-Identifier: Apache-2.0
"""Versioned KV survives an L2 round trip through the real storage stack.

A real ``StorageManager`` (CPU L1, ``MockL2Adapter``) backs a real server
context. Objects of two versions live only in L2; a versioned lookup must
load exactly the in-window version from L2 into L1 and report it. Waits are
event based: the adapter's store listener and the prefetch controller's
condition variable, never sleeps.
"""

# Standard
from collections.abc import Iterator
from types import SimpleNamespace
from unittest.mock import patch
import threading

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
from lmcache.v1.distributed.internal_api import L2AdapterListener
from lmcache.v1.distributed.l2_adapters.config import L2AdaptersConfig
from lmcache.v1.distributed.l2_adapters.mock_l2_adapter import MockL2AdapterConfig
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObjMetadata,
    TensorMemoryObj,
)
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.engine_context import MPCacheServerContext
from lmcache.v1.multiprocess.kv_version import (
    KV_VERSION_CONFIG,
    MAX_KV_STALENESS_CONFIG,
)
from lmcache.v1.multiprocess.modules.lookup import LookupModule

pytestmark = pytest.mark.no_shared_allocator

CHUNK = 4
NUM_CHUNKS = 3
OBJ_BYTES = 256
MODEL = "m"
TOKENS = list(range(CHUNK * NUM_CHUNKS))
LAYOUT = MemoryLayoutDesc(shapes=[torch.Size([OBJ_BYTES])], dtypes=[torch.uint8])


class _StoredKeys(L2AdapterListener):
    """Signals once ``expected`` keys have been stored in L2."""

    def __init__(self, expected: int) -> None:
        self._expected = expected
        self._stored = 0
        self._lock = threading.Lock()
        self.done = threading.Event()

    def on_l2_keys_stored(self, keys: list[ObjectKey], sizes: list[int]) -> None:
        with self._lock:
            self._stored += len(keys)
            if self._stored >= self._expected:
                self.done.set()

    def on_l2_keys_accessed(self, keys: list[ObjectKey]) -> None:
        pass

    def on_l2_keys_deleted(self, keys: list[ObjectKey]) -> None:
        pass


def _memory_obj(fill: int) -> TensorMemoryObj:
    raw = torch.full((OBJ_BYTES,), fill, dtype=torch.uint8)
    metadata = MemoryObjMetadata(
        shape=torch.Size([OBJ_BYTES]),
        dtype=torch.uint8,
        address=0,
        phy_size=OBJ_BYTES,
        fmt=MemoryFormat.KV_2LTD,
        ref_count=1,
    )
    return TensorMemoryObj(raw, metadata, parent_allocator=None)


def _key(
    request_id: str, version: int, staleness: int, worker_id: int | None
) -> IPCCacheServerKey:
    return IPCCacheServerKey.from_token_ids(
        model_name=MODEL,
        world_size=1,
        worker_id=worker_id,
        token_ids=TOKENS,
        start=0,
        end=len(TOKENS),
        request_id=request_id,
        request_configs={
            KV_VERSION_CONFIG: version,
            MAX_KV_STALENESS_CONFIG: staleness,
        },
        num_kv_readers=1,
    )


@pytest.fixture
def ctx(monkeypatch: pytest.MonkeyPatch) -> Iterator[MPCacheServerContext]:
    monkeypatch.setattr(
        "lmcache.v1.memory_management._allocate_cpu_memory",
        lambda size, *args, **kwargs: torch.empty(size, dtype=torch.uint8),
    )
    monkeypatch.setattr(
        "lmcache.v1.memory_management._free_cpu_memory", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        "lmcache.v1.memory_allocators.mixed_memory_allocator.torch_dev",
        SimpleNamespace(is_available=lambda: False),
    )
    config = StorageManagerConfig(
        l1_manager_config=L1ManagerConfig(
            memory_config=L1MemoryManagerConfig(
                size_in_bytes=1 << 20, use_lazy=False, align_bytes=64, shm_name=""
            )
        ),
        eviction_config=EvictionConfig(eviction_policy="LRU"),
        l2_adapter_config=L2AdaptersConfig(
            adapters=[MockL2AdapterConfig(max_size_gb=0.01, mock_bandwidth_gb=10.0)]
        ),
    )
    with patch("lmcache.v1.multiprocess.engine_context.get_event_bus"):
        context = MPCacheServerContext(config, chunk_size=CHUNK)
    context.layout_desc_registry.register(MODEL, 1, LAYOUT)
    try:
        yield context
    finally:
        context.close()


def _store_in_l2_only(ctx: MPCacheServerContext, version: int) -> list[ObjectKey]:
    """Put a version's objects into L2 without ever writing them to L1."""
    keys = ctx.resolve_obj_keys(_key(f"writer-{version}", version, 0, 0), [0])[0]
    ctx.session_manager.remove(f"writer-{version}")
    listener = _StoredKeys(len(keys))
    ctx.storage_manager.register_l2_listener(listener)
    _, adapter = ctx.storage_manager.l2_adapters()[0]
    adapter.submit_store_task(keys, [_memory_obj(version) for _ in keys])
    assert listener.done.wait(timeout=10.0)
    return keys


def test_versioned_lookup_loads_the_in_window_version_from_l2(
    ctx: MPCacheServerContext,
) -> None:
    stale = _store_in_l2_only(ctx, version=2)
    fresh = _store_in_l2_only(ctx, version=5)
    assert ctx.storage_manager.get_l1_usage()[0] == 0
    module = LookupModule(ctx)

    module.lookup(_key("r", 6, 1, None), tp_size=1)
    assert module.wait_prefetch_status("r", timeout=10.0) == NUM_CHUNKS
    assert module.query_kv_version_spans("r") == [(0, NUM_CHUNKS * CHUNK, 5)]

    retrieve_keys = ctx.resolve_obj_keys(_key("r", 6, 1, 0), [0])[0]
    assert retrieve_keys == fresh
    with ctx.storage_manager.read_prefetched_results(retrieve_keys) as objs:
        assert objs is not None
        assert all(bool((obj.tensor == 5).all()) for obj in objs)

    module.free_lookup_locks(_key("r", 6, 1, None), tp_size=1)

    # Window [3, 4]: version 2 is too old and version 5 too new, both in L2.
    module.lookup(_key("strict", 4, 1, None), tp_size=1)
    assert module.wait_prefetch_status("strict", timeout=10.0) == 0
    assert stale != fresh
