# SPDX-License-Identifier: Apache-2.0
"""Bounded-staleness lookup through ``LookupModule`` and the server context.

The server context is real (token hashing, sessions, layout registry and
``resolve_obj_keys``); only the storage manager is an in-memory two-tier
double that answers prefetches synchronously. KV is "stored" by resolving a
worker STORE key exactly as the store paths do, so every test exercises the
same keys the server would write and read.
"""

# Standard
from collections import Counter
from collections.abc import Iterator
from typing import Any
from unittest.mock import MagicMock, patch
import time

# Third Party
import pytest
import torch

# First Party
from lmcache.lmcache_native import Bitmap
from lmcache.v1.distributed.api import (
    AttnWindowDesc,
    MemoryLayoutDesc,
    ObjectKey,
    PrefetchHandle,
    PrefetchResult,
    PrefetchTaskSpec,
)
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.engine_context import MPCacheServerContext
from lmcache.v1.multiprocess.kv_version import (
    KV_VERSION_CONFIG,
    MAX_KV_STALENESS_CONFIG,
)
from lmcache.v1.multiprocess.modules.lookup import LookupModule

CHUNK = 4
NUM_CHUNKS = 6
MODEL = "m"
TOKENS = list(range(CHUNK * NUM_CHUNKS))


class _TwoTierStorage:
    """Answers prefetches from two key sets, like a prefix-policy prefetch.

    A submitted task completes immediately: the hit is the longest prefix
    present (L1 or L2) in every row; hit keys from L2 move to L1 and every
    hit key is read-locked ``num_kv_readers`` times.
    """

    is_multi_l1 = False

    def __init__(self) -> None:
        self.l1: set[ObjectKey] = set()
        self.l2: set[ObjectKey] = set()
        self.locks: Counter[ObjectKey] = Counter()
        self.submitted: list[PrefetchTaskSpec] = []
        self._results: dict[int, PrefetchResult] = {}

    def submit_prefetch_task(
        self, spec: PrefetchTaskSpec, external_request_id: str = ""
    ) -> PrefetchHandle:
        present = self.l1 | self.l2
        hit = min(
            next((i for i, k in enumerate(row.keys) if k not in present), len(row.keys))
            for row in spec.key_groups
        )
        hit_cells, l1_cells, l2_cells = [], [], []
        for row in spec.key_groups:
            size = len(row.keys)
            hit_row, l1_row, l2_row = Bitmap(size), Bitmap(size), Bitmap(size)
            for i, key in enumerate(row.keys[:hit]):
                hit_row.set(i)
                (l1_row if key in self.l1 else l2_row).set(i)
                self.l1.add(key)
                self.locks[key] += spec.num_kv_readers
            hit_cells.append(hit_row)
            l1_cells.append(l1_row)
            l2_cells.append(l2_row)
        task_id = len(self.submitted)
        self.submitted.append(spec)
        self._results[task_id] = PrefetchResult(
            hit_cells=hit_cells, l1_hit_cells=l1_cells, l2_hit_cells=l2_cells
        )
        return PrefetchHandle(
            prefetch_request_id=task_id,
            external_request_id=external_request_id,
            total_requested_keys=len(spec.key_groups) * spec.group_size,
            submit_time=time.monotonic(),
        )

    def query_prefetch_status(self, handle: PrefetchHandle) -> PrefetchResult | None:
        if handle.prefetch_request_id == -1:
            return PrefetchResult(hit_cells=[], l1_hit_cells=[], l2_hit_cells=[])
        return self._results.pop(handle.prefetch_request_id, None)

    def wait_prefetch_status(self, handle: PrefetchHandle, timeout: float) -> bool:
        return True

    def touch_cached_keys(self, keys: list[ObjectKey]) -> None:
        pass

    def finish_read_prefetched(
        self,
        keys: list[ObjectKey],
        read_locks: int = 1,
        l1_owners: dict[ObjectKey, int] | None = None,
    ) -> None:
        for key in keys:
            self.locks[key] -= read_locks

    def close(self) -> None:
        pass


@pytest.fixture
def server() -> Iterator[tuple[MPCacheServerContext, LookupModule, _TwoTierStorage]]:
    storage = _TwoTierStorage()
    config = MagicMock()
    config.l1_manager_config.gds_l1_config = None
    config.l1_manager_config.memory_config.shm_name = ""
    with (
        patch(
            "lmcache.v1.multiprocess.engine_context.StorageManager",
            return_value=storage,
        ),
        patch("lmcache.v1.multiprocess.engine_context.get_event_bus"),
    ):
        ctx = MPCacheServerContext(config, chunk_size=CHUNK)
    ctx.layout_desc_registry.register(
        MODEL,
        1,
        MemoryLayoutDesc(shapes=[torch.Size([CHUNK])], dtypes=[torch.uint8]),
    )
    try:
        yield ctx, LookupModule(ctx), storage
    finally:
        ctx.close()


def _configs(version: int | None, staleness: int | None = None) -> dict[str, Any]:
    configs: dict[str, Any] = {}
    if version is not None:
        configs[KV_VERSION_CONFIG] = version
    if staleness is not None:
        configs[MAX_KV_STALENESS_CONFIG] = staleness
    return configs


def _key(
    request_id: str,
    configs: dict[str, Any],
    *,
    worker_id: int | None = None,
    start_chunk: int = 0,
    end_chunk: int = NUM_CHUNKS,
) -> IPCCacheServerKey:
    return IPCCacheServerKey.from_token_ids(
        model_name=MODEL,
        world_size=1,
        worker_id=worker_id,
        token_ids=TOKENS,
        start=start_chunk * CHUNK,
        end=end_chunk * CHUNK,
        request_id=request_id,
        request_configs=configs or None,
        num_kv_readers=1,
    )


def _store(
    ctx: MPCacheServerContext,
    tier: set[ObjectKey],
    request_id: str,
    configs: dict[str, Any],
    num_chunks: int,
) -> list[ObjectKey]:
    """Write the keys a worker STORE of the first ``num_chunks`` resolves."""
    num_groups = ctx.layout_desc_registry.find_attn_desc(MODEL, 1).num_object_groups
    per_group = ctx.resolve_obj_keys(
        _key(request_id, configs, worker_id=0, end_chunk=num_chunks),
        list(range(num_groups)),
    )
    keys = [key for group_keys in per_group for key in group_keys]
    tier.update(keys)
    ctx.session_manager.remove(request_id)
    return keys


def _lookup(
    module: LookupModule, request_id: str, configs: dict[str, Any]
) -> int | None:
    module.lookup(_key(request_id, configs), tp_size=1)
    return module.wait_prefetch_status(request_id, timeout=1.0)


def test_no_hint_keys_are_unchanged_and_isolated_from_versioned(server) -> None:
    ctx, module, storage = server
    plain = _store(ctx, storage.l1, "writer", {}, NUM_CHUNKS)
    assert plain == ctx.resolve_obj_keys(_key("probe", {}, worker_id=0), [0])[0]

    assert _lookup(module, "plain", {}) == NUM_CHUNKS
    assert len(storage.submitted) == 1
    assert module.query_kv_version_spans("plain") == []

    # Unversioned KV is never served to a versioned request.
    assert _lookup(module, "versioned", _configs(3, 2)) == 0
    assert module.query_kv_version_spans("versioned") == []


def test_exact_version_hit(server) -> None:
    ctx, module, storage = server
    _store(ctx, storage.l1, "w", _configs(5), NUM_CHUNKS)

    assert _lookup(module, "r", _configs(5)) == NUM_CHUNKS
    assert module.query_kv_version_spans("r") == [(0, NUM_CHUNKS * CHUNK, 5)]
    assert len(storage.submitted) == 1


def test_version_outside_window_is_a_miss(server) -> None:
    ctx, module, storage = server
    _store(ctx, storage.l1, "w", _configs(3), NUM_CHUNKS)

    assert _lookup(module, "newer", _configs(6, 2)) == 0
    # Newer-than-requested KV is never served either.
    assert _lookup(module, "older", _configs(2, 5)) == 0
    # One round per version in the window, newest first, then it stops.
    assert len(storage.submitted) == 3 + 3


def test_older_version_in_window_is_served(server) -> None:
    ctx, module, storage = server
    _store(ctx, storage.l1, "w", _configs(4), NUM_CHUNKS)

    assert _lookup(module, "r", _configs(6, 2)) == NUM_CHUNKS
    assert module.query_kv_version_spans("r") == [(0, NUM_CHUNKS * CHUNK, 4)]


def test_mixed_versions_are_reported_and_out_of_window_stops_prefix(server) -> None:
    ctx, module, storage = server
    _store(ctx, storage.l1, "old", _configs(1), NUM_CHUNKS)  # outside [5, 7]
    _store(ctx, storage.l1, "mid", _configs(5), 4)
    _store(ctx, storage.l2, "new", _configs(7), 2)

    assert _lookup(module, "r", _configs(7, 2)) == 4
    assert module.query_kv_version_spans("r") == [
        (0, 2 * CHUNK, 7),
        (2 * CHUNK, 4 * CHUNK, 5),
    ]


def test_retrieve_and_lock_release_address_the_served_versions(server) -> None:
    ctx, module, storage = server
    mid = _store(ctx, storage.l1, "mid", _configs(5), 4)
    new = _store(ctx, storage.l1, "new", _configs(7), 2)
    configs = _configs(7, 2)
    assert _lookup(module, "r", configs) == 4

    served = new[:2] + mid[2:4]
    assert all(storage.locks[k] == 1 for k in served)
    retrieved = ctx.resolve_obj_keys(_key("r", configs, worker_id=0, end_chunk=4), [0])[
        0
    ]
    assert retrieved == served

    module.free_lookup_locks(_key("r", configs, end_chunk=4), tp_size=1)
    assert +storage.locks == Counter()


def test_store_after_stale_hit_is_stamped_with_oldest_served_version(server) -> None:
    ctx, module, storage = server
    _store(ctx, storage.l1, "mid", _configs(5), 2)
    configs = _configs(7, 2)
    assert _lookup(module, "r", configs) == 2

    # The request computes chunks 2.. on top of version-5 KV, so the new KV
    # is stamped 5: a request accepting only 7 must not reuse it.
    stored = ctx.resolve_obj_keys(_key("r", configs, worker_id=0), [0])[0]
    later = _key("probe", _configs(5), worker_id=0)
    assert stored == ctx.resolve_obj_keys(later, [0])[0]


def test_store_without_hit_is_stamped_with_request_version(server) -> None:
    ctx, module, storage = server
    _store(ctx, storage.l2, "w", _configs(8), NUM_CHUNKS)
    assert _lookup(module, "exact", _configs(8)) == NUM_CHUNKS
    assert _lookup(module, "window", _configs(9, 1)) == NUM_CHUNKS
    assert module.query_kv_version_spans("window") == [(0, NUM_CHUNKS * CHUNK, 8)]


def test_staleness_is_narrowed_for_sliding_window_layouts(server) -> None:
    ctx, module, storage = server
    ctx.layout_desc_registry.register(
        MODEL,
        1,
        MemoryLayoutDesc(shapes=[torch.Size([CHUNK])], dtypes=[torch.uint8]),
        attn_desc=AttnWindowDesc(num_chunks_in_sw=[-1, 2], world_size=1),
    )
    _store(ctx, storage.l1, "w", _configs(4), NUM_CHUNKS)

    assert _lookup(module, "r", _configs(5, 1)) == 0
    assert len(storage.submitted) == 1


def test_spans_are_unknown_without_a_session(server) -> None:
    _, module, _ = server
    assert module.query_kv_version_spans("never-seen") is None


def test_abandoned_lookup_locks_release_under_their_own_versions(server) -> None:
    ctx, module, storage = server
    _store(ctx, storage.l1, "old", _configs(6), NUM_CHUNKS)
    configs = _configs(7, 1)
    # An engine-covered lookup from chunk 2, abandoned for a full re-lookup.
    module.lookup(_key("r", configs, start_chunk=2), tp_size=1)
    assert module.wait_prefetch_status("r", timeout=1.0) == NUM_CHUNKS
    _store(ctx, storage.l1, "new", _configs(7), 3)
    assert _lookup(module, "r", configs) == NUM_CHUNKS
    assert module.query_kv_version_spans("r") == [
        (0, 3 * CHUNK, 7),
        (3 * CHUNK, NUM_CHUNKS * CHUNK, 6),
    ]

    # A range no lookup of this request served is left to the lock TTL.
    module.free_lookup_locks(
        _key("r", configs, end_chunk=0), tp_size=1, extra_ranges=((CHUNK, 3 * CHUNK),)
    )
    assert sum(storage.locks.values()) == 2 * NUM_CHUNKS - 2

    module.free_lookup_locks(
        _key("r", configs),
        tp_size=1,
        extra_ranges=((2 * CHUNK, NUM_CHUNKS * CHUNK),),
    )
    assert +storage.locks == Counter()
