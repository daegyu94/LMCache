# SPDX-License-Identifier: Apache-2.0
"""Bounded-staleness lookup through ``LookupModule`` and the server context.

The server context is real (token hashing, sessions, layout registry,
``resolve_obj_keys`` and the store gate); only the storage manager is an
in-memory two-tier double. KV is "stored" by resolving a worker STORE key
exactly as the store paths do, so every test exercises the keys the server
would write and read. Prefetch rounds complete at once unless a test holds
them, which makes the multi-round races deterministic.
"""

# Standard
from collections import Counter
from collections.abc import Iterator
from typing import Any
from unittest.mock import MagicMock, patch
import threading
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
from lmcache.v1.multiprocess.config import (
    CoordinatorConfig,
    MPServerConfig,
)
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.engine_context import MPCacheServerContext
from lmcache.v1.multiprocess.engine_module import ModuleBuildContext
from lmcache.v1.multiprocess.kv_version import (
    KV_VERSION_CAPABILITY,
    KV_VERSION_CONFIG,
    MAX_KV_STALENESS_CONFIG,
    KVVersioning,
)
from lmcache.v1.multiprocess.modules.lookup import LookupModule

CHUNK = 4
NUM_CHUNKS = 6
MODEL = "m"
TOKENS = list(range(CHUNK * NUM_CHUNKS))


class _TwoTierStorage:
    """Answers prefetches from two key sets, like a prefix-policy prefetch.

    A task's hit is the longest prefix present (L1 or L2) in every row; hit
    keys from L2 move to L1 and every hit key is read-locked
    ``num_kv_readers`` times. A task whose index is in ``held`` does not
    finish until ``release`` is set; ``held_submitted`` is set once such a
    task has been submitted.
    """

    is_multi_l1 = False

    def __init__(self) -> None:
        self.l1: set[ObjectKey] = set()
        self.l2: set[ObjectKey] = set()
        self.locks: Counter[ObjectKey] = Counter()
        self.submitted: list[PrefetchTaskSpec] = []
        self.held: set[int] = set()
        self.release = threading.Event()
        self.held_submitted = threading.Event()
        self.unlocked = threading.Event()
        self._results: dict[int, PrefetchResult] = {}
        self._lock = threading.Lock()

    def submit_prefetch_task(
        self, spec: PrefetchTaskSpec, external_request_id: str = ""
    ) -> PrefetchHandle:
        with self._lock:
            present = self.l1 | self.l2
            hit = min(
                next(
                    (i for i, k in enumerate(row.keys) if k not in present),
                    len(row.keys),
                )
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
        if task_id in self.held:
            self.held_submitted.set()
        return PrefetchHandle(
            prefetch_request_id=task_id,
            external_request_id=external_request_id,
            total_requested_keys=len(spec.key_groups) * spec.group_size,
            submit_time=time.monotonic(),
        )

    def query_prefetch_status(self, handle: PrefetchHandle) -> PrefetchResult | None:
        if handle.prefetch_request_id == -1:
            return PrefetchResult(hit_cells=[], l1_hit_cells=[], l2_hit_cells=[])
        if handle.prefetch_request_id in self.held and not self.release.is_set():
            return None
        with self._lock:
            return self._results.pop(handle.prefetch_request_id, None)

    def wait_prefetch_status(self, handle: PrefetchHandle, timeout: float) -> bool:
        if handle.prefetch_request_id in self.held:
            return self.release.wait(timeout)
        return True

    def touch_cached_keys(self, keys: list[ObjectKey]) -> None:
        pass

    def finish_read_prefetched(
        self,
        keys: list[ObjectKey],
        read_locks: int = 1,
        l1_owners: dict[ObjectKey, int] | None = None,
    ) -> None:
        with self._lock:
            for key in keys:
                self.locks[key] -= read_locks
        self.unlocked.set()

    def held_locks(self) -> Counter[ObjectKey]:
        with self._lock:
            return +self.locks

    def close(self) -> None:
        pass


Server = tuple[MPCacheServerContext, LookupModule, _TwoTierStorage]


@pytest.fixture
def server() -> Iterator[Server]:
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
    module = LookupModule(ctx, kv_versioning=KVVersioning.ENABLED)
    try:
        yield ctx, module, storage
    finally:
        storage.release.set()
        module.close()
        ctx.close()


def _configs(version: object, staleness: object = None) -> dict[str, Any]:
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
    num_chunks: int = NUM_CHUNKS,
) -> list[ObjectKey]:
    """Do what a STORE of the first ``num_chunks`` chunks writes.

    Honours the server's store gate like the LMCache-driven STORE does, and
    keeps the request's session (its lookup decides the gate).

    Returns:
        The keys written; empty when the gate refused the store.
    """
    key = _key(request_id, configs, worker_id=0, end_chunk=num_chunks)
    num_groups = ctx.layout_desc_registry.find_attn_desc(MODEL, 1).num_object_groups
    per_group = ctx.resolve_obj_keys(key, list(range(num_groups)))
    if not ctx.kv_version_allows_store(key):
        return []
    keys = [k for group_keys in per_group for k in group_keys]
    tier.update(keys)
    return keys


def _lookup(module: LookupModule, request_id: str, configs: dict[str, Any]) -> int:
    module.lookup(_key(request_id, configs), tp_size=1)
    hit = module.wait_prefetch_status(request_id, timeout=10.0)
    assert hit is not None
    return hit


def test_no_hint_keys_are_unchanged_and_isolated_from_versioned(server: Server) -> None:
    ctx, module, storage = server
    plain = _store(ctx, storage.l1, "writer", {})
    assert plain == ctx.resolve_obj_keys(_key("probe", {}, worker_id=0), [0])[0]

    assert _lookup(module, "plain", {}) == NUM_CHUNKS
    assert len(storage.submitted) == 1
    assert module.query_kv_version_spans("plain") == []

    # Unversioned KV is never served to a versioned request.
    assert _lookup(module, "versioned", _configs(3, 2)) == 0
    assert module.query_kv_version_spans("versioned") == []


def test_exact_version_is_one_prefetch(server: Server) -> None:
    ctx, module, storage = server
    _store(ctx, storage.l1, "w", _configs(5))

    assert _lookup(module, "r", _configs(5)) == NUM_CHUNKS
    assert module.query_kv_version_spans("r") == [(0, NUM_CHUNKS * CHUNK, 5)]
    assert len(storage.submitted) == 1


def test_version_outside_window_is_a_miss(server: Server) -> None:
    ctx, module, storage = server
    _store(ctx, storage.l1, "w", _configs(3))

    assert _lookup(module, "newer", _configs(6, 2)) == 0
    # Newer-than-requested KV is never served either.
    assert _lookup(module, "older", _configs(2, 5)) == 0
    # One round per version in the window, newest first, then it stops.
    assert len(storage.submitted) == 3 + 3


def test_rounds_advance_on_the_server_without_polling(server: Server) -> None:
    ctx, module, storage = server
    _store(ctx, storage.l2, "w", _configs(4))

    module.lookup(_key("r", _configs(6, 2)), tp_size=1)
    # One wait covers every round; nobody polls between them.
    assert module.wait_prefetch_status("r", timeout=10.0) == NUM_CHUNKS
    assert len(storage.submitted) == 3
    assert module.query_kv_version_spans("r") == [(0, NUM_CHUNKS * CHUNK, 4)]


def test_mixed_versions_are_reported_and_out_of_window_stops_prefix(
    server: Server,
) -> None:
    ctx, module, storage = server
    _store(ctx, storage.l1, "old", _configs(1))  # outside [5, 7]
    _store(ctx, storage.l1, "mid", _configs(5), 4)
    _store(ctx, storage.l2, "new", _configs(7), 2)

    assert _lookup(module, "r", _configs(7, 2)) == 4
    assert module.query_kv_version_spans("r") == [
        (0, 2 * CHUNK, 7),
        (2 * CHUNK, 4 * CHUNK, 5),
    ]


def test_retrieve_and_lock_release_address_the_served_versions(
    server: Server,
) -> None:
    ctx, module, storage = server
    mid = _store(ctx, storage.l1, "mid", _configs(5), 4)
    new = _store(ctx, storage.l1, "new", _configs(7), 2)
    configs = _configs(7, 2)
    assert _lookup(module, "r", configs) == 4

    served = new[:2] + mid[2:4]
    assert storage.held_locks() == Counter({k: 1 for k in served})
    retrieved = ctx.resolve_obj_keys(_key("r", configs, worker_id=0, end_chunk=4), [0])[
        0
    ]
    assert retrieved == served

    module.free_lookup_locks(_key("r", configs, end_chunk=4), tp_size=1)
    assert storage.held_locks() == Counter()


def test_kv_computed_on_older_kv_is_not_stored(server: Server) -> None:
    """A request served version-8 KV computes the rest under version 10. That
    KV belongs to neither version, so it is not stored: a later exact
    version-8 request must not find it, and neither must a version-10 one."""
    ctx, module, storage = server
    _store(ctx, storage.l1, "v8", _configs(8), 2)
    configs = _configs(10, 2)
    assert _lookup(module, "r", configs) == 2

    assert _store(ctx, storage.l1, "r", configs) == []
    assert _lookup(module, "exact-8", _configs(8)) == 2
    assert _lookup(module, "exact-10", _configs(10)) == 0


def test_kv_computed_on_own_version_kv_is_stored(server: Server) -> None:
    ctx, module, storage = server
    _store(ctx, storage.l1, "v10", _configs(10), 2)
    configs = _configs(10, 2)
    assert _lookup(module, "r", configs) == 2

    assert _store(ctx, storage.l1, "r", configs)
    assert _lookup(module, "exact-10", _configs(10)) == NUM_CHUNKS


@pytest.mark.parametrize("version", ["10", 10.0, True, -1])
def test_malformed_version_neither_loads_nor_stores(
    server: Server, version: object
) -> None:
    ctx, module, storage = server
    _store(ctx, storage.l1, "plain", {})
    configs = _configs(version)

    assert _lookup(module, "r", configs) == 0
    assert storage.submitted == []
    assert _store(ctx, storage.l1, "r", configs) == []


def test_disabled_versioning_fails_closed(server: Server) -> None:
    ctx, _, storage = server
    _store(ctx, storage.l1, "w", _configs(5))
    module = LookupModule(ctx)
    try:
        assert _lookup(module, "r", _configs(5)) == 0
        assert storage.submitted == []
    finally:
        module.close()


@pytest.mark.parametrize(
    "mode, advertised",
    [("lmcache_driven", True), ("engine_driven", False), ("auto", False)],
)
def test_capability_is_advertised_only_in_lmcache_driven_mode(
    server: Server, mode: str, advertised: bool
) -> None:
    ctx, _, _ = server
    build_ctx = ModuleBuildContext(
        ctx,
        MPServerConfig(supported_transfer_mode=mode),  # type: ignore[arg-type]
        CoordinatorConfig(),
    )
    module = LookupModule.create(build_ctx)
    assert module is not None
    module.close()
    assert (KV_VERSION_CAPABILITY in build_ctx.experimental) is advertised


def test_staleness_is_narrowed_for_sliding_window_layouts(server: Server) -> None:
    ctx, module, storage = server
    ctx.layout_desc_registry.register(
        MODEL,
        1,
        MemoryLayoutDesc(shapes=[torch.Size([CHUNK])], dtypes=[torch.uint8]),
        attn_desc=AttnWindowDesc(num_chunks_in_sw=[-1, 2], world_size=1),
    )
    _store(ctx, storage.l1, "w", _configs(4))

    assert _lookup(module, "r", _configs(5, 1)) == 0
    assert len(storage.submitted) == 1


def test_spans_are_unknown_until_the_lookup_result_is_consumed(
    server: Server,
) -> None:
    ctx, module, storage = server
    _store(ctx, storage.l1, "w", _configs(5))
    assert module.query_kv_version_spans("never-seen") is None

    module.lookup(_key("r", _configs(5)), tp_size=1)
    assert module.query_kv_version_spans("r") is None
    assert module.query_prefetch_status("r") == NUM_CHUNKS
    assert module.query_kv_version_spans("r") == [(0, NUM_CHUNKS * CHUNK, 5)]


def _hold_second_round(ctx: MPCacheServerContext, storage: _TwoTierStorage) -> None:
    """Version 7 holds chunks 0-1; the version-6 round is held."""
    _store(ctx, storage.l1, "v7", _configs(7), 2)
    _store(ctx, storage.l1, "v6", _configs(6))
    storage.held.add(1)


def test_free_mid_probe_releases_every_round_and_submits_nothing(
    server: Server,
) -> None:
    ctx, module, storage = server
    _hold_second_round(ctx, storage)
    configs = _configs(7, 2)
    module.lookup(_key("r", configs), tp_size=1)
    assert storage.held_submitted.wait(timeout=10.0)
    # The held round never finishes before the deadline.
    assert module.wait_prefetch_status("r", timeout=0.01) is None

    module.free_lookup_locks(_key("r", configs), tp_size=1)
    storage.release.set()

    assert storage.unlocked.wait(timeout=10.0)
    assert storage.held_locks() == Counter()
    assert len(storage.submitted) == 2
    assert module.query_prefetch_status("r") == 0


def test_newer_lookup_supersedes_a_running_probe(server: Server) -> None:
    ctx, module, storage = server
    _hold_second_round(ctx, storage)
    module.lookup(_key("r", _configs(7, 2)), tp_size=1)
    assert storage.held_submitted.wait(timeout=10.0)

    # The request looks up again, exact version only.
    module.lookup(_key("r", _configs(7)), tp_size=1)
    assert module.query_prefetch_status("r") == 2
    storage.release.set()

    assert storage.unlocked.wait(timeout=10.0)
    # Only the newer lookup's locks remain; the probe never resurfaces.
    newer = storage.submitted[2]
    assert storage.held_locks() == Counter({k: 1 for k in newer.key_groups[0].keys[:2]})
    assert module.query_kv_version_spans("r") == [(0, 2 * CHUNK, 7)]
    assert module.query_prefetch_status("r") == 0


def test_end_session_mid_probe_releases_its_locks(server: Server) -> None:
    ctx, module, storage = server
    _hold_second_round(ctx, storage)
    module.lookup(_key("r", _configs(7, 2)), tp_size=1)
    assert storage.held_submitted.wait(timeout=10.0)

    module.end_session("r")
    storage.release.set()

    assert storage.unlocked.wait(timeout=10.0)
    assert storage.held_locks() == Counter()
    assert ctx.session_manager.get("r") is None


def test_abandoned_lookup_locks_release_under_their_own_versions(
    server: Server,
) -> None:
    ctx, module, storage = server
    _store(ctx, storage.l1, "old", _configs(6))
    configs = _configs(7, 1)
    # An engine-covered lookup from chunk 2, abandoned for a full re-lookup.
    module.lookup(_key("r", configs, start_chunk=2), tp_size=1)
    assert module.wait_prefetch_status("r", timeout=10.0) == NUM_CHUNKS
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
    assert sum(storage.held_locks().values()) == 2 * NUM_CHUNKS - 2

    module.free_lookup_locks(
        _key("r", configs),
        tp_size=1,
        extra_ranges=((2 * CHUNK, NUM_CHUNKS * CHUNK),),
    )
    assert storage.held_locks() == Counter()
