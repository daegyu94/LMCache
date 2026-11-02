# SPDX-License-Identifier: Apache-2.0
"""Unit tests for generation-scoped chunk hashing and stale-L2 collection."""

# Standard
from collections.abc import Callable
import threading
import time

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.internal_api import L2AdapterListener
from lmcache.v1.multiprocess.cache_generation import (
    L2_DELETE_BATCH_SIZE,
    CacheGeneration,
    HashScope,
    StaleL2KeyCollector,
)
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.session import SessionManager
from lmcache.v1.multiprocess.token_codec import pack_token_ids
from lmcache.v1.multiprocess.token_hasher import TokenHasher

CHUNK_SIZE = 4
TOKENS = list(range(100, 100 + 3 * CHUNK_SIZE))
PACKED = pack_token_ids(TOKENS)


def _wait_for(predicate: Callable[[], bool], timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def _lookup_key(request_id: str) -> IPCCacheServerKey:
    return IPCCacheServerKey.from_token_ids(
        model_name="m",
        world_size=1,
        worker_id=None,
        token_ids=TOKENS,
        start=0,
        end=len(TOKENS),
        request_id=request_id,
    )


def _object_key(i: int) -> ObjectKey:
    return ObjectKey(
        chunk_hash=i.to_bytes(8, "big"),
        model_name="m",
        kv_rank=0,
        object_group_id=0,
        cache_salt="",
    )


def _hashes(hasher: TokenHasher, prefix_hash: bytes | int) -> list[bytes]:
    return hasher.compute_packed_chunk_hashes(PACKED, prefix_hash=prefix_hash)


# ---------------------------------------------------------------------------
# TokenHasher.domain_prefix_hash
# ---------------------------------------------------------------------------


def test_domain_prefix_is_deterministic_and_distinct_per_domain() -> None:
    first, second = TokenHasher(CHUNK_SIZE), TokenHasher(CHUNK_SIZE)
    seeds = [first.domain_prefix_hash([tag, 7]) for tag in range(4)]

    assert first.domain_prefix_hash([2, 7]) == second.domain_prefix_hash([2, 7])
    assert len(set(seeds)) == len(seeds)
    assert first.none_hash not in seeds


def test_domain_prefix_never_aliases_a_longer_unscoped_prefix() -> None:
    # With one-token chunks, a seed built from real chunks would make scoped
    # hashes of T equal unscoped hashes of a longer sequence ending in T.
    hasher = TokenHasher(chunk_size=1)
    scoped = hasher.compute_chunk_hashes(
        [7, 8], prefix_hash=hasher.domain_prefix_hash([1])
    )
    unscoped = set(hasher.compute_chunk_hashes([1, 0, 7, 8]))
    unscoped |= set(hasher.compute_chunk_hashes([1, 7, 8]))

    assert not set(scoped) & unscoped


def test_domain_prefix_rejects_non_uint32_words() -> None:
    with pytest.raises(ValueError):
        TokenHasher(CHUNK_SIZE).domain_prefix_hash([1 << 32])


# ---------------------------------------------------------------------------
# CacheGeneration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("algorithm", ["blake3", "builtin"])
def test_l1_mode_hashes_are_unchanged(algorithm: str) -> None:
    hasher = TokenHasher(chunk_size=CHUNK_SIZE, hash_algorithm=algorithm)
    generation = CacheGeneration("l1")
    scope = generation.current_scope(hasher)

    assert scope == HashScope(0, hasher.none_hash)
    assert generation.untrusted_store_prefix_hash(hasher) == hasher.none_hash
    assert _hashes(hasher, scope.prefix_hash) == hasher.compute_chunk_hashes(TOKENS)


def test_l1_mode_never_bumps() -> None:
    generation = CacheGeneration("l1")
    with pytest.raises(ValueError):
        generation.bump()
    assert generation.current == 0


def test_generation_mode_scopes_are_distinct() -> None:
    hasher = TokenHasher(CHUNK_SIZE)
    generation = CacheGeneration("generation")
    scopes = [generation.current_scope(hasher)]
    for _ in range(3):
        generation.bump()
        scopes.append(generation.current_scope(hasher))

    assert [s.generation for s in scopes] == [0, 1, 2, 3]
    seeds = [s.prefix_hash for s in scopes]
    seeds.append(generation.untrusted_store_prefix_hash(hasher))
    seeds.append(hasher.none_hash)
    assert len(set(seeds)) == len(seeds)
    every_hash = [h for seed in seeds for h in _hashes(hasher, seed)]
    assert len(set(every_hash)) == len(every_hash)


def test_generation_mode_scope_is_stable_within_a_process() -> None:
    hasher = TokenHasher(CHUNK_SIZE)
    generation = CacheGeneration("generation")
    generation.bump()
    assert generation.scope(hasher, 1) == generation.current_scope(hasher)


def test_restarted_server_never_reuses_a_generation_scope() -> None:
    hasher = TokenHasher(CHUNK_SIZE)
    before, after = CacheGeneration("generation"), CacheGeneration("generation")
    for generation in (0, 1, 5):
        assert (
            before.scope(hasher, generation).prefix_hash
            != after.scope(hasher, generation).prefix_hash
        )
    assert before.untrusted_store_prefix_hash(
        hasher
    ) != after.untrusted_store_prefix_hash(hasher)


def test_concurrent_bumps_are_never_lost() -> None:
    generation = CacheGeneration("generation")
    results: list[int] = []
    results_lock = threading.Lock()

    def bump_many() -> None:
        for _ in range(200):
            value = generation.bump()
            with results_lock:
                results.append(value)

    threads = [threading.Thread(target=bump_many) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert generation.current == 1600
    assert sorted(results) == list(range(1, 1601))


# ---------------------------------------------------------------------------
# Session scopes
# ---------------------------------------------------------------------------


@pytest.fixture
def scoped() -> tuple[TokenHasher, CacheGeneration, SessionManager]:
    hasher = TokenHasher(CHUNK_SIZE)
    generation = CacheGeneration("generation")
    manager = SessionManager(hasher, cache_generation=generation, cleanup_interval=None)
    return hasher, generation, manager


def test_lookup_created_session_stores_in_its_lookup_generation(scoped) -> None:
    hasher, generation, manager = scoped
    scope = generation.current_scope(hasher)
    session = manager.get_or_create_for_lookup("r", scope)
    session.set_tokens(PACKED)

    expected = _hashes(hasher, scope.prefix_hash)
    assert session.get_store_hashes(0, len(TOKENS)) == expected
    assert session.get_hashes(0) == expected


def test_store_scope_never_moves_to_a_newer_generation(scoped) -> None:
    hasher, generation, manager = scoped
    old = generation.current_scope(hasher)
    session = manager.get_or_create_for_lookup("spanning", old)
    session.begin_lookup(_lookup_key("spanning"), (-1,), old)
    session.set_tokens(PACKED)

    generation.bump()
    new = generation.current_scope(hasher)
    # The engine re-schedules the request after the bump: the new lookup's
    # RETRIEVE/release keys follow it, but a store that arrives now may carry
    # KV computed before the bump, so stores stay in the old generation.
    assert manager.get_or_create_for_lookup("spanning", new) is session
    session.begin_lookup(_lookup_key("spanning"), (-1,), new)

    assert session.lookup_scope == new
    assert session.get_hashes(0) == _hashes(hasher, new.prefix_hash)
    assert session.get_store_hashes(0, len(TOKENS)) == _hashes(hasher, old.prefix_hash)


def test_session_created_outside_a_lookup_stores_untrusted(scoped) -> None:
    hasher, generation, manager = scoped
    generation.bump()
    session = manager.get_or_create("sessionless-store")
    session.set_tokens(PACKED)

    stored = set(session.get_store_hashes(0, len(TOKENS)))
    for g in range(generation.current + 1):
        assert not stored & set(
            _hashes(hasher, generation.scope(hasher, g).prefix_hash)
        )
    assert session.lookup_scope == generation.current_scope(hasher)


def test_l1_mode_sessions_are_bit_identical() -> None:
    hasher = TokenHasher(CHUNK_SIZE)
    manager = SessionManager(
        hasher, cache_generation=CacheGeneration("l1"), cleanup_interval=None
    )
    expected = hasher.compute_chunk_hashes(TOKENS)
    for session in (
        manager.get_or_create("a"),
        manager.get_or_create_for_lookup("b", HashScope(0, hasher.none_hash)),
    ):
        session.set_tokens(PACKED)
        assert session.get_hashes(0) == expected
        assert session.get_store_hashes(0, len(TOKENS)) == expected


def test_abandoned_lookup_scope_tracks_each_lookups_generation(scoped) -> None:
    hasher, generation, manager = scoped
    old = generation.current_scope(hasher)
    session = manager.get_or_create_for_lookup("r", old)
    session.begin_lookup(_lookup_key("r"), (-1,), old)
    session.record_prefetch_result(3, (0,))

    generation.bump()
    new = generation.current_scope(hasher)
    session.begin_lookup(_lookup_key("r"), (-1,), new)
    session.record_prefetch_result(1, (0,))

    assert session.abandoned_lookup_scope(3) == old
    assert session.abandoned_lookup_scope(1) == new
    # An unrecorded hit is ambiguous once lookups spanned a bump.
    assert session.abandoned_lookup_scope(2) is None


def test_abandoned_lookup_scope_is_ambiguous_for_equal_hits(scoped) -> None:
    hasher, generation, manager = scoped
    session = manager.get_or_create_for_lookup("r", generation.current_scope(hasher))
    for _ in range(2):
        session.begin_lookup(_lookup_key("r"), (-1,), generation.current_scope(hasher))
        session.record_prefetch_result(3, (0,))
        generation.bump()

    assert session.abandoned_lookup_scope(3) is None


def test_failed_retrieve_snapshot_carries_the_lookup_scope(scoped) -> None:
    hasher, generation, manager = scoped
    scope = generation.current_scope(hasher)
    session = manager.get_or_create_for_lookup("r", scope)
    session.begin_lookup(_lookup_key("r"), (-1,), scope)
    session.record_prefetch_result(2, (0,))
    generation.bump()

    worker_key = IPCCacheServerKey.from_token_ids(
        model_name="m",
        world_size=1,
        worker_id=0,
        token_ids=TOKENS,
        start=0,
        end=len(TOKENS),
        request_id="r",
    )
    snapshot = session.prepare_failed_retrieve_release(worker_key)
    assert snapshot is not None
    assert snapshot[-1] == scope


# ---------------------------------------------------------------------------
# StaleL2KeyCollector
# ---------------------------------------------------------------------------


class _RecordingAdapter:
    """L2 adapter double: records delete batches, optionally blocking or
    failing."""

    def __init__(self) -> None:
        self.deleted_batches: list[list[ObjectKey]] = []
        self.release = threading.Event()
        self.release.set()
        self.fail = False

    def delete(self, keys: list[ObjectKey]) -> None:
        self.release.wait()
        if self.fail:
            raise OSError("backend unavailable")
        self.deleted_batches.append(list(keys))

    @property
    def deleted(self) -> list[ObjectKey]:
        return [key for batch in self.deleted_batches for key in batch]


class _StorageManager:
    """The storage-manager methods the collector relies on."""

    def __init__(self, adapter: _RecordingAdapter) -> None:
        self.adapter = adapter
        self.listeners: list[L2AdapterListener] = []

    def register_l2_listener(self, listener: L2AdapterListener) -> None:
        self.listeners.append(listener)

    def unregister_l2_listener(self, listener: L2AdapterListener) -> None:
        self.listeners.remove(listener)

    def l2_adapters(self) -> list[tuple[None, _RecordingAdapter]]:
        return [(None, self.adapter)]


def _collector(
    adapter: _RecordingAdapter, **bounds: int
) -> tuple[StaleL2KeyCollector, _StorageManager]:
    storage = _StorageManager(adapter)
    return StaleL2KeyCollector(storage, **bounds), storage  # type: ignore[arg-type]


@pytest.fixture
def collector_parts():
    adapter = _RecordingAdapter()
    collector, storage = _collector(adapter)
    yield collector, storage, adapter
    adapter.release.set()
    if collector in storage.listeners:
        collector.close()


def test_collector_registers_as_l2_listener(collector_parts) -> None:
    collector, storage, _ = collector_parts
    assert storage.listeners == [collector]


def test_retire_returns_before_slow_deletes_finish(collector_parts) -> None:
    collector, _, adapter = collector_parts
    keys = [_object_key(i) for i in range(10)]
    collector.on_l2_keys_stored(keys, [1] * len(keys))
    adapter.release.clear()  # every delete now blocks

    started = time.monotonic()
    assert collector.retire_tracked_keys() == len(keys)
    assert time.monotonic() - started < 0.5

    adapter.release.set()
    assert _wait_for(lambda: collector.report_status()["deleted_keys"] == len(keys))
    assert sorted(adapter.deleted, key=lambda k: k.chunk_hash) == keys


def test_deletes_run_in_bounded_batches(collector_parts) -> None:
    collector, _, adapter = collector_parts
    keys = [_object_key(i) for i in range(2 * L2_DELETE_BATCH_SIZE + 1)]
    collector.on_l2_keys_stored(keys, [1] * len(keys))
    collector.retire_tracked_keys()

    assert _wait_for(lambda: len(adapter.deleted) == len(keys))
    assert max(len(b) for b in adapter.deleted_batches) <= L2_DELETE_BATCH_SIZE
    assert _wait_for(lambda: collector.report_status()["deleted_keys"] == len(keys))
    status = collector.report_status()
    assert status["pending_delete_keys"] == 0
    assert status["retired_keys"] == len(keys)


def test_keys_stored_after_retirement_wait_for_the_next_one(collector_parts) -> None:
    collector, _, adapter = collector_parts
    before, after = _object_key(1), _object_key(2)
    collector.on_l2_keys_stored([before], [1])
    collector.retire_tracked_keys()
    collector.on_l2_keys_stored([after], [1])

    assert _wait_for(lambda: adapter.deleted == [before])
    assert collector.report_status()["tracked_keys"] == 1

    collector.retire_tracked_keys()
    assert _wait_for(lambda: adapter.deleted == [before, after])


def test_keys_already_deleted_by_l2_are_not_tracked(collector_parts) -> None:
    collector, _, adapter = collector_parts
    evicted, kept = _object_key(1), _object_key(2)
    collector.on_l2_keys_stored([evicted, kept], [1, 1])
    collector.on_l2_keys_deleted([evicted])

    assert collector.retire_tracked_keys() == 1
    assert _wait_for(lambda: adapter.deleted == [kept])


def test_failed_delete_is_counted_and_not_reported_deleted(collector_parts) -> None:
    collector, _, adapter = collector_parts
    adapter.fail = True
    collector.on_l2_keys_stored([_object_key(1)], [1])
    collector.retire_tracked_keys()

    assert _wait_for(lambda: collector.report_status()["failed_delete_calls"] == 1)
    status = collector.report_status()
    assert status["deleted_keys"] == 0
    assert status["pending_delete_keys"] == 0

    # The collector keeps working after a failure.
    adapter.fail = False
    collector.on_l2_keys_stored([_object_key(2)], [1])
    collector.retire_tracked_keys()
    assert _wait_for(lambda: collector.report_status()["deleted_keys"] == 1)


def test_tracked_and_pending_bounds_are_independent() -> None:
    adapter = _RecordingAdapter()
    adapter.release.clear()  # keep retired keys pending behind one batch
    collector, _ = _collector(adapter, max_tracked_keys=4, max_pending_keys=6)
    try:
        collector.on_l2_keys_stored([_object_key(i) for i in range(6)], [1] * 6)
        assert collector.report_status()["untracked_keys"] == 2

        collector.retire_tracked_keys()  # 4 keys: may be popped as a batch
        collector.on_l2_keys_stored([_object_key(i) for i in range(6, 10)], [1] * 4)
        collector.retire_tracked_keys()
        collector.on_l2_keys_stored([_object_key(i) for i in range(10, 14)], [1] * 4)

        status = collector.report_status()
        # A full pending queue never stops tracking new stores.
        assert status["tracked_keys"] == 4
        assert status["pending_delete_keys"] <= 6
        assert status["retired_keys"] + status["dropped_keys"] == 8  # both retirements
    finally:
        adapter.release.set()
        collector.close()


def test_close_unregisters_and_stops_deleting(collector_parts) -> None:
    collector, storage, adapter = collector_parts
    collector.close()

    assert storage.listeners == []
    collector.on_l2_keys_stored([_object_key(1)], [1])
    collector.retire_tracked_keys()
    time.sleep(0.2)
    assert adapter.deleted == []
