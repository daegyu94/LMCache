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
    StaleL2KeyCollector,
)
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.session import SessionManager
from lmcache.v1.multiprocess.token_codec import pack_token_ids
from lmcache.v1.multiprocess.token_hasher import TokenHasher

CHUNK_SIZE = 4
TOKENS = list(range(100, 100 + 3 * CHUNK_SIZE))


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


# ---------------------------------------------------------------------------
# TokenHasher.generation_prefix_hash
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("algorithm", ["blake3", "builtin"])
def test_generation_zero_hashes_are_unchanged(algorithm: str) -> None:
    hasher = TokenHasher(chunk_size=CHUNK_SIZE, hash_algorithm=algorithm)
    seed = hasher.generation_prefix_hash(0)

    assert seed == hasher.none_hash
    assert hasher.compute_chunk_hashes(TOKENS, prefix_hash=seed) == (
        hasher.compute_chunk_hashes(TOKENS)
    )
    assert hasher.compute_packed_chunk_hashes(
        pack_token_ids(TOKENS), prefix_hash=seed
    ) == hasher.compute_packed_chunk_hashes(pack_token_ids(TOKENS))


def test_each_generation_hashes_the_same_tokens_differently() -> None:
    hasher = TokenHasher(chunk_size=CHUNK_SIZE)
    per_generation = [
        hasher.compute_chunk_hashes(
            TOKENS, prefix_hash=hasher.generation_prefix_hash(generation)
        )
        for generation in range(4)
    ]

    every_hash = [h for hashes in per_generation for h in hashes]
    assert len(set(every_hash)) == len(every_hash)


def test_generation_hashes_are_deterministic_across_hashers() -> None:
    first = TokenHasher(chunk_size=CHUNK_SIZE)
    second = TokenHasher(chunk_size=CHUNK_SIZE)
    # Alternate generations so a memoized seed is never simply reused.
    for generation in (3, 1, 3):
        assert first.compute_chunk_hashes(
            TOKENS, prefix_hash=first.generation_prefix_hash(generation)
        ) == second.compute_chunk_hashes(
            TOKENS, prefix_hash=second.generation_prefix_hash(generation)
        )


def test_generation_seed_never_aliases_a_longer_generation_zero_prefix() -> None:
    # With one-token chunks, a seed built from a real chunk would make
    # generation-g hashes of T equal generation-0 hashes of [g] + T.
    hasher = TokenHasher(chunk_size=1)
    gen1 = hasher.compute_chunk_hashes(
        [7, 8], prefix_hash=hasher.generation_prefix_hash(1)
    )
    gen0_shifted = hasher.compute_chunk_hashes([1, 7, 8])

    assert not set(gen1) & set(gen0_shifted)


def test_negative_generation_is_rejected() -> None:
    with pytest.raises(ValueError):
        TokenHasher(chunk_size=CHUNK_SIZE).generation_prefix_hash(-1)


# ---------------------------------------------------------------------------
# CacheGeneration
# ---------------------------------------------------------------------------


def test_cache_generation_starts_at_zero_and_bumps_monotonically() -> None:
    generation = CacheGeneration()
    assert generation.current == 0
    assert generation.bump() == 1
    assert generation.bump() == 2
    assert generation.current == 2


def test_concurrent_bumps_are_never_lost() -> None:
    generation = CacheGeneration()
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
# Session generation semantics
# ---------------------------------------------------------------------------


def _expected_hashes(hasher: TokenHasher, generation: int) -> list[bytes]:
    return hasher.compute_packed_chunk_hashes(
        pack_token_ids(TOKENS), prefix_hash=hasher.generation_prefix_hash(generation)
    )


def test_new_sessions_hash_under_the_current_generation() -> None:
    hasher = TokenHasher(chunk_size=CHUNK_SIZE)
    generation = CacheGeneration()
    manager = SessionManager(hasher, cleanup_interval=None, cache_generation=generation)
    generation.bump()

    session = manager.get_or_create("fresh")
    session.set_tokens(pack_token_ids(TOKENS))

    assert session.cache_generation == 1
    assert session.get_hashes(0) == _expected_hashes(hasher, 1)


def test_pre_bump_session_keeps_its_generation_until_next_lookup() -> None:
    hasher = TokenHasher(chunk_size=CHUNK_SIZE)
    generation = CacheGeneration()
    manager = SessionManager(hasher, cleanup_interval=None, cache_generation=generation)
    session = manager.get_or_create("spanning")
    session.set_tokens(pack_token_ids(TOKENS))
    old_hashes = session.get_hashes(0)

    generation.bump()

    # A store resolved through the pre-bump session (KV computed under the
    # old weights) must not land under new-generation hashes.
    assert manager.get_or_create("spanning") is session
    assert session.get_hashes(0) == old_hashes == _expected_hashes(hasher, 0)

    # A lookup under the new generation re-scopes the session.
    session.begin_lookup(_lookup_key("spanning"), (-1,), cache_generation=1)
    assert session.cache_generation == 1
    assert session.get_hashes(0) == _expected_hashes(hasher, 1)


def test_default_session_manager_stays_at_generation_zero() -> None:
    hasher = TokenHasher(chunk_size=CHUNK_SIZE)
    session = SessionManager(hasher, cleanup_interval=None).get_or_create("r")
    session.set_tokens(pack_token_ids(TOKENS))

    assert session.cache_generation == 0
    assert session.get_hashes(0) == hasher.compute_chunk_hashes(TOKENS)


# ---------------------------------------------------------------------------
# StaleL2KeyCollector
# ---------------------------------------------------------------------------


class _RecordingAdapter:
    """L2 adapter double: records delete batches, optionally blocking."""

    def __init__(self) -> None:
        self.deleted_batches: list[list[ObjectKey]] = []
        self.release = threading.Event()
        self.release.set()

    def delete(self, keys: list[ObjectKey]) -> None:
        self.release.wait()
        self.deleted_batches.append(list(keys))

    @property
    def deleted(self) -> list[ObjectKey]:
        return [key for batch in self.deleted_batches for key in batch]


class _StorageManager:
    """The two storage-manager methods the collector relies on."""

    def __init__(self, adapter: _RecordingAdapter) -> None:
        self.adapter = adapter
        self.listeners: list[L2AdapterListener] = []

    def register_l2_listener(self, listener: L2AdapterListener) -> None:
        self.listeners.append(listener)

    def l2_adapters(self) -> list[tuple[None, _RecordingAdapter]]:
        return [(None, self.adapter)]


@pytest.fixture
def collector_parts():
    adapter = _RecordingAdapter()
    storage = _StorageManager(adapter)
    collector = StaleL2KeyCollector(storage)  # type: ignore[arg-type]
    yield collector, storage, adapter
    adapter.release.set()
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
    status = collector.report_status()
    assert status["pending_delete_keys"] == 0
    assert status["retired_keys"] == status["deleted_keys"] == len(keys)


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


def test_tracking_is_bounded_across_tracked_and_pending_keys() -> None:
    adapter = _RecordingAdapter()
    adapter.release.clear()  # keep retired keys pending
    collector = StaleL2KeyCollector(
        _StorageManager(adapter),  # type: ignore[arg-type]
        max_tracked_keys=8,
    )
    try:
        collector.on_l2_keys_stored([_object_key(i) for i in range(6)], [1] * 6)
        collector.retire_tracked_keys()
        collector.on_l2_keys_stored([_object_key(i) for i in range(6, 12)], [1] * 6)

        status = collector.report_status()
        assert status["tracked_keys"] + status["pending_delete_keys"] <= 8
        assert status["tracked_keys"] >= 2
        assert status["untracked_keys"] == 12 - 8
    finally:
        adapter.release.set()
        collector.close()
