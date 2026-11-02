# SPDX-License-Identifier: Apache-2.0
"""End-to-end tests for ``clear_mode="generation"`` on a real storage manager.

A real :class:`MPCacheServerContext` (L1 + mock L2 adapter) is driven through
the public ``LookupModule`` and ``ManagementModule`` handlers. Stores are
simulated by resolving a worker's store keys exactly as the STORE path does
(:meth:`MPCacheServerContext.resolve_store_obj_keys`) and committing them to
L1, from where the store controller copies them to L2.
"""

# Standard
from collections.abc import Callable, Iterator
from unittest.mock import MagicMock
import argparse
import threading
import time

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import EngineType
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
from lmcache.v1.multiprocess.config import (
    CoordinatorConfig,
    MPServerConfig,
    P2PConfig,
    add_mp_server_args,
    parse_args_to_mp_server_config,
    validate_clear_mode,
)
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.engine_context import MPCacheServerContext
from lmcache.v1.multiprocess.modules.lmcache_driven_transfer import (
    LMCacheDrivenTransferModule,
)
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


def _key(
    request_id: str, worker_id: int | None, tokens: list[int] = TOKENS
) -> IPCCacheServerKey:
    return IPCCacheServerKey.from_token_ids(
        model_name=MODEL,
        world_size=1,
        worker_id=worker_id,
        token_ids=tokens,
        start=0,
        end=len(tokens),
        request_id=request_id,
    )


class _Server:
    """The modules under test sharing one context."""

    def __init__(self, clear_mode: str, session_ttl_seconds: float = 600.0) -> None:
        self.ctx = MPCacheServerContext(
            _storage_config(),
            chunk_size=CHUNK_SIZE,
            clear_mode=clear_mode,  # type: ignore[arg-type]
            session_ttl_seconds=session_ttl_seconds,
        )
        self.ctx.layout_desc_registry.register(MODEL, 1, LAYOUT)
        self.lookup = LookupModule(self.ctx)
        self.management = ManagementModule(self.ctx)

    @property
    def l2(self) -> MockL2Adapter:
        adapter: L2AdapterInterface = self.ctx.storage_manager.l2_adapters()[0][1]
        assert isinstance(adapter, MockL2Adapter)
        return adapter

    def close(self) -> None:
        self.management.close()
        self.lookup.close()
        self.ctx.close()

    def store_keys(self, request_id: str) -> list[ObjectKey]:
        """Resolve worker 0's keys the way STORE does."""
        return self.ctx.resolve_store_obj_keys(_key(request_id, worker_id=0), [0])[0]

    def commit(self, keys: list[ObjectKey]) -> None:
        """Commit keys to L1 and wait until the store controller wrote L2."""
        sm = self.ctx.storage_manager
        reserved = sm.reserve_write(keys, LAYOUT)
        assert len(reserved) == len(keys)
        sm.finish_write(list(reserved), l1_owners_of(reserved))
        assert _wait_for(
            lambda: all(self.l2.debug_has_key(k) for k in keys)
            and all(self.evictable(k) for k in keys)
        )

    def evictable(self, key: ObjectKey) -> bool:
        return self.ctx.storage_manager.l1_managers()[0].is_key_evictable(key)

    def lookup_hits(self, request_id: str) -> int:
        """Run a scheduler lookup; its locks stay held. Return hit chunks."""
        self.lookup.lookup(_key(request_id, worker_id=None), 1)
        hits = self.lookup.wait_prefetch_status(request_id, 10.0)
        assert hits is not None
        return hits

    def free(self, request_id: str, extra_ranges: tuple = ()) -> None:
        self.lookup.free_lookup_locks(_key(request_id, worker_id=None), 1, extra_ranges)

    def read(self, request_id: str) -> int:
        """A complete reader: lookup, release, end session."""
        hits = self.lookup_hits(request_id)
        self.free(request_id)
        self.lookup.end_session(request_id)
        return hits

    def write(self, request_id: str) -> list[ObjectKey]:
        """A complete writer: lookup, store, release, end session."""
        self.lookup_hits(request_id)
        keys = self.store_keys(request_id)
        self.commit(keys)
        self.free(request_id)
        self.lookup.end_session(request_id)
        return keys


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


# ---------------------------------------------------------------------------
# Default mode
# ---------------------------------------------------------------------------


def test_default_mode_keeps_hashes_and_serves_l2_after_clear(l1_server) -> None:
    # A store with no preceding lookup is trusted in l1 mode, as before.
    keys = l1_server.store_keys("writer")
    l1_server.commit(keys)

    l1_server.management.clear()

    assert l1_server.ctx.cache_generation.current == 0
    assert "cache_generation" not in l1_server.management.report_status()
    # L2 is untouched by an L1 clear, so the old KV is promoted back.
    assert l1_server.read("reader") == NUM_CHUNKS
    assert all(l1_server.l2.debug_has_key(k) for k in keys)


# ---------------------------------------------------------------------------
# Generation mode
# ---------------------------------------------------------------------------


def test_generation_clear_makes_l1_and_l2_unreachable(generation_server) -> None:
    server = generation_server
    old_keys = server.write("writer")
    assert server.read("warm") == NUM_CHUNKS

    server.management.clear()

    assert server.ctx.cache_generation.current == 1
    assert server.read("reader") == 0
    # The old objects are garbage collected from L2 in the background.
    assert _wait_for(lambda: not any(server.l2.debug_has_key(k) for k in old_keys))
    status = server.management.report_status()["cache_generation"]
    assert isinstance(status, dict)
    assert status["generation"] == 1
    assert status["stale_l2_gc"]["deleted_keys"] >= len(old_keys)


def test_new_generation_data_round_trips(generation_server) -> None:
    server = generation_server
    server.management.clear()
    server.write("writer")

    assert server.read("reader") == NUM_CHUNKS


def test_pre_bump_store_committing_after_bump_is_not_visible(
    generation_server,
) -> None:
    server = generation_server
    server.lookup_hits("in-flight")
    in_flight_keys = server.store_keys("in-flight")  # resolved before the bump
    server.management.clear()
    server.commit(in_flight_keys)  # written after it

    assert server.read("reader") == 0
    # The late commit is tracked anew and collected by the next clear.
    server.management.clear()
    assert _wait_for(
        lambda: not any(server.l2.debug_has_key(k) for k in in_flight_keys)
    )


def test_store_arriving_after_relookup_stays_in_old_generation(
    generation_server,
) -> None:
    server = generation_server
    request_id = "spanning"
    server.lookup_hits(request_id)
    server.free(request_id)

    server.management.clear()
    # The engine re-schedules the request under the new weights; a store it
    # issued before the reset (lazy offload, delayed COMMIT) arrives only now.
    assert server.lookup_hits(request_id) == 0
    late_keys = server.store_keys(request_id)
    server.commit(late_keys)
    server.free(request_id)
    server.lookup.end_session(request_id)

    assert server.read("reader") == 0


def test_store_without_a_lookup_is_never_served(generation_server) -> None:
    server = generation_server
    server.management.clear()
    # E.g. the request's session was reaped by TTL: nothing proves which
    # weights computed this KV.
    keys = server.store_keys("sessionless")
    server.commit(keys)

    assert server.read("reader") == 0


def test_release_after_bump_frees_the_old_generation_locks(
    generation_server,
) -> None:
    server = generation_server
    old_keys = server.write("writer")
    assert server.lookup_hits("holder") == NUM_CHUNKS
    assert not any(server.evictable(k) for k in old_keys)

    server.management.clear()  # non-forced: locked L1 objects survive
    server.free("holder")

    assert all(server.evictable(k) for k in old_keys)


def test_abandoned_range_release_across_bump_uses_its_own_generation(
    generation_server,
) -> None:
    server = generation_server
    old_keys = server.write("writer")
    # Another reader also holds the old objects.
    assert server.lookup_hits("other") == NUM_CHUNKS
    # The request's first lookup locks the old objects, then is abandoned.
    assert server.lookup_hits("abandoning") == NUM_CHUNKS

    server.management.clear()
    assert server.lookup_hits("abandoning") == 0
    server.free("abandoning", extra_ranges=((0, NUM_CHUNKS * CHUNK_SIZE),))

    # Only the abandoned lookup's share was released.
    assert not any(server.evictable(k) for k in old_keys)
    server.free("other")
    assert all(server.evictable(k) for k in old_keys)


def test_lookup_racing_the_bump_releases_what_it_locked(generation_server) -> None:
    server = generation_server
    old_keys = server.write("writer")
    # Keep the old objects L1-resident through the clear, so the racer's
    # result does not depend on the background L2 deletes.
    assert server.lookup_hits("holder") == NUM_CHUNKS
    sm = server.ctx.storage_manager
    hashed = threading.Event()
    proceed = threading.Event()
    submit = sm.submit_prefetch_task

    def gated_submit(*args, **kwargs):
        hashed.set()
        proceed.wait()
        return submit(*args, **kwargs)

    sm.submit_prefetch_task = gated_submit  # type: ignore[method-assign]
    racer_hits: list[int] = []
    racer = threading.Thread(
        target=lambda: racer_hits.append(server.lookup_hits("racer"))
    )
    racer.start()
    try:
        assert hashed.wait(5.0)  # the lookup hashed under the old generation
        clear = threading.Thread(target=server.management.clear)
        clear.start()
        clear.join(5.0)
        assert not clear.is_alive()  # the clear never waits for the lookup
    finally:
        proceed.set()
        racer.join(10.0)
        sm.submit_prefetch_task = submit  # type: ignore[method-assign]

    # The racing lookup is linearized before the bump: it hit the
    # old-generation objects, and its release after the bump resolves the
    # old-generation keys it locked.
    assert racer_hits == [NUM_CHUNKS]
    server.free("racer")
    server.free("holder")
    assert all(server.evictable(k) for k in old_keys)
    assert server.read("after") == 0


def test_sub_chunk_lookup_then_decode_store_is_served(generation_server) -> None:
    server = generation_server
    request_id = "short-prompt"
    prompt = TOKENS[: CHUNK_SIZE // 2]  # no complete chunk to look up
    server.lookup.lookup(_key(request_id, None, prompt), 1)
    assert server.lookup.wait_prefetch_status(request_id, 10.0) == 0
    # Decode crosses chunk boundaries; the worker stores the full sequence.
    server.commit(server.store_keys(request_id))
    server.lookup.end_session(request_id)

    assert server.read("turn-2") == NUM_CHUNKS


def test_long_request_outliving_the_session_ttl_still_stores() -> None:
    server = _Server("generation", session_ttl_seconds=0.05)
    try:
        assert server.lookup_hits("long") == 0
        server.free("long")
        time.sleep(0.1)
        assert server.ctx.session_manager.cleanup_expired() == 1

        server.commit(server.store_keys("long"))  # store after reaping
        assert server.lookup_hits("long") == NUM_CHUNKS  # re-lookup hits
        server.free("long")
        server.lookup.end_session("long")
        assert server.read("turn-2") == NUM_CHUNKS
    finally:
        server.close()


def test_generation_mode_registers_only_vllm_workers(generation_server) -> None:
    module = LMCacheDrivenTransferModule(generation_server.ctx)
    try:
        with pytest.raises(ValueError, match="only vLLM"):
            module.register_kv_cache(
                1, MagicMock(), MODEL, 1, EngineType.SGLANG, MagicMock(), []
            )
    finally:
        module.close()


def test_close_stops_tracking_l2_keys(generation_server) -> None:
    server = generation_server
    server.management.close()
    keys = server.write("writer")
    status = server.management.report_status()["cache_generation"]
    assert isinstance(status, dict)
    assert status["stale_l2_gc"]["tracked_keys"] == 0
    assert all(server.l2.debug_has_key(k) for k in keys)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def _parse(argv: list[str]) -> MPServerConfig:
    parser = argparse.ArgumentParser()
    add_mp_server_args(parser)
    return parse_args_to_mp_server_config(parser.parse_args(argv))


def test_cli_parses_clear_mode() -> None:
    assert _parse([]).clear_mode == "l1"
    assert _parse(["--clear-mode", "generation"]).clear_mode == "generation"
    with pytest.raises(SystemExit):
        _parse(["--clear-mode", "all"])


def test_generation_mode_rejects_unsupported_combinations() -> None:
    with pytest.raises(ValueError, match="blend"):
        MPServerConfig(clear_mode="generation", engine_type="blend")
    with pytest.raises(ValueError, match="P2P"):
        MPServerConfig(
            clear_mode="generation",
            p2p_config=P2PConfig(advertise_url="tcp://10.0.0.1:6000"),
        )
    with pytest.raises(ValueError, match="coordinator"):
        validate_clear_mode(
            MPServerConfig(clear_mode="generation"),
            CoordinatorConfig(url="http://coordinator:9300"),
        )
    validate_clear_mode(MPServerConfig(), CoordinatorConfig(url="http://c:9300"))
    validate_clear_mode(MPServerConfig(clear_mode="generation"), CoordinatorConfig())
