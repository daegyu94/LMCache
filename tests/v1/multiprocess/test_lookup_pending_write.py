# SPDX-License-Identifier: Apache-2.0
"""LookupModule holds a lookup while its first missing chunk is being stored.

The L1 manager and the storage manager's ``has_pending_write`` are real; only
the prefetch submission is mocked, so the tests observe whether and when the
module submits.
"""

# Standard
from unittest.mock import MagicMock
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
    ipc_key_to_object_keys,
)
from lmcache.v1.distributed.config import (
    EvictionConfig,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    StorageManagerConfig,
)
from lmcache.v1.distributed.internal_api import L1KeyState
from lmcache.v1.distributed.l1_manager import L1Manager
from lmcache.v1.distributed.storage_manager import StorageManager
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.modules.lookup import LookupModule
import lmcache.v1.memory_management as memory_management

pytestmark = pytest.mark.no_shared_allocator

CHUNK_SIZE = 256
CHUNK_HASHES = [b"h0", b"h1", b"h2"]
LAYOUT = MemoryLayoutDesc(shapes=[torch.Size([1024])], dtypes=[torch.float32])
WAIT_MS = 200


def _key(request_id: str = "req-1", world_size: int = 1) -> IPCCacheServerKey:
    return IPCCacheServerKey(
        model_name="m",
        world_size=world_size,
        worker_id=None,
        token_ids=tuple(range(len(CHUNK_HASHES) * CHUNK_SIZE)),
        start=0,
        end=len(CHUNK_HASHES) * CHUNK_SIZE,
        request_id=request_id,
        num_kv_readers=1,
    )


def _object_keys(chunk: int, world_size: int = 1) -> list[ObjectKey]:
    """Every rank's key of one chunk."""
    return ipc_key_to_object_keys(_key(world_size=world_size), CHUNK_HASHES, [0])[0][
        chunk * world_size : (chunk + 1) * world_size
    ]


class _Env:
    """A LookupModule over a real StorageManager with ``num_l1`` L1 managers.

    Only prefetch submission/query is replaced, to observe when the module
    submits.
    """

    def __init__(
        self,
        monkeypatch: pytest.MonkeyPatch,
        wait_ms: int,
        world_size: int = 1,
        num_l1: int = 1,
        max_deferred: int = 256,
    ) -> None:
        self.world_size = world_size
        configs = [
            L1ManagerConfig(
                L1MemoryManagerConfig(
                    size_in_bytes=1024 * 1024, use_lazy=False, shm_name=""
                ),
                tag="_default" if i == 0 else f"l1-{i}",
            )
            for i in range(num_l1)
        ]
        self.l1s = tuple(L1Manager(config) for config in configs)
        self.l1 = self.l1s[0]
        self.storage = StorageManager(
            StorageManagerConfig(configs[0], EvictionConfig("noop")),
            _l1_managers=self.l1s,
        )
        self.hit_chunks = -1
        self.submit = MagicMock(side_effect=self._submit)
        monkeypatch.setattr(self.storage, "submit_prefetch_task", self.submit)
        monkeypatch.setattr(
            self.storage, "query_prefetch_status", MagicMock(side_effect=self._query)
        )
        monkeypatch.setattr(
            self.storage, "wait_prefetch_status", MagicMock(return_value=True)
        )
        ctx = MagicMock()
        ctx.chunk_size = CHUNK_SIZE
        ctx.storage_manager = self.storage
        ctx.event_bus.has_subscribers.return_value = False
        ctx.layout_desc_registry.find_attn_desc.return_value = AttnWindowDesc(
            num_chunks_in_sw=[-1], world_size=world_size
        )
        ctx.layout_desc_registry.find_group_layout_descs.return_value = {0: LAYOUT}
        ctx.token_hasher.compute_chunk_hashes.return_value = CHUNK_HASHES
        ctx.session_manager.get_or_create.return_value.prefetch_hit_chunks = -1
        self.ctx = ctx
        self.module = LookupModule(
            ctx, pending_write_wait_ms=wait_ms, max_deferred_lookups=max_deferred
        )

    def _states(self, keys: list[ObjectKey]) -> list[L1KeyState]:
        """Resident if any L1 holds the key, else the first L1's state."""
        per_l1 = [l1.probe_keys(keys) for l1 in self.l1s]
        return [
            L1KeyState.RESIDENT
            if L1KeyState.RESIDENT in states
            else max(states, key=lambda st: st is L1KeyState.STAGED)
            for states in zip(*per_l1, strict=False)
        ]

    def _submit(self, spec, external_request_id: str) -> PrefetchHandle:
        """Record the resident prefix at submit time, as a prefetch would."""
        hits = 0
        for i in range(spec.group_size):
            if any(
                state is not L1KeyState.RESIDENT
                for state in self._states([row.keys[i] for row in spec.key_groups])
            ):
                break
            hits += 1
        self.hit_chunks = hits
        return PrefetchHandle(
            prefetch_request_id=0,
            external_request_id=external_request_id,
            total_requested_keys=spec.group_size,
            submit_time=time.monotonic(),
        )

    def _query(self, handle: PrefetchHandle) -> PrefetchResult:
        rows = [Bitmap(len(CHUNK_HASHES), self.hit_chunks)] * self.world_size
        return PrefetchResult(hit_cells=rows, l1_hit_cells=rows, l2_hit_cells=[])

    def lookup(self, request_id: str = "req-1") -> None:
        self.module.lookup(_key(request_id, self.world_size), tp_size=1)

    @property
    def submitted(self) -> int:
        return self.submit.call_count

    def reserve(self, chunk: int) -> list[ObjectKey]:
        keys = _object_keys(chunk, self.world_size)
        self.l1.reserve_write(keys, [False] * len(keys), LAYOUT)
        return keys

    def commit(self, keys: list[ObjectKey]) -> None:
        self.l1.finish_write(keys)

    def close(self) -> None:
        self.storage.close()


@pytest.fixture
def make_env(monkeypatch: pytest.MonkeyPatch):
    # Real CPU allocators and L1 lifecycle; only host pinning is replaced.
    monkeypatch.setattr(
        memory_management,
        "_allocate_cpu_memory",
        lambda size, *args, **kwargs: torch.empty(size, dtype=torch.uint8),
    )
    monkeypatch.setattr(memory_management, "_free_cpu_memory", lambda *a, **k: None)
    envs: list[_Env] = []

    def _make(wait_ms: int = WAIT_MS, **kwargs: int) -> _Env:
        env = _Env(monkeypatch, wait_ms, **kwargs)
        envs.append(env)
        return env

    yield _make
    for env in envs:
        env.close()


def test_staged_prefix_defers_then_submits_after_commit(make_env) -> None:
    env = make_env()
    keys = env.reserve(0)
    env.lookup()

    assert env.submitted == 0
    assert env.module.query_prefetch_status("req-1") is None
    assert env.submitted == 0

    env.commit(keys)
    assert env.module.query_prefetch_status("req-1") == 1
    assert env.submitted == 1


def test_nothing_staged_submits_immediately(make_env) -> None:
    env = make_env()
    env.lookup()
    assert env.submitted == 1


def test_gap_before_staged_chunk_does_not_defer(make_env) -> None:
    """The prefix ends at the first absent chunk, so a later staged chunk
    is no reason to wait."""
    env = make_env()
    env.reserve(1)
    env.lookup()
    assert env.submitted == 1


def test_resident_prefix_then_staged_chunk_defers(make_env) -> None:
    env = make_env()
    first = env.reserve(0)
    env.commit(first)
    env.reserve(1)
    env.lookup()
    assert env.submitted == 0


def test_deadline_submits_even_if_still_staged(make_env) -> None:
    env = make_env(wait_ms=30)
    env.reserve(0)
    env.lookup()
    assert env.module.query_prefetch_status("req-1") is None

    time.sleep(0.05)
    assert env.module.query_prefetch_status("req-1") == 0  # still a miss
    assert env.submitted == 1


def test_zero_wait_never_probes_or_defers(make_env, monkeypatch) -> None:
    env = make_env(wait_ms=0)
    env.reserve(0)
    probe = MagicMock()
    monkeypatch.setattr(env.storage, "has_pending_write", probe)
    env.lookup()
    assert env.module.query_prefetch_status("req-1") == 0
    assert env.module.wait_prefetch_status("req-1", timeout=0.1) == 0
    env.module.free_lookup_locks(_key(), tp_size=1)
    env.module.end_session("req-1")

    assert env.submitted == 1
    probe.assert_not_called()


def test_one_staged_rank_defers_multi_rank_chunk(make_env) -> None:
    env = make_env(world_size=2)
    rank0, rank1 = _object_keys(0, 2)
    env.l1.reserve_write([rank0], [False], LAYOUT)
    env.commit([rank0])
    env.l1.reserve_write([rank1], [False], LAYOUT)
    env.lookup()
    assert env.submitted == 0

    env.commit([rank1])
    assert env.module.query_prefetch_status("req-1") == 1
    assert env.submitted == 1


@pytest.mark.parametrize("end", ["end_session", "free_lookup_locks"])
def test_ending_a_deferred_lookup_drops_it(make_env, end: str) -> None:
    env = make_env()
    keys = env.reserve(0)
    env.lookup()

    if end == "end_session":
        env.module.end_session("req-1")
    else:
        env.module.free_lookup_locks(_key(), tp_size=1)
    env.commit(keys)

    assert env.submitted == 0
    assert env.module.query_prefetch_status("req-1") == 0
    assert env.submitted == 0


def test_relookup_replaces_deferred_job(make_env) -> None:
    env = make_env()
    keys = env.reserve(0)
    env.lookup()
    env.lookup()
    env.commit(keys)

    assert env.module.query_prefetch_status("req-1") == 1
    assert env.submitted == 1


def test_cap_bypass_submits_without_deferring(make_env) -> None:
    env = make_env(max_deferred=1)
    env.reserve(0)
    env.lookup("req-1")
    env.lookup("req-2")

    assert env.submitted == 1  # req-2 was not held
    assert env.module.query_prefetch_status("req-1") is None


def test_wait_prefetch_status_blocks_until_commit(make_env) -> None:
    env = make_env()
    keys = env.reserve(0)
    env.lookup()

    assert env.module.wait_prefetch_status("req-1", timeout=0.02) is None
    threading.Timer(0.05, env.commit, args=(keys,)).start()
    assert env.module.wait_prefetch_status("req-1", timeout=2.0) == 1
    assert env.submitted == 1


def test_concurrent_writer_and_pollers_submit_exactly_once(make_env) -> None:
    env = make_env(wait_ms=2000)
    keys = env.reserve(0)
    env.lookup()
    results: list[int | None] = []

    def poll() -> None:
        while (r := env.module.query_prefetch_status("req-1")) is None:
            time.sleep(0.001)
        results.append(r)

    pollers = [threading.Thread(target=poll) for _ in range(4)]
    for t in pollers:
        t.start()
    time.sleep(0.05)
    env.commit(keys)
    for t in pollers:
        t.join(timeout=5)

    assert env.submitted == 1
    assert results.count(1) == 1  # the job is consumed exactly once


def test_has_pending_write_merges_all_l1_managers(make_env) -> None:
    env = make_env(num_l1=2)
    first, second = env.l1s
    key = _object_keys(0)

    second.reserve_write(key, [False], LAYOUT)
    assert env.storage.has_pending_write([key]) is True

    # Resident in another L1 wins over a reservation here.
    first.reserve_write(key, [False], LAYOUT)
    first.finish_write(key)
    assert env.storage.has_pending_write([key]) is False
