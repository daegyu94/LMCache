# SPDX-License-Identifier: Apache-2.0
"""``LookupModule`` defers a lookup that needs chunks another engine is computing.

The storage manager is mocked and driven through the module's public handlers:
a deferred lookup submits no prefetch, its status polls report "not yet", and
it runs once the chunks land, their owner stops computing them, or it gives up.
"""

# Standard
from unittest.mock import MagicMock
import time

# Third Party
import pytest

# First Party
from lmcache.lmcache_native import Bitmap
from lmcache.v1.distributed.api import (
    DEFAULT_SHARD_REPLICATION_DESC,
    AttnWindowDesc,
    PrefetchHandle,
    PrefetchResult,
)
from lmcache.v1.multiprocess.config import MPServerConfig
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.modules.lookup import (
    _LEASE_SWEEP_EVERY,
    LookupModule,
)

CHUNK_SIZE = 256
NUM_CHUNKS = 3
LONG_TIMEOUT_S = 60.0


class _Env:
    """A mocked server context with controllable residency and sessions."""

    def __init__(self) -> None:
        self.resident = False
        self.ended: set[str] = set()
        self.ctx = ctx = MagicMock()
        ctx.chunk_size = CHUNK_SIZE
        ctx.event_bus.has_subscribers.return_value = False
        ctx.layout_desc_registry.find_attn_desc.return_value = AttnWindowDesc(
            num_chunks_in_sw=[-1], world_size=1
        )
        ctx.layout_desc_registry.find_shard_replication_desc.return_value = (
            DEFAULT_SHARD_REPLICATION_DESC
        )
        ctx.layout_desc_registry.find_group_layout_descs.return_value = {0: MagicMock()}
        ctx.token_hasher.compute_packed_chunk_hashes.return_value = [
            b"h0",
            b"h1",
            b"h2",
        ]
        ctx.session_manager.get.side_effect = (
            lambda rid: None if rid in self.ended else MagicMock()
        )
        ctx.session_manager.remove.side_effect = self._end
        l1 = MagicMock()
        l1.is_resident.side_effect = lambda _key: self.resident
        ctx.storage_manager.l1_managers.return_value = [l1]
        ctx.storage_manager.submit_prefetch_task.side_effect = (
            lambda _spec, external_request_id: PrefetchHandle(
                prefetch_request_id=0,
                external_request_id=external_request_id,
                total_requested_keys=NUM_CHUNKS,
                submit_time=time.monotonic(),
                sliding_windows=(-1,),
            )
        )
        self.set_hits(0)

    def set_hits(self, chunks: int) -> None:
        """Make every finished prefetch report ``chunks`` leading hits."""
        hits = Bitmap(NUM_CHUNKS, chunks)
        self.ctx.storage_manager.query_prefetch_status.return_value = PrefetchResult(
            hit_cells=[hits],
            l1_hit_cells=[hits.copy()],
            l2_hit_cells=[Bitmap(NUM_CHUNKS)],
        )

    def submitted(self) -> list[str]:
        """Return the request ids that submitted a prefetch, in order."""
        submit = self.ctx.storage_manager.submit_prefetch_task
        return [c.kwargs["external_request_id"] for c in submit.call_args_list]

    def _end(self, request_id: str) -> None:
        self.ended.add(request_id)


def _key(request_id: str, engine: str | None) -> IPCCacheServerKey:
    configs = None if engine is None else {"lmcache.hint.engine_id": engine}
    return IPCCacheServerKey.from_token_ids(
        model_name="m",
        world_size=1,
        worker_id=None,
        token_ids=range(NUM_CHUNKS * CHUNK_SIZE),
        start=0,
        end=NUM_CHUNKS * CHUNK_SIZE,
        request_id=request_id,
        request_configs=configs,
    )


def _deferred_pair(env: _Env, timeout_s: float = LONG_TIMEOUT_S) -> LookupModule:
    """Engine e1's "owner" leases the chunks; engine e2's "waiter" is deferred."""
    module = LookupModule(env.ctx, prefill_lease_timeout_s=timeout_s)
    module.lookup(_key("owner", "e1"), tp_size=1)
    module.lookup(_key("waiter", "e2"), tp_size=1)
    assert env.submitted() == ["owner"]
    return module


def test_leases_are_off_by_default() -> None:
    env = _Env()
    module = LookupModule(env.ctx)
    module.lookup(_key("owner", "e1"), tp_size=1)
    module.lookup(_key("waiter", "e2"), tp_size=1)
    assert env.submitted() == ["owner", "waiter"]
    assert not any(k.startswith("prefill_lease") for k in module.report_status())


def test_deferred_lookup_runs_once_the_chunks_are_resident() -> None:
    env = _Env()
    module = _deferred_pair(env)
    assert module.query_prefetch_status("waiter") is None
    assert env.submitted() == ["owner"]

    env.resident = True
    env.set_hits(NUM_CHUNKS)
    assert module.query_prefetch_status("waiter") == NUM_CHUNKS
    assert env.submitted() == ["owner", "waiter"]
    status = module.report_status()
    assert status["prefill_lease_waits"] == 1
    assert status["prefill_lease_served"] == 1
    assert status["prefill_lease_waiting"] == 0


def test_wait_handler_does_not_block_on_a_deferred_lookup() -> None:
    env = _Env()
    module = _deferred_pair(env)
    assert module.wait_prefetch_status("waiter", timeout=LONG_TIMEOUT_S) is None
    env.ctx.storage_manager.wait_prefetch_status.assert_not_called()


def test_owner_hitting_the_chunks_releases_the_waiter() -> None:
    env = _Env()
    module = _deferred_pair(env)
    env.set_hits(NUM_CHUNKS)
    assert module.query_prefetch_status("owner") == NUM_CHUNKS
    assert module.query_prefetch_status("waiter") == NUM_CHUNKS


def test_owner_missing_the_chunks_keeps_the_waiter_deferred() -> None:
    env = _Env()
    module = _deferred_pair(env)
    assert module.query_prefetch_status("owner") == 0
    assert module.query_prefetch_status("waiter") is None


def test_owner_ending_releases_the_waiter() -> None:
    env = _Env()
    module = _deferred_pair(env)
    module.end_session("owner")
    assert module.query_prefetch_status("waiter") == 0
    assert env.submitted() == ["owner", "waiter"]


def test_reaped_owner_does_not_hold_the_waiter() -> None:
    env = _Env()
    module = _deferred_pair(env)
    env.ended.add("owner")
    assert module.query_prefetch_status("waiter") == 0


def test_waiter_ending_submits_nothing() -> None:
    env = _Env()
    module = _deferred_pair(env)
    module.end_session("waiter")
    assert module.query_prefetch_status("waiter") == 0
    assert env.submitted() == ["owner"]
    assert module.report_status()["prefill_lease_waiting"] == 0


def test_reaped_waiter_is_swept_by_later_lookups() -> None:
    env = _Env()
    module = _deferred_pair(env)
    env.ended.add("waiter")
    for i in range(_LEASE_SWEEP_EVERY):
        module.lookup(_key(f"other-{i}", "e1"), tp_size=1)
    assert module.report_status()["prefill_lease_waiting"] == 0
    assert module.query_prefetch_status("waiter") == 0


def test_waiter_gives_up_when_the_owner_stores_nothing() -> None:
    env = _Env()
    module = _deferred_pair(env, timeout_s=0.01)
    time.sleep(0.02)
    assert module.query_prefetch_status("waiter") == 0
    assert env.submitted() == ["owner", "waiter"]
    assert module.report_status()["prefill_lease_timeouts"] == 1


def test_requests_of_one_engine_do_not_wait() -> None:
    env = _Env()
    module = LookupModule(env.ctx, prefill_lease_timeout_s=LONG_TIMEOUT_S)
    module.lookup(_key("a", "e1"), tp_size=1)
    module.lookup(_key("b", "e1"), tp_size=1)
    assert env.submitted() == ["a", "b"]


def test_clients_that_name_no_engine_take_no_part() -> None:
    env = _Env()
    module = LookupModule(env.ctx, prefill_lease_timeout_s=LONG_TIMEOUT_S)
    module.lookup(_key("anonymous", None), tp_size=1)
    module.lookup(_key("named", "e1"), tp_size=1)
    module.lookup(_key("anonymous-2", None), tp_size=1)
    assert env.submitted() == ["anonymous", "named", "anonymous-2"]


@pytest.mark.parametrize("timeout", [-1.0, float("nan"), float("inf")])
def test_invalid_timeout_is_rejected(timeout: float) -> None:
    with pytest.raises(ValueError, match="prefill lease timeout"):
        MPServerConfig(prefill_lease_timeout_seconds=timeout)
