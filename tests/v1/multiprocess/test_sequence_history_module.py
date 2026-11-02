# SPDX-License-Identifier: Apache-2.0
"""SequenceHistoryModule observes admitted lookups through LookupModule.

The storage manager and hasher are mocked; tests check which events the
module publishes and what it asks of the storage manager.
"""

# Standard
from typing import Any
from unittest.mock import MagicMock
import logging
import time

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.api import (
    AttnWindowDesc,
    ObjectKey,
    PrefetchHandle,
    PrefetchResult,
)
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.multiprocess.config import CoordinatorConfig, MPServerConfig
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.engine_module import ModuleBuildContext
from lmcache.v1.multiprocess.modules.lookup import LookupModule
from lmcache.v1.multiprocess.modules.sequence_history import SequenceHistoryModule
from lmcache.v1.multiprocess.sequence_tracker import SEQUENCE_ID_HINT

CHUNK_SIZE = 4
SEQ = {SEQUENCE_ID_HINT: "traj-1"}


def _ctx(num_groups: int = 1, world_size: int = 1) -> MagicMock:
    ctx = MagicMock()
    ctx.chunk_size = CHUNK_SIZE
    ctx.event_bus.has_subscribers.return_value = False
    ctx.layout_desc_registry.find_attn_desc.return_value = AttnWindowDesc(
        num_chunks_in_sw=[-1] * num_groups, world_size=world_size
    )
    ctx.layout_desc_registry.find_group_layout_descs.return_value = {
        gid: MagicMock() for gid in range(num_groups)
    }
    ctx.storage_manager.submit_prefetch_task.return_value = PrefetchHandle(
        prefetch_request_id=0,
        external_request_id="r",
        total_requested_keys=0,
        submit_time=time.monotonic(),
    )
    ctx.storage_manager.query_prefetch_status.return_value = PrefetchResult(
        hit_cells=[], l1_hit_cells=[], l2_hit_cells=[]
    )
    return ctx


def _build(
    ctx: MagicMock, max_chunks: int = 100, demotion: bool = False
) -> tuple[LookupModule, SequenceHistoryModule | None]:
    """Build both modules the way the server does: lookup first."""
    build_ctx = ModuleBuildContext(
        ctx,
        MPServerConfig(
            sequence_tracker_max_chunks=max_chunks, sequence_demotion=demotion
        ),
        CoordinatorConfig(),
    )
    lookup = LookupModule.create(build_ctx)
    assert isinstance(lookup, LookupModule)
    build_ctx.register(LookupModule, lookup)
    history = SequenceHistoryModule.create(build_ctx)
    assert history is None or isinstance(history, SequenceHistoryModule)
    return lookup, history


def _lookup(
    lookup: LookupModule,
    ctx: MagicMock,
    request_id: str,
    hashes: list[bytes],
    request_configs: dict[str, Any] | None,
    cache_salt: str = "",
    world_size: int = 1,
) -> IPCCacheServerKey:
    ctx.token_hasher.compute_packed_chunk_hashes.return_value = hashes
    key = IPCCacheServerKey.from_token_ids(
        model_name="m",
        world_size=world_size,
        worker_id=None,
        token_ids=range(len(hashes) * CHUNK_SIZE),
        end=len(hashes) * CHUNK_SIZE,
        request_id=request_id,
        cache_salt=cache_salt,
        request_configs=request_configs,
        num_kv_readers=1,
    )
    lookup.lookup(key, tp_size=1)
    return key


def _transitions(ctx: MagicMock) -> list[Event]:
    return [
        call.args[0]
        for call in ctx.event_bus.publish.call_args_list
        if call.args[0].event_type is EventType.MP_SEQUENCE_TRANSITION
    ]


def test_module_is_not_built_without_a_budget() -> None:
    ctx = _ctx()
    lookup, history = _build(ctx, max_chunks=0)
    assert history is None

    _lookup(lookup, ctx, "r1", [b"a", b"b"], SEQ)
    _lookup(lookup, ctx, "r2", [b"x"], SEQ)
    assert _transitions(ctx) == []


def test_rewrite_publishes_transition() -> None:
    ctx = _ctx()
    lookup, _ = _build(ctx)
    _lookup(lookup, ctx, "r1", [b"a", b"b", b"c"], SEQ)
    _lookup(lookup, ctx, "r2", [b"a", b"x"], SEQ)

    ctx.storage_manager.demote_cached_keys.assert_not_called()
    [event] = _transitions(ctx)
    assert event.session_id == "r2"
    assert event.metadata == {
        "kind": "rewrite",
        "divergence_chunk": 1,
        "previous_chunks": 3,
        "current_chunks": 2,
        "superseded_chunks": 2,
        "demotion_requested_chunks": 0,
        "model_name": "m",
    }


def test_append_publishes_transition() -> None:
    ctx = _ctx()
    lookup, _ = _build(ctx)
    _lookup(lookup, ctx, "r1", [b"a"], SEQ)
    _lookup(lookup, ctx, "r2", [b"a", b"b"], SEQ)

    [event] = _transitions(ctx)
    assert event.metadata["kind"] == "append"
    assert event.metadata["superseded_chunks"] == 0
    ctx.storage_manager.demote_cached_keys.assert_not_called()


def test_first_request_and_repeat_lookup_publish_nothing() -> None:
    ctx = _ctx()
    lookup, _ = _build(ctx)
    _lookup(lookup, ctx, "r1", [b"a", b"b"], SEQ)
    _lookup(lookup, ctx, "r1", [b"a"], SEQ)
    assert _transitions(ctx) == []


def test_same_sequence_id_under_another_salt_is_another_sequence() -> None:
    ctx = _ctx()
    lookup, _ = _build(ctx)
    _lookup(lookup, ctx, "r1", [b"a", b"b"], SEQ, cache_salt="t1")
    _lookup(lookup, ctx, "r2", [b"x"], SEQ, cache_salt="t2")
    assert _transitions(ctx) == []


@pytest.mark.parametrize("configs", [None, {"lmcache.skip_save": True}])
def test_requests_without_hint_are_untouched(configs: dict | None) -> None:
    ctx = _ctx()
    lookup, history = _build(ctx)
    assert history is not None
    _lookup(lookup, ctx, "r1", [b"a", b"b"], configs)
    _lookup(lookup, ctx, "r2", [b"x"], configs)

    assert _transitions(ctx) == []
    assert history.report_status() == {"tracked_sequences": 0, "tracked_chunks": 0}
    # The lookup itself is unchanged: one prefetch per lookup.
    assert ctx.storage_manager.submit_prefetch_task.call_count == 2


def test_declined_lookup_is_not_observed() -> None:
    ctx = _ctx()
    lookup, _ = _build(ctx)
    _lookup(lookup, ctx, "r1", [b"a", b"b"], SEQ)
    _lookup(lookup, ctx, "r2", [b"x"], {**SEQ, "lmcache.skip_load": True})
    _lookup(lookup, ctx, "r3", [b"a", b"b", b"c"], SEQ)

    [event] = _transitions(ctx)
    assert event.metadata["kind"] == "append"


def test_malformed_hint_is_ignored_with_one_warning() -> None:
    # lmcache loggers do not propagate, so caplog cannot see the records.
    records: list[logging.LogRecord] = []

    class _ListHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    ctx = _ctx()
    lookup, _ = _build(ctx)
    bad = {SEQUENCE_ID_HINT: "bad value/secret"}
    handler = _ListHandler(level=logging.WARNING)
    module_logger = logging.getLogger(
        "lmcache.v1.multiprocess.modules.sequence_history"
    )
    module_logger.addHandler(handler)
    try:
        _lookup(lookup, ctx, "r1", [b"a", b"b"], bad)
        _lookup(lookup, ctx, "r2", [b"x"], bad)
    finally:
        module_logger.removeHandler(handler)

    assert _transitions(ctx) == []
    warnings = [r.getMessage() for r in records if SEQUENCE_ID_HINT in r.getMessage()]
    assert len(warnings) == 1
    assert "secret" not in warnings[0]
    # The lookup itself still proceeds.
    assert ctx.storage_manager.submit_prefetch_task.call_count == 2


# ---------------------------------------------------------------------------
# Demotion
# ---------------------------------------------------------------------------


def _demoted(ctx: MagicMock) -> list[ObjectKey]:
    ctx.storage_manager.demote_cached_keys.assert_called_once()
    return ctx.storage_manager.demote_cached_keys.call_args.args[0]


def test_rewrite_demotes_superseded_chunks_in_chain_order() -> None:
    ctx = _ctx()
    lookup, _ = _build(ctx, demotion=True)
    _lookup(lookup, ctx, "r1", [b"a", b"b", b"c"], SEQ)
    _lookup(lookup, ctx, "r2", [b"a", b"x"], SEQ)

    assert [k.chunk_hash for k in _demoted(ctx)] == [b"b", b"c"]
    [event] = _transitions(ctx)
    assert event.metadata["demotion_requested_chunks"] == 2


def test_demotion_fans_out_chunk_major_over_groups_and_ranks() -> None:
    """Every (group, kv rank) key of a chunk comes before the next chunk's,
    so the policies demote whole chunks in chain order."""
    ctx = _ctx(num_groups=2, world_size=2)
    lookup, _ = _build(ctx, demotion=True)
    _lookup(lookup, ctx, "r1", [b"a", b"b", b"c"], SEQ, world_size=2)
    _lookup(lookup, ctx, "r2", [b"a"], SEQ, world_size=2)

    keys = _demoted(ctx)
    assert len(keys) == 2 * 2 * 2  # chunks x groups x ranks
    assert [k.chunk_hash for k in keys] == [b"b"] * 4 + [b"c"] * 4
    for chunk_keys in (keys[:4], keys[4:]):
        pairs = {(k.object_group_id, k.kv_rank) for k in chunk_keys}
        assert len(pairs) == 4
        assert {group for group, _ in pairs} == {0, 1}


def test_chunk_shared_with_another_sequence_is_not_demoted() -> None:
    ctx = _ctx()
    lookup, _ = _build(ctx, demotion=True)
    _lookup(lookup, ctx, "a1", [b"p", b"q", b"a"], {SEQUENCE_ID_HINT: "A"})
    _lookup(lookup, ctx, "b1", [b"p", b"q", b"b"], {SEQUENCE_ID_HINT: "B"})
    _lookup(lookup, ctx, "a2", [b"p", b"z"], {SEQUENCE_ID_HINT: "A"})

    assert [k.chunk_hash for k in _demoted(ctx)] == [b"a"]
    [event] = _transitions(ctx)
    assert event.metadata["superseded_chunks"] == 2
    assert event.metadata["demotion_requested_chunks"] == 1


def test_no_demotion_without_hint_or_module() -> None:
    ctx = _ctx()
    lookup, _ = _build(ctx, max_chunks=0)
    _lookup(lookup, ctx, "r1", [b"a", b"b"], SEQ)
    _lookup(lookup, ctx, "r2", [b"x"], SEQ)
    ctx2 = _ctx()
    lookup2, _ = _build(ctx2, demotion=True)
    _lookup(lookup2, ctx2, "r1", [b"a", b"b"], None)
    _lookup(lookup2, ctx2, "r2", [b"x"], None)

    ctx.storage_manager.demote_cached_keys.assert_not_called()
    ctx2.storage_manager.demote_cached_keys.assert_not_called()
