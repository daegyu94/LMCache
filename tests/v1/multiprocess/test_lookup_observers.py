# SPDX-License-Identifier: Apache-2.0
"""LookupModule.add_lookup_observer contract: who is called, in which order,
and what a failing observer does to the lookup."""

# Standard
from unittest.mock import MagicMock
import time

# First Party
from lmcache.v1.distributed.api import AttnWindowDesc, PrefetchHandle
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.modules.lookup import LookupModule

CHUNK_SIZE = 4
HASHES = [b"h0", b"h1"]


class _Recorder:
    def __init__(self, name: str, calls: list, fail: bool = False) -> None:
        self._name = name
        self._calls = calls
        self._fail = fail

    def on_lookup_admitted(
        self, key: IPCCacheServerKey, chunk_hashes: list[bytes]
    ) -> None:
        self._calls.append((self._name, key.request_id, list(chunk_hashes)))
        if self._fail:
            raise RuntimeError("observer bug")


def _ctx() -> MagicMock:
    ctx = MagicMock()
    ctx.chunk_size = CHUNK_SIZE
    ctx.event_bus.has_subscribers.return_value = False
    ctx.layout_desc_registry.find_attn_desc.return_value = AttnWindowDesc(
        num_chunks_in_sw=[-1], world_size=1
    )
    ctx.layout_desc_registry.find_group_layout_descs.return_value = {0: MagicMock()}
    ctx.token_hasher.compute_packed_chunk_hashes.return_value = HASHES
    ctx.storage_manager.submit_prefetch_task.return_value = PrefetchHandle(
        prefetch_request_id=0,
        external_request_id="r1",
        total_requested_keys=len(HASHES),
        submit_time=time.monotonic(),
    )
    return ctx


def _key(request_configs: dict | None = None) -> IPCCacheServerKey:
    return IPCCacheServerKey.from_token_ids(
        model_name="m",
        world_size=1,
        worker_id=None,
        token_ids=range(len(HASHES) * CHUNK_SIZE),
        end=len(HASHES) * CHUNK_SIZE,
        request_id="r1",
        request_configs=request_configs,
        num_kv_readers=1,
    )


def test_observers_are_called_in_registration_order_with_the_chain() -> None:
    calls: list = []
    module = LookupModule(_ctx())
    module.add_lookup_observer(_Recorder("first", calls))
    module.add_lookup_observer(_Recorder("second", calls))

    module.lookup(_key(), tp_size=1)

    assert calls == [("first", "r1", HASHES), ("second", "r1", HASHES)]


def test_raising_observer_does_not_fail_the_lookup_or_later_observers() -> None:
    calls: list = []
    ctx = _ctx()
    module = LookupModule(ctx)
    module.add_lookup_observer(_Recorder("broken", calls, fail=True))
    module.add_lookup_observer(_Recorder("after", calls))

    module.lookup(_key(), tp_size=1)

    assert [name for name, _, _ in calls] == ["broken", "after"]
    ctx.storage_manager.submit_prefetch_task.assert_called_once()
    assert module.report_status() == {"active_prefetch_jobs": 1}


def test_declined_lookup_is_not_observed() -> None:
    calls: list = []
    module = LookupModule(_ctx())
    module.add_lookup_observer(_Recorder("only", calls))

    module.lookup(_key({"lmcache.skip_load": True}), tp_size=1)

    assert calls == []
