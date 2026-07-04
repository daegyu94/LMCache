# SPDX-License-Identifier: Apache-2.0

# Standard
from types import SimpleNamespace
from typing import cast
import threading

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.l1_manager import L1Manager
from lmcache.v1.distributed.l2_adapters.base import L2AdapterInterface
from lmcache.v1.distributed.storage_controllers.store_controller import StoreController
from lmcache.v1.distributed.storage_controllers.store_policy import (
    AdapterDescriptor,
    StorePolicy,
)
from lmcache.v1.mp_observability.event_bus import EventBus


def make_object_key(chunk_id: int) -> ObjectKey:
    """Create a test ObjectKey with the given chunk ID."""
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(chunk_id),
        model_name="test_model",
        kv_rank=0,
    )


class _FakeMemoryObj:
    def get_size(self) -> int:
        return 128


class _FakeL1Manager:
    def __init__(self) -> None:
        self.finished_reads: list[list[ObjectKey]] = []

    def reserve_read(
        self, keys: list[ObjectKey]
    ) -> dict[ObjectKey, tuple[L1Error, object]]:
        return {key: (L1Error.SUCCESS, _FakeMemoryObj()) for key in keys}

    def finish_read(self, keys: list[ObjectKey]) -> None:
        self.finished_reads.append(list(keys))


class _FakeEventBus:
    def __init__(self) -> None:
        self.events: list[object] = []

    def publish(self, event: object) -> None:
        self.events.append(event)


class _AllToZeroPolicy(StorePolicy):
    def select_store_targets(
        self, keys: list[ObjectKey], adapters: list[AdapterDescriptor]
    ) -> dict[int, list[ObjectKey]]:
        return {0: list(keys)}

    def select_l1_deletions(self, keys: list[ObjectKey]) -> list[ObjectKey]:
        return []


class _HintAwareAdapter:
    def __init__(self) -> None:
        self.calls: list[tuple[list[ObjectKey], list[str | None]]] = []

    def submit_store_task_with_lifetime_hints(
        self,
        keys: list[ObjectKey],
        objects: list[object],
        lifetime_hints: list[str | None],
    ) -> int:
        self.calls.append((list(keys), list(lifetime_hints)))
        return 41


class _LegacyAdapter:
    def __init__(self) -> None:
        self.calls: list[list[ObjectKey]] = []

    def submit_store_task(self, keys: list[ObjectKey], objects: list[object]) -> int:
        self.calls.append(list(keys))
        return 42

    def submit_store_task_with_lifetime_hints(
        self,
        keys: list[ObjectKey],
        objects: list[object],
        lifetime_hints: list[str | None],
    ) -> int:
        if any(hint is not None for hint in lifetime_hints):
            raise ValueError("lifetime hints are unsupported")
        return self.submit_store_task(keys, objects)


class _FailingHintAdapter:
    def submit_store_task_with_lifetime_hints(
        self,
        keys: list[ObjectKey],
        objects: list[object],
        lifetime_hints: list[str | None],
    ) -> int:
        raise ValueError("unknown lifetime hint")


def _make_controller(adapter: object) -> StoreController:
    ctrl = StoreController.__new__(StoreController)
    ctrl._l2_adapters = {0: cast(L2AdapterInterface, adapter)}
    ctrl._adapter_descriptors = {
        0: cast(AdapterDescriptor, SimpleNamespace(type_name="fake"))
    }
    ctrl._policy = _AllToZeroPolicy()
    ctrl._draining = {}
    ctrl._l1_manager = cast(L1Manager, _FakeL1Manager())
    ctrl._event_bus = cast(EventBus, _FakeEventBus())
    ctrl._in_flight_tasks = {}
    ctrl._status_in_flight_count = 0
    ctrl._lifetime_hints_lock = threading.Lock()
    ctrl._lifetime_hints_by_key = {}
    return ctrl


def test_lifetime_hints_are_consumed_for_hint_aware_adapter() -> None:
    adapter = _HintAwareAdapter()
    ctrl = _make_controller(adapter)
    keys = [make_object_key(1), make_object_key(2)]

    ctrl.record_lifetime_hint([keys[0]], "transient")
    ctrl._submit_store_for_single_shape(keys)

    assert adapter.calls == [(keys, ["transient", None])]
    assert ctrl._consume_lifetime_hints(keys) == [None, None]


def test_lifetime_hints_do_not_break_legacy_adapter_without_hints() -> None:
    adapter = _LegacyAdapter()
    ctrl = _make_controller(adapter)
    keys = [make_object_key(1)]

    ctrl._submit_store_for_single_shape(keys)

    assert adapter.calls == [keys]
    assert ctrl._consume_lifetime_hints(keys) == [None]


def test_lifetime_hint_submit_failure_releases_read_locks() -> None:
    adapter = _FailingHintAdapter()
    ctrl = _make_controller(adapter)
    keys = [make_object_key(1)]

    ctrl.record_lifetime_hint(keys, "unknown")
    ctrl._submit_store_for_single_shape(keys)

    l1_manager = cast(_FakeL1Manager, ctrl._l1_manager)
    assert l1_manager.finished_reads == [keys]
    assert ctrl._in_flight_tasks == {}
    assert ctrl._status_in_flight_count == 0
