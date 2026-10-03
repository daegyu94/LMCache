# SPDX-License-Identifier: Apache-2.0
"""Partial flight identity, temporary-data handoff and controller recovery."""

# Standard
from collections.abc import Iterator
from dataclasses import replace

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.l1_manager import L1Manager
from lmcache.v1.distributed.l2_adapters.base import L2TaskId
from lmcache.v1.distributed.l2_adapters.mock_l2_adapter import (
    MockL2Adapter,
    MockL2AdapterConfig,
)
from lmcache.v1.distributed.storage_controllers.prefetch_controller import (
    PrefetchController,
)
from lmcache.v1.distributed.storage_controllers.prefetch_flights import (
    FlightMode,
    PrefetchFlights,
)
from lmcache.v1.distributed.storage_controllers.prefetch_policy import (
    DefaultPrefetchPolicy,
)
from lmcache.v1.distributed.storage_controllers.utils import L1ManagerDescriptor
from lmcache.v1.memory_management import MemoryObj
from tests.v1.distributed.test_prefetch_controller import (
    make_descriptor,
    make_group,
    make_l1_config,
    make_layout,
    make_object_key,
    make_spec,
    single_row_spec,
    store_keys_in_l2,
    wait_for_condition,
)


class CountedAdapter(MockL2Adapter):
    """Count public load submissions and inject an optional first failure."""

    def __init__(self, fail_first: bool = False) -> None:
        """Initialize a bandwidth-delayed adapter to force overlap."""
        super().__init__(MockL2AdapterConfig(max_size_gb=0.1, mock_bandwidth_gb=0.01))
        self.loads: list[list[ObjectKey]] = []
        self.fail_first = fail_first

    def submit_load_task(
        self, keys: list[ObjectKey], objects: list[MemoryObj]
    ) -> L2TaskId:
        """Record load cells and optionally reject the first submission."""
        self.loads.append(list(keys))
        if self.fail_first and len(self.loads) == 1:
            raise RuntimeError("injected load submission failure")
        return super().submit_load_task(keys, objects)


class RejectingLookupAdapter(MockL2Adapter):
    """Reject lookup before accepting ownership of any asynchronous task."""

    def __init__(self) -> None:
        """Create an empty optional cache tier with a failed submit path."""
        super().__init__(MockL2AdapterConfig(max_size_gb=0.1, mock_bandwidth_gb=1.0))
        self.lookup_attempts = 0

    def submit_lookup_and_lock_task(
        self, keys: list[ObjectKey], group_layout_descs: dict[int, MemoryLayoutDesc]
    ) -> L2TaskId:
        """Raise a synchronous error without creating a task or taking locks."""
        self.lookup_attempts += 1
        raise RuntimeError("injected lookup submission rejection")


def make_flight_controller(
    l1: L1Manager, adapter: MockL2Adapter, mode: FlightMode, max_in_flight: int = 8
) -> PrefetchController:
    """Create a controller with Default temporary L1 retention semantics."""
    return PrefetchController(
        [l1],
        [L1ManagerDescriptor(0, make_l1_config())],
        [adapter],
        [make_descriptor(0)],
        DefaultPrefetchPolicy(),
        max_in_flight=max_in_flight,
        flight_mode=mode,
    )


def make_temporary(l1: L1Manager, keys: list[ObjectKey]) -> None:
    """Admit temporary objects with one leader read lock and known payload."""
    layout = make_layout()
    reserved = l1.reserve_write(keys, [True] * len(keys), layout)
    for key, (error, obj) in reserved.items():
        assert error == L1Error.SUCCESS and obj is not None
        assert obj.tensor is not None
        obj.tensor.fill_(ObjectKey.Bytes2IntHash(key.chunk_hash))
    l1.finish_write_and_reserve_read(keys, 1)


@pytest.fixture
def l1() -> Iterator[L1Manager]:
    """Provide an actual L1 manager, freed after each public-contract test."""
    manager = L1Manager(make_l1_config())
    yield manager
    manager.close()


@pytest.mark.parametrize("mode", ["off", "overlap", "handoff"])
def test_immediate_consumer_handoff(l1: L1Manager, mode: FlightMode) -> None:
    """Only handoff preserves temporary payload after the leader consumes it."""
    flights = PrefetchFlights(l1, mode)
    shared, suffix = make_object_key(1), make_object_key(2)
    leader = single_row_spec([shared])
    follower = single_row_spec([shared, suffix], num_kv_readers=3)
    flights.started(0, leader, set())
    assert flights.defer(1, follower) == (mode != "off")
    make_temporary(l1, [shared])
    flights.handoff(0)
    flights.retire(0)
    l1.finish_read([shared], 1)  # The caller consumes before follower admission.
    reserved = l1.reserve_read([shared], 3)
    resident = {k for k, (error, _) in reserved.items() if error == L1Error.SUCCESS}
    flights.started(1, follower, resident)
    if mode == "handoff":
        assert shared in resident
        obj = reserved[shared][1]
        assert obj is not None and obj.tensor is not None
        assert torch.all(obj.tensor == 1)
        l1.finish_read([shared], 2)
        assert not l1.is_key_evictable(shared)
        l1.finish_read([shared], 1)
    else:
        assert not resident
    assert l1.get_object_state(shared) is None
    flights.retire(1)
    status = flights.report_status()
    assert status["current_pins"] == 0
    assert status["current_waiters"] == 0
    assert status["indexed_cells"] == 0
    assert status["handoff_pins_acquired"] == (1 if mode == "handoff" else 0)
    assert status["handoff_pins_released"] == status["handoff_pins_acquired"]


@pytest.mark.parametrize(
    "field", ["salt", "rank", "group", "model", "layout", "window", "fetch"]
)
def test_incompatible_cells_do_not_wait(l1: L1Manager, field: str) -> None:
    """Identity and interpretation differences cannot create shared flights."""
    key = make_object_key(0)
    leader = single_row_spec([key])
    follower = single_row_spec([key])
    if field in ("salt", "rank", "group", "model"):
        new_key = {
            "salt": replace(key, cache_salt="new-policy"),
            "rank": replace(key, kv_rank=1),
            "group": replace(key, object_group_id=1),
            "model": replace(key, model_name="other-model"),
        }[field]
        follower = make_spec([make_group([new_key], gid=new_key.object_group_id)])
    elif field == "layout":
        follower = make_spec(
            [
                replace(
                    follower.key_groups[0],
                    layout_desc=MemoryLayoutDesc(
                        shapes=[torch.Size([50, 2, 1024])], dtypes=[torch.bfloat16]
                    ),
                )
            ]
        )
    elif field == "window":
        follower = make_spec([make_group([key], window=1)])
    else:
        follower = single_row_spec([key], fetching_policy="full")
    flights = PrefetchFlights(l1, "handoff")
    flights.started(0, leader, set())
    assert not flights.defer(1, follower)
    flights.close()


def test_waiter_cap_and_failure_release(l1: L1Manager) -> None:
    """Bounded admission falls back and owner failure wakes without false hits."""
    keys = [make_object_key(i) for i in range(3)]
    flights = PrefetchFlights(l1, "handoff", max_waiters=1)
    flights.started(0, single_row_spec(keys[:2]), set())
    assert flights.defer(1, single_row_spec(keys))
    assert not flights.defer(2, single_row_spec(keys))
    assert flights.report_status()["cap_bypasses"] == 1
    flights.handoff(0)  # No objects ever became resident: failed load.
    flights.retire(0)
    assert not flights.defer(1, single_row_spec(keys))
    assert flights.report_status()["current_pins"] == 0
    flights.started(1, single_row_spec(keys), set())
    flights.close()
    assert flights.report_status()["current_waiters"] == 0


def test_multiple_leaders_and_shutdown_read_pins(l1: L1Manager) -> None:
    """A waiter can depend on two flights; cleanup frees only completed data."""
    a, b, c = [make_object_key(i) for i in (1, 2, 3)]
    flights = PrefetchFlights(l1, "handoff")
    flights.started(0, single_row_spec([a]), set())
    flights.started(1, single_row_spec([b]), set())
    waiter = single_row_spec([a, b, c])
    assert flights.defer(2, waiter)
    make_temporary(l1, [a])
    flights.handoff(0)
    flights.retire(0)
    l1.finish_read([a])
    assert flights.defer(2, waiter)
    assert l1.get_object_state(a) is not None
    # There is no active adapter write in this test. It tests rendezvous pin
    # cleanup, not baseline issue #5263's adapter quiescence contract.
    flights.close()
    assert l1.get_object_state(a) is None
    assert flights.report_status()["current_pins"] == 0
    assert flights.report_status()["indexed_cells"] == 0


@pytest.mark.parametrize("mode", ["off", "overlap", "handoff"])
@pytest.mark.parametrize("slots", [1, 8])
def test_partial_branch_controller(l1: L1Manager, mode: FlightMode, slots: int) -> None:
    """Grouped branch continuations share only overlapping data and own locks."""
    adapter = CountedAdapter()
    common = [make_object_key(i) for i in range(2)]
    left = common + [make_object_key(2)]
    right = common + [make_object_key(3)]
    store_keys_in_l2(adapter, list(dict.fromkeys(left + right)), make_layout())
    ctrl = make_flight_controller(l1, adapter, mode, slots)
    try:
        ids = [
            ctrl.submit_prefetch_request(single_row_spec(left)),
            ctrl.submit_prefetch_request(single_row_spec(right, num_kv_readers=2)),
        ]
        ctrl.start()
        for task in ids:
            assert ctrl.wait_prefetch_result(task, 10)
            result = ctrl.query_prefetch_result(task)
            assert result is not None and result.hit_cells[0].popcount() == 3
        expected = 6 if mode == "off" and slots == 8 else 4
        assert sum(map(len, adapter.loads)) == expected
        if mode != "off":
            assert ctrl.report_status()["branch_flights"]["waits"] == 1
        l1.finish_read(left, 1)
        assert all(not l1.is_key_evictable(k) for k in right)
        l1.finish_read(right, 2)
        assert all(l1.get_object_state(k) is None for k in set(left + right))
        status = ctrl.report_status()["branch_flights"]
        assert status["current_pins"] == 0
        assert status["handoff_pins_acquired"] == (2 if mode == "handoff" else 0)
        assert wait_for_condition(lambda: adapter.debug_get_locked_key_count() == 0)
    finally:
        ctrl.stop()
        adapter.close()


def test_controller_failure_falls_back(l1: L1Manager) -> None:
    """A follower independently retries the cells after a leader load fails."""
    adapter = CountedAdapter(fail_first=True)
    keys = [make_object_key(i) for i in range(3)]
    store_keys_in_l2(adapter, keys, make_layout())
    ctrl = make_flight_controller(l1, adapter, "handoff")
    try:
        ids = [ctrl.submit_prefetch_request(single_row_spec(keys)) for _ in range(2)]
        ctrl.start()
        assert ctrl.wait_prefetch_result(ids[0], 10)
        first = ctrl.query_prefetch_result(ids[0])
        assert first is not None and first.hit_cells[0].popcount() == 0
        assert ctrl.wait_prefetch_result(ids[1], 10)
        second = ctrl.query_prefetch_result(ids[1])
        assert second is not None and second.hit_cells[0].popcount() == len(keys)
        assert len(adapter.loads) == 2
        l1.finish_read(keys)
        assert ctrl.report_status()["branch_flights"]["current_pins"] == 0
    finally:
        ctrl.stop()
        adapter.close()


def test_resident_cells_do_not_create_false_dependencies(l1: L1Manager) -> None:
    """Followers sharing resident data can load distinct suffixes concurrently."""
    flights = PrefetchFlights(l1, "handoff")
    root, a, b = [make_object_key(i) for i in range(3)]
    flights.started(0, single_row_spec([root, a]), {root})
    assert not flights.defer(1, single_row_spec([root, b]))
    flights.close()


@pytest.mark.parametrize("cap", [0, -1])
def test_invalid_capacity(l1: L1Manager, cap: int) -> None:
    """Invalid capacity must fail before any request ownership is acquired."""
    with pytest.raises(ValueError, match="positive"):
        PrefetchFlights(l1, max_waiters=cap)


def test_grouped_partial_grid_and_independent_progress(l1: L1Manager) -> None:
    """Two object-group rows fold independently while a disjoint flight passes."""
    adapter = CountedAdapter()
    rows = [[make_object_key(i, gid=g) for i in range(4)] for g in range(2)]
    for row in rows:
        store_keys_in_l2(adapter, row, make_layout())
    specs = [
        make_spec([make_group(row[:3], gid=g) for g, row in enumerate(rows)]),
        make_spec([make_group(row[:2] + row[3:], gid=g) for g, row in enumerate(rows)]),
    ]
    ctrl = make_flight_controller(l1, adapter, "handoff")
    try:
        ids = [ctrl.submit_prefetch_request(spec) for spec in specs]
        ctrl.start()
        for task in ids:
            assert ctrl.wait_prefetch_result(task, 10)
            result = ctrl.query_prefetch_result(task)
            assert result is not None
            assert [row.popcount() for row in result.hit_cells] == [3, 3]
        assert sum(map(len, adapter.loads)) == 8
        for spec in specs:
            l1.finish_read([key for group in spec.key_groups for key in group.keys])
        assert all(l1.get_object_state(key) is None for row in rows for key in row)
    finally:
        ctrl.stop()
        adapter.close()


def test_disjoint_waiter_passes_without_dependency(l1: L1Manager) -> None:
    """A blocked branch does not block an unrelated admitted flight."""
    flights = PrefetchFlights(l1, "handoff")
    a, b, c = [make_object_key(i) for i in range(3)]
    flights.started(0, single_row_spec([a]), set())
    assert flights.defer(1, single_row_spec([a, b]))
    assert not flights.defer(2, single_row_spec([c]))
    flights.started(2, single_row_spec([c]), set())
    assert flights.report_status()["indexed_cells"] == 2
    flights.retire(2)
    assert flights.defer(1, single_row_spec([a, b]))
    flights.close()


@pytest.mark.parametrize("mode", ["off", "handoff"])
def test_accepted_lookup_drains_when_later_adapter_rejects(
    l1: L1Manager, mode: FlightMode
) -> None:
    """A rejected optional tier cannot orphan an earlier accepted lookup."""
    accepted = CountedAdapter()
    rejected = RejectingLookupAdapter()
    keys = [make_object_key(i) for i in (1, 2, 3)]
    make_temporary(l1, keys)
    objects = []
    for error, obj in l1.unsafe_read(keys).values():
        assert error == L1Error.SUCCESS and obj is not None
        objects.append(obj)
    store_task = accepted.submit_store_task(keys, objects)
    assert wait_for_condition(
        lambda: store_task in accepted.pop_completed_store_tasks()
    )
    l1.finish_read(keys)
    assert all(l1.get_object_state(k) is None for k in keys)
    ctrl = PrefetchController(
        [l1],
        [L1ManagerDescriptor(0, make_l1_config())],
        [accepted, rejected],
        [make_descriptor(0), make_descriptor(1)],
        DefaultPrefetchPolicy(),
        flight_mode=mode,
    )
    try:
        task = ctrl.submit_prefetch_request(single_row_spec(keys))
        ctrl.start()
        assert ctrl.wait_prefetch_result(task, 10)
        result = ctrl.query_prefetch_result(task)
        assert result is not None and result.hit_cells[0].popcount() == len(keys)
        assert rejected.lookup_attempts == 1
        assert len(accepted.loads) == 1
        for key, (error, obj) in l1.unsafe_read(keys).items():
            assert error == L1Error.SUCCESS and obj is not None
            assert obj.tensor is not None
            assert torch.all(obj.tensor == ObjectKey.Bytes2IntHash(key.chunk_hash))
        l1.finish_read(keys)
        assert wait_for_condition(lambda: accepted.debug_get_locked_key_count() == 0)
        assert rejected.debug_get_locked_key_count() == 0
        assert wait_for_condition(
            lambda: ctrl.report_status()["in_flight_request_count"] == 0
        )
        assert l1.report_status()["staging_object_count"] == 0
        assert l1.report_status()["read_locked_count"] == 0
        assert ctrl.report_status()["branch_flights"]["current_pins"] == 0
    finally:
        ctrl.stop()
        accepted.close()
        rejected.close()


@pytest.mark.parametrize("mode", ["off", "handoff"])
def test_all_rejected_lookups_fall_back_to_resident_l1(
    l1: L1Manager, mode: FlightMode
) -> None:
    """All rejected submissions complete with fresh L1 hits and missing suffix."""
    adapter = RejectingLookupAdapter()
    keys = [make_object_key(i) for i in (1, 2)]
    make_temporary(l1, keys)
    missing = make_object_key(3)
    ctrl = make_flight_controller(l1, adapter, mode)
    try:
        task = ctrl.submit_prefetch_request(single_row_spec(keys + [missing]))
        ctrl.start()
        assert ctrl.wait_prefetch_result(task, 10)
        result = ctrl.query_prefetch_result(task)
        assert result is not None
        assert result.hit_cells[0].get_indices_list() == [0, 1]
        assert result.l1_hit_cells[0].get_indices_list() == [0, 1]
        assert result.l2_hit_cells[0].popcount() == 0
        l1.finish_read(keys, 2)
        assert all(l1.get_object_state(k) is None for k in keys)
        assert wait_for_condition(
            lambda: ctrl.report_status()["in_flight_request_count"] == 0
        )
        assert ctrl.report_status()["branch_flights"]["indexed_cells"] == 0
        assert l1.report_status()["staging_object_count"] == 0
        assert l1.report_status()["read_locked_count"] == 0
        assert adapter.debug_get_locked_key_count() == 0
    finally:
        ctrl.stop()
        adapter.close()
