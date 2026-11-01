# SPDX-License-Identifier: Apache-2.0
"""CPU-only admission tests with deterministic start/retire transitions.

The controller's single-threaded drain is exercised directly to avoid GPU
allocation and asynchronous completion timing in admission-policy tests.
"""

# Standard
from collections import deque
from types import SimpleNamespace
from unittest.mock import Mock

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import (
    GroupedObjectKeys,
    MemoryLayoutDesc,
    ObjectKey,
    PrefetchTaskSpec,
)
from lmcache.v1.distributed.storage_controllers.prefetch_controller import (
    PrefetchController,
    PrefetchPhase,
)


def make_spec() -> PrefetchTaskSpec:
    """Return one 100-byte load candidate."""
    return PrefetchTaskSpec(
        key_groups=[
            GroupedObjectKeys(
                keys=[ObjectKey(chunk_hash=b"one", model_name="m", kv_rank=0)],
                object_group_id=0,
                layout_desc=MemoryLayoutDesc([torch.Size([100])], [torch.uint8]),
            )
        ],
        num_kv_readers=1,
    )


def make_controller(capacity: int = 1000) -> PrefetchController:
    """Build the state needed by a deterministic admission drain."""
    ctrl = PrefetchController.__new__(PrefetchController)
    ctrl._pending_queue = deque()  # noqa: SLF001
    ctrl._in_flight_requests = {}  # noqa: SLF001
    ctrl._l2_adapters = {0: Mock()}  # noqa: SLF001
    ctrl._l2_affinity = {0: 0}  # noqa: SLF001
    ctrl._draining = {}  # noqa: SLF001
    manager = Mock()
    manager.get_memory_usage.return_value = (0, capacity)
    manager.get_staging_memory_usage.return_value = 0
    ctrl._l1_managers = {0: manager}  # noqa: SLF001
    ctrl._inflight_watermark = 0.5  # noqa: SLF001
    return ctrl


def lookup_request(size: int = 100) -> SimpleNamespace:
    """Return a lookup whose charge is observable through its public getter."""
    return SimpleNamespace(
        phase=PrefetchPhase.LOOKUP, get_max_load_bytes=Mock(return_value=size)
    )


def test_burst_stops_at_watermark_and_preserves_fifo() -> None:
    ctrl = make_controller()
    spec = make_spec()
    ctrl._pending_queue.extend((i, spec) for i in range(10))  # noqa: SLF001
    started = []

    def start(request_id: int, spec: PrefetchTaskSpec) -> None:
        started.append(request_id)
        ctrl._in_flight_requests[request_id] = lookup_request()  # noqa: SLF001

    ctrl._start_lookup_phase = start  # noqa: SLF001
    ctrl._start_pending_requests()  # noqa: SLF001
    assert started == list(range(5))
    assert [item[0] for item in ctrl._pending_queue] == list(range(5, 10))  # noqa: SLF001
    for request in ctrl._in_flight_requests.values():  # noqa: SLF001
        assert request.get_max_load_bytes.call_count == 1


@pytest.mark.parametrize("mode", ["immediate", "failure", "before_registration"])
def test_retired_requests_do_not_consume_admission_budget(mode: str) -> None:
    ctrl = make_controller()
    spec = make_spec()
    ctrl._pending_queue.extend((i, spec) for i in range(8))  # noqa: SLF001
    started = []

    def start(request_id: int, spec: PrefetchTaskSpec) -> None:
        started.append(request_id)
        if mode == "before_registration":
            raise RuntimeError("submit failed")
        ctrl._in_flight_requests[request_id] = lookup_request()  # noqa: SLF001
        if mode == "failure":
            raise RuntimeError("submit failed")
        ctrl._in_flight_requests.pop(request_id)  # noqa: SLF001

    def abort(request: SimpleNamespace) -> None:
        ctrl._in_flight_requests.clear()  # noqa: SLF001

    ctrl._start_lookup_phase = start  # noqa: SLF001
    ctrl._abort_request = abort  # noqa: SLF001
    ctrl._start_pending_requests()  # noqa: SLF001
    assert started == list(range(8))
    assert not ctrl._pending_queue  # noqa: SLF001


def test_staging_usage_is_refreshed_after_each_start() -> None:
    ctrl = make_controller()
    spec = make_spec()
    ctrl._pending_queue.extend((i, spec) for i in range(8))  # noqa: SLF001
    manager = ctrl._l1_managers[0]  # noqa: SLF001

    def start(request_id: int, spec: PrefetchTaskSpec) -> None:
        ctrl._in_flight_requests[request_id] = SimpleNamespace(  # noqa: SLF001
            phase=PrefetchPhase.PLAN_AND_LOAD, get_max_load_bytes=Mock(return_value=100)
        )
        manager.get_staging_memory_usage.return_value += 100

    ctrl._start_lookup_phase = start  # noqa: SLF001
    ctrl._start_pending_requests()  # noqa: SLF001
    assert len(ctrl._pending_queue) == 3  # noqa: SLF001


def test_new_drain_recomputes_existing_lookup_charge() -> None:
    ctrl = make_controller()
    spec = make_spec()
    existing = lookup_request(500)
    ctrl._in_flight_requests[99] = existing  # noqa: SLF001
    ctrl._pending_queue.append((0, spec))  # noqa: SLF001
    ctrl._start_lookup_phase = Mock()  # noqa: SLF001
    ctrl._start_pending_requests()  # noqa: SLF001
    ctrl._start_lookup_phase.assert_not_called()  # noqa: SLF001
    existing.phase = PrefetchPhase.PLAN_AND_LOAD
    ctrl._start_pending_requests()  # noqa: SLF001
    ctrl._start_lookup_phase.assert_called_once()  # noqa: SLF001
