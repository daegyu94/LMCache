# SPDX-License-Identifier: Apache-2.0
"""Tests for cache-salt-based shared-L2 QoS scheduling."""

# Standard
from unittest.mock import MagicMock
import argparse
import json
import select
import threading
import time

# Third Party
import pytest

# First Party
from lmcache.lmcache_native import Bitmap
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.config import L2QoSConfig
from lmcache.v1.distributed.internal_api import L2StoreResult
from lmcache.v1.distributed.l2_adapters.base import L2AdapterInterface
from lmcache.v1.distributed.l2_adapters.config import (
    L2AdapterConfigBase,
    add_l2_adapters_args,
    parse_args_to_l2_adapters_config,
)
from lmcache.v1.distributed.l2_qos_adapter import QosL2Adapter
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.multiprocess.qos import (
    CacheSaltQosManager,
    L2QoSDispatcher,
    L2QoSDispatcherPool,
    QosProfile,
    QosTaskHandle,
    QosWeightUpdate,
)
from lmcache.v1.platform import (
    EventNotifier,
    PipeNotifier,
    consume_fd,
    create_event_notifier,
)


def _parse_mock_adapter_config(
    extra: dict[str, object] | None = None,
) -> L2AdapterConfigBase:
    """Parse one mock adapter through the public repeatable-JSON CLI."""
    spec: dict[str, object] = {
        "type": "mock",
        "max_size_gb": 1,
        "mock_bandwidth_gb": 1,
    }
    if extra is not None:
        spec.update(extra)
    parser = argparse.ArgumentParser()
    add_l2_adapters_args(parser)
    args = parser.parse_args(["--l2-adapter", json.dumps(spec)])
    return parse_args_to_l2_adapters_config(args).adapters[0]


def _drain_dispatcher(
    dispatcher: L2QoSDispatcher,
    handles: list[QosTaskHandle],
    timeout: float = 3,
) -> None:
    """Wait for and complete dispatcher handles."""
    pending = list(handles)
    deadline = time.monotonic() + timeout
    while pending:
        progressed = False
        for handle in pending[:]:
            if not handle.future.done():
                continue
            handle.future.result()
            dispatcher.complete(handle.task_id)
            pending.remove(handle)
            progressed = True
        if not progressed:
            if time.monotonic() >= deadline:
                raise TimeoutError
            time.sleep(0.001)


def _make_mock_adapter() -> tuple[MagicMock, list[EventNotifier]]:
    """Create an asynchronous adapter mock with valid event descriptors."""
    notifiers = [create_event_notifier() for _ in range(3)]
    adapter = MagicMock(spec=L2AdapterInterface)
    adapter.get_store_event_fd.return_value = notifiers[0].fileno()
    adapter.get_lookup_and_lock_event_fd.return_value = notifiers[1].fileno()
    adapter.get_load_event_fd.return_value = notifiers[2].fileno()
    adapter.submit_store_task.return_value = 7
    return adapter, notifiers


class _ThreadUnsafeStoreAdapter(L2AdapterInterface):
    """Adapter-shaped test double whose store state has no internal lock."""

    def __init__(self, notifiers: list[EventNotifier]) -> None:
        super().__init__()
        self._notifiers = notifiers
        self._next_store_task_id = 0
        self._completed: dict[int, L2StoreResult] = {}
        self._submit_in_progress = threading.Event()
        self.submit_started = threading.Event()
        self.release_submit = threading.Event()
        self.pop_entered = threading.Event()
        self.overlap_detected = False

    def get_store_event_fd(self) -> int:
        return self._notifiers[0].fileno()

    def get_lookup_and_lock_event_fd(self) -> int:
        return self._notifiers[1].fileno()

    def get_load_event_fd(self) -> int:
        return self._notifiers[2].fileno()

    def submit_store_task(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
    ) -> int:
        del keys, objects
        self._submit_in_progress.set()
        self.submit_started.set()
        try:
            if not self.release_submit.wait(timeout=2):
                raise TimeoutError("test submit was not released")
            task_id = self._next_store_task_id
            self._next_store_task_id += 1
            self._completed[task_id] = L2StoreResult(True, 0)
            return task_id
        finally:
            self._submit_in_progress.clear()

    def pop_completed_store_tasks(self) -> dict[int, L2StoreResult]:
        self.pop_entered.set()
        if self._submit_in_progress.is_set():
            self.overlap_detected = True
        completed = self._completed
        self._completed = {}
        return completed

    def submit_lookup_and_lock_task(
        self,
        keys: list[ObjectKey],
        group_layout_descs: dict[int, MemoryLayoutDesc],
    ) -> int:
        del keys, group_layout_descs
        return 0

    def query_lookup_and_lock_result(self, task_id: int) -> Bitmap | None:
        del task_id
        return None

    def submit_unlock(self, keys: list[ObjectKey]) -> None:
        del keys

    def submit_load_task(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
    ) -> int:
        del keys, objects
        return 0

    def query_load_result(self, task_id: int) -> Bitmap | None:
        del task_id
        return None

    def close(self) -> None:
        return None


def test_l2_qos_adapter_serializes_concrete_adapter_calls() -> None:
    """Admission and controller polling never enter a raw adapter together."""
    notifiers = [create_event_notifier() for _ in range(3)]
    adapter = _ThreadUnsafeStoreAdapter(notifiers)
    dispatcher = L2QoSDispatcher(max_inflight_tasks=1)
    manager = CacheSaltQosManager()
    wrapper = QosL2Adapter(adapter, dispatcher, manager)
    key = ObjectKey(b"thread-safe", "model", 0, cache_salt="tenant-a")
    memory_obj = MagicMock(spec=MemoryObj)
    memory_obj.get_size.return_value = 1
    pop_results: list[dict[int, L2StoreResult]] = []

    def pop() -> None:
        pop_results.append(wrapper.pop_completed_store_tasks())

    pop_thread: threading.Thread | None = None
    try:
        wrapped_task_id = wrapper.submit_store_task([key], [memory_obj])
        assert adapter.submit_started.wait(timeout=1)
        pop_thread = threading.Thread(target=pop)
        pop_thread.start()
        assert not adapter.pop_entered.wait(timeout=0.05)

        adapter.release_submit.set()
        pop_thread.join(timeout=1)
        assert not pop_thread.is_alive()
        assert adapter.pop_entered.is_set()
        assert not adapter.overlap_detected

        completed: dict[int, L2StoreResult] = pop_results[0] if pop_results else {}
        deadline = time.monotonic() + 1
        while not completed and time.monotonic() < deadline:
            completed = wrapper.pop_completed_store_tasks()
            if not completed:
                time.sleep(0.001)
        assert completed[wrapped_task_id].is_successful()
    finally:
        adapter.release_submit.set()
        if pop_thread is not None:
            pop_thread.join(timeout=1)
        wrapper.close()
        dispatcher.close()
        for notifier in notifiers:
            notifier.close()


def test_l2_qos_config_validates_default_sched_weight() -> None:
    """The server default is configurable within the public weight range."""
    config = L2QoSConfig()
    assert config.default_sched_weight == 100
    assert config.max_inflight_tasks == 8
    assert config.max_inflight_bytes == 0
    assert L2QoSConfig(default_sched_weight=250).default_sched_weight == 250

    with pytest.raises(ValueError, match="default_sched_weight"):
        L2QoSConfig(default_sched_weight=0)


def test_l2_adapter_qos_resource_group_config() -> None:
    """Adapters are independent by default and may name a shared group."""
    independent = _parse_mock_adapter_config()
    grouped = _parse_mock_adapter_config({"qos_resource_group": "nvme0"})

    assert independent.qos_resource_group is None
    assert grouped.qos_resource_group == "nvme0"


@pytest.mark.parametrize("group", ["", " ", 7, []])
def test_l2_adapter_rejects_invalid_qos_resource_group(group: object) -> None:
    """A named QoS resource group must be a non-empty string."""
    with pytest.raises(ValueError, match="qos_resource_group"):
        _parse_mock_adapter_config({"qos_resource_group": group})


def test_dispatcher_pool_isolates_resource_groups() -> None:
    """An in-flight task in one group does not block another group."""
    manager = CacheSaltQosManager()
    pool = L2QoSDispatcherPool(manager, quantum_bytes=1, max_inflight_tasks=1)
    first = pool.acquire("adapter:0")
    second = pool.acquire("adapter:1")

    try:
        assert first is not second
        first_handle = first.submit(
            manager.get_profile("tenant-a"), "load", 1, lambda: 1
        )
        assert first_handle.future.result(timeout=1) == 1

        second_handle = second.submit(
            manager.get_profile("tenant-b"), "load", 1, lambda: 2
        )
        assert second_handle.future.result(timeout=1) == 2

        first.complete(first_handle.task_id)
        second.complete(second_handle.task_id)
    finally:
        pool.close()


def test_dispatcher_pool_shares_named_resource_group() -> None:
    """Adapters in one named group share dispatcher admission accounting."""
    manager = CacheSaltQosManager()
    pool = L2QoSDispatcherPool(manager, quantum_bytes=1, max_inflight_tasks=1)
    first = pool.acquire("named:nvme0")
    second = pool.acquire("named:nvme0")

    try:
        assert first is second
        first_handle = first.submit(
            manager.get_profile("tenant-a"), "load", 1, lambda: 1
        )
        assert first_handle.future.result(timeout=1) == 1

        second_handle = second.submit(
            manager.get_profile("tenant-b"), "load", 1, lambda: 2
        )
        with pytest.raises(TimeoutError):
            second_handle.future.result(timeout=0.05)

        first.complete(first_handle.task_id)
        assert second_handle.future.result(timeout=1) == 2
        second.complete(second_handle.task_id)

        pool.release("named:nvme0")
        pool.release("named:nvme0")
        with pytest.raises(RuntimeError, match="closed"):
            first.submit(manager.get_profile("tenant-a"), "load", 1, lambda: 3)
    finally:
        pool.close()


def test_dispatcher_pool_propagates_weights_to_every_group() -> None:
    """Runtime weight changes update all active resource-group queues."""
    manager = CacheSaltQosManager()
    pool = L2QoSDispatcherPool(manager, quantum_bytes=1)
    first = pool.acquire("adapter:0")
    second = pool.acquire("adapter:1")

    try:
        profile = manager.get_profile("tenant-a")
        first_handle = first.submit(profile, "load", 1, lambda: 1)
        second_handle = second.submit(profile, "load", 1, lambda: 2)
        assert first_handle.future.result(timeout=1) == 1
        assert second_handle.future.result(timeout=1) == 2

        manager.set_sched_weight("tenant-a", 500)
        assert first.snapshot()["tenant-a"]["weight"] == 500
        assert second.snapshot()["tenant-a"]["weight"] == 500

        first.complete(first_handle.task_id)
        second.complete(second_handle.task_id)
    finally:
        pool.close()


def test_dispatcher_advances_deficit_rounds_without_wall_clock_delay() -> None:
    """Large tasks and low weights advance through logical rounds immediately."""
    dispatcher = L2QoSDispatcher(quantum_bytes=1, max_inflight_tasks=0)
    handle = dispatcher.submit(
        QosProfile(domain_id="tenant-a", weight=1),
        "load",
        10,
        lambda: 7,
    )

    try:
        assert handle.future.result(timeout=0.5) == 7
        dispatcher.complete(handle.task_id)
    finally:
        dispatcher.close()


def test_cache_salt_qos_manager_uses_default_then_explicit_weight() -> None:
    """Unknown salts use the default and may be registered later."""
    manager = CacheSaltQosManager(default_sched_weight=125)

    initial = manager.get_profile("tenant-a")
    manager.set_sched_weight("tenant-a", 700)
    explicit = manager.get_profile("tenant-a")
    removed = manager.delete_sched_weight("tenant-a")
    restored = manager.get_profile("tenant-a")

    assert (initial.weight, initial.source) == (125, "default")
    assert (explicit.weight, explicit.source) == (700, "explicit")
    assert removed is True
    assert (restored.weight, restored.source) == (125, "default")


def test_default_weight_updates_only_active_default_domains() -> None:
    """Default changes update default domains without overriding explicit ones."""
    manager = CacheSaltQosManager(default_sched_weight=100)
    dispatcher = L2QoSDispatcher(max_inflight_tasks=0)
    manager.register_listener(dispatcher.update_profile)
    handle = dispatcher.submit(
        manager.get_profile("tenant-a"),
        "load",
        1,
        lambda: 1,
    )

    try:
        assert handle.future.result(timeout=1) == 1
        manager.set_default_sched_weight(250)
        assert dispatcher.snapshot()["tenant-a"]["weight"] == 250

        manager.set_sched_weight("tenant-a", 700)
        manager.set_default_sched_weight(400)
        assert dispatcher.snapshot()["tenant-a"]["weight"] == 700

        dispatcher.complete(handle.task_id)
        assert dispatcher.snapshot() == {}
    finally:
        dispatcher.close()


def test_dispatcher_prunes_high_cardinality_idle_domains() -> None:
    """Completed salts do not accumulate in dispatcher or registry state."""
    manager = CacheSaltQosManager()
    dispatcher = L2QoSDispatcher(max_inflight_tasks=0)

    try:
        for index in range(64):
            handle = dispatcher.submit(
                manager.get_profile(f"tenant-{index}"),
                "lookup",
                1,
                lambda: 1,
            )
            assert handle.future.result(timeout=1) == 1
            dispatcher.complete(handle.task_id)

        assert dispatcher.snapshot() == {}
        assert manager.list_sched_weights() == {}
    finally:
        dispatcher.close()


def test_qos_manager_isolates_listener_failures() -> None:
    """One broken resource-group listener does not suppress later listeners."""
    manager = CacheSaltQosManager()
    updates: list[QosWeightUpdate] = []

    def fail_listener(_update: QosWeightUpdate) -> None:
        raise RuntimeError("listener failed")

    manager.register_listener(fail_listener)
    manager.register_listener(updates.append)

    manager.set_sched_weight("tenant-a", 500)

    assert updates == [
        QosWeightUpdate(
            cache_salt="tenant-a",
            weight=500,
            source="explicit",
        )
    ]


@pytest.mark.parametrize("weight", [0, 10001, 1.5, True])
def test_cache_salt_qos_manager_rejects_invalid_weight(weight: object) -> None:
    """Only integer weights in the supported range are accepted."""
    manager = CacheSaltQosManager()

    with pytest.raises(ValueError):
        manager.set_sched_weight("tenant-a", weight)  # type: ignore[arg-type]


def test_runtime_registration_updates_already_queued_domain() -> None:
    """Registering a discovered salt updates its existing dispatcher queue."""
    manager = CacheSaltQosManager(default_sched_weight=100)
    dispatcher = L2QoSDispatcher(quantum_bytes=1, max_inflight_tasks=1)
    manager.register_listener(dispatcher.update_profile)
    blocker_started = threading.Event()
    release_blocker = threading.Event()

    def block() -> int:
        blocker_started.set()
        release_blocker.wait(timeout=3)
        return 0

    blocker = dispatcher.submit(
        manager.get_profile("blocker"),
        "load",
        1,
        block,
    )
    queued = dispatcher.submit(
        manager.get_profile("tenant-a"),
        "load",
        1,
        lambda: 1,
    )

    try:
        assert blocker_started.wait(timeout=1)
        manager.set_sched_weight("tenant-a", 900)
        assert dispatcher.snapshot()["tenant-a"]["weight"] == 900
        release_blocker.set()
        _drain_dispatcher(dispatcher, [blocker, queued])
    finally:
        release_blocker.set()
        dispatcher.close()


def test_dispatcher_preserves_weighted_service_share() -> None:
    """A busy shared backend gives more admissions to the higher weight."""
    dispatcher = L2QoSDispatcher(quantum_bytes=1, max_inflight_tasks=1)
    low = QosProfile(domain_id="low", weight=100)
    high = QosProfile(domain_id="high", weight=300)
    admissions: list[str] = []

    def admit(domain: str) -> int:
        admissions.append(domain)
        return 1

    handles = []
    try:
        for _ in range(40):
            handles.append(dispatcher.submit(low, "load", 1, lambda: admit("low")))
            handles.append(dispatcher.submit(high, "load", 1, lambda: admit("high")))
        _drain_dispatcher(dispatcher, handles)
    finally:
        dispatcher.close()

    sample = admissions[:32]
    assert sample.count("high") >= sample.count("low") * 2


def test_qos_l2_adapter_resolves_weight_from_object_key() -> None:
    """The wrapper charges a store task to its ObjectKey cache salt."""
    dispatcher = L2QoSDispatcher(max_inflight_tasks=1)
    manager = CacheSaltQosManager()
    manager.set_sched_weight("tenant-a", 500)
    manager.register_listener(dispatcher.update_profile)
    adapter, notifiers = _make_mock_adapter()
    adapter.report_status.return_value = {}
    wrapper = QosL2Adapter(adapter, dispatcher, manager, resource_group="nvme0")
    key = ObjectKey(b"hash", "model", 0, cache_salt="tenant-a")
    memory_obj = MagicMock()
    memory_obj.get_size.return_value = 16

    try:
        wrapped_task_id = wrapper.submit_store_task([key], [memory_obj])
        deadline = time.monotonic() + 2
        while not adapter.submit_store_task.called and time.monotonic() < deadline:
            time.sleep(0.001)
        assert adapter.submit_store_task.called
        assert dispatcher.snapshot()["tenant-a"]["weight"] == 500

        adapter.pop_completed_store_tasks.return_value = {7: L2StoreResult(True, 16)}
        completed: dict[int, L2StoreResult] = {}
        while not completed and time.monotonic() < deadline:
            completed = wrapper.pop_completed_store_tasks()
            if not completed:
                time.sleep(0.001)

        assert completed[wrapped_task_id].is_successful()
        assert dispatcher.snapshot() == {}
        assert wrapper.report_status()["qos_resource_group"] == "nvme0"
    finally:
        wrapper.close()
        dispatcher.close()
        for notifier in notifiers:
            notifier.close()


def test_qos_l2_adapter_partitions_mixed_lookup_and_load() -> None:
    """Mixed batches are scheduled per salt and merged in original key order."""
    dispatcher = L2QoSDispatcher(max_inflight_tasks=0)
    manager = CacheSaltQosManager()
    adapter, notifiers = _make_mock_adapter()
    wrapper = QosL2Adapter(adapter, dispatcher, manager)
    keys = [
        ObjectKey(b"a", "model", 0, cache_salt="tenant-a"),
        ObjectKey(b"b", "model", 0, cache_salt="tenant-b"),
        ObjectKey(b"c", "model", 0, cache_salt="tenant-a"),
    ]
    objects: list[MemoryObj] = []
    for _ in keys:
        memory_obj = MagicMock(spec=MemoryObj)
        memory_obj.get_size.return_value = 16
        objects.append(memory_obj)

    lookup_submissions: dict[int, list[ObjectKey]] = {}
    load_submissions: dict[int, tuple[list[ObjectKey], list[MemoryObj]]] = {}

    def submit_lookup(
        part_keys: list[ObjectKey],
        _group_layout_descs: dict[int, object],
    ) -> int:
        raw_task_id = 10 + len(lookup_submissions)
        lookup_submissions[raw_task_id] = part_keys
        return raw_task_id

    def query_lookup(raw_task_id: int) -> Bitmap:
        part_keys = lookup_submissions[raw_task_id]
        result = Bitmap(len(part_keys))
        if part_keys[0].cache_salt == "tenant-a":
            result.set(1)
        else:
            result.set(0)
        return result

    def submit_load(
        part_keys: list[ObjectKey],
        part_objects: list[MemoryObj],
    ) -> int:
        raw_task_id = 20 + len(load_submissions)
        load_submissions[raw_task_id] = (part_keys, part_objects)
        return raw_task_id

    def query_load(raw_task_id: int) -> Bitmap:
        part_keys, _part_objects = load_submissions[raw_task_id]
        result = Bitmap(len(part_keys))
        if part_keys[0].cache_salt == "tenant-a":
            result.set(0)
        return result

    adapter.submit_lookup_and_lock_task.side_effect = submit_lookup
    adapter.query_lookup_and_lock_result.side_effect = query_lookup
    adapter.submit_load_task.side_effect = submit_load
    adapter.query_load_result.side_effect = query_load

    try:
        lookup_task_id = wrapper.submit_lookup_and_lock_task(keys, {})
        deadline = time.monotonic() + 2
        while len(lookup_submissions) < 2 and time.monotonic() < deadline:
            time.sleep(0.001)
        assert len(lookup_submissions) == 2

        lookup_result = wrapper.query_lookup_and_lock_result(lookup_task_id)
        assert lookup_result is not None
        assert lookup_result.get_indices_list() == [1, 2]

        lookup_by_salt = {
            part_keys[0].cache_salt: part_keys
            for part_keys in lookup_submissions.values()
        }
        assert lookup_by_salt == {
            "tenant-a": [keys[0], keys[2]],
            "tenant-b": [keys[1]],
        }

        load_task_id = wrapper.submit_load_task(keys, objects)
        deadline = time.monotonic() + 2
        while len(load_submissions) < 2 and time.monotonic() < deadline:
            time.sleep(0.001)
        assert len(load_submissions) == 2

        load_result = wrapper.query_load_result(load_task_id)
        assert load_result is not None
        assert load_result.get_indices_list() == [0]

        load_by_salt = {
            part_keys[0].cache_salt: (part_keys, part_objects)
            for part_keys, part_objects in load_submissions.values()
        }
        assert load_by_salt == {
            "tenant-a": ([keys[0], keys[2]], [objects[0], objects[2]]),
            "tenant-b": ([keys[1]], [objects[1]]),
        }
        assert dispatcher.snapshot() == {}
    finally:
        wrapper.close()
        dispatcher.close()
        for notifier in notifiers:
            notifier.close()


def test_qos_l2_adapter_merges_mixed_store_results() -> None:
    """Store subtasks produce one aggregate result for the public task ID."""
    dispatcher = L2QoSDispatcher(max_inflight_tasks=0)
    manager = CacheSaltQosManager()
    adapter, notifiers = _make_mock_adapter()
    wrapper = QosL2Adapter(adapter, dispatcher, manager)
    keys = [
        ObjectKey(b"a", "model", 0, cache_salt="tenant-a"),
        ObjectKey(b"b", "model", 0, cache_salt="tenant-b"),
    ]
    objects: list[MemoryObj] = []
    for _ in keys:
        memory_obj = MagicMock(spec=MemoryObj)
        memory_obj.get_size.return_value = 16
        objects.append(memory_obj)
    submissions: dict[int, tuple[list[ObjectKey], list[MemoryObj]]] = {}

    def submit_store(
        part_keys: list[ObjectKey],
        part_objects: list[MemoryObj],
    ) -> int:
        raw_task_id = 30 + len(submissions)
        submissions[raw_task_id] = (part_keys, part_objects)
        return raw_task_id

    adapter.submit_store_task.side_effect = submit_store

    try:
        wrapped_task_id = wrapper.submit_store_task(keys, objects)
        deadline = time.monotonic() + 2
        while len(submissions) < 2 and time.monotonic() < deadline:
            time.sleep(0.001)
        assert len(submissions) == 2

        adapter.pop_completed_store_tasks.return_value = {
            raw_task_id: L2StoreResult(True, 16) for raw_task_id in submissions
        }
        result = wrapper.pop_completed_store_tasks()[wrapped_task_id]

        assert result.is_successful()
        assert result.bytes_transferred() == 32
        assert dispatcher.snapshot() == {}
    finally:
        wrapper.close()
        dispatcher.close()
        for notifier in notifiers:
            notifier.close()


def test_qos_l2_adapter_relays_pipe_notifications(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The wrapper relays concrete adapter completion notifications."""
    # First Party
    import lmcache.v1.distributed.l2_qos_adapter as qos_adapter_module
    import lmcache.v1.platform.event_notifier as event_notifier

    monkeypatch.setattr(event_notifier, "HAS_EVENTFD", False)
    monkeypatch.setattr(qos_adapter_module, "create_event_notifier", PipeNotifier)

    dispatcher = L2QoSDispatcher(max_inflight_tasks=1)
    manager = CacheSaltQosManager()
    notifiers = [PipeNotifier() for _ in range(3)]
    adapter = MagicMock(spec=L2AdapterInterface)
    adapter.get_store_event_fd.return_value = notifiers[0].fileno()
    adapter.get_lookup_and_lock_event_fd.return_value = notifiers[1].fileno()
    adapter.get_load_event_fd.return_value = notifiers[2].fileno()
    adapter.submit_store_task.return_value = 7
    wrapper = qos_adapter_module.QosL2Adapter(adapter, dispatcher, manager)
    wrapper_event_fd = wrapper.get_store_event_fd()
    poller = select.poll()
    poller.register(wrapper_event_fd, select.POLLIN)
    key = ObjectKey(b"hash", "model", 0, cache_salt="pipe-test")
    memory_obj = MagicMock()
    memory_obj.get_size.return_value = 16

    try:
        wrapped_task_id = wrapper.submit_store_task([key], [memory_obj])
        assert poller.poll(1000), "admission completion was not signaled"
        consume_fd(wrapper_event_fd)

        adapter.pop_completed_store_tasks.return_value = {7: L2StoreResult(True, 16)}
        notifiers[0].notify()
        assert poller.poll(1000), "adapter pipe notification was not relayed"
        consume_fd(wrapper_event_fd)
        assert wrapper.pop_completed_store_tasks()[wrapped_task_id].is_successful()
    finally:
        wrapper.close()
        dispatcher.close()
        for notifier in notifiers:
            notifier.close()
