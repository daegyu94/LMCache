# SPDX-License-Identifier: Apache-2.0
"""Tests for cgroup-derived shared-L2 QoS scheduling."""

# Standard
from pathlib import Path
import select
import threading
import time

# Third Party
import pytest

# First Party
from lmcache.v1.multiprocess.qos import (
    L2QoSDispatcher,
    QosProfile,
    get_current_qos_profile,
    read_cgroup_io_weight,
)


def test_read_cgroup_io_weight(tmp_path: Path) -> None:
    """Parse the cgroup v2 default and device-specific forms."""
    cgroup_file = tmp_path / "io.weight"
    cgroup_file.write_text("default 250\n8:0 700\n", encoding="utf-8")

    profile = read_cgroup_io_weight(cgroup_file)

    assert profile is not None
    assert profile.default_weight == 250
    assert profile.device_weights == {"8:0": 700}


def test_qos_profile_explicit_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicit deployment metadata overrides cgroup discovery."""
    monkeypatch.setenv("LMCACHE_QOS_DOMAIN", "tenant-a")
    monkeypatch.setenv("LMCACHE_QOS_WEIGHT", "700")

    profile = QosProfile.from_environment()

    assert profile.domain_id == "tenant-a"
    assert profile.weight == 700
    assert profile.source == "explicit"


def test_dispatcher_preserves_weighted_service_share() -> None:
    """A busy shared backend gives more admissions to the higher weight."""
    dispatcher = L2QoSDispatcher(
        quantum_bytes=1,
        max_inflight_tasks=1,
    )
    low = QosProfile(domain_id="low", weight=100)
    high = QosProfile(domain_id="high", weight=300)
    admissions: list[str] = []

    def admit_low() -> int:
        admissions.append("low")
        return 1

    def admit_high() -> int:
        admissions.append("high")
        return 1

    handles = []

    try:
        for _ in range(40):
            handles.append(
                dispatcher.submit(
                    low,
                    "load",
                    1,
                    admit_low,
                )
            )
            handles.append(
                dispatcher.submit(
                    high,
                    "load",
                    1,
                    admit_high,
                )
            )

        pending = list(handles)
        deadline = time.monotonic() + 2
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
    finally:
        dispatcher.close()

    sample = admissions[:32]
    assert sample.count("high") > sample.count("low")
    assert sample.count("high") >= sample.count("low") * 2


def test_dispatcher_does_not_accumulate_quantum_while_blocked() -> None:
    """Admission backpressure must not create a burst for one domain."""
    dispatcher = L2QoSDispatcher(
        quantum_bytes=1,
        max_inflight_tasks=1,
    )
    blocker_profile = QosProfile(domain_id="blocker", weight=100)
    first_profile = QosProfile(domain_id="first", weight=100)
    second_profile = QosProfile(domain_id="second", weight=100)
    blocker_started = threading.Event()
    release_blocker = threading.Event()
    admissions: list[str] = []

    def block() -> int:
        blocker_started.set()
        release_blocker.wait(timeout=5)
        admissions.append("blocker")
        return 0

    def admit_first() -> int:
        admissions.append("first")
        return 0

    def admit_second() -> int:
        admissions.append("second")
        return 0

    handles = [
        dispatcher.submit(blocker_profile, "load", 1, block),
        *[dispatcher.submit(first_profile, "load", 1, admit_first) for _ in range(20)],
        dispatcher.submit(second_profile, "load", 1, admit_second),
    ]

    try:
        assert blocker_started.wait(timeout=1)
        # The dispatcher polls blocked domains every 10 ms. Give it enough
        # time to expose the old unbounded-deficit behavior deterministically.
        time.sleep(0.1)
        release_blocker.set()

        pending = list(handles)
        deadline = time.monotonic() + 3
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
    finally:
        release_blocker.set()
        dispatcher.close()

    assert admissions[1:3] == ["first", "second"]


def test_message_queue_propagates_qos_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    """The MP handshake makes the client profile visible to handlers."""
    monkeypatch.setenv("LMCACHE_QOS_DOMAIN", "handshake-a")
    monkeypatch.setenv("LMCACHE_QOS_WEIGHT", "700")
    try:
        # Third Party
        import zmq

        # First Party
        from lmcache.v1.multiprocess.mq import (
            MessageQueueClient,
            MessageQueueServer,
        )
        from lmcache.v1.multiprocess.protocol import HandlerType, RequestType
    except ModuleNotFoundError as exc:
        pytest.skip(f"MP protocol dependencies unavailable: {exc}")

    context = zmq.Context.instance()
    server_url = f"inproc://qos-profile-{time.monotonic_ns()}"
    server = MessageQueueServer(server_url, context)

    def handler() -> str:
        profile = get_current_qos_profile()
        return f"{profile.domain_id}:{profile.weight}:{profile.source}"

    server.add_handler(RequestType.NOOP, [], HandlerType.SYNC, handler)
    server.start()
    client = MessageQueueClient(server_url, context)
    try:
        assert client.submit_request(RequestType.NOOP, []).result(timeout=5) == (
            "handshake-a:700:handshake"
        )
    finally:
        client.close()
        server.close()


def test_qos_l2_adapter_releases_dispatch_slot() -> None:
    """The adapter wrapper maps concrete completions back to wrapper IDs."""
    try:
        # Standard
        from unittest.mock import MagicMock

        # First Party
        from lmcache.v1.distributed.api import ObjectKey
        from lmcache.v1.distributed.internal_api import L2StoreResult
        from lmcache.v1.distributed.l2_adapters.base import L2AdapterInterface
        from lmcache.v1.distributed.l2_qos_adapter import QosL2Adapter
        from lmcache.v1.multiprocess.qos import qos_profile_context
        from lmcache.v1.platform import create_event_notifier
    except ModuleNotFoundError as exc:
        pytest.skip(f"L2 adapter dependencies unavailable: {exc}")

    dispatcher = L2QoSDispatcher(max_inflight_tasks=1)
    notifiers = [create_event_notifier() for _ in range(3)]
    adapter = MagicMock(spec=L2AdapterInterface)
    adapter.get_store_event_fd.return_value = notifiers[0].fileno()
    adapter.get_lookup_and_lock_event_fd.return_value = notifiers[1].fileno()
    adapter.get_load_event_fd.return_value = notifiers[2].fileno()
    adapter.submit_store_task.return_value = 7
    wrapper = QosL2Adapter(adapter, dispatcher)
    key = ObjectKey(b"hash", "model", 0)
    memory_obj = MagicMock()
    memory_obj.get_size.return_value = 16
    profile = QosProfile(domain_id="adapter-test", weight=200)

    try:
        with qos_profile_context(profile):
            wrapped_task_id = wrapper.submit_store_task([key], [memory_obj])

        deadline = time.monotonic() + 2
        while not adapter.submit_store_task.called and time.monotonic() < deadline:
            time.sleep(0.001)
        assert adapter.submit_store_task.called

        adapter.pop_completed_store_tasks.return_value = {7: L2StoreResult(True, 16)}
        completed: dict[int, L2StoreResult] = {}
        while not completed and time.monotonic() < deadline:
            completed = wrapper.pop_completed_store_tasks()
            if not completed:
                time.sleep(0.001)

        assert completed[wrapped_task_id].is_successful()
        assert dispatcher.snapshot()["adapter-test"]["inflight_tasks"] == 0
    finally:
        wrapper.close()
        dispatcher.close()
        for notifier in notifiers:
            notifier.close()


def test_qos_l2_adapter_relays_pipe_notifications(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The wrapper relays adapter completion signals from pipe notifiers."""
    try:
        # Standard
        from unittest.mock import MagicMock

        # First Party
        from lmcache.v1.distributed.api import ObjectKey
        from lmcache.v1.distributed.internal_api import L2StoreResult
        from lmcache.v1.distributed.l2_adapters.base import L2AdapterInterface
        from lmcache.v1.multiprocess.qos import qos_profile_context
        from lmcache.v1.platform import PipeNotifier, consume_fd
        import lmcache.v1.distributed.l2_qos_adapter as qos_adapter_module
        import lmcache.v1.platform.event_notifier as event_notifier
    except ModuleNotFoundError as exc:
        pytest.skip(f"L2 adapter dependencies unavailable: {exc}")

    monkeypatch.setattr(event_notifier, "HAS_EVENTFD", False)
    monkeypatch.setattr(qos_adapter_module, "create_event_notifier", PipeNotifier)

    dispatcher = L2QoSDispatcher(max_inflight_tasks=1)
    notifiers = [PipeNotifier() for _ in range(3)]
    adapter = MagicMock(spec=L2AdapterInterface)
    adapter.get_store_event_fd.return_value = notifiers[0].fileno()
    adapter.get_lookup_and_lock_event_fd.return_value = notifiers[1].fileno()
    adapter.get_load_event_fd.return_value = notifiers[2].fileno()
    adapter.submit_store_task.return_value = 7
    wrapper = qos_adapter_module.QosL2Adapter(adapter, dispatcher)
    wrapper_event_fd = wrapper.get_store_event_fd()
    poller = select.poll()
    poller.register(wrapper_event_fd, select.POLLIN)
    key = ObjectKey(b"hash", "model", 0)
    memory_obj = MagicMock()
    memory_obj.get_size.return_value = 16

    try:
        with qos_profile_context(QosProfile(domain_id="pipe-test")):
            wrapped_task_id = wrapper.submit_store_task([key], [memory_obj])

        assert poller.poll(1000), "admission completion was not signaled"
        consume_fd(wrapper_event_fd)
        adapter.pop_completed_store_tasks.return_value = {7: L2StoreResult(True, 16)}
        notifiers[0].notify()
        assert poller.poll(1000), "adapter pipe notification was not relayed"
        consume_fd(wrapper_event_fd)

        completed = wrapper.pop_completed_store_tasks()
        assert completed[wrapped_task_id].is_successful()
    finally:
        wrapper.close()
        dispatcher.close()
        for notifier in notifiers:
            notifier.close()
