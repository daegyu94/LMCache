# SPDX-License-Identifier: Apache-2.0
"""Tests for cgroup-derived shared-L2 QoS scheduling."""

# Standard
from pathlib import Path
import threading
import time

# Third Party
import pytest

# First Party
from lmcache.v1.multiprocess.qos import (
    L2QoSDispatcher,
    QosProfile,
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
