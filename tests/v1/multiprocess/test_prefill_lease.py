# SPDX-License-Identifier: Apache-2.0
"""Contract tests for ``PrefillLeaseTable``."""

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.multiprocess.prefill_lease import PrefillLeaseTable


def key(i: int) -> ObjectKey:
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(i),
        model_name="m",
        kv_rank=0,
        object_group_id=0,
    )


class Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def env():
    clock = Clock()
    resident: set[ObjectKey] = set()
    dead: set[str] = set()
    table = PrefillLeaseTable(
        10.0, resident.__contains__, lambda rid: rid not in dead, clock
    )
    return table, clock, resident, dead


def test_waiter_is_served_once_every_chunk_is_resident(env):
    table, clock, resident, _ = env
    keys = [key(0), key(1)]
    assert table.acquire_or_wait("a", keys, "e1")
    assert not table.acquire_or_wait("b", keys, "e2")
    resident.add(key(0))
    assert not table.ready("b")
    resident.add(key(1))
    clock.now += 1.5
    assert table.ready("b")
    assert not table.is_waiting("b")
    stats = table.stats()
    assert (stats["waits"], stats["served"], stats["timeouts"]) == (1, 1, 0)
    assert stats["wait_seconds"] == pytest.approx(1.5)


def test_progress_keeps_the_waiter_waiting_past_the_timeout(env):
    """Each chunk the owner stores restarts the idle timer, so a long but
    progressing prefill is waited for."""
    table, clock, resident, _ = env
    keys = [key(i) for i in range(3)]
    assert table.acquire_or_wait("a", keys, "e1")
    assert not table.acquire_or_wait("b", keys, "e2")
    for i in range(2):
        clock.now += 9.0
        resident.add(key(i))
        assert not table.ready("b")
    clock.now += 9.0
    assert not table.ready("b")
    resident.add(key(2))
    assert table.ready("b")
    assert table.stats()["served"] == 1


def test_waiter_gives_up_without_progress(env):
    table, clock, _, _ = env
    assert table.acquire_or_wait("a", [key(0)], "e1")
    assert not table.acquire_or_wait("b", [key(0)], "e2")
    clock.now += 9.9
    assert not table.ready("b")
    clock.now += 0.1
    assert table.ready("b")
    assert table.stats()["timeouts"] == 1


def test_waits_only_on_chunks_another_engine_leased(env):
    table, _, resident, _ = env
    assert table.acquire_or_wait("a", [key(0)], "e1")
    assert not table.acquire_or_wait("b", [key(0), key(1)], "e2")
    resident.add(key(0))
    assert table.ready("b")


def test_requests_of_the_same_engine_never_wait(env):
    """The engine's own prefix cache shares the prompt; the second request
    proceeds and the first keeps the lease."""
    table, _, _, _ = env
    assert table.acquire_or_wait("a", [key(0)], "e1")
    assert table.acquire_or_wait("a2", [key(0)], "e1")
    assert not table.acquire_or_wait("b", [key(0)], "e2")
    table.forget("a2")
    assert not table.ready("b")
    table.forget("a")
    assert table.ready("b")


def test_owner_releasing_a_hit_chunk_ends_the_wait(env):
    table, _, _, _ = env
    assert table.acquire_or_wait("a", [key(0), key(1)], "e1")
    assert not table.acquire_or_wait("b", [key(0)], "e2")
    table.release("a", [key(0)])
    assert table.ready("b")
    assert table.acquire_or_wait("c", [key(0)], "e3")


def test_dead_owner_neither_blocks_nor_keeps_its_leases(env):
    table, _, _, dead = env
    assert table.acquire_or_wait("a", [key(0)], "e1")
    assert not table.acquire_or_wait("b", [key(0)], "e2")
    dead.add("a")
    assert table.ready("b")
    assert table.acquire_or_wait("c", [key(0)], "e3")
    table.forget("c")
    table.sweep()
    assert table.stats()["leases"] == 0


def test_sweep_drops_waits_of_ended_requests(env):
    table, _, _, dead = env
    assert table.acquire_or_wait("a", [key(0)], "e1")
    assert not table.acquire_or_wait("b", [key(0)], "e2")
    dead.add("b")
    table.sweep()
    assert not table.is_waiting("b")
    assert table.stats()["leases"] == 1


def test_owner_may_look_up_its_own_chunks_again(env):
    table, _, _, _ = env
    assert table.acquire_or_wait("a", [key(0)], "e1")
    assert table.acquire_or_wait("a", [key(0)], "e1")


def test_request_not_waiting_is_ready(env):
    table, _, _, _ = env
    assert table.ready("never-waited")


def test_forget_drops_a_wait(env):
    table, _, _, _ = env
    assert table.acquire_or_wait("a", [key(0)], "e1")
    assert not table.acquire_or_wait("b", [key(0)], "e2")
    table.forget("b")
    assert not table.is_waiting("b")
    assert table.stats()["waiting"] == 0


def test_non_positive_timeout_is_rejected():
    with pytest.raises(ValueError):
        PrefillLeaseTable(0.0, lambda _key: False, lambda _rid: True)
