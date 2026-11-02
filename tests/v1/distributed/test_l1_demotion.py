# SPDX-License-Identifier: Apache-2.0
"""``demote_keys`` reorders L1 eviction without removing anything.

Runs a real ``DramL1Manager`` with each eviction policy bridged in through
``L1EvictionPolicy``, the same wiring the L1 eviction controller uses.
"""

# Standard
from collections.abc import Iterator
import threading

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.config import L1ManagerConfig, L1MemoryManagerConfig
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.eviction import EvictionPolicy, L1EvictionPolicy
from lmcache.v1.distributed.eviction_policy import (
    ARCEvictionPolicy,
    IsolatedLRUEvictionPolicy,
    LRUEvictionPolicy,
)
from lmcache.v1.distributed.internal_api import L1ManagerListener
from lmcache.v1.distributed.l1_manager import DramL1Manager
import lmcache.v1.memory_management as memory_management

pytestmark = pytest.mark.no_shared_allocator

ALIGN = 4096
LAYOUT = MemoryLayoutDesc([torch.Size([1024])], [torch.float32])  # 4 KiB


def _key(index: int, cache_salt: str = "") -> ObjectKey:
    return ObjectKey(
        ObjectKey.IntHash2Bytes(index), "demotion", 0, cache_salt=cache_salt
    )


@pytest.fixture
def l1(monkeypatch: pytest.MonkeyPatch) -> Iterator[DramL1Manager]:
    monkeypatch.setattr(
        memory_management,
        "_allocate_cpu_memory",
        lambda size, *args, **kwargs: torch.empty(size, dtype=torch.uint8),
    )
    monkeypatch.setattr(memory_management, "_free_cpu_memory", lambda *a, **k: None)
    manager = DramL1Manager(
        L1ManagerConfig(
            L1MemoryManagerConfig(size_in_bytes=64 * ALIGN, use_lazy=False, shm_name="")
        )
    )
    yield manager
    manager.close()


def _attach(l1: DramL1Manager, policy: EvictionPolicy) -> EvictionPolicy:
    l1.register_listener(L1EvictionPolicy(policy))
    return policy


def _write(l1: DramL1Manager, keys: list[ObjectKey]) -> None:
    reserved = l1.reserve_write(keys, [False] * len(keys), LAYOUT)
    assert all(err is L1Error.SUCCESS for err, _ in reserved.values())
    assert set(l1.finish_write(keys).values()) == {L1Error.SUCCESS}


def _eviction_order(policy: EvictionPolicy, cache_salt: str | None = None) -> list:
    actions = policy.get_eviction_actions(1.0, cache_salt=cache_salt)
    return [key for action in actions for key in action.keys]


def test_lru_demoted_keys_are_evicted_first_tail_first(l1: DramL1Manager) -> None:
    policy = _attach(l1, LRUEvictionPolicy())
    old_chain = [_key(i) for i in range(3)]
    _write(l1, old_chain)
    other = [_key(10), _key(11)]
    _write(l1, other)

    l1.demote_keys(old_chain[1:])

    order = _eviction_order(policy)
    assert order[:2] == [old_chain[2], old_chain[1]]
    assert set(order[2:]) == {old_chain[0], *other}


def test_demoted_key_stays_resident_and_touch_restores_recency(
    l1: DramL1Manager,
) -> None:
    """A key demoted for one consumer is still readable by another, and that
    consumer's use moves it back to the most recently used end."""
    policy = _attach(l1, LRUEvictionPolicy())
    shared, other = _key(1), _key(2)
    _write(l1, [shared, other])

    l1.demote_keys([shared])
    assert _eviction_order(policy)[0] == shared

    error, _ = l1.reserve_read([shared])[shared]
    assert error is L1Error.SUCCESS
    assert l1.finish_read([shared]) == {shared: L1Error.SUCCESS}
    l1.touch_keys([shared])
    assert _eviction_order(policy) == [other, shared]


def test_demoting_unknown_keys_is_a_no_op(l1: DramL1Manager) -> None:
    policy = _attach(l1, LRUEvictionPolicy())
    keys = [_key(1), _key(2)]
    _write(l1, keys)
    before = _eviction_order(policy)

    l1.demote_keys([_key(99)])

    assert _eviction_order(policy) == before


def test_isolated_lru_demotes_within_the_keys_salt(l1: DramL1Manager) -> None:
    policy = _attach(l1, IsolatedLRUEvictionPolicy())
    a_keys = [_key(1, "a"), _key(2, "a")]
    b_keys = [_key(3, "b"), _key(4, "b")]
    _write(l1, a_keys + b_keys)
    b_before = _eviction_order(policy, cache_salt="b")

    l1.demote_keys([a_keys[0]])

    assert _eviction_order(policy, cache_salt="a")[0] == a_keys[0]
    assert _eviction_order(policy, cache_salt="b") == b_before


def test_policy_without_demotion_support_keeps_its_order(l1: DramL1Manager) -> None:
    """ARC keeps the base-class no-op, so its eviction order is unchanged."""
    policy = _attach(l1, ARCEvictionPolicy())
    keys = [_key(i) for i in range(4)]
    _write(l1, keys)
    before = _eviction_order(policy)

    l1.demote_keys(keys[2:])

    assert _eviction_order(policy) == before


def test_read_locked_demoted_key_is_skipped_by_the_eviction_filter(
    l1: DramL1Manager,
) -> None:
    """Demotion puts a key first in line, but a reader's lock still wins:
    the controller's evictability filter skips it."""
    policy = _attach(l1, LRUEvictionPolicy())
    locked, other = _key(1), _key(2)
    _write(l1, [locked, other])
    assert l1.reserve_read([locked])[locked][0] is L1Error.SUCCESS

    l1.demote_keys([locked])
    actions = policy.get_eviction_actions(1.0, key_eligible_filter=l1.is_key_evictable)

    assert [key for action in actions for key in action.keys] == [other]
    assert l1.finish_read([locked]) == {locked: L1Error.SUCCESS}


class _GateListener(L1ManagerListener):
    """Pauses the first demotion until the test lets it continue."""

    def __init__(self) -> None:
        self.demotion_started = threading.Event()
        self.may_continue = threading.Event()

    def on_l1_keys_demoted(self, keys: list[ObjectKey]) -> None:
        self.demotion_started.set()
        assert self.may_continue.wait(timeout=10)

    def on_l1_keys_reserved_read(self, keys: list[ObjectKey]) -> None:
        pass

    def on_l1_keys_read_finished(self, keys: list[ObjectKey]) -> None:
        pass

    def on_l1_keys_reserved_write(self, keys: list[ObjectKey]) -> None:
        pass

    def on_l1_keys_write_finished(self, keys: list[ObjectKey]) -> None:
        pass

    def on_l1_keys_finish_write_and_reserve_read(self, keys: list[ObjectKey]) -> None:
        pass

    def on_l1_keys_deleted_by_manager(self, keys: list[ObjectKey]) -> None:
        pass

    def on_l1_keys_accessed(self, keys: list[ObjectKey]) -> None:
        pass


def test_demotion_racing_an_eviction_does_not_resurrect_the_key(
    l1: DramL1Manager,
) -> None:
    """A demotion that reaches the policy after the eviction controller
    deleted the key must not bring the key back into the eviction order.

    The interleaving is forced with events: the demotion is held between
    listeners while the eviction runs on the main thread.
    """
    gate = _GateListener()
    l1.register_listener(gate)  # runs before the policy's listener
    policy = _attach(l1, LRUEvictionPolicy())
    victim, survivor = _key(1), _key(2)
    _write(l1, [victim])
    _write(l1, [survivor])  # victim is now the least recently used

    demoter = threading.Thread(target=l1.demote_keys, args=([victim],))
    demoter.start()
    assert gate.demotion_started.wait(timeout=10)
    for action in policy.get_eviction_actions(
        0.5, key_eligible_filter=l1.is_key_evictable
    ):
        assert action.keys == [victim]
        assert l1.delete(action.keys) == {victim: L1Error.SUCCESS}
    gate.may_continue.set()
    demoter.join(timeout=10)
    assert not demoter.is_alive()

    assert _eviction_order(policy) == [survivor]
    assert l1.reserve_read([victim])[victim][0] is L1Error.KEY_NOT_EXIST
