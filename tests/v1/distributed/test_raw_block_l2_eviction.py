# SPDX-License-Identifier: Apache-2.0
"""Raw-block L2 adapter and L2 eviction controller together: stores that find
no contiguous space below the watermark get an eviction round."""

# Standard
import logging
import os

# First Party
from lmcache.v1.distributed.config import EvictionConfig
from lmcache.v1.distributed.quota_manager import QuotaManager
from lmcache.v1.distributed.storage_controllers.eviction_controller import (
    L2AdapterEvictionState,
    L2EvictionController,
)
from tests.v1.distributed.test_raw_block_l2_adapter import (
    RawBlockL2Adapter,
    _create_memory_obj,
    _create_object_key,
    _make_config,
    _run_lookup,
    _run_store,
    requires_raw_block_ext,
)

controller_logger = L2EvictionController.__module__
SMALL = 8 * 1024  # header + one 1024-float payload
N = 32


def _adapter(tmp_path) -> RawBlockL2Adapter:
    dev = os.path.join(tmp_path, "dev.bin")
    with open(dev, "wb") as f:
        f.truncate(8 << 20)
    # Metadata, the format-marker block, then room for N small objects.
    return RawBlockL2Adapter(
        _make_config(dev, capacity_bytes=(1 << 20) + 4096 + N * SMALL)
    )


def _fill_and_punch(adapter: RawBlockL2Adapter) -> None:
    """alice at even positions, bob at odd; bob's are deleted, leaving free
    space as holes of one small object each, at 50 % usage."""
    bob = []
    for i in range(N):
        salt = "alice" if i % 2 == 0 else "bob"
        key = _create_object_key(i, cache_salt=salt)
        assert _run_store(adapter, [key], [_create_memory_obj(1024)])
        if salt == "bob":
            bob.append(key)
    adapter.delete(bob)


def _store_until_it_fits(adapter, controller, state, key) -> int:
    rounds = 0
    while not _run_store(adapter, [key], [_create_memory_obj(4096)]):
        controller._check_and_evict(state)
        rounds += 1
        assert rounds < 10
    return rounds


@requires_raw_block_ext
def test_a_store_that_finds_no_range_gets_eviction_below_the_watermark(tmp_path):
    adapter = _adapter(str(tmp_path))
    try:
        cfg = EvictionConfig(
            eviction_policy="LRU", trigger_watermark=0.8, eviction_ratio=0.2
        )
        state = L2AdapterEvictionState(0, adapter, cfg)
        controller = L2EvictionController([state], quota_manager=None)
        _fill_and_punch(adapter)
        assert adapter.get_usage().usage_fraction < 0.8
        controller._check_and_evict(state)  # nothing failed: nothing evicted
        used = adapter.get_usage().total_bytes_used

        rounds = _store_until_it_fits(
            adapter, controller, state, _create_object_key(99)
        )
        assert rounds >= 1
        assert adapter.get_usage().total_bytes_used < used + 5 * SMALL
        status = controller.report_status()["adapters"][0]
        assert status["space_exhausted_rounds"] == rounds
    finally:
        adapter.close()


@requires_raw_block_ext
def test_under_quotas_only_the_salt_closest_to_its_quota_gives_up_space(
    tmp_path, caplog
):
    adapter = _adapter(str(tmp_path))
    try:
        cfg = EvictionConfig(
            eviction_policy="IsolatedLRU", trigger_watermark=0.8, eviction_ratio=0.2
        )
        state = L2AdapterEvictionState(0, adapter, cfg)
        qm = QuotaManager()
        for salt in ("alice", "bob", "carol"):
            qm.set_quota(salt, 1 << 20)
        controller = L2EvictionController([state], quota_manager=qm)
        _fill_and_punch(adapter)
        carol = _create_object_key(500, cache_salt="carol")
        assert _run_store(adapter, [carol], [_create_memory_obj(1024)])  # a hole
        before = dict(adapter.get_usage().bytes_by_cache_salt)

        with caplog.at_level(logging.INFO, logger=controller_logger):
            _store_until_it_fits(
                adapter, controller, state, _create_object_key(99, cache_salt="bob")
            )
        after = adapter.get_usage().bytes_by_cache_salt
        # alice holds most of her quota and fragments the space; carol keeps hers.
        assert after["alice"] < before["alice"]
        assert after["carol"] == before["carol"]
        messages = [r.getMessage() for r in caplog.records]
        assert not any("over quota" in m for m in messages)
        assert controller.report_status()["adapters"][0]["space_exhausted_rounds"]
    finally:
        adapter.close()


@requires_raw_block_ext
def test_a_salt_whose_data_is_locked_is_passed_over(tmp_path):
    """The salt closest to its quota holds only locked keys, which eviction
    skips; the next salt gives up the round instead."""
    adapter = _adapter(str(tmp_path))
    try:
        cfg = EvictionConfig(
            eviction_policy="IsolatedLRU", trigger_watermark=0.8, eviction_ratio=0.2
        )
        state = L2AdapterEvictionState(0, adapter, cfg)
        qm = QuotaManager()
        qm.set_quota("alice", N * SMALL)  # half of it used: the fullest
        qm.set_quota("bob", 4 * N * SMALL)
        controller = L2EvictionController([state], quota_manager=qm)
        alice = [_create_object_key(i, cache_salt="alice") for i in range(N // 2)]
        bob = [_create_object_key(100 + i, cache_salt="bob") for i in range(N // 2)]
        for key in alice + bob:
            assert _run_store(adapter, [key], [_create_memory_obj(1024)])
        _run_lookup(adapter, alice)  # locks alice's keys
        before = dict(adapter.get_usage().bytes_by_cache_salt)

        _store_until_it_fits(adapter, controller, state, _create_object_key(99))
        after = adapter.get_usage().bytes_by_cache_salt
        assert after["alice"] == before["alice"]
        assert after["bob"] < before["bob"]
        rounds = controller.report_status()["adapters"][0]["space_exhausted_rounds"]
        assert rounds >= 1
    finally:
        adapter.close()


@requires_raw_block_ext
def test_an_over_quota_salt_with_only_locked_keys_does_not_stop_the_round(tmp_path):
    """eve has no quota, so she is over it every pass, but her only key is
    locked: evicting her frees nothing, so a salt under its quota still gives
    up the round."""
    adapter = _adapter(str(tmp_path))
    try:
        cfg = EvictionConfig(
            eviction_policy="IsolatedLRU", trigger_watermark=0.8, eviction_ratio=0.2
        )
        state = L2AdapterEvictionState(0, adapter, cfg)
        qm = QuotaManager()
        qm.set_quota("alice", 1 << 20)
        qm.set_quota("bob", 1 << 20)
        controller = L2EvictionController([state], quota_manager=qm)
        _fill_and_punch(adapter)
        eve = _create_object_key(700, cache_salt="eve")
        assert _run_store(adapter, [eve], [_create_memory_obj(1024)])
        _run_lookup(adapter, [eve])  # locks it
        _store_until_it_fits(
            adapter, controller, state, _create_object_key(99, cache_salt="bob")
        )
        assert adapter.get_usage().bytes_by_cache_salt["eve"] > 0
        rounds = controller.report_status()["adapters"][0]["space_exhausted_rounds"]
        assert rounds >= 1
    finally:
        adapter.close()


@requires_raw_block_ext
def test_a_round_that_frees_nothing_is_not_counted(tmp_path):
    """Fragmented below the watermark, and every remaining key is locked: the
    round deletes nothing, so it is not reported as one."""
    adapter = _adapter(str(tmp_path))
    try:
        cfg = EvictionConfig(
            eviction_policy="LRU", trigger_watermark=0.8, eviction_ratio=0.2
        )
        state = L2AdapterEvictionState(0, adapter, cfg)
        controller = L2EvictionController([state], quota_manager=None)
        _fill_and_punch(adapter)  # alice's keys remain, at 50 % usage
        alice = [_create_object_key(i, cache_salt="alice") for i in range(0, N, 2)]
        _run_lookup(adapter, alice)
        used = adapter.get_usage().total_bytes_used
        assert not _run_store(
            adapter, [_create_object_key(99)], [_create_memory_obj(4096)]
        )
        controller._check_and_evict(state)
        assert adapter.get_usage().total_bytes_used == used
        status = controller.report_status()["adapters"][0]
        assert status["space_exhausted_rounds"] == 0
    finally:
        adapter.close()
