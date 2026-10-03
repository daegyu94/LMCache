# SPDX-License-Identifier: Apache-2.0
"""Behavioral tests for dependency utility, bounded advice and ownership."""

# Standard
from concurrent.futures import ThreadPoolExecutor

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.eviction_policy.lru import LRUEvictionPolicy
from lmcache.v1.distributed.eviction_policy.prefix_utility import (
    PrefixUtilityPolicy,
    read_prefix_advice,
)


def key(number: int, salt: str = "g0") -> ObjectKey:
    """Create a namespace-qualified object for the contract tests."""
    return ObjectKey(number.to_bytes(8, "big"), "model", 0, cache_salt=salt)


def victim(policy: PrefixUtilityPolicy) -> ObjectKey:
    """Select one candidate without pretending deletion succeeded."""
    return policy.get_eviction_actions(0.001)[0].keys[0]


def test_shared_interest_survives_one_completion() -> None:
    p = PrefixUtilityPolicy()
    root, a, b = [key(i) for i in range(3)]
    p.on_keys_created([root, a, b])
    p.advise_prefix("a", [root, a], "soon")
    p.advise_prefix("b", [root, b], "soon")
    p.advise_prefix("a", [root, a], "terminal")
    assert victim(p) == a
    p.on_keys_removed([a])
    assert victim(p) == b
    assert p.get_prefix_stats()["active_manifests"] == 1


def test_hole_makes_suffix_zero_value() -> None:
    p = PrefixUtilityPolicy()
    root, missing, tail, live = [key(i) for i in range(4)]
    p.on_keys_created([root, tail, live])
    p.advise_prefix("hole", [root, missing, tail], "soon")
    p.advise_prefix("live", [live], "soon")
    assert victim(p) == tail


def test_greedy_batch_recalculates_frontier_without_deleting() -> None:
    p = PrefixUtilityPolicy()
    root, mid, tail = [key(i) for i in range(3)]
    p.on_keys_created([root, mid, tail])
    p.advise_prefix("a", [root, mid, tail], "soon")
    assert p.get_eviction_actions(1)[0].keys == [tail, mid, root]
    assert p.get_eviction_actions(1)[0].keys == [tail, mid, root]


def test_locked_leaf_does_not_override_authoritative_filter() -> None:
    p = PrefixUtilityPolicy()
    root, tail = key(1), key(2)
    p.on_keys_created([root, tail])
    p.advise_prefix("a", [root, tail], "soon")
    assert p.get_eviction_actions(1, lambda k: k != tail)[0].keys == [root]


def test_staging_is_not_a_readable_prefix() -> None:
    p = PrefixUtilityPolicy()
    root, tail, live = key(1), key(2), key(3)
    p.on_keys_reserved([root])
    p.on_keys_created([tail, live])
    p.advise_prefix("a", [root, tail], "soon")
    p.advise_prefix("b", [live], "soon")
    assert p.get_eviction_actions(0.01, lambda k: k != root)[0].keys == [tail]


def test_generation_qualified_interest() -> None:
    p = PrefixUtilityPolicy()
    old, new = key(1, "g0"), key(1, "g1")
    p.on_keys_created([old, new])
    p.advise_prefix("same-id", [old], "soon")
    p.advise_prefix("same-id", [new], "soon")
    p.advise_prefix("same-id", [new], "terminal")
    assert victim(p) == new
    assert p.get_prefix_stats()["manifests"] == 2


@pytest.mark.parametrize("mode", ["PrefixLRU", "SessionRefcount", "PrefixUtility"])
def test_no_advice_matches_lru(mode: str) -> None:
    p = PrefixUtilityPolicy(mode)  # type: ignore[arg-type]
    lru = LRUEvictionPolicy()
    keys = [key(i) for i in range(12)]
    for policy in (p, lru):
        policy.on_keys_created(keys)
        policy.on_keys_touched(keys[3:7])
        policy.on_keys_removed([keys[1]])
    assert p.get_eviction_actions(1)[0].keys == lru.get_eviction_actions(1)[0].keys


def test_metadata_budget_and_malformed_advice() -> None:
    p = PrefixUtilityPolicy(max_cohorts=2, max_references=4)
    for i in range(10):
        p.advise_prefix(str(i), [key(i), key(i + 1)], "soon")
    assert p.get_prefix_stats()["key_references"] == 4
    assert p.get_prefix_stats()["discarded_manifests"] == 8
    before = p.get_prefix_stats()
    p.advise_prefix("mixed", [key(1), key(2, "other")], "soon")
    p.advise_prefix("big", [key(i) for i in range(5)], "soon")
    p.advise_prefix("bad", [key(1), key(1)], "soon")
    assert p.get_prefix_stats() == before
    assert read_prefix_advice(None) is None
    assert read_prefix_advice({"lmcache.agentrl.trajectory_id": []}) is None


def test_leaf_comparator_and_utility_have_distinct_objectives() -> None:
    root, tail, single = key(1), key(2), key(3)
    choices = []
    for mode in ("PrefixLRU", "PrefixUtility"):
        p = PrefixUtilityPolicy(mode)
        p.on_keys_created([single, root, tail])
        p.on_keys_touched([root, tail])
        p.advise_prefix("live", [single], "soon")
        p.advise_prefix("done", [root, tail], "terminal")
        choices.append(victim(p))
    assert choices == [single, tail]


def test_concurrent_advice_respects_hard_limits() -> None:
    p = PrefixUtilityPolicy(max_cohorts=8, max_references=16)

    def advise(i: int) -> None:
        p.advise_prefix(str(i), [key(i), key(i + 1)], "soon")
        p.get_eviction_actions(0.5)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(advise, range(100)))
    assert p.get_prefix_stats()["manifests"] <= 8
    assert p.get_prefix_stats()["key_references"] <= 16


def test_no_advice_repeated_reservations_match_lru() -> None:
    p = PrefixUtilityPolicy()
    lru = LRUEvictionPolicy()
    keys = [key(i) for i in range(8)]
    for policy in (p, lru):
        policy.on_keys_created(keys)
        policy.on_keys_reserved(keys[2:5])
        policy.on_keys_reserved(keys[3:6])
    assert p.get_eviction_actions(1)[0].keys == lru.get_eviction_actions(1)[0].keys


def test_late_and_oversized_advice_remain_performance_hints() -> None:
    p = PrefixUtilityPolicy(max_references=2)
    p.advise_prefix("a", [key(1)], "terminal")
    p.advise_prefix("a", [key(1)], "soon")
    p.advise_prefix("a", [key(1), key(2), key(3)], "terminal")
    assert p.get_prefix_stats()["active_manifests"] == 1
    assert p.get_prefix_stats()["advice_calls"] == 2
