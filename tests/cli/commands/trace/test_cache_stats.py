# SPDX-License-Identifier: Apache-2.0

"""Tests for replay-scoped cache outcome statistics."""

# Future
from __future__ import annotations

# Standard
import json

# First Party
from lmcache.cli.commands.trace.cache_stats import CacheOutcomeStatsSubscriber
from lmcache.v1.mp_observability.event import Event, EventType


def _event(
    event_type: EventType,
    metadata: dict[str, int],
) -> Event:
    return Event(event_type=event_type, metadata=metadata)


def test_cache_outcomes_are_reported_by_key(tmp_path):
    subscriber = CacheOutcomeStatsSubscriber(speedup=4.0)
    callbacks = subscriber.get_subscriptions()

    subscriber.record_l1_retrieve(requested_keys=3, hit_keys=3)
    subscriber.record_l1_retrieve(requested_keys=2, hit_keys=0)
    callbacks[EventType.L2_PREFETCH_LOOKUP_SUBMITTED](
        _event(
            EventType.L2_PREFETCH_LOOKUP_SUBMITTED,
            {"key_count": 5},
        )
    )
    callbacks[EventType.L2_PREFETCH_LOOKUP_COMPLETED](
        _event(
            EventType.L2_PREFETCH_LOOKUP_COMPLETED,
            {"prefix_hit_count": 2},
        )
    )
    callbacks[EventType.L2_PREFETCH_LOAD_SUBMITTED](
        _event(
            EventType.L2_PREFETCH_LOAD_SUBMITTED,
            {"key_count": 3},
        )
    )
    callbacks[EventType.L2_PREFETCH_LOAD_COMPLETED](
        _event(
            EventType.L2_PREFETCH_LOAD_COMPLETED,
            {"loaded_count": 2, "failed_count": 1},
        )
    )

    output_path = tmp_path / "cache_replay_stats.json"
    subscriber.write_json(output_path)
    result = json.loads(output_path.read_text(encoding="utf-8"))

    assert result["speedup"] == 4.0
    assert result["l1_retrieve"] == {
        "hit_keys": 3,
        "hit_rate": 0.6,
        "miss_keys": 2,
        "requested_keys": 5,
        "requests": 2,
    }
    assert result["l2_lookup"]["hit_keys"] == 2
    assert result["l2_lookup"]["miss_keys"] == 3
    assert result["l2_lookup"]["hit_rate"] == 0.4
    assert result["l2_load"]["loaded_keys"] == 2
    assert result["l2_load"]["failed_keys"] == 1
    assert result["l2_load"]["success_rate"] == 2 / 3


def test_empty_cache_outcomes_have_null_rates():
    result = CacheOutcomeStatsSubscriber().snapshot()

    assert result["l1_retrieve"]["hit_rate"] is None
    assert result["l2_lookup"]["hit_rate"] is None
    assert result["l2_load"]["success_rate"] is None
