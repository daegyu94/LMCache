# SPDX-License-Identifier: Apache-2.0

"""Tests for replay-scoped exact L2 statistics."""

# Future
from __future__ import annotations

# Standard
import json

# Third Party
import pytest

# First Party
from lmcache.cli.commands.trace.l2_stats import L2LatencyStatsSubscriber
from lmcache.v1.mp_observability.event import Event, EventType


def test_stats_are_exact_and_adapter_agnostic(tmp_path):
    subscriber = L2LatencyStatsSubscriber()

    for task_id, latency_us in enumerate((1, 2, 3), start=1):
        subscriber.get_subscriptions()[EventType.L2_STORE_SUBMITTED](
            Event(
                EventType.L2_STORE_SUBMITTED,
                timestamp=10.0,
                metadata={
                    "adapter_index": 0,
                    "task_id": task_id,
                    "l2_name": "fs_native",
                    "total_bytes": 1000,
                },
            )
        )
        subscriber.get_subscriptions()[EventType.L2_STORE_COMPLETED](
            Event(
                EventType.L2_STORE_COMPLETED,
                timestamp=10.0 + latency_us / 1_000_000,
                metadata={
                    "adapter_index": 0,
                    "task_id": task_id,
                    "l2_name": "fs_native",
                },
            )
        )

    subscriber.get_subscriptions()[EventType.L2_LOAD_TASK_SUBMITTED](
        Event(
            EventType.L2_LOAD_TASK_SUBMITTED,
            timestamp=20.0,
            metadata={
                "request_id": 1,
                "adapter_index": 0,
                "task_id": 1,
                "l2_name": "HF3FS",
                "total_bytes": 2000,
            },
        )
    )
    subscriber.get_subscriptions()[EventType.L2_LOAD_TASK_COMPLETED](
        Event(
            EventType.L2_LOAD_TASK_COMPLETED,
            timestamp=20.000004,
            metadata={
                "request_id": 1,
                "adapter_index": 0,
                "task_id": 1,
                "l2_name": "HF3FS",
            },
        )
    )

    output_path = tmp_path / "l2_replay_stats.json"
    subscriber.write_json(output_path)
    result = json.loads(output_path.read_text(encoding="utf-8"))

    write_stats = result["operations"]["write"]["fs_native"]
    assert write_stats["completed"] == 3
    assert write_stats["average_latency_us"] == 2.0
    assert write_stats["p90_latency_us"] == pytest.approx(2.8)
    assert write_stats["p99_latency_us"] == pytest.approx(2.98)
    assert result["operations"]["read"]["HF3FS"]["p99_latency_us"] == 4.0
    assert result["sample_encoding"] == "uint32"
