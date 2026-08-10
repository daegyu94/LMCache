# SPDX-License-Identifier: Apache-2.0

"""Replay-scoped cache hit and load outcome statistics."""

# Future
from __future__ import annotations

# Standard
from pathlib import Path
from typing import Any
import json

# First Party
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import EventCallback, EventSubscriber


def _rate(numerator: int, denominator: int) -> float | None:
    """Return a ratio, or ``None`` when no observations exist."""
    if denominator == 0:
        return None
    return numerator / denominator


class CacheOutcomeStatsSubscriber(EventSubscriber):
    """Collect replay-scoped L1/L2 cache outcomes.

    L1 retrieve outcomes are recorded at the dispatcher boundary because
    ``read_prefetched_results`` returns the actual objects visible to the
    caller.  L2 lookup and load outcomes use request-level EventBus events;
    lookup hits are the retained prefix count and load outcomes are the
    counts reported by the prefetch controller.
    """

    def __init__(self, speedup: float = 1.0) -> None:
        self._speedup = speedup
        self._l1_requests = 0
        self._l1_requested_keys = 0
        self._l1_hit_keys = 0
        self._l2_lookup_requests = 0
        self._l2_lookup_completed = 0
        self._l2_lookup_requested_keys = 0
        self._l2_lookup_hit_keys = 0
        self._l2_load_requests = 0
        self._l2_load_requested_keys = 0
        self._l2_load_loaded_keys = 0
        self._l2_load_failed_keys = 0

    def get_subscriptions(self) -> dict[EventType, EventCallback]:
        """Return callbacks for request-level lookup and load outcomes."""
        return {
            EventType.L2_PREFETCH_LOOKUP_SUBMITTED: self._on_lookup_submitted,
            EventType.L2_PREFETCH_LOOKUP_COMPLETED: self._on_lookup_completed,
            EventType.L2_PREFETCH_LOAD_SUBMITTED: self._on_load_submitted,
            EventType.L2_PREFETCH_LOAD_COMPLETED: self._on_load_completed,
        }

    def record_l1_retrieve(self, requested_keys: int, hit_keys: int) -> None:
        """Record one replayed L1 retrieve result.

        Args:
            requested_keys: Number of keys passed to the retrieve call.
            hit_keys: Number of objects returned by L1.  The current
                StorageManager API returns ``None`` for a partial read, so
                replay records such a result as zero hits.
        """
        requested_keys = max(0, requested_keys)
        hit_keys = min(requested_keys, max(0, hit_keys))
        self._l1_requests += 1
        self._l1_requested_keys += requested_keys
        self._l1_hit_keys += hit_keys

    def snapshot(self) -> dict[str, Any]:
        """Return JSON-serializable cache outcome aggregates."""
        l1_miss_keys = self._l1_requested_keys - self._l1_hit_keys
        l2_lookup_miss_keys = self._l2_lookup_requested_keys - self._l2_lookup_hit_keys
        return {
            "schema_version": 1,
            "speedup": self._speedup,
            "l1_retrieve": {
                "requests": self._l1_requests,
                "requested_keys": self._l1_requested_keys,
                "hit_keys": self._l1_hit_keys,
                "miss_keys": l1_miss_keys,
                "hit_rate": _rate(self._l1_hit_keys, self._l1_requested_keys),
            },
            "l2_lookup": {
                "submitted_requests": self._l2_lookup_requests,
                "completed_requests": self._l2_lookup_completed,
                "requested_keys": self._l2_lookup_requested_keys,
                "hit_keys": self._l2_lookup_hit_keys,
                "miss_keys": l2_lookup_miss_keys,
                "hit_rate": _rate(
                    self._l2_lookup_hit_keys,
                    self._l2_lookup_requested_keys,
                ),
            },
            "l2_load": {
                "submitted_requests": self._l2_load_requests,
                "requested_keys": self._l2_load_requested_keys,
                "loaded_keys": self._l2_load_loaded_keys,
                "failed_keys": self._l2_load_failed_keys,
                "success_rate": _rate(
                    self._l2_load_loaded_keys,
                    self._l2_load_requested_keys,
                ),
            },
        }

    def write_json(self, path: str | Path) -> None:
        """Write cache outcome aggregates to *path*."""
        output_path = Path(path).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(self.snapshot(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _on_lookup_submitted(self, event: Event) -> None:
        self._l2_lookup_requests += 1
        self._l2_lookup_requested_keys += int(event.metadata.get("key_count", 0))

    def _on_lookup_completed(self, event: Event) -> None:
        self._l2_lookup_completed += 1
        self._l2_lookup_hit_keys += int(event.metadata.get("prefix_hit_count", 0))

    def _on_load_submitted(self, event: Event) -> None:
        self._l2_load_requests += 1
        self._l2_load_requested_keys += int(event.metadata.get("key_count", 0))

    def _on_load_completed(self, event: Event) -> None:
        self._l2_load_loaded_keys += int(event.metadata.get("loaded_count", 0))
        self._l2_load_failed_keys += int(event.metadata.get("failed_count", 0))
