# SPDX-License-Identifier: Apache-2.0

"""Exact, replay-scoped L2 latency and throughput statistics.

This collector is intentionally used by ``lmcache trace replay`` only.  A
long-running LMCache server should use the existing OTel metrics subscribers
instead of retaining every latency sample until process exit.
"""

# Future
from __future__ import annotations

# Standard
from array import array
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import json

# Third Party
import numpy as np

# First Party
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import EventCallback, EventSubscriber

_MAX_UINT32 = (1 << 32) - 1


@dataclass
class _LatencyAccumulator:
    """Exact latency samples and streaming totals for one adapter/op pair."""

    submitted: int = 0
    completed: int = 0
    unmatched_completed: int = 0
    total_us: int = 0
    total_bytes: int = 0
    samples_us: array = field(default_factory=lambda: array("I"))

    def record(self, latency_us: int, total_bytes: int) -> None:
        """Record one completed task using a four-byte microsecond sample."""
        self.completed += 1
        self.total_us += latency_us
        self.total_bytes += max(0, total_bytes)
        self.samples_us.append(min(latency_us, _MAX_UINT32))

    def percentile_us(self, quantile: float) -> float | None:
        """Return an exact sample percentile using in-place selection."""
        if not self.samples_us:
            return None
        values = np.frombuffer(self.samples_us, dtype=np.uint32)
        rank = (len(values) - 1) * quantile
        lower = int(rank)
        upper = min(lower + 1, len(values) - 1)
        values.partition((lower, upper))
        if lower == upper:
            return float(values[lower])
        fraction = rank - lower
        return float(values[lower] * (1.0 - fraction) + values[upper] * fraction)

    def as_dict(self) -> dict[str, Any]:
        """Return JSON-serializable aggregate and percentile statistics."""
        average_us = self.total_us / self.completed if self.completed else None
        throughput_gbps = (
            self.total_bytes / (self.total_us * 1000.0) if self.total_us > 0 else None
        )
        return {
            "submitted": self.submitted,
            "completed": self.completed,
            "unmatched_completed": self.unmatched_completed,
            "samples": len(self.samples_us),
            "total_bytes": self.total_bytes,
            "average_latency_us": average_us,
            "p50_latency_us": self.percentile_us(0.50),
            "p90_latency_us": self.percentile_us(0.90),
            "p99_latency_us": self.percentile_us(0.99),
            "min_latency_us": min(self.samples_us) if self.samples_us else None,
            "max_latency_us": max(self.samples_us) if self.samples_us else None,
            "aggregate_throughput_gbps": throughput_gbps,
        }


class L2LatencyStatsSubscriber(EventSubscriber):
    """Collect exact L2 task latency statistics for a finite replay.

    Store latency is measured from ``L2_STORE_SUBMITTED`` to
    ``L2_STORE_COMPLETED``. Load latency is measured from
    ``L2_LOAD_TASK_SUBMITTED`` to ``L2_LOAD_TASK_COMPLETED``. Samples are
    retained as unsigned 32-bit microseconds, so percentile values are exact
    for the stored microsecond resolution while using four bytes per sample.
    The statistics are separated by operation and adapter ``l2_name``.
    """

    def __init__(self) -> None:
        self._pending_store: dict[tuple[int, int], tuple[float, int, str]] = {}
        self._pending_load: dict[tuple[int, int], tuple[float, int, str]] = {}
        self._stats: dict[tuple[str, str], _LatencyAccumulator] = {}

    def get_subscriptions(self) -> dict[EventType, EventCallback]:
        """Return callbacks for L2 store/load submission and completion."""
        return {
            EventType.L2_STORE_SUBMITTED: self._on_store_submitted,
            EventType.L2_STORE_COMPLETED: self._on_store_completed,
            EventType.L2_LOAD_TASK_SUBMITTED: self._on_load_submitted,
            EventType.L2_LOAD_TASK_COMPLETED: self._on_load_completed,
        }

    def snapshot(self) -> dict[str, Any]:
        """Return exact JSON-serializable replay statistics.

        This method should be called after the owning EventBus has stopped,
        which guarantees that all queued completion events were dispatched.
        """
        operations: dict[str, dict[str, dict[str, Any]]] = {
            "read": {},
            "write": {},
        }
        for (operation, l2_name), accumulator in sorted(self._stats.items()):
            operations[operation][l2_name] = accumulator.as_dict()
        return {
            "schema_version": 1,
            "latency_unit": "microseconds",
            "sample_encoding": "uint32",
            "operations": operations,
            "pending": {
                "read": len(self._pending_load),
                "write": len(self._pending_store),
            },
        }

    def write_json(self, path: str | Path) -> None:
        """Write the current replay statistics to a JSON file."""
        output_path = Path(path).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(self.snapshot(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _on_store_submitted(self, event: Event) -> None:
        key = self._store_key(event)
        if key is None:
            return
        l2_name = str(event.metadata.get("l2_name", "unknown"))
        self._accumulator("write", l2_name).submitted += 1
        self._pending_store[key] = (
            event.timestamp,
            int(event.metadata.get("total_bytes", 0)),
            l2_name,
        )

    def _on_store_completed(self, event: Event) -> None:
        key = self._store_key(event)
        if key is None:
            return
        self._record("write", key, event, self._pending_store)

    def _on_load_submitted(self, event: Event) -> None:
        key = self._load_key(event)
        if key is None:
            return
        l2_name = str(event.metadata.get("l2_name", "unknown"))
        self._accumulator("read", l2_name).submitted += 1
        self._pending_load[key] = (
            event.timestamp,
            int(event.metadata.get("total_bytes", 0)),
            l2_name,
        )

    def _on_load_completed(self, event: Event) -> None:
        key = self._load_key(event)
        if key is None:
            return
        self._record("read", key, event, self._pending_load)

    def _record(
        self,
        operation: str,
        key: tuple[int, int],
        event: Event,
        pending: dict[tuple[int, int], tuple[float, int, str]],
    ) -> None:
        entry = pending.pop(key, None)
        l2_name = str(event.metadata.get("l2_name", "unknown"))
        accumulator = self._accumulator(operation, l2_name)
        if entry is None:
            accumulator.unmatched_completed += 1
            return
        start_timestamp, total_bytes, submitted_l2_name = entry
        if submitted_l2_name != l2_name:
            l2_name = submitted_l2_name
            accumulator = self._accumulator(operation, l2_name)
        latency_us = max(0, round((event.timestamp - start_timestamp) * 1_000_000))
        accumulator.record(latency_us, total_bytes)

    def _accumulator(self, operation: str, l2_name: str) -> _LatencyAccumulator:
        return self._stats.setdefault((operation, l2_name), _LatencyAccumulator())

    @staticmethod
    def _store_key(event: Event) -> tuple[int, int] | None:
        adapter_index = event.metadata.get("adapter_index")
        task_id = event.metadata.get("task_id")
        if adapter_index is None or task_id is None:
            return None
        return int(adapter_index), int(task_id)

    @staticmethod
    def _load_key(event: Event) -> tuple[int, int] | None:
        request_id = event.metadata.get("request_id")
        adapter_index = event.metadata.get("adapter_index")
        if request_id is None or adapter_index is None:
            return None
        return int(request_id), int(adapter_index)
