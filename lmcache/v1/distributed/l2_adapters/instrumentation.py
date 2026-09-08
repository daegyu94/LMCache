# SPDX-License-Identifier: Apache-2.0
"""Opt-in, thread-safe latency collection for L2 adapter operations."""

# Future
from __future__ import annotations

# Standard
from collections import defaultdict
import math
import threading
import time


class L2AdapterInstrumentation:
    """Collect named elapsed-time spans from a single L2 adapter instance.

    The collector is deliberately passed to an adapter instance instead of
    installed globally.  It is therefore safe to use in a benchmark process
    with more than one adapter and has zero recording work until enabled.
    Spans can overlap (for example, concurrent NIXL file transfers), so their
    percentiles must not be summed or interpreted as an E2E decomposition.
    """

    def __init__(self) -> None:
        self._samples: dict[str, list[float]] = defaultdict(list)
        self._lock = threading.Lock()

    def record(self, name: str, seconds: float) -> None:
        """Record one finite non-negative span under ``name``."""
        if not math.isfinite(seconds) or seconds < 0:
            return
        with self._lock:
            self._samples[name].append(seconds)

    def start(self) -> float:
        """Return a monotonic timestamp for use with :meth:`finish`."""
        return time.monotonic()

    def finish(self, name: str, started_at: float) -> None:
        """Record time elapsed since a timestamp returned by :meth:`start`."""
        self.record(name, time.monotonic() - started_at)

    @staticmethod
    def _percentile(values: list[float], fraction: float) -> float | None:
        if not values:
            return None
        ordered = sorted(values)
        rank = (len(ordered) - 1) * fraction
        low = int(rank)
        high = min(low + 1, len(ordered) - 1)
        return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)

    def snapshot(self) -> dict[str, dict[str, float | int | None]]:
        """Return JSON-compatible per-span count and percentiles."""
        with self._lock:
            copied = {name: list(values) for name, values in self._samples.items()}
        return {
            name: {
                "count": len(values),
                "p50": self._percentile(values, 0.5),
                "p95": self._percentile(values, 0.95),
                "p99": self._percentile(values, 0.99),
            }
            for name, values in copied.items()
        }
