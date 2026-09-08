# SPDX-License-Identifier: Apache-2.0
"""Metrics aggregation for the synthetic L2 benchmark."""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass, field
import math


@dataclass(frozen=True)
class LatencySample:
    """Timing spans for one request.

    ``e2e_seconds`` is authoritative. The other spans overlap and are not an
    additive latency breakdown.
    """

    queue_seconds: float
    lookup_seconds: float
    submit_seconds: float
    completion_seconds: float
    e2e_seconds: float


@dataclass
class Metrics:
    """Collect successful bytes, failures, and request latency samples."""

    read_requests: int = 0
    write_requests: int = 0
    read_bytes: int = 0
    write_bytes: int = 0
    read_hits: int = 0
    read_misses: int = 0
    errors: int = 0
    timeouts: int = 0
    integrity_failures: int = 0
    samples: dict[str, list[LatencySample]] = field(default_factory=dict)

    def add(
        self,
        operation: str,
        sample: LatencySample,
        *,
        read_bytes: int = 0,
        write_bytes: int = 0,
        hits: int = 0,
        misses: int = 0,
        error: bool = False,
        timeout: bool = False,
        integrity: bool = False,
    ) -> None:
        """Add one measured request; warmup callers must not call this method."""
        self.samples.setdefault(operation, []).append(sample)
        if read_bytes:
            self.read_requests += 1
        if write_bytes:
            self.write_requests += 1
        self.read_bytes += read_bytes
        self.write_bytes += write_bytes
        self.read_hits += hits
        self.read_misses += misses
        self.errors += int(error)
        self.timeouts += int(timeout)
        self.integrity_failures += int(integrity)

    @staticmethod
    def percentile(values: list[float], fraction: float) -> float | None:
        """Return an interpolated finite percentile, or ``None`` if empty."""
        finite = sorted(value for value in values if math.isfinite(value))
        if not finite:
            return None
        rank = (len(finite) - 1) * fraction
        low = int(rank)
        high = min(low + 1, len(finite) - 1)
        return finite[low] + (finite[high] - finite[low]) * (rank - low)

    def summary(self, elapsed_seconds: float) -> dict[str, object]:
        """Return JSON-compatible counters and percentiles for measured I/O."""
        elapsed = max(elapsed_seconds, 0.0)
        result: dict[str, object] = {
            "read_requests": self.read_requests,
            "write_requests": self.write_requests,
            "read_bytes": self.read_bytes,
            "write_bytes": self.write_bytes,
            "read_hits": self.read_hits,
            "read_misses": self.read_misses,
            "errors": self.errors,
            "timeouts": self.timeouts,
            "integrity_failures": self.integrity_failures,
            "elapsed_seconds": elapsed,
            "read_bytes_per_second": self.read_bytes / elapsed if elapsed else 0.0,
            "write_bytes_per_second": self.write_bytes / elapsed if elapsed else 0.0,
            "successful_bytes_per_second": (self.read_bytes + self.write_bytes)
            / elapsed
            if elapsed
            else 0.0,
        }
        latencies: dict[str, dict[str, float | None]] = {}
        for operation, samples in self.samples.items():
            operation_latency: dict[str, float | None] = {}
            for name in (
                "queue_seconds",
                "lookup_seconds",
                "submit_seconds",
                "completion_seconds",
                "e2e_seconds",
            ):
                values = [getattr(sample, name) for sample in samples]
                for fraction, suffix in ((0.5, "p50"), (0.95, "p95"), (0.99, "p99")):
                    operation_latency[f"{name}_{suffix}"] = self.percentile(
                        values, fraction
                    )
            latencies[operation] = operation_latency
        result["latency_seconds"] = latencies
        return result
