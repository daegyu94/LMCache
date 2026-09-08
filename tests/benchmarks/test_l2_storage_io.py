# SPDX-License-Identifier: Apache-2.0
"""Focused tests for the CPU-only L2 storage I/O benchmark helpers."""

# Future
from __future__ import annotations

# Standard
from typing import cast

# Third Party
import pytest

# First Party
from benchmarks.l2_storage_io.benchmark import StoreCompletionDispatcher
from benchmarks.l2_storage_io.metrics import LatencySample, Metrics
from benchmarks.l2_storage_io.workload import WorkloadSpec, build_workload


def test_mixed_workload_reuses_prefix_and_writes_fresh_chunks() -> None:
    """Mixed requests have established reads and never reuse a measured write."""
    spec = WorkloadSpec(
        operation="mixed",
        requests=4,
        chunks_per_request=4,
        working_set_chunks=8,
        prefix_reuse=1.0,
        seed=7,
    )
    warm, requests = build_workload(spec)
    measured = [request for request in requests if not request.warmup]
    assert len(warm) == 8
    assert all(request.read_indices and request.write_indices for request in measured)
    assert all(request.chunks[0] == warm[0] for request in measured)
    writes = [
        request.chunks[index].chunk_id
        for request in measured
        for index in request.write_indices
    ]
    assert len(writes) == len(set(writes))
    assert min(writes) >= len(warm)


def test_mixed_single_chunk_is_rejected() -> None:
    """A one-chunk mixed request cannot represent both a hit and a fresh write."""
    with pytest.raises(ValueError, match="chunks_per_request"):
        WorkloadSpec(operation="mixed", chunks_per_request=1)


def test_metrics_percentiles_and_errors_are_visible() -> None:
    """Metrics include interpolated per-span percentiles and bad-run counters."""
    metrics = Metrics()
    metrics.add(
        "read",
        LatencySample(0.1, 0.2, 0.3, 0.4, 1.0),
        read_bytes=4096,
        hits=1,
    )
    metrics.add(
        "read",
        LatencySample(0.3, 0.4, 0.5, 0.6, 2.0),
        misses=1,
        error=True,
        integrity=True,
    )
    summary = metrics.summary(2.0)
    assert summary["read_bytes"] == 4096
    assert summary["errors"] == 1
    assert summary["integrity_failures"] == 1
    latencies = cast(dict[str, dict[str, float]], summary["latency_seconds"])
    assert latencies["read"]["e2e_seconds_p50"] == 1.5


def test_dynamic_nixl_memory_indices_are_relative_to_registered_arena() -> None:
    """Dynamic NIXL descriptors use zero-based pages within the L1 registration."""
    pytest.importorskip("nixl")
    # First Party
    from lmcache.v1.distributed.l2_adapters.nixl_store_dynamic_l2_adapter import (
        DynamicNixlStorageAgent,
    )

    agent = object.__new__(DynamicNixlStorageAgent)
    agent.l1_memory_base = 0x10000
    agent.l1_memory_size = 0x3000
    agent.l1_align_bytes = 0x1000
    assert agent.get_memory_indices(0x11000, 0x2000) == [1, 2]
    with pytest.raises(ValueError, match="outside"):
        agent.get_memory_indices(0x13000, 0x1000)


def test_warmup_writes_are_bounded_unique_and_cold_selection_reaches_tail() -> None:
    """Warmup does real fresh writes and non-prefix reads select outside the hot set."""
    spec = WorkloadSpec(
        operation="read",
        requests=8,
        warmup_requests=2,
        chunks_per_request=2,
        working_set_chunks=8,
        prefix_reuse=0.0,
        hot_set_fraction=0.25,
        seed=11,
    )
    warm, requests = build_workload(spec)
    warmups = [request for request in requests if request.warmup]
    assert all(len(request.chunks) == 2 for request in warmups)
    warmup_ids = [chunk.chunk_id for request in warmups for chunk in request.chunks]
    assert len(warmup_ids) == len(set(warmup_ids))
    assert min(warmup_ids) >= len(warm)
    measured_starts = [
        request.chunks[0].chunk_id for request in requests if not request.warmup
    ]
    assert min(measured_starts) >= 2


def test_store_completion_dispatcher_retains_out_of_order_results() -> None:
    """Polling one task does not drop a completion belonging to another worker."""

    class Adapter:
        def __init__(self) -> None:
            self.calls = 0

        def pop_completed_store_tasks(self) -> dict[int, int]:
            self.calls += 1
            return {2: 200, 1: 100} if self.calls == 1 else {}

    dispatcher = StoreCompletionDispatcher(Adapter())
    assert dispatcher.wait(1, 0.1) == (100, False)
    assert dispatcher.wait(2, 0.1) == (200, False)
