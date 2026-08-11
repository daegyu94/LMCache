# SPDX-License-Identifier: Apache-2.0

"""Tests for direct causal L2 trace replay."""

# Standard
from pathlib import Path
import struct
import time

# Third Party
import pytest
import torch

# First Party
from lmcache.cli.commands.trace.l2_driver import L2ReplayDriver, L2TracePlan
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.config import (
    EvictionConfig,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    StorageManagerConfig,
)
from lmcache.v1.distributed.l2_adapters.config import L2AdaptersConfig
from lmcache.v1.distributed.l2_adapters.mock_l2_adapter import MockL2AdapterConfig
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import EventBus, EventBusConfig
from lmcache.v1.mp_observability.trace.recorder import L2TraceRecorder
import lmcache.v1.mp_observability.event_bus as event_bus_module


def _config() -> StorageManagerConfig:
    memory = L1MemoryManagerConfig(
        size_in_bytes=32 * 1024 * 1024,
        use_lazy=False,
        init_size_in_bytes=32 * 1024 * 1024,
        align_bytes=4096,
    )
    return StorageManagerConfig(
        l1_manager_config=L1ManagerConfig(
            memory_config=memory,
            write_ttl_seconds=600,
            read_ttl_seconds=300,
        ),
        eviction_config=EvictionConfig(eviction_policy="noop"),
        l2_adapter_config=L2AdaptersConfig(
            [MockL2AdapterConfig(max_size_gb=0.1, mock_bandwidth_gb=10.0)]
        ),
    )


def _key(value: int) -> ObjectKey:
    return ObjectKey(
        chunk_hash=value.to_bytes(4, "big"),
        model_name="test",
        kv_rank=0,
    )


def _layout(size: int = 4096) -> MemoryLayoutDesc:
    return MemoryLayoutDesc(shapes=[torch.Size([size])], dtypes=[torch.uint8])


def _write_trace(path: Path, events: list[tuple[EventType, dict]]) -> None:
    saved_bus = event_bus_module._global_bus
    bus = EventBus(EventBusConfig(enabled=True))
    event_bus_module._global_bus = bus
    bus.start()
    recorder = L2TraceRecorder(str(path))
    recorder.attach_storage_config(_config())
    bus.register_subscriber(recorder)
    try:
        for event_type, metadata in events:
            bus.publish(
                Event(
                    event_type=event_type,
                    metadata={"trace_t_mono": time.monotonic(), **metadata},
                )
            )
        time.sleep(0.2)
        bus._drain_all()
    finally:
        bus.stop()
        event_bus_module._global_bus = saved_bus


def _store_then_read_events(key: ObjectKey) -> list[tuple[EventType, dict]]:
    common = {"adapter_index": 0, "l2_name": "mock"}
    return [
        (
            EventType.L2_STORE_SUBMITTED,
            {**common, "task_id": 1, "keys": [key], "object_sizes": [4096]},
        ),
        (
            EventType.L2_STORE_COMPLETED,
            {
                **common,
                "task_id": 1,
                "succeeded_count": 1,
                "failed_count": 0,
                "bytes_transferred": 4096,
            },
        ),
        (
            EventType.L2_LOOKUP_TASK_SUBMITTED,
            {
                **common,
                "request_id": 11,
                "task_id": 2,
                "keys": [key],
                "layout_desc": _layout(),
            },
        ),
        (
            EventType.L2_LOOKUP_TASK_COMPLETED,
            {**common, "request_id": 11, "task_id": 2, "hit_indices": [0]},
        ),
        (
            EventType.L2_LOAD_TASK_SUBMITTED,
            {
                **common,
                "request_id": 11,
                "task_id": 3,
                "keys": [key],
                "object_sizes": [4096],
            },
        ),
        (
            EventType.L2_LOAD_TASK_COMPLETED,
            {**common, "request_id": 11, "task_id": 3, "success_indices": [0]},
        ),
        (
            EventType.L2_UNLOCK_SUBMITTED,
            {**common, "request_id": 11, "keys": [key]},
        ),
    ]


def test_plan_derives_store_lookup_load_dependencies(tmp_path):
    trace = tmp_path / "l2.lct"
    _write_trace(trace, _store_then_read_events(_key(1)))

    plan = L2TracePlan.from_file(str(trace))
    store, lookup, load, unlock = plan.operations

    assert lookup.dependencies == {store.task_key}
    assert load.dependencies == {lookup.task_key}
    assert unlock.dependencies == {load.task_key}
    assert plan.prepare_objects == {}


def test_prepare_and_replay_read_before_trace(tmp_path):
    trace = tmp_path / "l2.lct"
    key = _key(2)
    events = _store_then_read_events(key)[2:]
    _write_trace(trace, events)

    with L2ReplayDriver(_config(), str(trace), speedup=10.0) as driver:
        prepared = driver.prepare()
        result = driver.run()

    assert prepared["prepared_objects"] == 1
    assert prepared["prepared_bytes"] == 4096
    assert result["valid"] is True
    assert result["operations_submitted"] == {
        "lookup_task": 1,
        "load_task": 1,
        "unlock": 1,
    }


def test_causal_replay_store_then_read_is_valid(tmp_path):
    trace = tmp_path / "l2.lct"
    _write_trace(trace, _store_then_read_events(_key(3)))

    with L2ReplayDriver(_config(), str(trace), speedup=10.0) as driver:
        result = driver.run()

    assert result["valid"] is True
    assert result["total_bytes_submitted"] == 8192
    assert result["actual_submission_window_seconds"] >= 0
    assert result["drain_seconds"] >= 0
    assert result["total_dependency_wait_seconds"] >= 0
    assert result["total_buffer_wait_seconds"] >= 0


def test_plan_rejects_trace_without_end_marker(tmp_path):
    trace = tmp_path / "l2.lct"
    _write_trace(trace, _store_then_read_events(_key(4)))

    data = trace.read_bytes()
    offset = 0
    frame_offsets = []
    while offset < len(data):
        frame_offsets.append(offset)
        frame_size = struct.unpack(">I", data[offset : offset + 4])[0]
        offset += 4 + frame_size
    trace.write_bytes(data[: frame_offsets[-1]])

    with pytest.raises(ValueError, match="missing end marker"):
        L2TracePlan.from_file(str(trace))
