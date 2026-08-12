# SPDX-License-Identifier: Apache-2.0

"""Direct, causal replay for L2 adapter-level traces."""

# Future
from __future__ import annotations

# Standard
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import json
import math
import time

# Third Party
import torch

# First Party
from lmcache.cli.commands.trace.l2_stats import L2LatencyStatsSubscriber
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.config import StorageManagerConfig
from lmcache.v1.distributed.l2_adapters.base import L2AdapterInterface
from lmcache.v1.distributed.storage_manager import StorageManager
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.mp_observability.trace import codecs
from lmcache.v1.mp_observability.trace.reader import TraceReader

logger = init_logger(__name__)

TaskKey = tuple[str, int, int]

_STORE_SUBMIT = "l2.store.submitted"
_STORE_COMPLETE = "l2.store.completed"
_LOOKUP_SUBMIT = "l2.lookup_task.submitted"
_LOOKUP_COMPLETE = "l2.lookup_task.completed"
_LOAD_SUBMIT = "l2.load_task.submitted"
_LOAD_COMPLETE = "l2.load_task.completed"
_UNLOCK = "l2.unlock.submitted"
_DELETE = "l2.delete.submitted"
_TRACE_END = "l2.trace.end"
_OUTCOME_MISMATCH_SAMPLE_LIMIT = 10
_PROGRESS_LOG_INTERVAL_SECONDS = 5.0


def _task_key(operation: str, args: dict[str, Any]) -> TaskKey:
    return (operation, int(args["adapter_index"]), int(args["task_id"]))


def _layout_size(layout: MemoryLayoutDesc) -> int:
    size = 0
    for shape, dtype in zip(layout.shapes, layout.dtypes, strict=True):
        size += math.prod(shape) * dtype.itemsize
    return size


def _lookup_group_layout_descs(args: dict[str, Any]) -> dict[int, MemoryLayoutDesc]:
    """Return v0.5.3 group layouts, accepting legacy single-layout traces."""
    layouts = args.get("group_layout_descs")
    if layouts:
        return {int(group_id): layout for group_id, layout in layouts.items()}
    layout = args.get("layout_desc")
    if layout is None:
        raise ValueError("L2 lookup submission is missing layout metadata")
    return {0: layout}


@dataclass
class ReplayOperation:
    """One source-side L2 submission to reproduce."""

    sequence: int
    t_mono: float
    operation: str
    args: dict[str, Any]
    task_key: TaskKey | None = None
    dependencies: set[TaskKey] = field(default_factory=set)


@dataclass
class L2TracePlan:
    """Parsed submissions, outcomes, dependencies, and safe prepare objects."""

    operations: list[ReplayOperation]
    completions: dict[TaskKey, dict[str, Any]]
    prepare_objects: dict[ObjectKey, int]
    trace_percent: float
    source_operations_total: int

    @classmethod
    def from_file(
        cls,
        trace_path: str,
        *,
        trace_percent: float = 100.0,
    ) -> "L2TracePlan":
        """Parse an L2 trace and derive causal dependencies.

        Args:
            trace_path: L2 ``.lct`` file.
            trace_percent: Percentage of source submissions to select from
                the beginning of the trace.

        Returns:
            Replay plan ordered by source submission sequence.

        Raises:
            ValueError: The trace or percentage is invalid, or the trace uses
                multiple adapters.
        """
        if (
            not math.isfinite(trace_percent)
            or trace_percent <= 0
            or trace_percent > 100
        ):
            raise ValueError("trace_percent must be finite and in (0, 100]")
        decoded: list[tuple[int, float, str, dict[str, Any]]] = []
        footer: dict[str, Any] | None = None
        with TraceReader(trace_path) as reader:
            if reader.header.level != "l2":
                raise ValueError(
                    f"L2 replay requires header.level='l2', got {reader.header.level!r}"
                )
            for sequence, record in enumerate(reader.records()):
                args = codecs.decode_args(record.args)
                if record.qualname == _TRACE_END:
                    if footer is not None:
                        raise ValueError("invalid L2 trace: duplicate end marker")
                    footer = args
                    continue
                decoded.append(
                    (
                        sequence,
                        record.t_mono,
                        record.qualname,
                        args,
                    )
                )

        if footer is None:
            raise ValueError("incomplete L2 trace: missing end marker")
        recorder_dropped = int(footer.get("recorder_dropped_count", 0))
        event_bus_dropped = int(footer.get("event_bus_dropped_count", 0))
        if recorder_dropped or event_bus_dropped:
            raise ValueError(
                "incomplete L2 trace: "
                f"recorder_dropped={recorder_dropped}, "
                f"event_bus_dropped={event_bus_dropped}"
            )

        adapter_indices = {
            int(args["adapter_index"])
            for _, _, name, args in decoded
            if name in {_STORE_SUBMIT, _LOOKUP_SUBMIT, _LOAD_SUBMIT}
        }
        if len(adapter_indices) > 1:
            raise ValueError("L2 replay currently supports one source adapter")

        operations: list[ReplayOperation] = []
        by_task: dict[TaskKey, ReplayOperation] = {}
        completions: dict[TaskKey, dict[str, Any]] = {}
        completion_sequence: dict[TaskKey, int] = {}
        lookup_tasks_by_request: dict[int, list[TaskKey]] = defaultdict(list)
        load_tasks_by_request: dict[int, list[TaskKey]] = defaultdict(list)
        size_by_key: dict[ObjectKey, int] = {}

        submit_names = {_STORE_SUBMIT, _LOOKUP_SUBMIT, _LOAD_SUBMIT, _UNLOCK, _DELETE}
        source_operations_total = sum(
            1 for _, _, name, _ in decoded if name in submit_names
        )
        selected_operations = math.ceil(source_operations_total * trace_percent / 100.0)
        for sequence, t_mono, name, args in decoded:
            if name in submit_names:
                if len(operations) >= selected_operations:
                    continue
                operation = name.split(".")[1]
                key = None
                if name in {_STORE_SUBMIT, _LOOKUP_SUBMIT, _LOAD_SUBMIT}:
                    key = _task_key(operation, args)
                op = ReplayOperation(sequence, t_mono, operation, args, key)
                operations.append(op)
                if key is not None:
                    by_task[key] = op
                    request_id = args.get("request_id")
                    if request_id is not None and operation == "lookup_task":
                        lookup_tasks_by_request[int(request_id)].append(key)
                    elif request_id is not None and operation == "load_task":
                        load_tasks_by_request[int(request_id)].append(key)
                object_sizes = args.get("object_sizes")
                if object_sizes is not None:
                    for object_key, size in zip(
                        args.get("keys", []), object_sizes, strict=True
                    ):
                        size_by_key[object_key] = int(size)
            elif name in {_STORE_COMPLETE, _LOOKUP_COMPLETE, _LOAD_COMPLETE}:
                operation = name.split(".")[1]
                key = _task_key(operation, args)
                completions[key] = args
                completion_sequence[key] = sequence

        # A later hit must not prepare an object whose first source lookup was
        # a miss. Track the first selected lookup submission even when its
        # completion is absent, so an unknown first outcome stays unprepared.
        first_lookup_by_object: dict[ObjectKey, tuple[int, TaskKey, int]] = {}
        for replay_operation in operations:
            if (
                replay_operation.operation != "lookup_task"
                or replay_operation.task_key is None
            ):
                continue
            for index, object_key in enumerate(replay_operation.args["keys"]):
                first_lookup_by_object.setdefault(
                    object_key,
                    (replay_operation.sequence, replay_operation.task_key, index),
                )

        successful_store_by_key: dict[ObjectKey, list[tuple[int, TaskKey]]] = (
            defaultdict(list)
        )
        for key, completion in completions.items():
            if key[0] != "store" or int(completion.get("succeeded_count", 0)) <= 0:
                continue
            submit = by_task.get(key)
            if submit is None:
                continue
            for object_key in submit.args["keys"]:
                successful_store_by_key[object_key].append(
                    (completion_sequence[key], key)
                )

        for key, completion in completions.items():
            if key[0] != "lookup_task":
                continue
            lookup = by_task.get(key)
            if lookup is None:
                continue
            complete_seq = completion_sequence[key]
            hit_indices = {int(index) for index in completion.get("hit_indices", [])}
            for index, object_key in enumerate(lookup.args["keys"]):
                if index not in hit_indices:
                    continue
                completed_store_candidates = [
                    item
                    for item in successful_store_by_key.get(object_key, [])
                    if item[0] < complete_seq
                ]
                # A store that completes while lookup is in flight may explain
                # the hit, but it is not a source happens-before dependency.
                dependency_candidates = [
                    item
                    for item in completed_store_candidates
                    if item[0] < lookup.sequence
                ]
                if dependency_candidates:
                    lookup.dependencies.add(max(dependency_candidates)[1])

        prepare_objects: dict[ObjectKey, int] = {}
        for object_key, (_, lookup_task_key, index) in first_lookup_by_object.items():
            first_lookup_completion = completions.get(lookup_task_key)
            if first_lookup_completion is None:
                continue
            hit_indices = {
                int(index) for index in first_lookup_completion.get("hit_indices", [])
            }
            if index not in hit_indices:
                continue
            lookup = by_task[lookup_task_key]
            complete_seq = completion_sequence[lookup_task_key]
            completed_store_candidates = [
                item
                for item in successful_store_by_key.get(object_key, [])
                if item[0] < complete_seq
            ]
            if completed_store_candidates:
                continue
            size = size_by_key.get(object_key)
            if size is None:
                layouts = _lookup_group_layout_descs(lookup.args)
                layout = layouts[object_key.object_group_id]
                size = _layout_size(layout)
            prepare_objects[object_key] = size

        for op in operations:
            request_id = op.args.get("request_id")
            if request_id is None:
                continue
            request_id = int(request_id)
            if op.operation == "load_task":
                op.dependencies.update(lookup_tasks_by_request[request_id])
            elif op.operation == "unlock":
                dependencies = load_tasks_by_request[request_id]
                if not dependencies:
                    dependencies = lookup_tasks_by_request[request_id]
                op.dependencies.update(dependencies)

        return cls(
            operations=operations,
            completions={
                key: completion
                for key, completion in completions.items()
                if key in by_task
            },
            prepare_objects=prepare_objects,
            trace_percent=trace_percent,
            source_operations_total=source_operations_total,
        )


class ReplayBufferPool:
    """Reusable, L1-backed buffers registered with the target adapter."""

    def __init__(self, storage_manager: StorageManager) -> None:
        self._storage_manager = storage_manager
        self._free: dict[int, list[MemoryObj]] = defaultdict(list)
        self._counter = 0

    def acquire(self, sizes: list[int]) -> list[MemoryObj] | None:
        """Acquire one replay buffer per byte size, or return ``None`` on OOM."""
        acquired: list[MemoryObj] = []
        for size in sizes:
            if self._free[size]:
                acquired.append(self._free[size].pop())
                continue
            key = ObjectKey(
                chunk_hash=self._counter.to_bytes(32, "big"),
                model_name="__lmcache_l2_replay__",
                kv_rank=0,
            )
            self._counter += 1
            layout = MemoryLayoutDesc(
                shapes=[torch.Size([size])],
                dtypes=[torch.uint8],
            )
            reserved = self._storage_manager.reserve_write([key], layout, mode="new")
            obj = reserved.get(key)
            if obj is None:
                self.release(acquired)
                return None
            acquired.append(obj)
        return acquired

    def release(self, objects: list[MemoryObj]) -> None:
        """Return completed-task buffers to the pool."""
        for obj in objects:
            self._free[obj.get_size()].append(obj)


class L2ReplayDriver:
    """Replay L2 task submissions directly against one target adapter."""

    def __init__(
        self,
        sm_config: StorageManagerConfig,
        trace_path: str,
        *,
        speedup: float = 1.0,
        trace_percent: float = 100.0,
        drain_timeout: float = 60.0,
    ) -> None:
        if not math.isfinite(speedup) or speedup <= 0:
            raise ValueError("speedup must be a finite positive number")
        self._plan = L2TracePlan.from_file(
            trace_path,
            trace_percent=trace_percent,
        )
        self._storage_manager = StorageManager(sm_config, start_controllers=False)
        adapters = self._storage_manager.l2_adapters_snapshot()
        if len(adapters) != 1:
            self._storage_manager.close()
            raise ValueError("L2 replay requires exactly one target adapter")
        self._adapter: L2AdapterInterface = adapters[0][2]
        adapter_status = self._adapter.report_status()
        adapter_type = adapter_status.get("type")
        self._adapter_name = str(adapter_type or type(self._adapter).__name__)
        self._latency_stats = L2LatencyStatsSubscriber()
        self._buffers = ReplayBufferPool(self._storage_manager)
        self._speedup = speedup
        self._drain_timeout = drain_timeout
        self._closed = False

    def __enter__(self) -> "L2ReplayDriver":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def close(self) -> None:
        """Release adapter and L1 replay buffers."""
        if self._closed:
            return
        self._closed = True
        self._storage_manager.close()

    def prepare(self) -> dict[str, Any]:
        """Store safely inferred source-resident objects before replay.

        Only objects whose first selected source lookup was a hit and whose
        hit cannot be explained by a preceding successful store completion
        are prepared.
        """
        started = time.monotonic()
        items = list(self._plan.prepare_objects.items())
        tasks: dict[int, tuple[list[MemoryObj], int]] = {}
        submitted_bytes = 0
        for offset in range(0, len(items), 64):
            batch = items[offset : offset + 64]
            keys = [key for key, _ in batch]
            sizes = [size for _, size in batch]
            objects = self._buffers.acquire(sizes)
            if objects is None:
                raise RuntimeError("L2 prepare exhausted the replay buffer pool")
            task_id = self._adapter.submit_store_task(keys, objects)
            tasks[task_id] = (objects, sum(sizes))
            submitted_bytes += sum(sizes)

        succeeded = 0
        deadline = time.monotonic() + self._drain_timeout
        while tasks and time.monotonic() < deadline:
            for (
                task_id,
                store_result,
            ) in self._adapter.pop_completed_store_tasks().items():
                task = tasks.pop(task_id, None)
                if task is None:
                    continue
                objects, _ = task
                self._buffers.release(objects)
                if store_result.is_successful():
                    succeeded += 1
            if tasks:
                time.sleep(0.001)
        if tasks:
            raise RuntimeError(f"L2 prepare timed out with {len(tasks)} task(s)")
        return {
            "prepared_objects": len(items),
            "prepared_bytes": submitted_bytes,
            "store_tasks_succeeded": succeeded,
            "elapsed_seconds": time.monotonic() - started,
        }

    def run(self) -> dict[str, Any]:
        """Run causal timestamp-scaled L2 replay and return JSON-safe stats."""
        pending = list(self._plan.operations)
        schedule_origin = (
            self._plan.operations[0].t_mono if self._plan.operations else 0.0
        )
        completed: set[TaskKey] = set()
        completion_times: dict[TaskKey, float] = {}
        stores: dict[int, tuple[ReplayOperation, list[MemoryObj], float]] = {}
        lookups: dict[int, tuple[ReplayOperation, list[MemoryObj], float]] = {}
        loads: dict[int, tuple[ReplayOperation, list[MemoryObj], float]] = {}
        dispatch_times: list[float] = []
        schedule_lags: list[float] = []
        dependency_waits: list[float] = []
        buffer_waits: list[float] = []
        outcome_comparisons: dict[str, int] = defaultdict(int)
        outcome_mismatch_counts: dict[str, int] = defaultdict(int)
        outcome_mismatch_count = 0
        outcome_mismatch_samples: list[str] = []
        counts: dict[str, int] = defaultdict(int)
        bytes_submitted: dict[str, int] = defaultdict(int)

        def record_outcome_comparison(
            operation: str, task_key: TaskKey, matches: bool
        ) -> None:
            nonlocal outcome_mismatch_count
            outcome_comparisons[operation] += 1
            if matches:
                return
            outcome_mismatch_count += 1
            outcome_mismatch_counts[operation] += 1
            if len(outcome_mismatch_samples) < _OUTCOME_MISMATCH_SAMPLE_LIMIT:
                outcome_mismatch_samples.append(f"{operation}:{task_key}")

        started = time.monotonic()
        last_progress = started
        last_progress_log = started

        def log_progress(now: float) -> None:
            nonlocal last_progress_log
            if now - last_progress_log < _PROGRESS_LOG_INTERVAL_SECONDS:
                return
            dispatched = len(self._plan.operations) - len(pending)
            in_flight = len(stores) + len(lookups) + len(loads)
            logger.info(
                "L2 replay progress: elapsed=%.1fs dispatched=%d/%d "
                "completed=%d pending=%d in_flight(store=%d lookup=%d load=%d) "
                "bytes_submitted=%d",
                now - started,
                dispatched,
                len(self._plan.operations),
                dispatched - in_flight,
                len(pending),
                len(stores),
                len(lookups),
                len(loads),
                sum(bytes_submitted.values()),
            )
            last_progress_log = now

        while pending or stores or lookups or loads:
            now = time.monotonic()
            progress = False
            for op in list(pending):
                target = started + (op.t_mono - schedule_origin) / self._speedup
                if now < target or not op.dependencies.issubset(completed):
                    continue
                objects: list[MemoryObj] = []
                if op.operation in {"store", "load_task"}:
                    sizes = [int(size) for size in op.args["object_sizes"]]
                    acquired = self._buffers.acquire(sizes)
                    if acquired is None:
                        continue
                    objects = acquired
                    bytes_submitted[op.operation] += sum(sizes)
                dispatched = time.monotonic()
                if op.operation == "store":
                    task_id = self._adapter.submit_store_task(op.args["keys"], objects)
                    self._latency_stats.record_submission("write", self._adapter_name)
                    stores[task_id] = (op, objects, dispatched)
                elif op.operation == "lookup_task":
                    task_id = self._adapter.submit_lookup_and_lock_task(
                        op.args["keys"], _lookup_group_layout_descs(op.args)
                    )
                    lookups[task_id] = (op, objects, dispatched)
                elif op.operation == "load_task":
                    task_id = self._adapter.submit_load_task(op.args["keys"], objects)
                    self._latency_stats.record_submission("read", self._adapter_name)
                    loads[task_id] = (op, objects, dispatched)
                elif op.operation == "unlock":
                    self._adapter.submit_unlock(op.args["keys"])
                elif op.operation == "delete":
                    self._adapter.delete(op.args["keys"])
                else:
                    raise ValueError(f"unsupported L2 operation {op.operation!r}")
                pending.remove(op)
                counts[op.operation] += 1
                dispatch_times.append(dispatched)
                schedule_lags.append(max(0.0, dispatched - target))
                dependency_ready = max(
                    [target]
                    + [completion_times[dependency] for dependency in op.dependencies]
                )
                dependency_waits.append(max(0.0, dependency_ready - target))
                buffer_waits.append(max(0.0, dispatched - dependency_ready))
                progress = True

            for (
                task_id,
                store_result,
            ) in self._adapter.pop_completed_store_tasks().items():
                state = stores.pop(task_id, None)
                if state is None:
                    continue
                op, objects, dispatched = state
                completion_time = time.monotonic()
                self._latency_stats.record_completion(
                    "write",
                    self._adapter_name,
                    round((completion_time - dispatched) * 1_000_000),
                    sum(int(size) for size in op.args["object_sizes"]),
                )
                self._buffers.release(objects)
                assert op.task_key is not None
                expected = self._plan.completions.get(op.task_key, {})
                expected_ok = int(expected.get("succeeded_count", 0)) > 0
                record_outcome_comparison(
                    "store", op.task_key, store_result.is_successful() == expected_ok
                )
                if op.task_key is not None:
                    completed.add(op.task_key)
                    completion_times[op.task_key] = completion_time
                progress = True

            for target_id in list(lookups):
                lookup_result = self._adapter.query_lookup_and_lock_result(target_id)
                if lookup_result is None:
                    continue
                op, _, _ = lookups.pop(target_id)
                assert op.task_key is not None
                expected = self._plan.completions.get(op.task_key, {})
                record_outcome_comparison(
                    "lookup_task",
                    op.task_key,
                    lookup_result.get_indices_list() == expected.get("hit_indices", []),
                )
                if op.task_key is not None:
                    completed.add(op.task_key)
                    completion_times[op.task_key] = time.monotonic()
                progress = True

            for target_id in list(loads):
                load_result = self._adapter.query_load_result(target_id)
                if load_result is None:
                    continue
                op, objects, dispatched = loads.pop(target_id)
                completion_time = time.monotonic()
                self._latency_stats.record_completion(
                    "read",
                    self._adapter_name,
                    round((completion_time - dispatched) * 1_000_000),
                    sum(int(size) for size in op.args["object_sizes"]),
                )
                self._buffers.release(objects)
                assert op.task_key is not None
                expected = self._plan.completions.get(op.task_key, {})
                record_outcome_comparison(
                    "load_task",
                    op.task_key,
                    load_result.get_indices_list()
                    == expected.get("success_indices", []),
                )
                if op.task_key is not None:
                    completed.add(op.task_key)
                    completion_times[op.task_key] = completion_time
                progress = True

            log_progress(time.monotonic())
            if progress:
                last_progress = time.monotonic()
                continue

            if time.monotonic() - last_progress > self._drain_timeout:
                raise RuntimeError(
                    "L2 replay made no progress before drain timeout: "
                    f"pending={len(pending)} "
                    f"in_flight={len(stores) + len(lookups) + len(loads)}"
                )
            time.sleep(0.001)

        finished = time.monotonic()
        submission_window = (
            max(dispatch_times) - min(dispatch_times)
            if len(dispatch_times) > 1
            else 0.0
        )
        source_window = (
            (self._plan.operations[-1].t_mono - schedule_origin) / self._speedup
            if self._plan.operations
            else 0.0
        )
        total_bytes = sum(bytes_submitted.values())
        elapsed = finished - started
        drain_seconds = (
            max(0.0, finished - max(dispatch_times)) if dispatch_times else 0.0
        )
        return {
            **self._latency_stats.snapshot(),
            "speedup": self._speedup,
            "trace_percent": self._plan.trace_percent,
            "source_operations_total": self._plan.source_operations_total,
            "operations_selected": len(self._plan.operations),
            "source_submission_window_seconds": source_window,
            "actual_submission_window_seconds": submission_window,
            "total_replay_seconds": elapsed,
            "drain_seconds": drain_seconds,
            "max_schedule_lag_seconds": max(schedule_lags, default=0.0),
            "mean_schedule_lag_seconds": (
                sum(schedule_lags) / len(schedule_lags) if schedule_lags else 0.0
            ),
            "max_dependency_wait_seconds": max(dependency_waits, default=0.0),
            "total_dependency_wait_seconds": sum(dependency_waits),
            "max_buffer_wait_seconds": max(buffer_waits, default=0.0),
            "total_buffer_wait_seconds": sum(buffer_waits),
            "operations_submitted": dict(counts),
            "bytes_submitted": dict(bytes_submitted),
            "total_bytes_submitted": total_bytes,
            "throughput_bytes_per_second": total_bytes / elapsed if elapsed else 0.0,
            "outcome_matches_source": outcome_mismatch_count == 0,
            "outcome_comparisons": dict(outcome_comparisons),
            "outcome_mismatch_count": outcome_mismatch_count,
            "outcome_mismatch_counts": dict(outcome_mismatch_counts),
            "outcome_mismatch_rate": (
                outcome_mismatch_count / sum(outcome_comparisons.values())
                if outcome_comparisons
                else 0.0
            ),
            "outcome_mismatch_samples": outcome_mismatch_samples,
            "operations_without_outcome_comparison": {
                "unlock": counts.get("unlock", 0),
                "delete": counts.get("delete", 0),
            },
        }


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Write an L2 prepare or replay result as formatted JSON."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
