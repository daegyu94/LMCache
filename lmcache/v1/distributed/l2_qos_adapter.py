# SPDX-License-Identifier: Apache-2.0
"""Weighted L2 request scheduling wrapper for asynchronous L2 adapters.

The wrapper keeps the public asynchronous adapter contract intact. The
controller receives a wrapper task ID immediately, while the concrete adapter
submission is admitted by :class:`L2QoSDispatcher` in a separate thread.
Calls into the concrete adapter are serialized because admission and
completion polling run on different threads.
The policy controls request admission at the LMCache adapter boundary; backend
buffering, network routing, and remote scheduler behavior remain outside its
control.
"""

# Standard
from concurrent.futures import Future
from dataclasses import dataclass
from functools import partial
from typing import Callable
import select
import threading

# First Party
from lmcache.lmcache_native import Bitmap
from lmcache.v1.distributed.api import KeyListPage, MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.internal_api import L2AdapterListener, L2StoreResult
from lmcache.v1.distributed.l2_adapters.base import (
    AdapterUsage,
    L2AdapterInterface,
    L2TaskId,
)
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.multiprocess.qos import (
    CacheSaltQosManager,
    L2QoSDispatcher,
    QosProfile,
    QosTaskHandle,
)
from lmcache.v1.platform import (
    EventNotifier,
    consume_fd,
    create_event_notifier,
)

_TaskResult = L2StoreResult | Bitmap


@dataclass
class _PendingSubtask:
    """One salt-homogeneous concrete adapter submission."""

    positions: list[int]
    handle: QosTaskHandle | None
    raw_task_id: L2TaskId | None = None
    result: _TaskResult | None = None


@dataclass
class _PendingTask:
    """One public wrapper task composed of salt-homogeneous subtasks."""

    key_count: int
    subtasks: list[_PendingSubtask]


@dataclass(frozen=True)
class _TaskPart:
    """Inputs needed to enqueue one salt-homogeneous subtask."""

    profile: QosProfile
    positions: list[int]
    cost_bytes: int
    action: Callable[[], L2TaskId]


class QosL2Adapter(L2AdapterInterface):
    """Apply the shared QoS dispatcher to one asynchronous L2 adapter.

    Args:
        adapter: The concrete adapter to wrap.
        dispatcher: Dispatcher shared by L2 adapters in the same resource
            group. Sharing it makes weights apply across store, lookup, and
            load traffic contending for that resource.
        qos_manager: Registry resolving each key's cache salt to a scheduling
            weight.
        resource_group: Effective local contention-domain name for status
            reporting.
    """

    _KINDS = ("store", "lookup", "load")

    def __init__(
        self,
        adapter: L2AdapterInterface,
        dispatcher: L2QoSDispatcher,
        qos_manager: CacheSaltQosManager,
        resource_group: str | None = None,
    ) -> None:
        super().__init__()
        self._adapter = adapter
        self._dispatcher = dispatcher
        self._qos_manager = qos_manager
        self._resource_group = resource_group
        self._lock = threading.Lock()
        self._adapter_call_lock = threading.Lock()
        self._next_task_id: L2TaskId = 0
        self._tasks: dict[str, dict[L2TaskId, _PendingTask]] = {
            kind: {} for kind in self._KINDS
        }
        self._raw_to_wrapped: dict[str, dict[L2TaskId, tuple[L2TaskId, int]]] = {
            kind: {} for kind in self._KINDS
        }
        self._orphan_results: dict[str, dict[L2TaskId, _TaskResult]] = {
            kind: {} for kind in self._KINDS
        }
        self._completed_results: dict[str, dict[L2TaskId, _TaskResult]] = {
            kind: {} for kind in self._KINDS
        }
        self._closed = False
        self._event_notifiers: dict[str, EventNotifier] = {
            kind: create_event_notifier() for kind in self._KINDS
        }
        self._underlying_event_fds = {
            "store": adapter.get_store_event_fd(),
            "lookup": adapter.get_lookup_and_lock_event_fd(),
            "load": adapter.get_load_event_fd(),
        }
        self._relay_stop = threading.Event()
        self._relay_thread = threading.Thread(
            target=self._relay_events,
            daemon=True,
            name="lmcache-l2-qos-events",
        )
        self._relay_thread.start()

    @property
    def inner_adapter(self) -> L2AdapterInterface:
        """Return the wrapped adapter for management/introspection code."""
        return self._adapter

    # Event FDs belong to the wrapper.  A relay thread consumes concrete
    # adapter notifications and forwards them so the wrapper can also signal
    # synthetic completion when dispatcher admission fails.
    def get_store_event_fd(self) -> int:
        """Return the store completion event FD."""
        return self._event_notifiers["store"].fileno()

    def get_lookup_and_lock_event_fd(self) -> int:
        """Return the lookup completion event FD."""
        return self._event_notifiers["lookup"].fileno()

    def get_load_event_fd(self) -> int:
        """Return the load completion event FD."""
        return self._event_notifiers["load"].fileno()

    def submit_store_task(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
    ) -> L2TaskId:
        """Queue a store request under weighted L2 request scheduling."""
        self._validate_key_object_lengths(keys, objects)
        parts = []
        for cache_salt, positions in self._partition_keys(keys):
            part_keys = [keys[index] for index in positions]
            part_objects = [objects[index] for index in positions]
            parts.append(
                _TaskPart(
                    profile=self._qos_manager.get_profile(cache_salt),
                    positions=positions,
                    cost_bytes=sum(obj.get_size() for obj in part_objects),
                    action=partial(
                        self._submit_store_task,
                        part_keys,
                        part_objects,
                    ),
                )
            )
        return self._queue_task(
            kind="store",
            key_count=len(keys),
            operation="store",
            parts=parts,
        )

    def pop_completed_store_tasks(self) -> dict[L2TaskId, L2StoreResult]:
        """Return completed store tasks using wrapper task IDs."""
        with self._adapter_call_lock:
            raw_completed = self._adapter.pop_completed_store_tasks()
        handles: list[QosTaskHandle] = []
        with self._lock:
            for raw_task_id, result in raw_completed.items():
                mapping = self._raw_to_wrapped["store"].get(raw_task_id)
                if mapping is None:
                    self._orphan_results["store"][raw_task_id] = result
                    continue
                wrapped_task_id, subtask_index = mapping
                pending = self._tasks["store"].get(wrapped_task_id)
                if pending is None:
                    self._raw_to_wrapped["store"].pop(raw_task_id, None)
                    continue
                subtask = pending.subtasks[subtask_index]
                handle = self._finish_subtask_locked(
                    "store",
                    wrapped_task_id,
                    pending,
                    subtask,
                    result,
                )
                if handle is not None:
                    handles.append(handle)

        for handle in handles:
            self._dispatcher.complete(handle.task_id)

        completed: dict[L2TaskId, L2StoreResult] = {}
        for task_id, task_result in self._drain_completed("store").items():
            assert isinstance(task_result, L2StoreResult)
            completed[task_id] = task_result
        return completed

    def submit_lookup_and_lock_task(
        self,
        keys: list[ObjectKey],
        group_layout_descs: dict[int, MemoryLayoutDesc],
    ) -> L2TaskId:
        """Queue a lookup-and-lock request under weighted L2 request scheduling."""
        parts = []
        for cache_salt, positions in self._partition_keys(keys):
            part_keys = [keys[index] for index in positions]
            parts.append(
                _TaskPart(
                    profile=self._qos_manager.get_profile(cache_salt),
                    positions=positions,
                    cost_bytes=max(1, len(part_keys)),
                    action=partial(
                        self._submit_lookup_and_lock_task,
                        part_keys,
                        group_layout_descs,
                    ),
                )
            )
        return self._queue_task(
            kind="lookup",
            key_count=len(keys),
            operation="lookup",
            parts=parts,
        )

    def query_lookup_and_lock_result(self, task_id: L2TaskId) -> Bitmap | None:
        """Query a lookup result and release its QoS accounting on completion."""
        result = self._query_result(
            kind="lookup",
            task_id=task_id,
            query=self._adapter.query_lookup_and_lock_result,
        )
        return result

    def submit_unlock(self, keys: list[ObjectKey]) -> None:
        """Forward unlock control traffic without data-plane admission."""
        with self._adapter_call_lock:
            self._adapter.submit_unlock(keys)

    def submit_load_task(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
    ) -> L2TaskId:
        """Queue a load request under weighted L2 request scheduling."""
        self._validate_key_object_lengths(keys, objects)
        parts = []
        for cache_salt, positions in self._partition_keys(keys):
            part_keys = [keys[index] for index in positions]
            part_objects = [objects[index] for index in positions]
            parts.append(
                _TaskPart(
                    profile=self._qos_manager.get_profile(cache_salt),
                    positions=positions,
                    cost_bytes=sum(obj.get_size() for obj in part_objects),
                    action=partial(
                        self._submit_load_task,
                        part_keys,
                        part_objects,
                    ),
                )
            )
        return self._queue_task(
            kind="load",
            key_count=len(keys),
            operation="load",
            parts=parts,
        )

    def query_load_result(self, task_id: L2TaskId) -> Bitmap | None:
        """Query a load result and release its QoS accounting on completion."""
        result = self._query_result(
            kind="load",
            task_id=task_id,
            query=self._adapter.query_load_result,
        )
        return result

    def register_listener(self, listener: L2AdapterListener) -> None:
        """Register a listener on the concrete adapter."""
        with self._adapter_call_lock:
            self._adapter.register_listener(listener)

    def set_backend_identity(self, name: str, shared: bool = False) -> None:
        """Set backend identity on the concrete adapter."""
        with self._adapter_call_lock:
            self._adapter.set_backend_identity(name, shared=shared)

    @property
    def supports_global_eviction(self) -> bool:
        """Return whether the concrete adapter supports global eviction."""
        with self._adapter_call_lock:
            return self._adapter.supports_global_eviction

    def get_usage(self) -> AdapterUsage:
        """Return usage reported by the concrete adapter."""
        with self._adapter_call_lock:
            return self._adapter.get_usage()

    def delete(self, keys: list[ObjectKey]) -> None:
        """Delete keys through the concrete adapter."""
        with self._adapter_call_lock:
            self._adapter.delete(keys)

    def list_l2_keys(
        self,
        model_name: str | None = None,
        page_size: int = 500,
        cursor: str | None = None,
    ) -> KeyListPage:
        """List keys through the concrete adapter."""
        with self._adapter_call_lock:
            return self._adapter.list_l2_keys(model_name, page_size, cursor)

    def report_status(self) -> dict[str, object]:
        """Return concrete status plus QoS state.

        The returned mapping contains the concrete adapter's status keys, a
        ``qos`` mapping keyed by cache salt, and ``qos_resource_group`` when
        the wrapper belongs to a named resource group.
        """
        with self._adapter_call_lock:
            status = dict(self._adapter.report_status())
        status["qos"] = self._dispatcher.snapshot()
        if self._resource_group is not None:
            status["qos_resource_group"] = self._resource_group
        return status

    def close(self) -> None:
        """Close the concrete adapter."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._relay_stop.set()
        self._relay_thread.join()
        for notifier in self._event_notifiers.values():
            notifier.close()
        with self._adapter_call_lock:
            self._adapter.close()

    def _submit_store_task(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
    ) -> L2TaskId:
        """Submit one concrete store task under the adapter-call lock."""
        with self._adapter_call_lock:
            return self._adapter.submit_store_task(keys, objects)

    def _submit_lookup_and_lock_task(
        self,
        keys: list[ObjectKey],
        group_layout_descs: dict[int, MemoryLayoutDesc],
    ) -> L2TaskId:
        """Submit one concrete lookup task under the adapter-call lock."""
        with self._adapter_call_lock:
            return self._adapter.submit_lookup_and_lock_task(keys, group_layout_descs)

    def _submit_load_task(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
    ) -> L2TaskId:
        """Submit one concrete load task under the adapter-call lock."""
        with self._adapter_call_lock:
            return self._adapter.submit_load_task(keys, objects)

    def _queue_task(
        self,
        kind: str,
        key_count: int,
        operation: str,
        parts: list[_TaskPart],
    ) -> L2TaskId:
        """Create one wrapper task and enqueue its salt-homogeneous parts."""
        with self._lock:
            task_id = self._next_task_id
            self._next_task_id += 1
            subtasks = [
                _PendingSubtask(
                    positions=part.positions,
                    handle=None,
                )
                for part in parts
            ]
            pending = _PendingTask(
                key_count=key_count,
                subtasks=subtasks,
            )
            self._tasks[kind][task_id] = pending

        admission_failed = False
        for subtask_index, (part, subtask) in enumerate(
            zip(
                parts,
                pending.subtasks,
                strict=True,
            )
        ):
            try:
                handle = self._dispatcher.submit(
                    profile=part.profile,
                    operation=operation,
                    cost_bytes=part.cost_bytes,
                    action=part.action,
                )
            except Exception:
                with self._lock:
                    subtask.result = self._failure_result(kind, len(subtask.positions))
                admission_failed = True
                continue

            with self._lock:
                subtask.handle = handle
            handle.future.add_done_callback(
                partial(
                    self._on_submission_done,
                    kind,
                    task_id,
                    subtask_index,
                )
            )

        with self._lock:
            finalized = self._finalize_task_locked(kind, task_id, pending)
        if admission_failed or finalized:
            self._notify_completion(kind)
        return task_id

    def _on_submission_done(
        self,
        kind: str,
        task_id: L2TaskId,
        subtask_index: int,
        future: Future[int],
    ) -> None:
        """Record the concrete adapter task ID or synthesize a failure."""
        handle: QosTaskHandle | None = None
        try:
            raw_task_id = future.result()
        except Exception:
            with self._lock:
                pending = self._tasks[kind].get(task_id)
                if pending is None:
                    return
                subtask = pending.subtasks[subtask_index]
                handle = self._finish_subtask_locked(
                    kind,
                    task_id,
                    pending,
                    subtask,
                    self._failure_result(kind, len(subtask.positions)),
                )
        else:
            with self._lock:
                pending = self._tasks[kind].get(task_id)
                if pending is None:
                    return
                subtask = pending.subtasks[subtask_index]
                subtask.raw_task_id = raw_task_id
                self._raw_to_wrapped[kind][raw_task_id] = (
                    task_id,
                    subtask_index,
                )
                orphan = self._orphan_results[kind].pop(raw_task_id, None)
                if orphan is not None:
                    handle = self._finish_subtask_locked(
                        kind,
                        task_id,
                        pending,
                        subtask,
                        orphan,
                    )

        if handle is not None:
            self._dispatcher.complete(handle.task_id)
        self._notify_completion(kind)

    def _query_result(
        self,
        kind: str,
        task_id: L2TaskId,
        query: Callable[[L2TaskId], Bitmap | None],
    ) -> Bitmap | None:
        """Query admitted subtasks and merge completed bitmaps."""
        with self._lock:
            completed = self._completed_results[kind].pop(task_id, None)
            if completed is not None:
                assert isinstance(completed, Bitmap)
                return completed
            pending = self._tasks[kind].get(task_id)
            if pending is None:
                return None
            raw_subtasks = [
                (index, subtask.raw_task_id)
                for index, subtask in enumerate(pending.subtasks)
                if subtask.result is None and subtask.raw_task_id is not None
            ]

        for subtask_index, raw_task_id in raw_subtasks:
            with self._adapter_call_lock:
                result = query(raw_task_id)
            if result is None:
                continue
            handle: QosTaskHandle | None = None
            with self._lock:
                pending = self._tasks[kind].get(task_id)
                if pending is None:
                    break
                subtask = pending.subtasks[subtask_index]
                if subtask.result is not None or subtask.raw_task_id != raw_task_id:
                    continue
                handle = self._finish_subtask_locked(
                    kind,
                    task_id,
                    pending,
                    subtask,
                    result,
                )
            if handle is not None:
                self._dispatcher.complete(handle.task_id)

        with self._lock:
            completed = self._completed_results[kind].pop(task_id, None)
        if completed is None:
            return None
        assert isinstance(completed, Bitmap)
        return completed

    def _drain_completed(self, kind: str) -> dict[L2TaskId, _TaskResult]:
        """Pop synthetic results created by failed or raced submissions."""
        with self._lock:
            completed = self._completed_results[kind]
            self._completed_results[kind] = {}
            return completed

    def _finish_subtask_locked(
        self,
        kind: str,
        task_id: L2TaskId,
        pending: _PendingTask,
        subtask: _PendingSubtask,
        result: _TaskResult,
    ) -> QosTaskHandle | None:
        """Record one subtask result and finalize its parent when complete."""
        if subtask.result is not None:
            return None
        subtask.result = result
        if subtask.raw_task_id is not None:
            self._raw_to_wrapped[kind].pop(subtask.raw_task_id, None)
        handle = subtask.handle
        subtask.handle = None
        self._finalize_task_locked(kind, task_id, pending)
        return handle

    def _finalize_task_locked(
        self,
        kind: str,
        task_id: L2TaskId,
        pending: _PendingTask,
    ) -> bool:
        """Publish a parent result after every subtask has completed."""
        if any(subtask.result is None for subtask in pending.subtasks):
            return False
        self._tasks[kind].pop(task_id, None)
        self._completed_results[kind][task_id] = self._merge_results(
            kind,
            pending.key_count,
            pending.subtasks,
        )
        return True

    def _notify_completion(self, kind: str) -> None:
        """Wake the controller after admission or concrete completion."""
        try:
            self._event_notifiers[kind].notify()
        except OSError:
            # The controller may already be shutting down.
            pass

    def _relay_events(self) -> None:
        """Forward concrete adapter completion notifications to the wrapper."""
        poller = select.poll()
        fd_to_kind: dict[int, str] = {}
        for kind, event_fd in self._underlying_event_fds.items():
            poller.register(event_fd, select.POLLIN)
            fd_to_kind[event_fd] = kind

        while not self._relay_stop.is_set():
            try:
                ready = poller.poll(100)
            except OSError:
                if self._relay_stop.is_set():
                    return
                raise

            for event_fd, events in ready:
                event_kind = fd_to_kind.get(event_fd)
                if event_kind is None:
                    continue
                if events & (select.POLLIN | select.POLLERR | select.POLLHUP):
                    if events & select.POLLIN:
                        try:
                            consume_fd(event_fd)
                        except OSError:
                            if self._relay_stop.is_set():
                                return
                            continue
                    self._event_notifiers[event_kind].notify()

    @staticmethod
    def _failure_result(kind: str, key_count: int) -> _TaskResult:
        """Create the interface result used when admission fails."""
        if kind == "store":
            return L2StoreResult(False, 0)
        return Bitmap(key_count)

    @staticmethod
    def _merge_results(
        kind: str,
        key_count: int,
        subtasks: list[_PendingSubtask],
    ) -> _TaskResult:
        """Merge salt-homogeneous results into the public task result."""
        if kind == "store":
            store_results: list[L2StoreResult] = []
            for subtask in subtasks:
                result = subtask.result
                assert isinstance(result, L2StoreResult)
                store_results.append(result)
            successful = all(result.is_successful() for result in store_results)
            bytes_transferred = sum(
                result.bytes_transferred() for result in store_results
            )
            return L2StoreResult(successful, bytes_transferred)

        merged = Bitmap(key_count)
        for subtask in subtasks:
            result = subtask.result
            assert isinstance(result, Bitmap)
            for local_index, original_index in enumerate(subtask.positions):
                if result.test(local_index):
                    merged.set(original_index)
        return merged

    @staticmethod
    def _partition_keys(keys: list[ObjectKey]) -> list[tuple[str, list[int]]]:
        """Group key positions by cache salt while preserving input order."""
        groups: dict[str, list[int]] = {}
        for index, key in enumerate(keys):
            groups.setdefault(key.cache_salt, []).append(index)
        if groups:
            return list(groups.items())
        return [("", [])]

    @staticmethod
    def _validate_key_object_lengths(
        keys: list[ObjectKey],
        objects: list[MemoryObj],
    ) -> None:
        """Reject key/object batches that cannot be partitioned in parallel."""
        if len(keys) != len(objects):
            raise ValueError("keys and objects must have the same length")
