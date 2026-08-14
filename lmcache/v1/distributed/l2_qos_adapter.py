# SPDX-License-Identifier: Apache-2.0
"""Weighted QoS wrapper for asynchronous L2 adapters.

The wrapper keeps the public asynchronous adapter contract intact.  The
controller receives a wrapper task ID immediately, while the concrete adapter
submission is admitted by :class:`L2QoSDispatcher` in a separate thread.
"""

# Standard
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any, Callable
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
    L2QoSDispatcher,
    QosTaskHandle,
    get_current_qos_profile,
)
from lmcache.v1.platform import (
    EventNotifier,
    consume_fd,
    create_event_notifier,
)


@dataclass
class _PendingTask:
    """One wrapper task waiting for or running an adapter submission."""

    handle: QosTaskHandle | None
    key_count: int
    raw_task_id: L2TaskId | None = None


class QosL2Adapter(L2AdapterInterface):
    """Apply the shared QoS dispatcher to one asynchronous L2 adapter.

    Args:
        adapter: The concrete adapter to wrap.
        dispatcher: Dispatcher shared by every L2 adapter in one storage
            manager.  Sharing it makes weights apply across store, lookup, and
            load traffic instead of independently per backend.
    """

    _KINDS = ("store", "lookup", "load")

    def __init__(
        self,
        adapter: L2AdapterInterface,
        dispatcher: L2QoSDispatcher,
    ) -> None:
        super().__init__()
        self._adapter = adapter
        self._dispatcher = dispatcher
        self._lock = threading.Lock()
        self._next_task_id: L2TaskId = 0
        self._tasks: dict[str, dict[L2TaskId, _PendingTask]] = {
            kind: {} for kind in self._KINDS
        }
        self._raw_to_wrapped: dict[str, dict[L2TaskId, L2TaskId]] = {
            kind: {} for kind in self._KINDS
        }
        self._orphan_results: dict[str, dict[L2TaskId, Any]] = {
            kind: {} for kind in self._KINDS
        }
        self._completed_results: dict[str, dict[L2TaskId, Any]] = {
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
        """Queue a weighted store submission and return a wrapper task ID."""
        cost_bytes = sum(obj.get_size() for obj in objects)
        return self._queue_task(
            kind="store",
            key_count=len(keys),
            cost_bytes=cost_bytes,
            operation="store",
            action=lambda: self._adapter.submit_store_task(keys, objects),
        )

    def pop_completed_store_tasks(self) -> dict[L2TaskId, L2StoreResult]:
        """Return completed store tasks using wrapper task IDs."""
        raw_completed = self._adapter.pop_completed_store_tasks()
        completed = self._drain_completed("store")
        with self._lock:
            for raw_task_id, result in raw_completed.items():
                wrapped_task_id = self._raw_to_wrapped["store"].pop(raw_task_id, None)
                if wrapped_task_id is None:
                    self._orphan_results["store"][raw_task_id] = result
                    continue
                pending = self._tasks["store"].pop(wrapped_task_id, None)
                if pending is not None:
                    self._complete_dispatch_task(pending)
                completed[wrapped_task_id] = result
        return completed

    def submit_lookup_and_lock_task(
        self,
        keys: list[ObjectKey],
        group_layout_descs: dict[int, MemoryLayoutDesc],
    ) -> L2TaskId:
        """Queue a weighted lookup-and-lock submission."""
        return self._queue_task(
            kind="lookup",
            key_count=len(keys),
            cost_bytes=max(1, len(keys)),
            operation="lookup",
            action=lambda: self._adapter.submit_lookup_and_lock_task(
                keys, group_layout_descs
            ),
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
        self._adapter.submit_unlock(keys)

    def submit_load_task(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
    ) -> L2TaskId:
        """Queue a weighted load submission and return a wrapper task ID."""
        cost_bytes = sum(obj.get_size() for obj in objects)
        return self._queue_task(
            kind="load",
            key_count=len(keys),
            cost_bytes=cost_bytes,
            operation="load",
            action=lambda: self._adapter.submit_load_task(keys, objects),
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
        self._adapter.register_listener(listener)

    def set_backend_identity(self, name: str, shared: bool = False) -> None:
        """Set backend identity on the concrete adapter."""
        self._adapter.set_backend_identity(name, shared=shared)

    @property
    def supports_global_eviction(self) -> bool:
        """Return whether the concrete adapter supports global eviction."""
        return self._adapter.supports_global_eviction

    def get_usage(self) -> AdapterUsage:
        """Return usage reported by the concrete adapter."""
        return self._adapter.get_usage()

    def delete(self, keys: list[ObjectKey]) -> None:
        """Delete keys through the concrete adapter."""
        self._adapter.delete(keys)

    def list_l2_keys(
        self,
        model_name: str | None = None,
        page_size: int = 500,
        cursor: str | None = None,
    ) -> KeyListPage:
        """List keys through the concrete adapter."""
        return self._adapter.list_l2_keys(model_name, page_size, cursor)

    def report_status(self) -> dict:
        """Return concrete adapter status with QoS queue state attached."""
        status = dict(self._adapter.report_status())
        status["qos"] = self._dispatcher.snapshot()
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
        self._adapter.close()

    def _queue_task(
        self,
        kind: str,
        key_count: int,
        cost_bytes: int,
        operation: str,
        action: Callable[[], L2TaskId],
    ) -> L2TaskId:
        """Create a wrapper task and submit its action to the dispatcher."""
        with self._lock:
            task_id = self._next_task_id
            self._next_task_id += 1
            pending = _PendingTask(handle=None, key_count=key_count)
            self._tasks[kind][task_id] = pending

        try:
            handle = self._dispatcher.submit(
                profile=get_current_qos_profile(),
                operation=operation,
                cost_bytes=cost_bytes,
                action=action,
            )
        except BaseException:
            with self._lock:
                self._tasks[kind].pop(task_id, None)
            raise

        with self._lock:
            pending.handle = handle
        handle.future.add_done_callback(
            lambda future: self._on_submission_done(kind, task_id, pending, future)
        )
        return task_id

    def _on_submission_done(
        self,
        kind: str,
        task_id: L2TaskId,
        pending: _PendingTask,
        future: Future[int],
    ) -> None:
        """Record the concrete adapter task ID or synthesize a failure."""
        try:
            raw_task_id = future.result()
        except BaseException:
            with self._lock:
                self._tasks[kind].pop(task_id, None)
                self._completed_results[kind][task_id] = self._failure_result(
                    kind, pending.key_count
                )
                self._complete_dispatch_task(pending)
        else:
            with self._lock:
                pending.raw_task_id = raw_task_id
                self._raw_to_wrapped[kind][raw_task_id] = task_id
                orphan = self._orphan_results[kind].pop(raw_task_id, None)
                if orphan is not None:
                    self._raw_to_wrapped[kind].pop(raw_task_id, None)
                    self._tasks[kind].pop(task_id, None)
                    self._completed_results[kind][task_id] = orphan
                    self._complete_dispatch_task(pending)

        try:
            self._event_notifiers[kind].notify()
        except OSError:
            # The controller may already be shutting down.
            pass

    def _query_result(
        self,
        kind: str,
        task_id: L2TaskId,
        query: Callable[[L2TaskId], Any],
    ) -> Any | None:
        """Query one concrete task after dispatcher admission."""
        with self._lock:
            completed = self._completed_results[kind].pop(task_id, None)
            if completed is not None:
                return completed
            pending = self._tasks[kind].get(task_id)
            if pending is None or pending.raw_task_id is None:
                return None
            raw_task_id = pending.raw_task_id

        result = query(raw_task_id)
        if result is None:
            return None

        with self._lock:
            self._tasks[kind].pop(task_id, None)
            self._raw_to_wrapped[kind].pop(raw_task_id, None)
            self._complete_dispatch_task(pending)
        return result

    def _drain_completed(self, kind: str) -> dict[L2TaskId, Any]:
        """Pop synthetic results created by failed or raced submissions."""
        with self._lock:
            completed = self._completed_results[kind]
            self._completed_results[kind] = {}
            return completed

    def _complete_dispatch_task(self, pending: _PendingTask) -> None:
        """Release dispatcher accounting for a finished wrapper task."""
        if pending.handle is not None:
            self._dispatcher.complete(pending.handle.task_id)

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
    def _failure_result(kind: str, key_count: int) -> Any:
        """Create the interface result used when admission fails."""
        if kind == "store":
            return L2StoreResult(False, 0)
        return Bitmap(key_count)
