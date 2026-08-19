# SPDX-License-Identifier: Apache-2.0
"""Cache-salt-based weighted L2 request scheduling."""

# Standard
from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Callable
import threading

# First Party
from lmcache.logging import init_logger

MIN_SCHED_WEIGHT = 1
MAX_SCHED_WEIGHT = 10000
DEFAULT_SCHED_WEIGHT = 100

logger = init_logger(__name__)


@dataclass(frozen=True)
class QosProfile:
    """Scheduling metadata for one cache-salt domain.

    Args:
        domain_id: Exact cache_salt value. The empty string represents
            requests that do not specify a salt.
        weight: Relative scheduling weight in the range [1, 10000].
        source: Either explicit for a registered weight or default.
    """

    domain_id: str = ""
    weight: int = DEFAULT_SCHED_WEIGHT
    source: str = "default"

    def __post_init__(self) -> None:
        if not MIN_SCHED_WEIGHT <= self.weight <= MAX_SCHED_WEIGHT:
            raise ValueError(
                f"weight must be in [{MIN_SCHED_WEIGHT}, {MAX_SCHED_WEIGHT}], "
                f"got {self.weight}"
            )


@dataclass(frozen=True)
class QosWeightUpdate:
    """One explicit-salt or default scheduling-weight update.

    ``cache_salt=None`` denotes a default-weight change. Dispatchers apply it
    only to their currently active domains whose profile source is ``default``.
    """

    cache_salt: str | None
    weight: int
    source: str


class CacheSaltQosManager:
    """Manage dynamic scheduling weights keyed by cache_salt.

    Unknown salts immediately use the configured default weight. Registering
    a salt later changes subsequent and queued dispatch decisions through
    listeners, so provisioning may happen before or after the first request.

    Args:
        default_sched_weight: Weight used by salts without an explicit entry.
    """

    def __init__(self, default_sched_weight: int = DEFAULT_SCHED_WEIGHT) -> None:
        self._validate_weight(default_sched_weight)
        self._default_sched_weight = default_sched_weight
        self._weights: dict[str, int] = {}
        self._listeners: list[Callable[[QosWeightUpdate], None]] = []
        self._lock = threading.Lock()

    @property
    def default_sched_weight(self) -> int:
        """Return the weight used for unregistered salts."""
        with self._lock:
            return self._default_sched_weight

    def set_default_sched_weight(self, weight: int) -> None:
        """Set the fallback weight and notify active default-weight domains.

        Args:
            weight: Relative scheduling weight in the range [1, 10000].

        Raises:
            ValueError: If weight is outside the supported range.
        """
        self._validate_weight(weight)
        with self._lock:
            self._default_sched_weight = weight
            listeners = list(self._listeners)
        self._notify(
            listeners,
            [QosWeightUpdate(cache_salt=None, weight=weight, source="default")],
        )

    def set_sched_weight(self, cache_salt: str, weight: int) -> None:
        """Create or update the explicit weight for one cache salt.

        Args:
            cache_salt: Tenant namespace carried by each ObjectKey.
            weight: Relative scheduling weight in the range [1, 10000].

        Raises:
            ValueError: If weight is outside the supported range.
        """
        self._validate_weight(weight)
        with self._lock:
            self._weights[cache_salt] = weight
            listeners = list(self._listeners)
        self._notify(
            listeners,
            [
                QosWeightUpdate(
                    cache_salt=cache_salt,
                    weight=weight,
                    source="explicit",
                )
            ],
        )

    def delete_sched_weight(self, cache_salt: str) -> bool:
        """Delete an explicit weight and restore the default for that salt.

        Args:
            cache_salt: Tenant namespace to remove.

        Returns:
            True when an explicit entry existed.
        """
        with self._lock:
            existed = self._weights.pop(cache_salt, None) is not None
            update = QosWeightUpdate(
                cache_salt=cache_salt,
                weight=self._default_sched_weight,
                source="default",
            )
            listeners = list(self._listeners)
        if existed:
            self._notify(listeners, [update])
        return existed

    def has_sched_weight(self, cache_salt: str) -> bool:
        """Return whether a salt has an explicit weight."""
        with self._lock:
            return cache_salt in self._weights

    def get_profile(self, cache_salt: str) -> QosProfile:
        """Return the effective scheduling profile for a cache salt.

        Args:
            cache_salt: Tenant namespace carried by an ObjectKey.

        Returns:
            An explicit or default-derived profile.
        """
        with self._lock:
            weight = self._weights.get(cache_salt, self._default_sched_weight)
            source = "explicit" if cache_salt in self._weights else "default"
        return QosProfile(domain_id=cache_salt, weight=weight, source=source)

    def list_sched_weights(self) -> dict[str, int]:
        """Return a stable copy of all explicit salt-to-weight entries."""
        with self._lock:
            return dict(sorted(self._weights.items()))

    def register_listener(
        self,
        listener: Callable[[QosWeightUpdate], None],
    ) -> None:
        """Register a callback for effective profile changes.

        Args:
            listener: Callback invoked after a weight update. It must not call
                back into this manager while holding unrelated locks.
        """
        with self._lock:
            self._listeners.append(listener)
            updates = [
                QosWeightUpdate(cache_salt=salt, weight=weight, source="explicit")
                for salt, weight in self._weights.items()
            ]
        self._notify([listener], updates)

    def unregister_listener(
        self,
        listener: Callable[[QosWeightUpdate], None],
    ) -> bool:
        """Remove a previously registered profile-change callback.

        Args:
            listener: Callback previously passed to ``register_listener``.

        Returns:
            ``True`` when the listener was registered and removed.
        """
        with self._lock:
            try:
                self._listeners.remove(listener)
            except ValueError:
                return False
        return True

    @staticmethod
    def _validate_weight(weight: int) -> None:
        """Validate one scheduling weight."""
        if isinstance(weight, bool) or not isinstance(weight, int):
            raise ValueError("sched_weight must be an integer")
        if not MIN_SCHED_WEIGHT <= weight <= MAX_SCHED_WEIGHT:
            raise ValueError(
                f"sched_weight must be in [{MIN_SCHED_WEIGHT}, {MAX_SCHED_WEIGHT}]"
            )

    @staticmethod
    def _notify(
        listeners: list[Callable[[QosWeightUpdate], None]],
        updates: list[QosWeightUpdate],
    ) -> None:
        """Notify listeners without holding the registry lock."""
        for update in updates:
            for listener in listeners:
                try:
                    listener(update)
                except Exception:
                    logger.exception("L2 QoS weight-update listener failed")


@dataclass
class _QueuedQosTask:
    """Internal task state owned by :class:`L2QoSDispatcher`."""

    task_id: int
    profile: QosProfile
    operation: str
    cost_bytes: int
    action: Callable[[], int]
    future: Future[int]


@dataclass(frozen=True)
class QosTaskHandle:
    """Handle returned when a task is admitted to the L2 dispatcher."""

    task_id: int
    domain_id: str
    cost_bytes: int
    future: Future[int]


@dataclass
class _DomainQueue:
    """Internal per-domain queue and fair-share state."""

    profile: QosProfile
    tasks: deque[_QueuedQosTask] = field(default_factory=deque)
    deficit_bytes: int = 0
    inflight_bytes: int = 0
    inflight_tasks: int = 0
    quantum_pending: bool = True


class L2QoSDispatcher:
    """Implement weighted L2 request scheduling for L2 tasks.

    The dispatcher is intentionally independent of a concrete L2 adapter. An
    adapter wrapper can submit the actual adapter call as ``action`` and call
    :meth:`complete` when the adapter reports completion. This lets one
    dispatcher coordinate store, lookup, and load tasks across adapters in the
    same resource group.
    Scheduling controls request admission and ordering before LMCache submits
    work to the concrete adapter. It does not propagate weights into
    filesystem writeback, device queues, network queues, or remote backend
    schedulers. Weights arbitrate only contended queued work; bounded in-flight
    limits keep work at this boundary long enough for that arbitration to
    influence downstream submission order.

    Args:
        quantum_bytes: Base byte quantum granted to a weight-100 domain per
            fair scheduling round.
        max_inflight_tasks: Resource-group concurrent task limit. ``0`` means no limit.
        max_inflight_bytes: Resource-group concurrent byte limit. ``0`` means no limit.
    """

    def __init__(
        self,
        quantum_bytes: int = 1 << 20,
        max_inflight_tasks: int = 8,
        max_inflight_bytes: int = 0,
    ) -> None:
        if quantum_bytes <= 0:
            raise ValueError("quantum_bytes must be positive")
        if max_inflight_tasks < 0:
            raise ValueError("max_inflight_tasks must be non-negative")
        if max_inflight_bytes < 0:
            raise ValueError("max_inflight_bytes must be non-negative")

        self._quantum_bytes = quantum_bytes
        self._max_inflight_tasks = max_inflight_tasks
        self._max_inflight_bytes = max_inflight_bytes
        self._domains: dict[str, _DomainQueue] = {}
        self._active_domains: deque[str] = deque()
        self._running: dict[int, _QueuedQosTask] = {}
        self._next_task_id = 0
        self._closed = False
        self._condition = threading.Condition()
        self._thread = threading.Thread(
            target=self._dispatch_loop,
            daemon=True,
            name="lmcache-l2-qos-dispatcher",
        )
        self._thread.start()

    def submit(
        self,
        profile: QosProfile,
        operation: str,
        cost_bytes: int,
        action: Callable[[], int],
    ) -> QosTaskHandle:
        """Queue one L2 request for weighted L2 request scheduling.

        Args:
            profile: Domain profile used for fair scheduling and quotas.
            operation: Operation label such as ``store`` or ``load``.
            cost_bytes: Estimated bytes consumed by the operation.
            action: Callable that submits the operation to the concrete adapter
                and returns its adapter task ID.

        Returns:
            A handle whose future resolves to the concrete adapter task ID.

        Raises:
            RuntimeError: If the dispatcher has been closed.
            ValueError: If ``cost_bytes`` is negative.
        """
        if cost_bytes < 0:
            raise ValueError("cost_bytes must be non-negative")
        cost_bytes = max(1, cost_bytes)
        future: Future[int] = Future()
        with self._condition:
            if self._closed:
                raise RuntimeError("L2QoSDispatcher is closed")
            domain = self._domains.get(profile.domain_id)
            if domain is None:
                domain = _DomainQueue(profile=profile)
                self._domains[profile.domain_id] = domain
            else:
                domain.profile = profile
            if not domain.tasks:
                self._active_domains.append(profile.domain_id)
                domain.quantum_pending = True
            task_id = self._next_task_id
            self._next_task_id += 1
            domain.tasks.append(
                _QueuedQosTask(
                    task_id=task_id,
                    profile=profile,
                    operation=operation,
                    cost_bytes=cost_bytes,
                    action=action,
                    future=future,
                )
            )
            self._condition.notify()
        return QosTaskHandle(task_id, profile.domain_id, cost_bytes, future)

    def update_profile(self, update: QosWeightUpdate) -> None:
        """Apply an explicit-salt or default weight change to queued work.

        Args:
            update: Registry update for one salt, or all default-weight domains.
        """
        with self._condition:
            if update.cache_salt is None:
                domains = [
                    domain
                    for domain in self._domains.values()
                    if domain.profile.source == "default"
                ]
            else:
                domain = self._domains.get(update.cache_salt)
                if domain is None:
                    return
                domains = [domain]
            for domain in domains:
                profile = QosProfile(
                    domain_id=domain.profile.domain_id,
                    weight=update.weight,
                    source=update.source,
                )
                domain.profile = profile
                for task in domain.tasks:
                    task.profile = profile
                # Re-evaluate the next round with the new weight. Retaining the
                # current deficit avoids throwing away service already earned.
                domain.quantum_pending = True
            self._condition.notify_all()

    def complete(self, task_id: int) -> None:
        """Release the in-flight accounting for a completed adapter task.

        Args:
            task_id: Dispatcher task ID returned in :class:`QosTaskHandle`.
        """
        with self._condition:
            task = self._running.pop(task_id, None)
            if task is None:
                return
            domain = self._domains[task.profile.domain_id]
            domain.inflight_bytes -= task.cost_bytes
            domain.inflight_tasks -= 1
            if not domain.tasks and domain.inflight_tasks == 0:
                del self._domains[task.profile.domain_id]
            self._condition.notify_all()

    def snapshot(self) -> dict[str, dict[str, int]]:
        """Return queue and service accounting grouped by domain."""
        with self._condition:
            return {
                domain_id: {
                    "weight": domain.profile.weight,
                    "queued_tasks": len(domain.tasks),
                    "inflight_tasks": domain.inflight_tasks,
                    "inflight_bytes": domain.inflight_bytes,
                }
                for domain_id, domain in self._domains.items()
            }

    def close(self) -> None:
        """Stop the dispatcher and fail tasks that were not dispatched."""
        with self._condition:
            if self._closed:
                return
            self._closed = True
            error = RuntimeError("L2QoSDispatcher is closed")
            for domain in self._domains.values():
                while domain.tasks:
                    task = domain.tasks.popleft()
                    task.future.set_exception(error)
            self._active_domains.clear()
            self._condition.notify_all()
        self._thread.join()

    def _can_dispatch(self, task: _QueuedQosTask) -> bool:
        """Check resource-group in-flight limits."""
        if self._max_inflight_tasks and len(self._running) >= self._max_inflight_tasks:
            return False
        if (
            self._max_inflight_bytes
            and self._running
            and sum(item.cost_bytes for item in self._running.values())
            + task.cost_bytes
            > self._max_inflight_bytes
        ):
            return False
        return True

    def _pop_next_locked(self) -> _QueuedQosTask | None:
        """Select and account for the next task. Caller holds the lock."""
        if not self._active_domains:
            return None

        for _ in range(len(self._active_domains)):
            domain_id = self._active_domains[0]
            domain = self._domains[domain_id]
            if not domain.tasks:
                self._active_domains.popleft()
                continue

            if domain.quantum_pending:
                domain.deficit_bytes += max(
                    1,
                    self._quantum_bytes * domain.profile.weight // DEFAULT_SCHED_WEIGHT,
                )
                domain.quantum_pending = False

            task = domain.tasks[0]
            if not self._can_dispatch(task):
                # Admission backpressure is not a scheduling round. Keep the
                # current deficit so a blocked domain does not accumulate an
                # unbounded burst while another task occupies the shared
                # in-flight budget.
                self._active_domains.rotate(-1)
                continue

            if domain.deficit_bytes < task.cost_bytes:
                self._active_domains.rotate(-1)
                domain.quantum_pending = True
                continue

            task = domain.tasks.popleft()
            domain.inflight_bytes += task.cost_bytes
            domain.inflight_tasks += 1
            self._running[task.task_id] = task
            domain.deficit_bytes -= task.cost_bytes
            if domain.tasks:
                if domain.deficit_bytes < domain.tasks[0].cost_bytes:
                    self._active_domains.rotate(-1)
                    domain.quantum_pending = True
            else:
                self._active_domains.popleft()
            return task
        return None

    def _has_admissible_head_locked(self) -> bool:
        """Return whether any queued head passes admission limits.

        Used by the dispatch loop to decide between advancing deficit rounds
        immediately and waiting on the condition. Deficit is intentionally not
        consulted: a deficit-blocked domain must keep advancing through logical
        rounds (without waiting), while only admission backpressure — which a
        completion or new submission can release — should block on the
        condition.
        """
        return any(
            domain.tasks and self._can_dispatch(domain.tasks[0])
            for domain in self._domains.values()
        )

    def _dispatch_loop(self) -> None:
        """Run adapter submission actions in weighted order."""
        while True:
            with self._condition:
                task = self._pop_next_locked()
                while task is None and not self._closed:
                    # Deficit rounds are logical scheduling rounds, not a
                    # passage of wall-clock time. Advance them immediately;
                    # wait only when admission limits block every head task or
                    # no work is queued. Completion and submission both notify
                    # the condition.
                    if self._has_admissible_head_locked():
                        task = self._pop_next_locked()
                        continue
                    self._condition.wait()
                    task = self._pop_next_locked()
                if task is None and self._closed:
                    return

            assert task is not None
            try:
                task.future.set_result(task.action())
            except Exception as exc:
                task.future.set_exception(exc)
                self.complete(task.task_id)


@dataclass
class _DispatcherPoolEntry:
    """One resource-group dispatcher and its adapter reference count."""

    dispatcher: L2QoSDispatcher
    adapter_count: int = 1


class L2QoSDispatcherPool:
    """Manage one shared dispatcher per L2 QoS resource group.

    The scheduling-weight registry remains common to all resource groups.
    Each group receives an independent DRR queue and independent in-flight
    limits. Adapters that acquire the same group share its dispatcher.

    Args:
        qos_manager: Registry that publishes cache-salt weight updates.
        quantum_bytes: Base byte quantum granted to a weight-100 domain.
        max_inflight_tasks: Per-group concurrent task limit. Zero is unlimited.
        max_inflight_bytes: Per-group concurrent byte limit. Zero is unlimited.
    """

    def __init__(
        self,
        qos_manager: CacheSaltQosManager,
        quantum_bytes: int = 1 << 20,
        max_inflight_tasks: int = 8,
        max_inflight_bytes: int = 0,
    ) -> None:
        self._qos_manager = qos_manager
        self._quantum_bytes = quantum_bytes
        self._max_inflight_tasks = max_inflight_tasks
        self._max_inflight_bytes = max_inflight_bytes
        self._entries: dict[str, _DispatcherPoolEntry] = {}
        self._closed = False
        self._lock = threading.Lock()

    def acquire(self, resource_group: str) -> L2QoSDispatcher:
        """Acquire the dispatcher for one resource group.

        Args:
            resource_group: Storage-manager-local group identifier.

        Returns:
            The existing group dispatcher, or a newly created dispatcher.

        Raises:
            ValueError: If ``resource_group`` is empty.
            RuntimeError: If the pool has been closed.
        """
        if not resource_group:
            raise ValueError("resource_group must be non-empty")
        with self._lock:
            if self._closed:
                raise RuntimeError("L2QoSDispatcherPool is closed")
            entry = self._entries.get(resource_group)
            if entry is not None:
                entry.adapter_count += 1
                return entry.dispatcher

            dispatcher = L2QoSDispatcher(
                quantum_bytes=self._quantum_bytes,
                max_inflight_tasks=self._max_inflight_tasks,
                max_inflight_bytes=self._max_inflight_bytes,
            )
            self._qos_manager.register_listener(dispatcher.update_profile)
            self._entries[resource_group] = _DispatcherPoolEntry(dispatcher)
            return dispatcher

    def release(self, resource_group: str) -> None:
        """Release one adapter reference to a resource-group dispatcher.

        The dispatcher and its profile listener are closed after the last
        adapter leaves the group.

        Args:
            resource_group: Identifier previously passed to ``acquire``.

        Raises:
            KeyError: If the group has no acquired dispatcher.
        """
        with self._lock:
            entry = self._entries.get(resource_group)
            if entry is None:
                raise KeyError(f"unknown L2 QoS resource group: {resource_group}")
            entry.adapter_count -= 1
            if entry.adapter_count:
                return
            del self._entries[resource_group]
            self._qos_manager.unregister_listener(entry.dispatcher.update_profile)

        entry.dispatcher.close()

    def close(self) -> None:
        """Close every resource-group dispatcher and listener."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            entries = list(self._entries.values())
            self._entries.clear()

        for entry in entries:
            self._qos_manager.unregister_listener(entry.dispatcher.update_profile)
            entry.dispatcher.close()
