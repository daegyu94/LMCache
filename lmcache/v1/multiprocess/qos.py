# SPDX-License-Identifier: Apache-2.0
"""QoS metadata, cgroup discovery, and weighted L2 task dispatching."""

# Standard
from collections import deque
from concurrent.futures import Future
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator
import contextvars
import hashlib
import os
import re
import threading

_MIN_CGROUP_WEIGHT = 1
_MAX_CGROUP_WEIGHT = 10000
_DEFAULT_CGROUP_WEIGHT = 100


@dataclass(frozen=True)
class CgroupIOWeight:
    """The cgroup v2 ``io.weight`` values visible to one process.

    Args:
        default_weight: Weight used for devices without an override.
        device_weights: Device-specific weights keyed by ``major:minor``.
        cgroup_path: The process cgroup path, if it could be discovered.
    """

    default_weight: int
    device_weights: dict[str, int] = field(default_factory=dict)
    cgroup_path: str | None = None


@dataclass(frozen=True)
class QosProfile:
    """Scheduling metadata for one logical LLM instance.

    ``domain_id`` identifies a workload, not a cache key. All ranks and
    connector sockets belonging to one LLM instance should use the same
    domain. ``weight`` is a relative service weight in the cgroup v2 range.

    Args:
        domain_id: Stable logical identifier for the workload.
        weight: Relative scheduling weight in the range [1, 10000].
        source: Origin of the profile, such as ``cgroup`` or ``explicit``.
        max_inflight_bytes: Optional per-domain byte limit.
        max_outstanding_ops: Optional per-domain operation limit.
    """

    domain_id: str = "default"
    weight: int = _DEFAULT_CGROUP_WEIGHT
    source: str = "default"
    max_inflight_bytes: int | None = None
    max_outstanding_ops: int | None = None

    def __post_init__(self) -> None:
        if not self.domain_id:
            raise ValueError("domain_id must not be empty")
        if not _MIN_CGROUP_WEIGHT <= self.weight <= _MAX_CGROUP_WEIGHT:
            raise ValueError(
                f"weight must be in [{_MIN_CGROUP_WEIGHT}, "
                f"{_MAX_CGROUP_WEIGHT}], got {self.weight}"
            )
        if self.max_inflight_bytes is not None and self.max_inflight_bytes <= 0:
            raise ValueError("max_inflight_bytes must be positive")
        if self.max_outstanding_ops is not None and self.max_outstanding_ops <= 0:
            raise ValueError("max_outstanding_ops must be positive")

    @classmethod
    def from_environment(cls) -> "QosProfile":
        """Build a profile from explicit variables or the current cgroup.

        The explicit ``LMCACHE_QOS_WEIGHT`` and ``LMCACHE_QOS_DOMAIN``
        variables take precedence. If they are absent, the default cgroup v2
        ``io.weight`` is used and a stable domain identifier is derived from
        the visible cgroup path and container hostname.

        Returns:
            A validated QoS profile. The default profile is returned when no
            cgroup information is available.
        """
        cgroup_weight = read_cgroup_io_weight()
        explicit_weight = os.getenv("LMCACHE_QOS_WEIGHT")
        explicit_domain = os.getenv("LMCACHE_QOS_DOMAIN")

        if explicit_weight is not None:
            try:
                weight = int(explicit_weight)
            except ValueError as exc:
                raise ValueError("LMCACHE_QOS_WEIGHT must be an integer") from exc
            source = "explicit"
        elif cgroup_weight is not None:
            weight = cgroup_weight.default_weight
            source = "cgroup"
        else:
            weight = _DEFAULT_CGROUP_WEIGHT
            source = "default"

        if explicit_domain is not None:
            domain_id = explicit_domain
        elif cgroup_weight is not None:
            hostname = os.getenv("HOSTNAME", "")
            identity = f"{hostname}:{cgroup_weight.cgroup_path or '/'}"
            digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
            domain_id = f"cgroup-{digest}"
        else:
            domain_id = "default"

        return cls(domain_id=domain_id, weight=weight, source=source)


DEFAULT_QOS_PROFILE = QosProfile()

_current_qos_profile: contextvars.ContextVar[QosProfile] = contextvars.ContextVar(
    "lmcache_current_qos_profile",
    default=DEFAULT_QOS_PROFILE,
)


@contextmanager
def qos_profile_context(profile: QosProfile) -> Iterator[None]:
    """Temporarily associate the current thread with a QoS profile.

    Args:
        profile: Profile to make visible to adapter submission code.

    Yields:
        Nothing. The previous profile is restored on exit.
    """
    token = _current_qos_profile.set(profile)
    try:
        yield
    finally:
        _current_qos_profile.reset(token)


def get_current_qos_profile() -> QosProfile:
    """Return the QoS profile associated with the current execution context."""
    return _current_qos_profile.get()


def _unescape_mountinfo_path(value: str) -> str:
    """Decode the octal escapes used for paths in ``mountinfo``."""
    return re.sub(
        r"\\([0-7]{3})",
        lambda match: chr(int(match.group(1), 8)),
        value,
    )


def _read_current_cgroup_path() -> str | None:
    """Read the current process path in the unified cgroup hierarchy."""
    try:
        lines = Path("/proc/self/cgroup").read_text(encoding="utf-8").splitlines()
    except OSError:
        return None

    for line in lines:
        hierarchy, _, path = line.partition(":")
        controllers, _, path = path.partition(":")
        if hierarchy == "0" and controllers == "":
            return path or "/"
    return None


def _find_cgroup2_mount() -> Path | None:
    """Find the cgroup v2 mount point from ``/proc/self/mountinfo``."""
    try:
        lines = Path("/proc/self/mountinfo").read_text(encoding="utf-8").splitlines()
    except OSError:
        return None

    for line in lines:
        before, separator, after = line.partition(" - ")
        if not separator:
            continue
        post_mount_fields = after.split()
        if not post_mount_fields or post_mount_fields[0] != "cgroup2":
            continue
        mount_fields = before.split()
        if len(mount_fields) < 5:
            continue
        return Path(_unescape_mountinfo_path(mount_fields[4]))
    return None


def _parse_io_weight_file(
    path: Path,
    cgroup_path: str | None,
) -> CgroupIOWeight | None:
    """Parse one cgroup v2 ``io.weight`` file."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None

    default_weight = _DEFAULT_CGROUP_WEIGHT
    device_weights: dict[str, int] = {}
    found_value = False
    for line in lines:
        fields = line.split()
        if len(fields) == 1:
            device = "default"
            raw_weight = fields[0]
        elif len(fields) == 2:
            device, raw_weight = fields
        else:
            continue
        try:
            weight = int(raw_weight)
        except ValueError:
            continue
        if not _MIN_CGROUP_WEIGHT <= weight <= _MAX_CGROUP_WEIGHT:
            continue
        found_value = True
        if device == "default":
            default_weight = weight
        elif re.fullmatch(r"\d+:\d+", device):
            device_weights[device] = weight

    if not found_value:
        return None
    return CgroupIOWeight(
        default_weight=default_weight,
        device_weights=device_weights,
        cgroup_path=cgroup_path,
    )


def read_cgroup_io_weight(
    cgroup_file: str | Path | None = None,
) -> CgroupIOWeight | None:
    """Read cgroup v2 ``io.weight`` for the current process.

    Args:
        cgroup_file: Optional explicit file path, primarily useful for tests.
            When omitted, the cgroup v2 mount and the current process cgroup
            path are discovered from procfs.

    Returns:
        Parsed cgroup weights, or ``None`` if cgroup v2 or ``io.weight`` is not
        visible.
    """
    if cgroup_file is not None:
        return _parse_io_weight_file(Path(cgroup_file), None)

    cgroup_path = _read_current_cgroup_path()
    mount_point = _find_cgroup2_mount()
    if cgroup_path is None or mount_point is None:
        return None

    relative_path = cgroup_path.lstrip("/")
    io_weight_path = mount_point / relative_path / "io.weight"
    return _parse_io_weight_file(io_weight_path, cgroup_path)


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
    """Dispatch L2 tasks using weighted, byte-based fair scheduling.

    The dispatcher is intentionally independent of a concrete L2 adapter. An
    adapter wrapper can submit the actual adapter call as ``action`` and call
    :meth:`complete` when the adapter reports completion. This lets one
    dispatcher coordinate store, lookup, and load tasks across adapters.

    Args:
        quantum_bytes: Base byte quantum granted to a weight-100 domain per
            fair scheduling round.
        max_inflight_tasks: Global concurrent task limit. ``0`` means no limit.
        max_inflight_bytes: Global concurrent byte limit. ``0`` means no limit.
    """

    def __init__(
        self,
        quantum_bytes: int = 1 << 20,
        max_inflight_tasks: int = 4,
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
        """Queue one L2 action for weighted dispatch.

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
        """Check global and per-domain in-flight limits."""
        domain = self._domains[task.profile.domain_id]
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
        if (
            task.profile.max_outstanding_ops
            and domain.inflight_tasks >= task.profile.max_outstanding_ops
        ):
            return False
        if (
            task.profile.max_inflight_bytes
            and domain.inflight_bytes
            and domain.inflight_bytes + task.cost_bytes
            > task.profile.max_inflight_bytes
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
                domain.deficit_bytes += (
                    self._quantum_bytes
                    * domain.profile.weight
                    // _DEFAULT_CGROUP_WEIGHT
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

    def _dispatch_loop(self) -> None:
        """Run adapter submission actions in weighted order."""
        while True:
            with self._condition:
                task = self._pop_next_locked()
                while task is None and not self._closed:
                    timeout = 0.01 if self._active_domains else None
                    self._condition.wait(timeout=timeout)
                    task = self._pop_next_locked()
                if task is None and self._closed:
                    return

            assert task is not None
            try:
                task.future.set_result(task.action())
            except BaseException as exc:
                task.future.set_exception(exc)
                self.complete(task.task_id)
