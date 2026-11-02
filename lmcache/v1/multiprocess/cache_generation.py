# SPDX-License-Identifier: Apache-2.0
"""Generation-scoped cache reset for the multiprocess cache server.

The server mixes a :class:`CacheGeneration` counter into the seed of every
chunk hash it computes (see :meth:`TokenHasher.generation_prefix_hash`).
Bumping the counter makes every object stored under an earlier generation
unreachable in O(1), in L1 and L2 alike, without per-key deletes.
:class:`StaleL2KeyCollector` then reclaims the unreachable L2 objects in the
background. Design: ``docs/design/v1/multiprocess/cache_generation.md``.
"""

# Standard
from collections import deque
import threading

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.internal_api import L2AdapterListener
from lmcache.v1.distributed.storage_manager import StorageManager
from lmcache.v1.periodic_thread import (
    PeriodicThread,
    PeriodicThreadRegistry,
    ThreadLevel,
    ThreadRunSummary,
    create_periodic_thread,
)

logger = init_logger(__name__)

MAX_TRACKED_L2_KEYS = 1_000_000
"""Upper bound on L2 keys :class:`StaleL2KeyCollector` holds (tracked plus
awaiting deletion). Keys stored past the bound are left to L2 eviction."""

L2_DELETE_BATCH_SIZE = 1024
"""Keys passed to one ``L2AdapterInterface.delete`` call."""

_MAX_BATCHES_PER_RUN = 64
"""Delete batches per collector cycle, so ``close`` is never stuck behind a
long backlog; a cycle that leaves work re-wakes the thread."""

_GC_INTERVAL_SECONDS = 1.0


class CacheGeneration:
    """Thread-safe monotonic counter naming the current cache generation.

    Starts at 0, the generation whose chunk hashes equal the hashes the
    server computed before generations existed. Only :meth:`bump` changes
    it; nothing ever decreases it within a process.
    """

    def __init__(self) -> None:
        self._value = 0
        self._lock = threading.Lock()

    @property
    def current(self) -> int:
        """The current generation."""
        with self._lock:
            return self._value

    def bump(self) -> int:
        """Advance to the next generation.

        Returns:
            The new generation. Chunk hashes computed after this returns
            never address objects stored under an earlier generation.
        """
        with self._lock:
            self._value += 1
            return self._value


class StaleL2KeyCollector(L2AdapterListener):
    """Track keys stored to L2 and delete them in the background once retired.

    Registered as an L2 listener on every current and future adapter of the
    storage manager. :meth:`retire_tracked_keys` hands everything stored so
    far to a background thread, which deletes it from every L2 adapter in
    batches of :data:`L2_DELETE_BATCH_SIZE`; the caller never waits for L2.

    The collector does not know generations. Its contract is "every key whose
    store completed before ``retire_tracked_keys`` returned is deleted"; the
    caller must only retire when all those keys are unreachable.

    Keys are not attributed to the adapter that stored them (the listener
    API carries no adapter), so each batch is offered to every adapter;
    adapters skip keys they do not hold. Adapters whose ``delete`` is the
    base-class no-op keep their objects until their own eviction removes
    them.

    Memory is bounded by ``max_tracked_keys``: keys stored past the bound
    are counted in ``untracked_keys`` and left to L2 eviction.

    Thread-safe: listener callbacks arrive from adapter threads, retirement
    from an RPC thread, deletion runs on the collector's own thread, and the
    lock is never held across an adapter call (a ``delete`` re-enters
    :meth:`on_l2_keys_deleted`).

    Args:
        storage_manager: Supplies the L2 adapters and their listener hook.
        max_tracked_keys: Upper bound on keys held, tracked plus awaiting
            deletion.
    """

    def __init__(
        self,
        storage_manager: StorageManager,
        max_tracked_keys: int = MAX_TRACKED_L2_KEYS,
    ) -> None:
        self._storage_manager = storage_manager
        self._max_tracked_keys = max_tracked_keys
        self._lock = threading.Lock()
        # Keys stored since the last retirement; may still be reachable.
        self._tracked: set[ObjectKey] = set()
        # Retired keys awaiting deletion, oldest first; all unreachable.
        self._retired: deque[ObjectKey] = deque()
        self._retired_total = 0
        self._deleted_total = 0
        self._untracked_total = 0
        self._failed_batches = 0
        self._thread: PeriodicThread = create_periodic_thread(
            name=f"lmcache-mp-stale-l2-gc-{id(self):x}",
            interval=_GC_INTERVAL_SECONDS,
            execute_fn=self._collect_cycle,
            level=ThreadLevel.LOW,
        )
        storage_manager.register_l2_listener(self)
        self._thread.start()

    def retire_tracked_keys(self) -> int:
        """Schedule every key stored so far for background deletion.

        Returns immediately; the deletion itself runs on the collector
        thread. A key whose store completes after this call is tracked anew
        and deleted by the next retirement.

        Returns:
            The number of keys newly scheduled for deletion.
        """
        with self._lock:
            retired = self._tracked
            self._tracked = set()
            self._retired.extend(retired)
            self._retired_total += len(retired)
        if retired:
            self._thread.wake()
        return len(retired)

    def on_l2_keys_stored(self, keys: list[ObjectKey], sizes: list[int]) -> None:
        """Track newly stored keys, up to ``max_tracked_keys``."""
        with self._lock:
            room = self._max_tracked_keys - len(self._tracked) - len(self._retired)
            if room >= len(keys):
                self._tracked.update(keys)
                return
            accepted = keys[: max(room, 0)]
            self._tracked.update(accepted)
            self._untracked_total += len(keys) - len(accepted)

    def on_l2_keys_accessed(self, keys: list[ObjectKey]) -> None:
        """No-op: access does not change whether a key is stale."""

    def on_l2_keys_deleted(self, keys: list[ObjectKey]) -> None:
        """Stop tracking keys the adapter already removed."""
        with self._lock:
            self._tracked.difference_update(keys)

    def report_status(self) -> dict[str, int]:
        """Return collector counters.

        Returns:
            ``tracked_keys`` (stored since the last retirement),
            ``pending_delete_keys`` (retired, not yet offered to the
            adapters), and cumulative ``retired_keys``, ``deleted_keys``
            (offered to every adapter's ``delete``), ``untracked_keys``
            (dropped by the memory bound) and ``failed_delete_batches``.
        """
        with self._lock:
            return {
                "tracked_keys": len(self._tracked),
                "pending_delete_keys": len(self._retired),
                "retired_keys": self._retired_total,
                "deleted_keys": self._deleted_total,
                "untracked_keys": self._untracked_total,
                "failed_delete_batches": self._failed_batches,
            }

    def close(self) -> None:
        """Stop the collector thread; keys still pending are left to L2
        eviction."""
        PeriodicThreadRegistry.get_instance().unregister(self._thread.name)
        self._thread.stop()

    def _collect_cycle(self) -> ThreadRunSummary:
        """Delete up to ``_MAX_BATCHES_PER_RUN`` batches of retired keys."""
        deleted = 0
        for _ in range(_MAX_BATCHES_PER_RUN):
            with self._lock:
                count = min(L2_DELETE_BATCH_SIZE, len(self._retired))
                batch = [self._retired.popleft() for _ in range(count)]
            if not batch:
                break
            for _desc, adapter in self._storage_manager.l2_adapters():
                try:
                    adapter.delete(batch)
                except Exception:
                    logger.exception(
                        "Stale L2 key deletion failed for %d key(s); "
                        "leaving them to L2 eviction",
                        len(batch),
                    )
                    with self._lock:
                        self._failed_batches += 1
            deleted += len(batch)
            with self._lock:
                self._deleted_total += len(batch)
        with self._lock:
            backlog = len(self._retired)
        if backlog:
            self._thread.wake()
        if deleted:
            logger.debug(
                "Deleted %d stale L2 key(s); %d still pending", deleted, backlog
            )
        return ThreadRunSummary(
            success=True,
            message=f"deleted={deleted} pending={backlog}",
        )
