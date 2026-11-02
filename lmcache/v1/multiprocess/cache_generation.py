# SPDX-License-Identifier: Apache-2.0
"""Generation-scoped cache reset for the multiprocess cache server.

In ``clear_mode="generation"`` the server mixes a :class:`CacheGeneration`
(a random per-process incarnation plus a counter) into the seed of every chunk
hash it computes. Bumping the counter makes every object stored under an
earlier generation unreachable in O(1), in L1 and L2 alike, without per-key
deletes. :class:`StaleL2KeyCollector` then reclaims the unreachable L2
objects in the background. Design:
``docs/design/v1/multiprocess/cache_generation.md``.
"""

# Standard
from collections import deque
from dataclasses import dataclass
import secrets
import threading

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.internal_api import L2AdapterListener
from lmcache.v1.distributed.storage_manager import StorageManager
from lmcache.v1.multiprocess.config import ClearMode
from lmcache.v1.multiprocess.token_hasher import ChunkHash, TokenHasher
from lmcache.v1.periodic_thread import (
    PeriodicThread,
    PeriodicThreadRegistry,
    ThreadLevel,
    ThreadRunSummary,
    create_periodic_thread,
)

logger = init_logger(__name__)

MAX_TRACKED_L2_KEYS = 1_000_000
"""Default bound on L2 keys stored since the last retirement that
:class:`StaleL2KeyCollector` tracks. Keys past it are left to L2 eviction."""

MAX_PENDING_L2_KEYS = 1_000_000
"""Default bound on retired keys awaiting deletion. When deletes fall behind
the clear rate, retiring past it drops the excess to L2 eviction."""

L2_DELETE_BATCH_SIZE = 1024
"""Keys passed to one ``L2AdapterInterface.delete`` call."""

_GC_INTERVAL_SECONDS = 1.0

# Leading tag words that keep the two seed domains disjoint.
_GENERATION_DOMAIN = 0x47454E31  # "GEN1"
_UNTRUSTED_STORE_DOMAIN = 0x5354524E  # "STRN"


@dataclass(frozen=True)
class HashScope:
    """The cache generation a set of chunk hashes was computed under.

    Attributes:
        generation: The generation number (always 0 in ``"l1"`` mode).
        prefix_hash: The initial prefix hash that seeds the rolling chunk
            hashes of this generation.
    """

    generation: int
    prefix_hash: ChunkHash


class CacheGeneration:
    """Thread-safe monotonic counter naming the current cache generation.

    In ``"l1"`` mode the generation stays 0 and every seed is the hasher's
    ``none_hash``, so hashes are bit-identical to generation-free hashing. In
    ``"generation"`` mode each seed also mixes a random 64-bit incarnation
    chosen at construction: a restarted server never reproduces an earlier
    process's keys, so a persistent or shared L2 cannot serve objects stored
    under the same generation number by a previous incarnation (or by another
    server).

    Args:
        clear_mode: The server's clear mode.
    """

    def __init__(self, clear_mode: ClearMode) -> None:
        self._clear_mode: ClearMode = clear_mode
        self._incarnation = secrets.randbits(64) if clear_mode == "generation" else 0
        self._value = 0
        self._lock = threading.Lock()

    @property
    def clear_mode(self) -> ClearMode:
        """The clear mode this counter serves."""
        return self._clear_mode

    @property
    def current(self) -> int:
        """The current generation."""
        with self._lock:
            return self._value

    def bump(self) -> int:
        """Advance to the next generation.

        Returns:
            The new generation. Chunk hashes computed from its scope never
            address objects stored under an earlier generation.

        Raises:
            ValueError: In ``"l1"`` mode, whose generation is fixed at 0.
        """
        if self._clear_mode != "generation":
            raise ValueError("cache generation only advances in 'generation' mode")
        with self._lock:
            self._value += 1
            return self._value

    def current_scope(self, hasher: TokenHasher) -> HashScope:
        """Return the scope of the current generation, read once.

        Args:
            hasher: The hasher whose ``chunk_size`` and algorithm the seed
                must match.

        Returns:
            The current generation and its seed.
        """
        return self.scope(hasher, self.current)

    def scope(self, hasher: TokenHasher, generation: int) -> HashScope:
        """Return the scope of ``generation``.

        Args:
            hasher: The hasher the seed must match.
            generation: A generation this counter has reached.

        Returns:
            The generation and its seed.
        """
        if self._clear_mode != "generation":
            return HashScope(generation=generation, prefix_hash=hasher.none_hash)
        return HashScope(
            generation=generation,
            prefix_hash=hasher.domain_prefix_hash(
                [_GENERATION_DOMAIN, *_split_u64(self._incarnation)]
                + list(_split_u64(generation))
            ),
        )

    def untrusted_store_prefix_hash(self, hasher: TokenHasher) -> ChunkHash:
        """Return the seed for stores whose generation cannot be proven.

        No lookup ever seeds with it, so objects stored under it are never
        served; they are collected like any other stale key. In ``"l1"``
        mode it is ``none_hash`` (every store is trusted).

        Args:
            hasher: The hasher the seed must match.

        Returns:
            The seed.
        """
        if self._clear_mode != "generation":
            return hasher.none_hash
        return hasher.domain_prefix_hash(
            [_UNTRUSTED_STORE_DOMAIN, *_split_u64(self._incarnation)]
        )


def _split_u64(value: int) -> tuple[int, int]:
    """Split a non-negative 64-bit integer into ``(high, low)`` uint32 words."""
    return value >> 32, value & 0xFFFFFFFF


class StaleL2KeyCollector(L2AdapterListener):
    """Track keys stored to L2 and delete them in the background once retired.

    Registered as an L2 listener on every current and future adapter of the
    storage manager. :meth:`retire_tracked_keys` hands everything stored so
    far to a background thread, which deletes it from every L2 adapter in
    batches of :data:`L2_DELETE_BATCH_SIZE`; the caller never waits for L2.

    The collector does not know generations. Its contract is "every key whose
    store completed before ``retire_tracked_keys`` returned is deleted,
    unless a bound or a failure drops it to L2 eviction"; the caller must only
    retire when all those keys are unreachable.

    Keys are not attributed to the adapter that stored them (the listener
    API carries no adapter), so each batch is offered to every adapter;
    adapters skip keys they do not hold. Adapters whose ``delete`` is the
    base-class no-op keep their objects until their own eviction removes
    them.

    Memory is bounded twice: at most ``max_tracked_keys`` keys stored since
    the last retirement (excess counted as ``untracked_keys``) and at most
    ``max_pending_keys`` retired keys awaiting deletion (excess counted as
    ``dropped_keys``), so a slow adapter cannot stop tracking new stores.

    Thread-safe: listener callbacks arrive from adapter threads, retirement
    from an RPC thread, deletion runs on the collector's own thread, and the
    lock is never held across an adapter call (a ``delete`` re-enters
    :meth:`on_l2_keys_deleted`).

    Args:
        storage_manager: Supplies the L2 adapters and their listener hooks.
        max_tracked_keys: Bound on keys tracked since the last retirement.
        max_pending_keys: Bound on retired keys awaiting deletion.
    """

    def __init__(
        self,
        storage_manager: StorageManager,
        max_tracked_keys: int = MAX_TRACKED_L2_KEYS,
        max_pending_keys: int = MAX_PENDING_L2_KEYS,
    ) -> None:
        self._storage_manager = storage_manager
        self._max_tracked_keys = max_tracked_keys
        self._max_pending_keys = max_pending_keys
        self._lock = threading.Lock()
        # Keys stored since the last retirement; may still be reachable.
        self._tracked: set[ObjectKey] = set()
        # Retired keys awaiting deletion, oldest first; all unreachable.
        self._pending: deque[ObjectKey] = deque()
        self._retired_total = 0
        self._deleted_total = 0
        self._untracked_total = 0
        self._dropped_total = 0
        self._failed_delete_calls = 0
        # Set while the collector thread is inside an adapter ``delete``;
        # ``close`` waits on it so no batch outlives the adapters.
        self._deleting = False
        self._idle = threading.Condition(self._lock)
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
            The number of keys newly scheduled for deletion (excluding any
            dropped by the pending bound).
        """
        with self._lock:
            retired = self._tracked
            self._tracked = set()
            room = max(0, self._max_pending_keys - len(self._pending))
            accepted = list(retired)[:room] if len(retired) > room else retired
            self._pending.extend(accepted)
            self._retired_total += len(accepted)
            self._dropped_total += len(retired) - len(accepted)
        if accepted:
            self._thread.wake()
        return len(accepted)

    def on_l2_keys_stored(self, keys: list[ObjectKey], sizes: list[int]) -> None:
        """Track newly stored keys, up to ``max_tracked_keys``."""
        with self._lock:
            room = max(0, self._max_tracked_keys - len(self._tracked))
            accepted = keys if len(keys) <= room else keys[:room]
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
            ``tracked_keys`` (stored since the last retirement) and
            ``pending_delete_keys`` (retired, not yet offered for deletion),
            plus cumulative counters: ``retired_keys`` (scheduled),
            ``deleted_keys`` (in batches every adapter's ``delete`` accepted
            without raising), ``failed_delete_calls`` (adapter ``delete``
            calls that raised; those batches are left to L2 eviction),
            ``untracked_keys`` and ``dropped_keys`` (excess past the tracked
            and pending bounds, left to L2 eviction).
        """
        with self._lock:
            return {
                "tracked_keys": len(self._tracked),
                "pending_delete_keys": len(self._pending),
                "retired_keys": self._retired_total,
                "deleted_keys": self._deleted_total,
                "failed_delete_calls": self._failed_delete_calls,
                "untracked_keys": self._untracked_total,
                "dropped_keys": self._dropped_total,
            }

    def close(self) -> None:
        """Stop tracking and stop the collector thread.

        Must run before the storage manager closes its adapters. The thread
        checks for stop between batches; this call then waits, without a
        timeout, for a batch already inside an adapter ``delete`` to return,
        so no delete reaches an adapter after ``close`` returns. That wait is
        bounded only by the adapters' own delete timeouts (e.g. 30 s for the
        FS adapter). Keys still pending are left to L2 eviction.
        """
        self._storage_manager.unregister_l2_listener(self)
        PeriodicThreadRegistry.get_instance().unregister(self._thread.name)
        self._thread.stop()
        with self._idle:
            while self._deleting:
                self._idle.wait()

    def _collect_cycle(self) -> ThreadRunSummary:
        """Delete pending keys batch by batch until empty or stopped."""
        deleted = 0
        failed_calls = 0
        while True:
            with self._lock:
                # Checked under the lock ``close`` waits on, so a batch starts
                # only if ``close`` will see ``_deleting`` and wait for it.
                if self._thread.stop_requested:
                    break
                count = min(L2_DELETE_BATCH_SIZE, len(self._pending))
                batch = [self._pending.popleft() for _ in range(count)]
                self._deleting = bool(batch)
            if not batch:
                break
            batch_failed = False
            for _desc, adapter in self._storage_manager.l2_adapters():
                try:
                    adapter.delete(batch)
                except Exception:
                    logger.exception(
                        "Stale L2 key deletion failed for %d key(s); "
                        "leaving them to L2 eviction",
                        len(batch),
                    )
                    batch_failed = True
                    failed_calls += 1
            with self._idle:
                self._deleting = False
                self._idle.notify_all()
                self._failed_delete_calls += failed_calls
                if not batch_failed:
                    self._deleted_total += len(batch)
            failed_calls = 0
            if not batch_failed:
                deleted += len(batch)
        with self._lock:
            backlog = len(self._pending)
        if deleted:
            logger.debug(
                "Deleted %d stale L2 key(s); %d still pending", deleted, backlog
            )
        return ThreadRunSummary(
            success=True,
            message=f"deleted={deleted} pending={backlog}",
        )
