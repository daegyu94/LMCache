# SPDX-License-Identifier: Apache-2.0
"""Prefill leases: let one request compute a prompt that several engines need.

Concurrent requests with the same prompt that reach different engines sharing
one LMCache server (an RL group's samples spread over data-parallel replicas by
a load balancer) all miss, because none of them has stored the prompt yet, and
every engine prefills it. The engine's own prefix cache only deduplicates within
one engine.

A lease records which request is about to compute a chunk. A later lookup from
another engine that needs a leased chunk waits while the owner makes progress:
it is released once every chunk it waits for is resident or no longer leased
(the owner hit it, or ended), and gives up, computing the chunks itself as
without leases, once the owner has stored none of them for ``idle_timeout_s``
(the owner is stuck in its engine's queue, or will never store them).
"""

# Future
from __future__ import annotations

# Standard
from collections.abc import Callable
from dataclasses import dataclass
import time

# First Party
from lmcache.v1.distributed.api import ObjectKey


@dataclass
class _Lease:
    owner: str
    engine: str


@dataclass
class _Wait:
    keys: list[ObjectKey]
    engine: str
    started_at: float
    progress_at: float


class PrefillLeaseTable:
    """Leases on chunks being computed, and the requests waiting on them.

    A lease lives as long as its owner: ``forget`` drops it when the owner
    ends, and ``sweep`` drops the leases and waits of requests ``is_alive``
    no longer reports.
    Not thread-safe: the caller serializes every call.

    Args:
        idle_timeout_s: How long a waiter waits without the owner storing any
            chunk it waits for; must be positive.
        is_resident: Whether a key is resident in the cache. Must have no side
            effects.
        is_alive: Whether a request is still live (its session exists).
        clock: Monotonic time source, in seconds.

    Raises:
        ValueError: If ``idle_timeout_s`` is not positive.
    """

    def __init__(
        self,
        idle_timeout_s: float,
        is_resident: Callable[[ObjectKey], bool],
        is_alive: Callable[[str], bool],
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if idle_timeout_s <= 0:
            raise ValueError(f"idle_timeout_s must be positive, got {idle_timeout_s}")
        self._idle_timeout_s = idle_timeout_s
        self._is_resident = is_resident
        self._is_alive = is_alive
        self._clock = clock
        self._leases: dict[ObjectKey, _Lease] = {}
        self._owned: dict[str, list[ObjectKey]] = {}
        self._waits: dict[str, _Wait] = {}
        self._stats = {"waits": 0, "served": 0, "timeouts": 0, "wait_seconds": 0.0}

    def acquire_or_wait(
        self, request_id: str, keys: list[ObjectKey], engine: str
    ) -> bool:
        """Lease ``keys`` to ``request_id``, unless a live request of another
        engine holds a lease on any of them.

        A request never waits on its own engine: the engine's prefix cache
        shares a prompt among its requests, and a wait would defeat that.
        Keys another live request of the same engine leased stay its own.

        Args:
            request_id: The looking-up request.
            keys: Every key of the chunks it is about to look up.
            engine: The engine the request runs on.

        Returns:
            True when the request may look up at once (its free keys are now
            leased to it); False when it must wait (``ready`` says when).
        """
        waiting = [k for k in keys if self._held_by_other_engine(k, engine)]
        if waiting:
            now = self._clock()
            self._waits[request_id] = _Wait(
                keys=waiting, engine=engine, started_at=now, progress_at=now
            )
            self._stats["waits"] += 1
            return False
        owned = self._owned.setdefault(request_id, [])
        for key in keys:
            lease = self._leases.get(key)
            if lease is not None and lease.owner != request_id and self._live(lease):
                continue  # held by another live request of this engine
            self._leases[key] = _Lease(owner=request_id, engine=engine)
            owned.append(key)
        return True

    def ready(self, request_id: str) -> bool:
        """Return whether ``request_id``'s wait is over, ending it if so.

        The wait is over once every awaited key is resident or no longer
        leased by a live request of another engine, or once none of them
        became resident for ``idle_timeout_s``. A request that is not
        waiting is ready.

        Args:
            request_id: The waiting request.

        Returns:
            True when the request may look up now.
        """
        wait = self._waits.get(request_id)
        if wait is None:
            return True
        now = self._clock()
        pending = [
            k
            for k in wait.keys
            if self._held_by_other_engine(k, wait.engine) and not self._is_resident(k)
        ]
        if len(pending) < len(wait.keys):
            wait.progress_at = now
        wait.keys = pending
        timed_out = bool(pending) and now - wait.progress_at >= self._idle_timeout_s
        if pending and not timed_out:
            return False
        del self._waits[request_id]
        self._stats["timeouts" if timed_out else "served"] += 1
        self._stats["wait_seconds"] += now - wait.started_at
        return True

    def release(self, request_id: str, keys: list[ObjectKey]) -> None:
        """Release ``request_id``'s leases on ``keys`` (it found them
        cached, so it will not compute them)."""
        for key in keys:
            lease = self._leases.get(key)
            if lease is not None and lease.owner == request_id:
                del self._leases[key]

    def forget(self, request_id: str) -> None:
        """Drop every lease and the wait of a finished request."""
        self._waits.pop(request_id, None)
        self.release(request_id, self._owned.pop(request_id, []))

    def sweep(self) -> None:
        """Drop the leases and waits of requests that ended without
        ``forget``."""
        for request_id in [
            r for r in {*self._owned, *self._waits} if not self._is_alive(r)
        ]:
            self.forget(request_id)

    def is_waiting(self, request_id: str) -> bool:
        """Return whether ``request_id`` waits on another request's lease."""
        return request_id in self._waits

    def stats(self) -> dict[str, float]:
        """Return counters: waits started, waits served (every awaited chunk
        landed or was released), waits given up for lack of progress, total
        wait seconds, and the live lease and wait counts."""
        return {
            **self._stats,
            "leases": len(self._leases),
            "waiting": len(self._waits),
        }

    def _live(self, lease: _Lease) -> bool:
        return self._is_alive(lease.owner)

    def _held_by_other_engine(self, key: ObjectKey, engine: str) -> bool:
        lease = self._leases.get(key)
        return lease is not None and lease.engine != engine and self._live(lease)
