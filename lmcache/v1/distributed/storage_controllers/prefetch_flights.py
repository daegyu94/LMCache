# SPDX-License-Identifier: Apache-2.0
"""Bounded partial-overlap prefetch rendezvous and temporary-object ownership.

All mutating methods run on the PrefetchController loop thread. A flight is
advisory: every follower performs fresh L1 locking and ordinary L2 planning.
No success bitmap or data pointer is transferred between requests.
"""

# Standard
from collections import defaultdict
from dataclasses import dataclass
from typing import Literal

# First Party
from lmcache.v1.distributed.api import FetchingPolicy, ObjectKey, PrefetchTaskSpec
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.l1_manager import L1Manager

FlightMode = Literal["off", "overlap", "handoff"]


@dataclass(frozen=True)
class _Cell:
    """Identity and interpretation of one candidate shared load."""

    key: ObjectKey
    shapes: tuple[tuple[int, ...], ...]
    dtypes: tuple[str, ...]
    window: int
    fetching_policy: FetchingPolicy


def _cells(spec: PrefetchTaskSpec) -> set[_Cell]:
    """Return deduplicated identity/layout/window qualified request cells."""
    cells: set[_Cell] = set()
    for group in spec.key_groups:
        shapes = tuple(tuple(shape) for shape in group.layout_desc.shapes)
        dtypes = tuple(str(dtype) for dtype in group.layout_desc.dtypes)
        cells.update(
            _Cell(key, shapes, dtypes, group.sliding_window_size, spec.fetching_policy)
            for key in group.keys
        )
    return cells


class PrefetchFlights:
    """Coordinate overlapping requests without changing cache identity.

    Args:
        l1_manager: The single L1 that receives loads from the controller.
        mode: ``off`` bypasses coordination; ``overlap`` waits for matching
            cells; ``handoff`` also pins temporary data for each waiter.
        max_waiters: Maximum requests waiting or holding a handoff pin.

    Raises:
        ValueError: If the mode is unknown or max_waiters is not positive.

    Note:
        Waiters delay their entire request, including the unique suffix.
        Pins use existing L1 read-lock TTL semantics; this does not fix the
        independent active-I/O reclamation/TTL contract described in #5263.
    """

    def __init__(
        self,
        l1_manager: L1Manager,
        mode: FlightMode = "off",
        max_waiters: int = 64,
    ) -> None:
        if mode not in ("off", "overlap", "handoff"):
            raise ValueError(f"Unknown prefetch flight mode: {mode}")
        if max_waiters <= 0:
            raise ValueError("max_waiters must be positive")
        self._l1 = l1_manager
        self._mode = mode
        self._max_waiters = max_waiters
        # A cell has at most one owner, and only admitted requests own cells.
        self._owners: dict[_Cell, int] = {}
        self._owned: dict[int, set[_Cell]] = {}
        # Waiter -> owner -> overlapping cells. Waiting requests own no cells,
        # so a cycle cannot form. Empty dependencies mean ready for admission.
        self._waiting: dict[int, dict[int, set[_Cell]]] = {}
        # One extra read lock per distinct (waiter, key), never a payload copy.
        self._pins: dict[int, set[ObjectKey]] = {}
        self._stats: dict[str, int] = {
            "waits": 0,
            "cap_bypasses": 0,
            "handoff_pins_acquired": 0,
            "handoff_pins_released": 0,
            "max_pins": 0,
        }
        self._pin_count = 0

    @property
    def enabled(self) -> bool:
        """Return whether overlap tracking is enabled."""
        return self._mode != "off"

    def defer(self, request_id: int, spec: PrefetchTaskSpec) -> bool:
        """Register a bounded waiter if an admitted load overlaps this request.

        Args:
            request_id: Unique controller request identifier.
            spec: Request identity, layout and window semantics.

        Returns:
            Whether the request must stay pending. A full waiter table falls
            back to ordinary independent loading without blocking admission.
        """
        if self._mode == "off":
            return False
        if request_id in self._waiting:
            return bool(self._waiting[request_id])
        dependencies: dict[int, set[_Cell]] = defaultdict(set)
        for cell in _cells(spec):
            owner = self._owners.get(cell)
            if owner is not None:
                dependencies[owner].add(cell)
        if not dependencies:
            return False
        if len(self._waiting) >= self._max_waiters:
            self._stats["cap_bypasses"] += 1
            return False
        self._waiting[request_id] = dict(dependencies)
        self._stats["waits"] += 1
        return True

    def started(
        self, request_id: int, spec: PrefetchTaskSpec, resident_keys: set[ObjectKey]
    ) -> None:
        """Return handoff pins after fresh locks and index only nonresident cells.

        Args:
            request_id: Newly admitted request identifier.
            spec: Request whose missing cells may become flights.
            resident_keys: Keys protected by the request's freshly acquired
                L1 locks. The controller must acquire those locks first.
        """
        if self._mode == "off":
            return
        self._release_pins(request_id)
        self._waiting.pop(request_id, None)
        owned = {
            cell
            for cell in _cells(spec)
            if cell.key not in resident_keys and cell not in self._owners
        }
        self._owned[request_id] = owned
        for cell in owned:
            self._owners[cell] = request_id

    def handoff(self, owner_id: int) -> None:
        """Pin a leader's resident cells before releasing/publishing its result.

        Args:
            owner_id: Request about to settle its own L1 locks. Unregistered
                requests (including synchronous L1-only calls) are a no-op.

        Note:
            Missing/failed cells acquire no pin. Followers still take fresh
            locks and recover through ordinary L2 loading after retirement.
        """
        if self._mode != "handoff" or owner_id not in self._owned:
            return
        for waiter_id, dependencies in self._waiting.items():
            cells = dependencies.get(owner_id, set())
            already_pinned = self._pins.get(waiter_id, set())
            keys = {cell.key for cell in cells} - already_pinned
            if not keys:
                continue
            reserved = self._l1.reserve_read(list(keys), 1)
            acquired = {
                key for key, (error, _) in reserved.items() if error == L1Error.SUCCESS
            }
            if acquired:
                self._pins.setdefault(waiter_id, set()).update(acquired)
                self._pin_count += len(acquired)
                self._stats["handoff_pins_acquired"] += len(acquired)
                self._stats["max_pins"] = max(self._stats["max_pins"], self._pin_count)

    def retire(self, request_id: int) -> None:
        """Remove completed/failed flight ownership and wake dependent waiters.

        Args:
            request_id: A completed or failed request. Calling twice is safe.
        """
        for cell in self._owned.pop(request_id, set()):
            del self._owners[cell]
        for dependencies in self._waiting.values():
            dependencies.pop(request_id, None)
        self._waiting.pop(request_id, None)
        self._release_pins(request_id)

    def close(self) -> None:
        """Release only rendezvous read pins after the controller thread stops.

        This does not reclaim or cancel active L2 load buffers. Adapter
        quiescence remains the independent existing controller contract.
        """
        for request_id in list(self._pins):
            self._release_pins(request_id)
        self._waiting.clear()
        self._owned.clear()
        self._owners.clear()

    def report_status(self) -> dict[str, int | str]:
        """Return approximate counters safe to sample from another thread.

        Returns:
            Mode, cumulative waits/cap bypasses/pins, and current bounded
            waiter, pin and indexed-cell counts. Snapshots are not atomic.
        """
        return {
            **self._stats.copy(),
            "mode": self._mode,
            "max_waiters": self._max_waiters,
            "current_waiters": len(self._waiting),
            "current_pins": self._pin_count,
            "indexed_cells": len(self._owners),
        }

    def _release_pins(self, request_id: int) -> None:
        keys = self._pins.pop(request_id, set())
        if keys:
            self._l1.finish_read(list(keys), 1)
            self._pin_count -= len(keys)
            self._stats["handoff_pins_released"] += len(keys)
