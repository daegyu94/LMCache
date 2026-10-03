# SPDX-License-Identifier: Apache-2.0
"""Bounded trajectory interest and marginal contiguous-prefix replacement."""

# Standard
from collections import OrderedDict
from collections.abc import Callable, Mapping
from typing import Literal
import sys
import threading

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.eviction import EvictionPolicy
from lmcache.v1.distributed.internal_api import EvictionAction, EvictionDestination

PrefixMode = Literal["PrefixLRU", "SessionRefcount", "PrefixUtility"]
CohortKey = tuple[str, int, int, str, str]


def read_prefix_advice(config: Mapping[str, object] | None) -> tuple[str, str] | None:
    """Validate bounded advisory metadata, without interpreting cache authority.

    Args:
        config: Existing request configs with trajectory_id and reuse_class.

    Returns:
        ID and soon/terminal state, or None for absent/malformed advice.
    """
    if config is None:
        return None
    trajectory = config.get("lmcache.agentrl.trajectory_id")
    state = config.get("lmcache.agentrl.reuse_class")
    if (
        isinstance(trajectory, str)
        and 0 < len(trajectory) <= 128
        and isinstance(state, str)
        and state in ("soon", "terminal")
    ):
        return trajectory, state
    return None


class PrefixUtilityPolicy(EvictionPolicy):
    """Greedy prefix utility with bounded manifests and an LRU fallback.

    Advice never changes payload identity, visibility, or reader ownership.
    PrefixLRU and SessionRefcount are controlled comparison algorithms, not
    reproductions of a production radix cache. All mutable state is locked.
    """

    def __init__(
        self,
        mode: PrefixMode = "PrefixUtility",
        max_cohorts: int = 256,
        max_references: int = 32768,
    ) -> None:
        """Configure policy and hard manifest limits.

        Args:
            mode: Marginal utility, leaf-LRU, or session-reference comparator.
            max_cohorts: Maximum retained trajectory manifests.
            max_references: Maximum total ordered-key references in manifests.

        Raises:
            ValueError: Limits are not positive or mode is unsupported.
        """
        if mode not in ("PrefixLRU", "SessionRefcount", "PrefixUtility"):
            raise ValueError("Unsupported prefix policy mode")
        if max_cohorts < 1 or max_references < 1:
            raise ValueError("Manifest limits must be positive")
        self._mode = mode
        self._max_cohorts = max_cohorts
        self._max_references = max_references
        self._order: OrderedDict[ObjectKey, None] = OrderedDict()
        self._ready: set[ObjectKey] = set()
        self._cohorts: OrderedDict[CohortKey, tuple[tuple[ObjectKey, ...], bool]] = (
            OrderedDict()
        )
        self._references = 0
        self._discarded = 0
        self._advised = 0
        self._destinations: list[EvictionDestination] = []
        self._lock = threading.Lock()

    @property
    def prefix_advice_enabled(self) -> bool:
        """Whether a caller should resolve a full prefix for this policy."""
        return True

    def register_eviction_destination(self, destination: EvictionDestination) -> None:
        """Register an eviction destination; the first registered one is used."""
        with self._lock:
            if destination not in self._destinations:
                self._destinations.append(destination)

    def on_keys_reserved(self, keys: list[ObjectKey]) -> None:
        """Track staging keys without treating them as readable prefix data."""
        with self._lock:
            for key in reversed(keys):
                self._order[key] = None
                self._order.move_to_end(key)

    def on_keys_created(self, keys: list[ObjectKey]) -> None:
        """Track admitted keys with the same reverse-batch ordering as LRU."""
        with self._lock:
            for key in reversed(keys):
                self._order[key] = None
                self._order.move_to_end(key)
                self._ready.add(key)

    def on_keys_touched(self, keys: list[ObjectKey]) -> None:
        """Update recency for actual usable accesses."""
        with self._lock:
            for key in reversed(keys):
                if key in self._order:
                    self._order.move_to_end(key)

    def on_keys_removed(self, keys: list[ObjectKey]) -> None:
        """Forget only confirmed deletions; manifests may describe holes."""
        with self._lock:
            for key in keys:
                self._order.pop(key, None)
                self._ready.discard(key)

    def advise_prefix(
        self, trajectory_id: str, keys: list[ObjectKey], state: str
    ) -> None:
        """Replace one namespace-qualified full-attention prefix interest.

        Args:
            trajectory_id: Stable ID, at most 128 characters; not an authority.
            keys: Entire ordered prefix from chunk zero, all in one namespace.
            state: soon marks interest; terminal withdraws this cohort only.

        Notes:
            Invalid, duplicate-key, or oversized advice is ignored. Oldest
            manifests are discarded at limits. Terminal manifests remain for
            topology-only comparators, within the same bounds.
        """
        if (
            not keys
            or not 0 < len(trajectory_id) <= 128
            or state not in ("soon", "terminal")
        ):
            return
        if len(keys) > self._max_references or len(set(keys)) != len(keys):
            return
        first = keys[0]
        namespace = (
            first.model_name,
            first.kv_rank,
            first.object_group_id,
            first.cache_salt,
        )
        if any(
            (k.model_name, k.kv_rank, k.object_group_id, k.cache_salt) != namespace
            for k in keys
        ):
            return
        cohort = (*namespace, trajectory_id)
        with self._lock:
            previous = self._cohorts.pop(cohort, None)
            if previous is not None:
                self._references -= len(previous[0])
            while self._cohorts and (
                len(self._cohorts) >= self._max_cohorts
                or self._references + len(keys) > self._max_references
            ):
                _, removed = self._cohorts.popitem(last=False)
                self._references -= len(removed[0])
                self._discarded += 1
            self._cohorts[cohort] = (tuple(keys), state == "soon")
            self._references += len(keys)
            self._advised += 1

    def get_prefix_stats(self) -> dict[str, int]:
        """Return bounded metadata counters and container-only memory estimate.

        Returns:
            Counters; container_bytes excludes referenced ObjectKey/string data.
        """
        with self._lock:
            return {
                "manifests": len(self._cohorts),
                "key_references": self._references,
                "active_manifests": sum(active for _, active in self._cohorts.values()),
                "discarded_manifests": self._discarded,
                "advice_calls": self._advised,
                "container_bytes": sys.getsizeof(self._cohorts)
                + sum(
                    sys.getsizeof(identity)
                    + sys.getsizeof(value)
                    + sys.getsizeof(value[0])
                    for identity, value in self._cohorts.items()
                ),
            }

    def get_eviction_actions(
        self,
        expected_ratio: float,
        key_eligible_filter: Callable[[ObjectKey], bool] | None = None,
        cache_salt: str | None = None,
    ) -> list[EvictionAction]:
        """Choose greedy victims without mutating resident membership.

        Args:
            expected_ratio: Fraction of tracked objects to evict, clamped to [0,1].
            key_eligible_filter: Authoritative eligibility (e.g. reader locks).
            cache_salt: Unused; this policy has no tenant-isolation guarantee.

        Returns:
            At most one action. Failed deletion leaves tracking unchanged.

        Notes:
            This exact reference algorithm rescans manifests per victim and is
            O(victims * references). It is not a constant-cost production index.
        """
        with self._lock:
            ratio = max(0.0, min(1.0, expected_ratio))
            count = max(1, int(len(self._order) * ratio)) if ratio else 0
            candidates = [
                k
                for k in self._order
                if key_eligible_filter is None or key_eligible_filter(k)
            ]
            available = set(self._ready)
            victims: list[ObjectKey] = []
            for _ in range(min(count, len(candidates))):
                scores: dict[ObjectKey, int] = {}
                blocked: set[ObjectKey] = set()
                for path, active in self._cohorts.values():
                    if self._mode == "PrefixUtility":
                        if not active:
                            continue
                        end = 0
                        for key in path:
                            if key not in available:
                                break
                            end += 1
                        for position, key in enumerate(path[:end]):
                            scores[key] = scores.get(key, 0) + end - position
                    else:
                        resident = [k for k in path if k in available]
                        blocked.update(resident[:-1])
                        if active and self._mode == "SessionRefcount":
                            for key in resident:
                                scores[key] = scores.get(key, 0) + 1
                leaves = [k for k in candidates if k not in blocked]
                pool = leaves or candidates
                victim = min(pool, key=lambda k: scores.get(k, 0))
                victims.append(victim)
                candidates.remove(victim)
                available.discard(victim)
            if not victims:
                return []
            destination = (
                self._destinations[0]
                if self._destinations
                else EvictionDestination.DISCARD
            )
            return [EvictionAction(keys=victims, destination=destination)]
