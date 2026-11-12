# SPDX-License-Identifier: Apache-2.0
"""Per-instance Bloom digests of L1 chunk hashes, maintained incrementally.

Owned by :class:`KeyDirectory`, which calls the hooks under its own lock as
L1 placements come and go. Inserts set bits as they arrive. A Bloom filter
cannot delete, so removals are only counted; once enough of an instance's
keys have left (or more have arrived than the filter was sized for), the next
read rebuilds that instance's filter from its current keys. This is the
Summary Cache scheme: stale bits cost a redundant lookup RPC, never a wrong
answer, so rebuilding on a change threshold rather than on every change is
safe.
"""

# Standard
from collections.abc import Mapping, Set
from dataclasses import dataclass, field

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.bloom_digest import BloomDigest, BloomFilter

# Rebuild once removals since the last build exceed this fraction of the keys
# the filter was built with. Summary Cache reports 1-10% thresholds costing
# little hit rate; here a stale bit only costs one redundant lookup RPC.
_REBUILD_REMOVED_FRACTION = 0.1
# Size each rebuilt filter for this multiple of the current keys, so the
# inserts that follow do not force an immediate rebuild.
_CAPACITY_HEADROOM = 2
_MIN_CAPACITY = 1024


@dataclass
class _InstanceDigest:
    bloom: BloomFilter
    built_with: int
    removed: int = 0
    pending: list[bytes] = field(default_factory=list)
    snapshot: BloomDigest | None = None


class L1DigestIndex:
    """Incrementally maintained L1 digests, one per instance.

    Not thread-safe: every method must be called under the owning
    directory's lock.
    """

    def __init__(self) -> None:
        self._digests: dict[str, _InstanceDigest] = {}

    def on_add(self, instance_id: str, chunk_hash: bytes) -> None:
        """Record a key newly placed in ``instance_id``'s L1.

        Args:
            instance_id: The instance that stored the key.
            chunk_hash: The key's chunk hash.
        """
        state = self._digests.get(instance_id)
        if state is None:
            return
        if len(state.pending) >= state.bloom.capacity:
            # Nobody has read for a long while: stop buffering, rebuild later.
            del self._digests[instance_id]
            return
        # Batched into the filter on the next read: one vectorized insert
        # per poll instead of one per event under the directory lock.
        state.pending.append(chunk_hash)

    def on_remove(self, instance_id: str) -> None:
        """Record that a key left ``instance_id``'s L1.

        Args:
            instance_id: The instance that dropped the key.
        """
        state = self._digests.get(instance_id)
        if state is not None:
            state.removed += 1

    def drop(self, instance_id: str) -> None:
        """Forget ``instance_id``'s digest (its L1 placements were fenced).

        Args:
            instance_id: The fenced instance.
        """
        self._digests.pop(instance_id, None)

    def clear(self) -> None:
        """Forget every digest; the next read rebuilds from the keys."""
        self._digests.clear()

    def digests(
        self, l1_keys_by_instance: Mapping[str, Set[ObjectKey]]
    ) -> dict[str, BloomDigest]:
        """Return a digest per instance, rebuilding or flushing as needed.

        Args:
            l1_keys_by_instance: The directory's current L1 keys per instance.

        Returns:
            ``instance_id`` → digest of its L1 chunk hashes.
        """
        result: dict[str, BloomDigest] = {}
        for instance_id, keys in l1_keys_by_instance.items():
            state = self._digests.get(instance_id)
            if (
                state is None
                or state.removed > _REBUILD_REMOVED_FRACTION * state.built_with
                or len(keys) > state.bloom.capacity
            ):
                state = self._rebuild(instance_id, keys)
            elif state.pending:
                state.bloom.add_many(state.pending)
                state.pending.clear()
                state.snapshot = None
            if state.snapshot is None:
                state.snapshot = state.bloom.snapshot()
            result[instance_id] = state.snapshot
        for instance_id in self._digests.keys() - l1_keys_by_instance.keys():
            del self._digests[instance_id]
        return result

    def _rebuild(self, instance_id: str, keys: Set[ObjectKey]) -> _InstanceDigest:
        """Build a fresh filter for ``instance_id`` from its current keys."""
        bloom = BloomFilter(max(_MIN_CAPACITY, _CAPACITY_HEADROOM * len(keys)))
        bloom.add_many([key.chunk_hash for key in keys])
        state = _InstanceDigest(bloom=bloom, built_with=len(keys))
        self._digests[instance_id] = state
        return state
