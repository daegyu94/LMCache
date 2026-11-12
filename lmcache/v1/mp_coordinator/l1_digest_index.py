# SPDX-License-Identifier: Apache-2.0
"""Per-instance Bloom digests of L1 chunk hashes, maintained incrementally.

Owned by :class:`KeyDirectory`, which calls every method under its own lock.
Inserts are buffered as they arrive and folded in on the next read. A Bloom
filter cannot delete, so removals are only counted; once enough of an
instance's keys have left (or more have arrived than the filter was sized
for), that instance's filter is rebuilt from its current keys. This is the
Summary Cache scheme: a stale bit costs a redundant lookup RPC, never a wrong
answer, so rebuilding on a change threshold rather than on every change is
safe.

A rebuild hashes every key of the instance, which takes ~0.4 us per key, so
it runs outside the directory lock in three steps::

    with lock:  jobs = index.begin_rebuilds(keys_by_instance)   # copy hashes
    built = L1DigestIndex.build(jobs)                            # no lock
    with lock:  digests = index.finish(built, keys_by_instance)  # install

Inserts and removals that land between the first and the last step are
recorded against the rebuild and applied when it is installed.

Each installed filter starts a new *generation*, and the hashes folded into it
afterwards are kept in order. A reader that already holds generation ``g`` up
to ``n`` folded hashes therefore needs only the hashes after ``n``, not the
whole filter (see :class:`L1DigestView`).
"""

# Standard
from collections.abc import Mapping, Set
from dataclasses import dataclass, field
import itertools

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
class _Rebuild:
    """Changes that landed while a rebuild of one instance was in flight."""

    added: list[bytes] = field(default_factory=list)
    removed: int = 0


@dataclass
class RebuildJob:
    """One instance's rebuild input, taken under the directory lock.

    Attributes:
        chunk_hashes: The instance's L1 chunk hashes when the job was taken.
        capacity: Number of keys the new filter is sized for.
    """

    chunk_hashes: list[bytes]
    capacity: int
    _token: _Rebuild


@dataclass(frozen=True)
class L1DigestView:
    """One instance's digest as of a read.

    Attributes:
        generation: Identifies the filter build; changes on every rebuild and
            never repeats, even across instances.
        digest: The filter with every hash folded in so far.
        added: Hashes folded in since the build, in order. Only the first
            ``count`` are part of ``digest``; the list may grow later.
        count: Number of hashes of ``added`` included in ``digest``.
    """

    generation: int
    digest: BloomDigest
    added: list[bytes]
    count: int


@dataclass
class _InstanceDigest:
    bloom: BloomFilter | None = None  # None until the first build is installed
    generation: int = 0
    built_with: int = 0
    removed: int = 0
    pending: list[bytes] = field(default_factory=list)
    # Folded since the build. Replaced, never cleared, on a rebuild, so a
    # view handed out earlier keeps a stable prefix.
    log: list[bytes] = field(default_factory=list)
    snapshot: BloomDigest | None = None
    rebuild: _Rebuild | None = None


class L1DigestIndex:
    """Incrementally maintained L1 digests, one per instance.

    Not thread-safe: every method except :meth:`build` must be called under
    the owning directory's lock.
    """

    def __init__(self) -> None:
        self._digests: dict[str, _InstanceDigest] = {}
        self._generations = itertools.count(1)

    def on_add(self, instance_id: str, chunk_hash: bytes) -> None:
        """Record a key newly placed in ``instance_id``'s L1.

        Args:
            instance_id: The instance that stored the key.
            chunk_hash: The key's chunk hash.
        """
        state = self._digests.get(instance_id)
        if state is None:
            return  # no digest yet: the first build reads the keys
        if state.rebuild is not None:
            state.rebuild.added.append(chunk_hash)
        if state.bloom is None:
            return
        if len(state.pending) >= state.bloom.capacity:
            # Nobody has read for a long while: force a rebuild on the next read.
            state.removed = state.built_with + 1
            state.pending.clear()
            return
        # Folded in on the next read: one vectorized insert per poll instead of
        # one per event under the directory lock.
        state.pending.append(chunk_hash)

    def on_remove(self, instance_id: str) -> None:
        """Record that a key left ``instance_id``'s L1.

        Args:
            instance_id: The instance that dropped the key.
        """
        state = self._digests.get(instance_id)
        if state is None:
            return
        state.removed += 1
        if state.rebuild is not None:
            state.rebuild.removed += 1

    def drop(self, instance_id: str) -> None:
        """Forget ``instance_id``'s digest (its L1 placements were fenced).

        Args:
            instance_id: The fenced instance.
        """
        self._digests.pop(instance_id, None)

    def clear(self) -> None:
        """Forget every digest; the next read rebuilds from the keys."""
        self._digests.clear()

    def begin_rebuilds(
        self, l1_keys_by_instance: Mapping[str, Set[ObjectKey]]
    ) -> dict[str, RebuildJob]:
        """Take the rebuild input of every instance whose filter needs one.

        An instance with a rebuild already in flight (another reader started
        it) is skipped.

        Args:
            l1_keys_by_instance: The directory's current L1 keys per instance.

        Returns:
            ``instance_id`` → rebuild job, to pass to :meth:`build`.
        """
        for instance_id in self._digests.keys() - l1_keys_by_instance.keys():
            del self._digests[instance_id]
        jobs: dict[str, RebuildJob] = {}
        for instance_id, keys in l1_keys_by_instance.items():
            state = self._digests.setdefault(instance_id, _InstanceDigest())
            stale = (
                state.bloom is None
                or state.removed > _REBUILD_REMOVED_FRACTION * state.built_with
                or len(keys) > state.bloom.capacity
            )
            if not stale or state.rebuild is not None:
                continue
            state.rebuild = _Rebuild()
            jobs[instance_id] = RebuildJob(
                chunk_hashes=[key.chunk_hash for key in keys],
                capacity=max(_MIN_CAPACITY, _CAPACITY_HEADROOM * len(keys)),
                _token=state.rebuild,
            )
        return jobs

    @staticmethod
    def build(jobs: dict[str, RebuildJob]) -> dict[str, tuple[RebuildJob, BloomFilter]]:
        """Build the filters for ``jobs``; call without holding the lock.

        Args:
            jobs: The output of :meth:`begin_rebuilds`.

        Returns:
            ``instance_id`` → (job, its new filter), to pass to :meth:`finish`.
        """
        built = {}
        for instance_id, job in jobs.items():
            bloom = BloomFilter(job.capacity)
            bloom.add_many(job.chunk_hashes)
            built[instance_id] = (job, bloom)
        return built

    def finish(
        self, built: dict[str, tuple[RebuildJob, BloomFilter]]
    ) -> dict[str, L1DigestView]:
        """Install built filters and return the current digest per instance.

        A filter whose instance was dropped, cleared or re-fenced while it was
        built is discarded. An instance whose first build is still in flight
        in another reader has no digest yet and is left out.

        Args:
            built: The output of :meth:`build`.

        Returns:
            ``instance_id`` → view of its current digest.
        """
        for instance_id, (job, bloom) in built.items():
            state = self._digests.get(instance_id)
            if state is None or state.rebuild is not job._token:
                continue
            bloom.add_many(job._token.added)
            state.bloom = bloom
            state.generation = next(self._generations)
            state.log = []
            state.built_with = len(job.chunk_hashes)
            state.removed = job._token.removed
            state.pending.clear()
            state.snapshot = None
            state.rebuild = None
        result: dict[str, L1DigestView] = {}
        for instance_id, state in self._digests.items():
            if state.bloom is None:
                continue
            if state.pending:
                state.bloom.add_many(state.pending)
                state.log.extend(state.pending)
                state.pending.clear()
                state.snapshot = None
            if state.snapshot is None:
                state.snapshot = state.bloom.snapshot()
            result[instance_id] = L1DigestView(
                generation=state.generation,
                digest=state.snapshot,
                added=state.log,
                count=len(state.log),
            )
        return result
