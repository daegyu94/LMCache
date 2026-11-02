# SPDX-License-Identifier: Apache-2.0
"""Per-sequence chunk-chain tracking for requests that rewrite their history.

A client may tag consecutive requests of one conversation or trajectory with
``lmcache.sequence_id``. The tracker remembers the chunk-hash chain of each
sequence's latest lookup and, on the next request of the same sequence,
finds the first chunk where the chains differ (the divergence chunk). The
previous request's chunks at or after that point will not be hit again by
this sequence: they are *superseded*.

The module is pure bookkeeping: no I/O and no cache side effects. See
``docs/design/v1/multiprocess/sequence_lifecycle.md``.
"""

# Standard
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
import enum
import re
import threading

SEQUENCE_ID_HINT = "lmcache.sequence_id"
"""Request config key naming the sequence a request continues."""

SEQUENCE_ID_MAX_LEN = 128
_SEQUENCE_ID_PATTERN = re.compile(r"[A-Za-z0-9._:\-]{1,%d}" % SEQUENCE_ID_MAX_LEN)


def _common_prefix_length(a: tuple[bytes, ...], b: tuple[bytes, ...]) -> int:
    """Return the number of leading elements ``a`` and ``b`` share."""
    n = 0
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        n += 1
    return n


def parse_sequence_id(request_configs: Mapping[str, Any] | None) -> str | None:
    """Return the request's ``lmcache.sequence_id`` hint.

    Args:
        request_configs: The request's ``lmcache.*`` entries as the engine
            forwarded them, or None when it carries none.

    Returns:
        The sequence id, or None when the request carries no hint.

    Raises:
        ValueError: If the hint is present but is not a string of 1 to
            ``SEQUENCE_ID_MAX_LEN`` characters from ``[A-Za-z0-9._:-]``.
            The message never includes the raw value.
    """
    if not request_configs or SEQUENCE_ID_HINT not in request_configs:
        return None
    value = request_configs[SEQUENCE_ID_HINT]
    if not isinstance(value, str):
        raise ValueError(f"{SEQUENCE_ID_HINT} must be a string")
    if _SEQUENCE_ID_PATTERN.fullmatch(value) is None:
        raise ValueError(
            f"{SEQUENCE_ID_HINT} must be 1-{SEQUENCE_ID_MAX_LEN} characters "
            "from [A-Za-z0-9._:-]"
        )
    return value


@dataclass(frozen=True)
class SequenceKey:
    """Identity of a tracked sequence.

    The model, world size and cache salt are part of the identity because
    they are part of every chunk's object key: two tenants (salts) using the
    same sequence id never compare each other's chains.
    """

    model_name: str
    world_size: int
    cache_salt: str
    sequence_id: str


class TransitionKind(enum.Enum):
    """How a request relates to its sequence's previous request."""

    FIRST = "first"
    """No previous request is tracked (new or forgotten sequence)."""

    REPEAT = "repeat"
    """The same request was looked up again (e.g. an engine re-lookup)."""

    APPEND = "append"
    """The previous chain is a prefix of the new one: nothing superseded."""

    REWRITE = "rewrite"
    """The chains differ before the end of the previous chain."""


@dataclass(frozen=True)
class SequenceTransition:
    """The outcome of one :meth:`SequenceTracker.observe` call.

    Attributes:
        kind: How the request relates to the previous one.
        divergence_chunk: Length of the common chunk prefix of the previous
            and the new chain; 0 for ``FIRST``.
        previous_chunks: Length of the previous chain; 0 for ``FIRST``.
        superseded_hashes: The previous chain's hashes at or after
            ``divergence_chunk``, in chain order, for ``APPEND`` (always
            empty) and ``REWRITE``; empty for ``FIRST`` and ``REPEAT``.
    """

    kind: TransitionKind
    divergence_chunk: int
    previous_chunks: int
    superseded_hashes: tuple[bytes, ...]


@dataclass
class _SequenceEntry:
    request_id: str
    chunk_hashes: tuple[bytes, ...]


class SequenceTracker:
    """Bounded memory of each sequence's latest chunk-hash chain.

    The chains of all tracked sequences hold at most ``max_tracked_chunks``
    hashes in total. When a new chain would exceed the budget, the least
    recently observed sequences are forgotten first; a chain longer than the
    whole budget is not stored. Forgetting a sequence only means its next
    request reports ``FIRST``.

    Thread-safe: every public method holds one internal lock.

    Args:
        max_tracked_chunks: Total chunk hashes kept across sequences; must
            be positive (the server does not build a tracker when tracking
            is disabled).

    Raises:
        ValueError: If ``max_tracked_chunks`` is not positive.
    """

    def __init__(self, max_tracked_chunks: int) -> None:
        if max_tracked_chunks < 1:
            raise ValueError(
                f"max_tracked_chunks must be positive; got {max_tracked_chunks}"
            )
        self._max_chunks = max_tracked_chunks
        self._lock = threading.Lock()
        # Ordered by last observation, oldest first.
        self._entries: OrderedDict[SequenceKey, _SequenceEntry] = OrderedDict()
        # Invariant: sum of len(entry.chunk_hashes) <= _max_chunks.
        self._num_chunks = 0

    def observe(
        self,
        key: SequenceKey,
        request_id: str,
        chunk_hashes: list[bytes],
    ) -> SequenceTransition:
        """Compare a request's chain with its sequence's previous request.

        The new chain replaces the previous one as the sequence's latest
        chain (an empty chain forgets the sequence).

        Args:
            key: The sequence the request belongs to.
            request_id: The engine request id; a repeat of the previous
                request id is reported as ``REPEAT``.
            chunk_hashes: The request's full chunk-hash chain from chunk 0,
                as the lookup computed it.

        Returns:
            The transition from the previous request to this one.
        """
        new_chain = tuple(chunk_hashes)
        with self._lock:
            previous = self._entries.pop(key, None)
            if previous is not None:
                self._num_chunks -= len(previous.chunk_hashes)
            if new_chain and len(new_chain) <= self._max_chunks:
                self._entries[key] = _SequenceEntry(request_id, new_chain)
                self._num_chunks += len(new_chain)
                self._shrink_to_budget_locked()

        if previous is None:
            return SequenceTransition(TransitionKind.FIRST, 0, 0, ())
        prev_chain = previous.chunk_hashes
        divergence = _common_prefix_length(prev_chain, new_chain)
        if previous.request_id == request_id:
            return SequenceTransition(
                TransitionKind.REPEAT, divergence, len(prev_chain), ()
            )
        kind = (
            TransitionKind.APPEND
            if divergence == len(prev_chain)
            else TransitionKind.REWRITE
        )
        return SequenceTransition(
            kind, divergence, len(prev_chain), prev_chain[divergence:]
        )

    def num_tracked_sequences(self) -> int:
        """Return how many sequences currently have a tracked chain."""
        with self._lock:
            return len(self._entries)

    def num_tracked_chunks(self) -> int:
        """Return the total chunk hashes held across tracked chains."""
        with self._lock:
            return self._num_chunks

    def _shrink_to_budget_locked(self) -> None:
        """Forget the oldest sequences until the budget holds; caller holds lock.

        The newest entry fits the budget on its own, so it is never dropped.
        """
        while self._num_chunks > self._max_chunks:
            _, entry = self._entries.popitem(last=False)
            self._num_chunks -= len(entry.chunk_hashes)
