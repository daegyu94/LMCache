# SPDX-License-Identifier: Apache-2.0
"""Version-qualified cache keys for bounded-staleness KV reuse.

A request may declare which producer version its KV belongs to and how much
older KV it accepts:

- ``lmcache.kv_version`` (int >= 0): the version of the KV this request
  computes, e.g. the weight version an engine serves it with.
- ``lmcache.max_kv_staleness`` (int >= 0, default 0): accept cached KV whose
  version is in ``[kv_version - max_kv_staleness, kv_version]``.

A request with ``lmcache.kv_version`` addresses every chunk under a
version-qualified chunk hash (:func:`versioned_chunk_hash`), so the version is
part of the object identity in L1 and in every L2 adapter, and KV of one
version can neither overwrite nor be served as another. A request without the
hint keeps the plain chunk hashes, so its keys are exactly today's.

The lookup probes the window newest-first. A served prefix is a sequence of
runs (:class:`KVVersionRun`) whose versions never increase along the prompt;
:class:`KVVersionAssignment` maps every chunk of a request to the version it
is addressed under for retrieve, lock release and store.

This module is pure: no I/O and no server state.
"""

# Standard
from collections.abc import Mapping
from dataclasses import dataclass
import enum
import hashlib
import threading
import time

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)

KV_VERSION_CONFIG = "lmcache.kv_version"
"""Request config key: the version of the KV the request computes."""

MAX_KV_STALENESS_CONFIG = "lmcache.max_kv_staleness"
"""Request config key: how many versions older than ``kv_version`` to accept."""

MAX_KV_STALENESS_LIMIT = 16
"""Upper bound on ``lmcache.max_kv_staleness``. A lookup costs up to
``max_kv_staleness + 1`` prefetch rounds, so the bound caps lookup latency."""

_MAX_KV_VERSION = (1 << 63) - 1
# blake2b personalization (at most 16 bytes) that separates version-qualified
# hashes from every other use of the chunk hash.
_VERSIONED_HASH_PERSON = b"lmcache.kv_ver"


class KVVersionHintProblem(enum.Enum):
    """A malformed or unsupported version hint and how the server treats it.

    The values are log messages; they never contain the client's raw value.
    """

    INVALID_VERSION = (
        "lmcache.kv_version is not an int in [0, 2^63); the request is served "
        "without version isolation"
    )
    STALENESS_WITHOUT_VERSION = (
        "lmcache.max_kv_staleness is set without lmcache.kv_version; ignored"
    )
    INVALID_STALENESS = (
        "lmcache.max_kv_staleness is not an int >= 0; only the exact "
        "kv_version is accepted"
    )
    STALENESS_CAPPED = (
        f"lmcache.max_kv_staleness exceeds {MAX_KV_STALENESS_LIMIT}; capped"
    )
    STALENESS_UNSUPPORTED_LAYOUT = (
        "lmcache.max_kv_staleness needs every object group to use full "
        "attention; only the exact kv_version is accepted"
    )


@dataclass(frozen=True)
class KVVersionWindow:
    """The KV versions one request accepts.

    Attributes:
        version: The version of the KV the request computes; KV it stores is
            stamped with at most this version.
        max_staleness: How many versions older than ``version`` the request
            accepts, ``0 <= max_staleness <= MAX_KV_STALENESS_LIMIT``.
    """

    version: int
    max_staleness: int

    @property
    def oldest(self) -> int:
        """The oldest accepted version; versions never go below 0."""
        return max(0, self.version - self.max_staleness)

    def contains(self, version: int) -> bool:
        """Return whether KV of ``version`` may be served to this request.

        Args:
            version: A KV version.

        Returns:
            True when ``oldest <= version <= self.version``.
        """
        return self.oldest <= version <= self.version


@dataclass(frozen=True)
class KVVersionHints:
    """The result of parsing a request's version hints.

    Attributes:
        window: The accepted versions, or None when the request carries no
            usable ``lmcache.kv_version`` (its keys stay unversioned).
        problems: Every malformed hint found, in check order; empty when the
            hints are well formed or absent.
    """

    window: KVVersionWindow | None
    problems: tuple[KVVersionHintProblem, ...]


@dataclass(frozen=True)
class KVVersionRun:
    """A run of consecutive chunks served from one KV version.

    Attributes:
        start_chunk: First chunk of the run (absolute chunk index).
        end_chunk: One past the last chunk, ``end_chunk > start_chunk``.
        version: The KV version every chunk of the run is cached under.
    """

    start_chunk: int
    end_chunk: int
    version: int


@dataclass(frozen=True)
class KVVersionAssignment:
    """Maps every chunk of a versioned request to the version it addresses.

    Chunks inside a served run keep that run's version, so retrieve and lock
    release address exactly the objects the lookup locked, and a re-store of
    a served chunk hits the existing object instead of re-stamping it. Every
    other chunk is addressed under ``write_version``.

    Attributes:
        runs: Served runs in chunk order, non-overlapping; empty when the
            lookup served nothing (or has not run).
        write_version: The version stored KV is stamped with:
            ``min(kv_version, oldest served version)``. A chunk computed on
            top of older cached KV attends to it, so it is stamped with the
            oldest version that influenced it; staleness therefore cannot
            accumulate across turns that each reuse the previous turn's KV.
    """

    runs: tuple[KVVersionRun, ...]
    write_version: int

    @classmethod
    def for_window(
        cls, window: KVVersionWindow, runs: tuple[KVVersionRun, ...]
    ) -> "KVVersionAssignment":
        """Build the assignment of a request from its window and served runs.

        Args:
            window: The request's accepted versions.
            runs: The runs its lookup served, in chunk order.

        Returns:
            The assignment, with ``write_version`` derived from ``window`` and
            the oldest served run.
        """
        return cls(
            runs=runs,
            write_version=min([window.version, *(run.version for run in runs)]),
        )

    def version_of(self, chunk: int) -> int:
        """Return the version chunk ``chunk`` is addressed under.

        Args:
            chunk: Absolute chunk index.

        Returns:
            The covering run's version, else ``write_version``.
        """
        for run in self.runs:
            if run.start_chunk <= chunk < run.end_chunk:
                return run.version
        return self.write_version

    def apply(self, chunk_hashes: list[bytes], start_chunk: int) -> list[bytes]:
        """Version-qualify consecutive chunk hashes.

        Args:
            chunk_hashes: Unversioned hashes of chunks ``start_chunk``,
                ``start_chunk + 1``, ...
            start_chunk: Absolute index of ``chunk_hashes[0]``.

        Returns:
            One versioned hash per input hash, in the same order.
        """
        return [
            versioned_chunk_hash(chunk_hash, self.version_of(start_chunk + i))
            for i, chunk_hash in enumerate(chunk_hashes)
        ]


class KVVersionHintWarner:
    """Rate-limited warnings for malformed version hints.

    Hints come from clients, so one misconfigured client could otherwise
    flood the server log. Each problem is reported at most once per
    ``interval_s`` with the number of occurrences since its last report; the
    raw client value is never logged.

    Thread-safe.
    """

    def __init__(self, interval_s: float) -> None:
        """Create the warner.

        Args:
            interval_s: Minimum seconds between two reports of one problem.
        """
        self._interval_s = interval_s
        self._lock = threading.Lock()
        # Per problem: occurrences since the last report, and when it was.
        self._pending: dict[KVVersionHintProblem, int] = {}
        self._last_report: dict[KVVersionHintProblem, float] = {}

    def parse(
        self, request_configs: Mapping[str, object] | None
    ) -> KVVersionWindow | None:
        """Parse a request's hints and report any problem found.

        Args:
            request_configs: The request's ``lmcache.*`` entries, or None.

        Returns:
            The window of :func:`parse_kv_version_hints`.
        """
        hints = parse_kv_version_hints(request_configs)
        if hints.problems:
            self.warn(hints.problems)
        return hints.window

    def warn(self, problems: tuple[KVVersionHintProblem, ...]) -> None:
        """Count ``problems`` and log those whose interval has passed.

        Args:
            problems: Problems found in one request's hints.
        """
        now = time.monotonic()
        due: list[tuple[KVVersionHintProblem, int]] = []
        with self._lock:
            for problem in problems:
                self._pending[problem] = self._pending.get(problem, 0) + 1
                last = self._last_report.get(problem)
                if last is None or now - last >= self._interval_s:
                    due.append((problem, self._pending.pop(problem)))
                    self._last_report[problem] = now
        for problem, count in due:
            logger.warning(
                "Malformed KV version hint: %s (%d request(s) since last report)",
                problem.value,
                count,
            )


def versioned_chunk_hash(chunk_hash: bytes, version: int) -> bytes:
    """Return the hash a chunk is addressed under for one KV version.

    The digest keeps the input length, so adapters that size or encode keys
    by hash length see no difference.

    Args:
        chunk_hash: The chunk's unversioned hash (1 to 64 bytes).
        version: The KV version, ``0 <= version < 2**63``.

    Returns:
        A hash that differs per version and from the unversioned hash.
    """
    digest = hashlib.blake2b(
        chunk_hash, digest_size=len(chunk_hash), person=_VERSIONED_HASH_PERSON
    )
    digest.update(version.to_bytes(8, byteorder="big"))
    return digest.digest()


def _as_int(value: object) -> int | None:
    """Return ``value`` if it is an int and not a bool, else None.

    JSON ``true`` decodes to a bool, which must not count as version 1.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def parse_kv_version_hints(
    request_configs: Mapping[str, object] | None,
) -> KVVersionHints:
    """Parse and validate a request's version hints.

    A malformed ``lmcache.kv_version`` disables versioning for the request (it
    is served as if it sent no hint). A malformed ``lmcache.max_kv_staleness``
    narrows the window to the exact version, and one above
    :data:`MAX_KV_STALENESS_LIMIT` is capped: both stay inside what the
    client asked for.

    Args:
        request_configs: The request's ``lmcache.*`` entries, or None.

    Returns:
        The window (None without a usable version) and every problem found.
    """
    configs = request_configs or {}
    has_staleness = MAX_KV_STALENESS_CONFIG in configs
    if KV_VERSION_CONFIG not in configs:
        problems = (
            (KVVersionHintProblem.STALENESS_WITHOUT_VERSION,) if has_staleness else ()
        )
        return KVVersionHints(window=None, problems=problems)

    version = _as_int(configs[KV_VERSION_CONFIG])
    if version is None or not 0 <= version <= _MAX_KV_VERSION:
        return KVVersionHints(
            window=None, problems=(KVVersionHintProblem.INVALID_VERSION,)
        )

    staleness = _as_int(configs.get(MAX_KV_STALENESS_CONFIG, 0))
    found: tuple[KVVersionHintProblem, ...] = ()
    if staleness is None or staleness < 0:
        staleness = 0
        found = (KVVersionHintProblem.INVALID_STALENESS,)
    elif staleness > MAX_KV_STALENESS_LIMIT:
        staleness = MAX_KV_STALENESS_LIMIT
        found = (KVVersionHintProblem.STALENESS_CAPPED,)
    return KVVersionHints(
        window=KVVersionWindow(version=version, max_staleness=staleness),
        problems=found,
    )
