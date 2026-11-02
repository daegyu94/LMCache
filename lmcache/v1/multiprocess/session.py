# SPDX-License-Identifier: Apache-2.0
"""
Session and SessionManager for tracking per-request state
in the multiprocess cache server.
"""

# Standard
from collections import OrderedDict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Optional, overload
import threading
import time

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.multiprocess.cache_generation import CacheGeneration, HashScope
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.token_codec import TOKEN_STRIDE, num_packed_tokens
from lmcache.v1.multiprocess.token_hasher import ChunkHash, TokenHasher
from lmcache.v1.periodic_thread import (
    PeriodicThread,
    PeriodicThreadRegistry,
    ThreadLevel,
    ThreadRunSummary,
    create_periodic_thread,
)

logger = init_logger(__name__)


_MAX_LOOKUP_HISTORY = 16
"""Earlier lookups per session whose lock scope stays resolvable for
``free_lookup_locks`` extra ranges."""

_MAX_REAPED_STORE_SEEDS = 65536
"""TTL-reaped sessions whose store seed is remembered in generation clear
mode, so a recreated session keeps storing in its original scope."""


@dataclass
class _ChunkHashChain:
    """Incrementally computed rolling chunk hashes from one seed."""

    prefix_hash: ChunkHash
    hashes: list[ChunkHash] = field(default_factory=list)

    def extend_to(self, hasher: TokenHasher, tokens: bytes, end_chunk: int) -> None:
        """Hash ``tokens`` up to (not including) ``end_chunk``, reusing the
        chunks already hashed."""
        stride = hasher.chunk_size * TOKEN_STRIDE
        # Slicing a memoryview does not copy; slicing the bytes would.
        view = memoryview(tokens)
        while len(self.hashes) < end_chunk:
            cs = len(self.hashes) * stride
            prefix = self.hashes[-1] if self.hashes else self.prefix_hash
            self.hashes.append(hasher.hash_packed_chunk(view[cs : cs + stride], prefix))


@dataclass
class Session:
    """Tracks accumulated token IDs and computed chunk hashes for a request.

    Tokens are held in the packed form of
    :mod:`lmcache.v1.multiprocess.token_codec`, the layout that arrives on
    the wire and that blake3 hashes; only callers needing ints unpack.

    Chunk hashes live in two scopes (identical outside generation clear
    mode, see ``docs/design/v1/multiprocess/cache_generation.md``):

    - **Lookup scope** (:meth:`get_hashes`): the generation of the latest
      lookup. RETRIEVE, lock release and touches address it, matching the
      keys that lookup prefetched and locked.
    - **Store scope** (:meth:`get_store_hashes`): fixed for the session's
      life by ``store_prefix_hash``. A session never moves its stores to a
      newer generation, because a store arriving after a bump may carry KV
      computed before it.

    Thread-safe: all public methods are protected by an internal lock
    to allow concurrent access from multiple TP worker threads.
    """

    request_id: str
    hasher: TokenHasher
    store_prefix_hash: ChunkHash
    """Seed of every store key this session resolves; never changes."""
    lookup_scope: HashScope
    """Generation and seed of the latest lookup (the session's creation
    scope until one runs)."""
    tokens: bytes = b""
    created_at: float = field(default_factory=time.time)
    lookup_ipc_key: Optional[IPCCacheServerKey] = None
    prefetch_hit_chunks: int = -1
    prefetch_locked_gids: tuple = ()
    prefetch_group_windows: tuple[int, ...] = ()
    _prefetch_owners: dict[ObjectKey, int] = field(default_factory=dict, repr=False)
    extras: dict[str, Any] = field(default_factory=dict)
    _lookup_generation: int = field(default=0, repr=False)
    _failed_retrieve_releases: set[tuple[int, int, int, int, int]] = field(
        default_factory=set, repr=False
    )
    # (hit_chunks, scope) of lookups whose result was recorded, newest last.
    _lookup_history: deque[tuple[int, HashScope]] = field(
        default_factory=lambda: deque(maxlen=_MAX_LOOKUP_HISTORY), repr=False
    )
    # Every scope a lookup of this session ran under (one per generation).
    _lookup_scopes_seen: set[HashScope] = field(default_factory=set, repr=False)
    # Scopes of history entries that aged out; any of them may have had the
    # hit an abandoned range names, so they keep a match ambiguous.
    _aged_out_scopes: set[HashScope] = field(default_factory=set, repr=False)
    _store_chain: _ChunkHashChain = field(init=False, repr=False)
    _lookup_chain: _ChunkHashChain = field(init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self) -> None:
        self._store_chain = _ChunkHashChain(self.store_prefix_hash)
        self._lookup_chain = self._chain_for(self.lookup_scope)

    @property
    def num_chunks_processed(self) -> int:
        """Lookup-scope chunks hashed and cached so far."""
        with self._lock:
            return len(self._lookup_chain.hashes)

    def set_tokens(self, packed_token_ids: bytes) -> None:
        """Update the token sequence (idempotent, replaces not extends).

        Args:
            packed_token_ids: Complete packed token sequence, starting at
                position 0.
        """
        with self._lock:
            self.tokens = packed_token_ids

    @overload
    def get_hashes(self, start: int, end: int) -> list: ...

    @overload
    def get_hashes(self, start: int) -> list: ...

    def get_hashes(self, start: int, end: int | None = None) -> list:
        """Compute and return lookup-scope chunk hashes for [start, end).

        Internally computes rolling hashes up to end_chunk, skipping
        already-computed chunks. These are the keys the latest lookup
        addressed; store paths must use :meth:`get_store_hashes`.

        Two calling conventions are supported (declared via ``@overload``)::

            get_hashes(start, end)  # explicit end
            get_hashes(start)       # end = last full-chunk boundary

        Args:
            start: Start token index (must be aligned to chunk_size).
            end: End token index (must be aligned to chunk_size).
                When omitted (``None``), automatically set to the last
                full-chunk boundary of the current token sequence.

        Returns:
            List of hash values for chunks in [start_chunk, end_chunk).

        Raises:
            ValueError: If an explicit ``end`` exceeds the current tokens
                (hashing past them yields valid-looking garbage).
        """
        with self._lock:
            return self._hashes_locked(self._lookup_chain, start, end)

    def get_store_hashes(self, start: int, end: int) -> list:
        """Compute and return store-scope chunk hashes for [start, end).

        Same contract as :meth:`get_hashes` with an explicit ``end``, but
        seeded with ``store_prefix_hash``, so every store of this session
        lands in the scope it was created in, however many bumps or lookups
        happened since.

        Args:
            start: Start token index (must be aligned to chunk_size).
            end: End token index (must be aligned to chunk_size).

        Returns:
            List of hash values for chunks in [start_chunk, end_chunk).

        Raises:
            ValueError: If ``end`` exceeds the current tokens.
        """
        with self._lock:
            return self._hashes_locked(self._store_chain, start, end)

    def begin_lookup(
        self,
        key: IPCCacheServerKey,
        group_windows: tuple[int, ...],
        scope: HashScope,
    ) -> None:
        """Record a new lookup and reset its per-lookup release state.

        Args:
            key: The lookup's IPC key.
            group_windows: Sliding window of each object group, in order.
            scope: The scope the lookup hashed under. Later RETRIEVEs and
                lock releases of this lookup resolve keys in it; the store
                scope is unaffected.
        """
        with self._lock:
            if scope != self.lookup_scope:
                self.lookup_scope = scope
                self._lookup_chain = self._chain_for(scope)
            self._lookup_scopes_seen.add(scope)
            self.lookup_ipc_key = key
            self.prefetch_hit_chunks = -1
            self.prefetch_locked_gids = ()
            self.prefetch_group_windows = group_windows
            self._lookup_generation += 1
            self._failed_retrieve_releases.clear()
            self._prefetch_owners.clear()

    def abandoned_lookup_scope(self, hit_chunks: int) -> HashScope | None:
        """Return the scope an earlier lookup locked a hit range under.

        ``free_lookup_locks`` extra ranges name an abandoned lookup only by
        its hit boundary, and that lookup may predate a bump. The scope is
        taken from recorded lookups with the same hit, together with the
        scopes of entries that aged out of the bounded history (any of which
        may have had that hit); if none is recorded, from the session's only
        lookup scope.

        Args:
            hit_chunks: The abandoned lookup's hit boundary, in chunks.

        Returns:
            The unique matching scope, or ``None`` when the candidates span
            several generations. A caller must then not release the range:
            L1 locks are anonymous refcounts, and resolving the wrong
            generation would decrement other readers' locks.
        """
        with self._lock:
            matches = {s for hit, s in self._lookup_history if hit == hit_chunks}
            if matches:
                matches |= self._aged_out_scopes
            else:
                matches = set(self._lookup_scopes_seen)
            return next(iter(matches)) if len(matches) == 1 else None

    def record_prefetch_result(
        self,
        hit_chunks: int,
        locked_gids: tuple[int, ...],
        l1_owners: dict[ObjectKey, int] | None = None,
    ) -> None:
        """Record the lock set acquired by the current lookup."""
        with self._lock:
            if hit_chunks > 0:
                if len(self._lookup_history) == self._lookup_history.maxlen:
                    self._aged_out_scopes.add(self._lookup_history[0][1])
                self._lookup_history.append((hit_chunks, self.lookup_scope))
            self.prefetch_hit_chunks = hit_chunks
            self.prefetch_locked_gids = locked_gids
            self._prefetch_owners = dict(l1_owners) if l1_owners is not None else {}

    def get_prefetch_owners(self) -> dict[ObjectKey, int]:
        """Snapshot the L1 owners whose read locks this session's lookup retained."""
        with self._lock:
            return self._prefetch_owners.copy()

    def prepare_failed_retrieve_release(
        self,
        key: IPCCacheServerKey,
    ) -> tuple[int, tuple[int, ...], tuple[int, ...], int, HashScope] | None:
        """Return a stable snapshot for a failed worker's lock release.

        Returns:
            ``(hit_chunks, locked_gids, group_windows, lookup_generation,
            lookup_scope)`` of the current lookup -- ``lookup_generation``
            being the per-session lookup counter checked by
            :meth:`claim_failed_retrieve_release` and ``lookup_scope`` the
            cache generation the locks were taken under -- or ``None`` when
            the session cannot prove ownership of ``key``'s range.
        """
        if key.worker_id is None:
            return None

        with self._lock:
            lookup_key = self.lookup_ipc_key
            hit_chunks = self.prefetch_hit_chunks
            if lookup_key is None or hit_chunks < 0:
                return None

            same_lookup = (
                key.model_name == lookup_key.model_name
                and key.world_size == lookup_key.world_size
                and key.token_bytes == lookup_key.token_bytes
                and key.cache_salt == lookup_key.cache_salt
                and key.start >= lookup_key.start
                and key.end <= lookup_key.end
            )
            if not same_lookup:
                return None

            return (
                hit_chunks,
                self.prefetch_locked_gids,
                self.prefetch_group_windows,
                self._lookup_generation,
                self.lookup_scope,
            )

    def claim_failed_retrieve_release(
        self,
        instance_id: int,
        key: IPCCacheServerKey,
        lookup_generation: int,
    ) -> bool:
        """Atomically claim one failed worker's prepared lock release.

        A scheduler lookup acquires read locks for every KV worker (or one
        reader count per TP worker for MLA), while RETRIEVE responses are
        per worker instance.  When a worker has lost its GPU registration,
        only that instance's share may be released.  Claiming after key
        resolution makes duplicate failed RETRIEVEs idempotent without
        consuming the claim when resolution itself fails.

        ``False`` is also returned when the session cannot prove ownership of
        the requested range.  In that case it is safer to leave the lock to
        its TTL than to decrement a concurrent request's anonymous L1 count.
        """
        if key.worker_id is None:
            return False

        with self._lock:
            lookup_key = self.lookup_ipc_key
            if lookup_key is None or lookup_generation != self._lookup_generation:
                return False

            same_lookup = (
                key.model_name == lookup_key.model_name
                and key.world_size == lookup_key.world_size
                and key.token_bytes == lookup_key.token_bytes
                and key.cache_salt == lookup_key.cache_salt
                and key.start >= lookup_key.start
                and key.end <= lookup_key.end
            )
            if not same_lookup:
                return False

            owner = (
                lookup_generation,
                instance_id,
                key.worker_id,
                key.start,
                key.end,
            )
            if owner in self._failed_retrieve_releases:
                return False
            self._failed_retrieve_releases.add(owner)
            return True

    def _chain_for(self, scope: HashScope) -> _ChunkHashChain:
        """Share the store chain when ``scope`` seeds identically to it."""
        if scope.prefix_hash == self.store_prefix_hash:
            return self._store_chain
        return _ChunkHashChain(scope.prefix_hash)

    def _hashes_locked(
        self, chain: _ChunkHashChain, start: int, end: int | None
    ) -> list[ChunkHash]:
        """Return ``chain``'s hashes for [start, end); caller holds the lock."""
        chunk_size = self.hasher.chunk_size
        assert start % chunk_size == 0, (
            f"start ({start}) must be a multiple of chunk_size ({chunk_size})"
        )
        held = num_packed_tokens(self.tokens)
        if end is not None and end > held:
            raise ValueError(
                f"get_hashes end ({end}) exceeds the session's "
                f"{held} token(s); the session may have "
                "been recreated after request cleanup"
            )
        if end is None:
            # No explicit end: use the last full-chunk boundary.
            end = held - (held % chunk_size)
        assert end % chunk_size == 0, (
            f"end ({end}) must be a multiple of chunk_size ({chunk_size})"
        )
        end_chunk = end // chunk_size
        chain.extend_to(self.hasher, self.tokens, end_chunk)
        return chain.hashes[start // chunk_size : end_chunk]


class SessionManager:
    """Thread-safe manager for per-request sessions."""

    DEFAULT_SESSION_TTL = 600  # 10 minutes
    DEFAULT_CLEANUP_INTERVAL = 60.0

    def __init__(
        self,
        hasher: TokenHasher,
        *,
        cache_generation: CacheGeneration,
        ttl: float = DEFAULT_SESSION_TTL,
        cleanup_interval: float | None = DEFAULT_CLEANUP_INTERVAL,
    ) -> None:
        """Create a session manager.

        Args:
            hasher: Hasher shared by every session.
            cache_generation: The server's generation counter; scopes the
                hashes of every session created here.
            ttl: Seconds after creation before a session is reaped.
            cleanup_interval: Seconds between reaper runs; ``None`` or a
                non-positive value disables the reaper thread.
        """
        self._hasher = hasher
        self._cache_generation = cache_generation
        # Generation mode only: store seeds of TTL-reaped sessions, oldest
        # first, bounded by _MAX_REAPED_STORE_SEEDS. A request outliving the
        # TTL keeps storing in the scope its lookup proved (or stays in its
        # old generation across a bump) instead of becoming untrusted.
        self._reaped_store_seeds: OrderedDict[str, ChunkHash] = OrderedDict()
        self._ttl = ttl
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()
        #: Callbacks invoked for every destroyed session, so owners of
        #: per-request state (e.g. entries in ``Session.extras``) can release
        #: it however the request ended. See :meth:`_notify_destroyed`.
        self.destroy_listeners: list[Callable[[Session], None]] = []
        self._cleanup_interval = cleanup_interval
        self._cleanup_thread: PeriodicThread | None = None
        if cleanup_interval is not None and cleanup_interval > 0:
            self._cleanup_thread = self._create_cleanup_thread(cleanup_interval)
            self._cleanup_thread.start()

    def get_or_create(self, request_id: str) -> Session:
        """Get existing session or create one outside a lookup.

        A session created here has no lookup proving which generation its
        KV was computed in (it may be a store arriving after its lookup's
        session was reaped). In generation clear mode its stores are
        therefore seeded with the untrusted-store seed, which no lookup
        uses -- unless the request's earlier session was reaped by TTL,
        whose store seed it inherits; its lookup scope is the current
        generation. In ``"l1"`` mode both are ``none_hash``.

        Args:
            request_id: Unique request identifier.

        Returns:
            The Session for this request_id.
        """
        with self._lock:
            session = self._sessions.get(request_id)
            if session is None:
                session = Session(
                    request_id=request_id,
                    hasher=self._hasher,
                    store_prefix_hash=self._reaped_store_seeds.pop(
                        request_id,
                        self._cache_generation.untrusted_store_prefix_hash(
                            self._hasher
                        ),
                    ),
                    lookup_scope=self._cache_generation.current_scope(self._hasher),
                )
                self._sessions[request_id] = session
                logger.debug("Created session for request_id=%s", request_id)
            return session

    def get_or_create_for_lookup(self, request_id: str, scope: HashScope) -> Session:
        """Get existing session or create one for a lookup in ``scope``.

        A session created here stores in ``scope`` for its whole life; an
        existing session keeps its store scope (see :class:`Session`), and so
        does one recreated after TTL reaping in generation clear mode.

        Args:
            request_id: Unique request identifier.
            scope: The scope the lookup hashed under, read once by the
                caller before hashing.

        Returns:
            The Session for this request_id.
        """
        with self._lock:
            session = self._sessions.get(request_id)
            if session is None:
                session = Session(
                    request_id=request_id,
                    hasher=self._hasher,
                    store_prefix_hash=self._reaped_store_seeds.pop(
                        request_id, scope.prefix_hash
                    ),
                    lookup_scope=scope,
                )
                self._sessions[request_id] = session
                logger.debug("Created session for request_id=%s", request_id)
            return session

    def get(self, request_id: str) -> Optional[Session]:
        """Return an existing session without creating ownership state."""
        with self._lock:
            return self._sessions.get(request_id)

    def remove(self, request_id: str) -> Optional[Session]:
        """Remove a session by request_id.

        Args:
            request_id: Unique request identifier.

        Returns:
            The removed session, or None if no session was found.
        """
        with self._lock:
            # The request ended: a later session with this id is unrelated.
            self._reaped_store_seeds.pop(request_id, None)
            if request_id not in self._sessions:
                return None
            session = self._sessions[request_id]
            del self._sessions[request_id]
            logger.debug("Removed session for request_id=%s", request_id)
        self._notify_destroyed(session)
        return session

    def cleanup_expired(self) -> int:
        """Remove sessions that have exceeded their TTL.

        Returns:
            Number of sessions removed.
        """
        now = time.time()
        expired: list[Session] = []
        with self._lock:
            for session in self._sessions.values():
                if now - session.created_at > self._ttl:
                    expired.append(session)
            for session in expired:
                del self._sessions[session.request_id]
                if self._cache_generation.clear_mode == "generation":
                    self._remember_store_seed_locked(session)

        for session in expired:
            self._notify_destroyed(session)
        if expired:
            logger.info("Cleaned up %d expired sessions", len(expired))
        return len(expired)

    def _remember_store_seed_locked(self, session: Session) -> None:
        """Record a reaped session's store seed; caller holds the lock."""
        self._reaped_store_seeds[session.request_id] = session.store_prefix_hash
        self._reaped_store_seeds.move_to_end(session.request_id)
        while len(self._reaped_store_seeds) > _MAX_REAPED_STORE_SEEDS:
            self._reaped_store_seeds.popitem(last=False)

    def _notify_destroyed(self, session: Session) -> None:
        """Run every destroy listener for one removed session.

        Called without the manager lock held: a listener may take other locks
        (e.g. the storage manager's). A listener that raises is logged and
        skipped so destruction always completes.

        Args:
            session: The session that was just removed.
        """
        for listener in list(self.destroy_listeners):
            try:
                listener(session)
            except Exception:
                logger.exception(
                    "Session destroy listener failed for request_id=%s",
                    session.request_id,
                )

    def active_count(self) -> int:
        """Return the number of active sessions.

        Returns:
            Number of currently tracked sessions.
        """
        with self._lock:
            return len(self._sessions)

    def close(self) -> None:
        """Stop the background cleanup thread and unregister it."""
        if self._cleanup_thread is None:
            return

        PeriodicThreadRegistry.get_instance().unregister(self._cleanup_thread.name)
        self._cleanup_thread.stop()
        self._cleanup_thread = None

    def _create_cleanup_thread(self, cleanup_interval: float) -> PeriodicThread:
        def execute_cleanup() -> ThreadRunSummary:
            removed = self.cleanup_expired()
            return ThreadRunSummary(
                success=True,
                message=f"Removed {removed} expired sessions",
                extra_info={"removed_sessions": str(removed)},
            )

        return create_periodic_thread(
            name=f"SessionManager-cleanup-thread-{id(self):x}",
            interval=cleanup_interval,
            execute_fn=execute_cleanup,
            level=ThreadLevel.MEDIUM,
        )
