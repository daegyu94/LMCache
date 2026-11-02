# SPDX-License-Identifier: Apache-2.0
"""LookupModule: lookup, prefetch polling, and session lifecycle."""

# Standard
from dataclasses import dataclass, field, replace
from functools import partial
import threading
import time

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import (
    DEFAULT_ATTN_WINDOW_DESC,
    FULL_ATTENTION_WINDOW_CHUNKS,
    AttnWindowDesc,
    MemoryLayoutDesc,
    ObjectKey,
    PrefetchHandle,
    PrefetchResult,
    PrefetchTaskSpec,
    ipc_key_to_grouped_object_keys,
    ipc_key_to_object_keys,
)
from lmcache.v1.distributed.bitmap_ops.fold import fold_unfold_grouped
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.otel_init import register_gauge
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.engine_context import (
    KV_VERSION_HINT_WARN_INTERVAL_S,
    MPCacheServerContext,
)
from lmcache.v1.multiprocess.engine_module import (
    DiscoverableModule,
    EngineModule,
    ModuleBuildContext,
)
from lmcache.v1.multiprocess.kv_load_policy import (
    DefaultKVLoadPolicy,
    KVLoadContext,
    KVLoadPolicy,
    create_kv_load_policy,
)
from lmcache.v1.multiprocess.kv_version import (
    KVVersionAssignment,
    KVVersionHintProblem,
    KVVersionHintWarner,
    KVVersionRun,
    KVVersionWindow,
    parse_kv_version_hints,
    versioned_chunk_hash,
)
from lmcache.v1.multiprocess.request_handler import HandlerType, request_handler
from lmcache.v1.multiprocess.token_codec import num_packed_tokens, unpack_token_ids
from lmcache.v1.multiprocess.token_hasher import TokenHasher

logger = init_logger(__name__)

# Stateless, so one instance can serve every LookupModule.
_DEFAULT_LOAD_POLICY = DefaultKVLoadPolicy({})


def resolve_prefetched_obj_keys(
    ctx: MPCacheServerContext,
    key: IPCCacheServerKey,
    hit_chunks: int,
    locked_gids: tuple,
    *,
    kv_versions: KVVersionAssignment | None,
    group_windows: tuple[int, ...] | None = None,
) -> list[ObjectKey]:
    """Resolve the subset of a request range that lookup actually locked.

    ``key.worker_id=None`` resolves every KV rank for scheduler-owned cleanup.
    A worker-specific key resolves only that worker's shard (or one MLA reader
    share), which is required for per-instance RETRIEVE failure cleanup.

    ``kv_versions`` must describe the lookup that took the locks: None for an
    unversioned request, else the version runs that lookup served (see
    :meth:`Session.kv_version_assignment`), so the released keys are exactly
    the version-qualified keys it locked.
    """
    chunk_hashes = ctx.token_hasher.compute_packed_chunk_hashes(
        key.token_bytes, start=key.start, end=key.end
    )
    if not chunk_hashes:
        return []

    start_chunk = key.start // ctx.chunk_size
    if kv_versions is not None:
        chunk_hashes = kv_versions.apply(chunk_hashes, start_chunk)
    end_chunk = start_chunk + len(chunk_hashes)
    if group_windows is None:
        group_windows = tuple(
            ctx.layout_desc_registry.find_attn_desc(
                key.model_name, key.world_size
            ).num_chunks_in_sw
        )

    obj_keys: list[ObjectKey] = []
    for group_idx, window in enumerate(group_windows):
        if locked_gids and group_idx not in locked_gids:
            continue
        if hit_chunks < 0:
            if window >= 0:
                continue
            lo, hi = start_chunk, end_chunk
        else:
            # Locked range per ``unfold``: the whole hit prefix for full
            # attention, its trailing ``window`` chunks otherwise.
            lo = 0 if window < 0 else max(0, hit_chunks - window)
            lo = max(lo, start_chunk)
            hi = min(hit_chunks, end_chunk)
        if lo >= hi:
            continue
        group_hashes = chunk_hashes[lo - start_chunk : hi - start_chunk]
        obj_keys.extend(ipc_key_to_object_keys(key, group_hashes, [group_idx])[0])
    return obj_keys


@dataclass
class _KVVersionProbe:
    """A versioned lookup that probes its window newest-first.

    Each prefetch round looks up chunks ``[p, n)`` under one version, where
    ``p`` is where the earlier rounds' hits ended, and the next round lowers
    the version by one. A lookup therefore runs at most
    ``max_staleness + 1`` rounds, and the served versions never increase
    along the prompt. Only one round is in flight at a time; the thread that
    consumes a round's result is the only one that mutates the probe.
    """

    window: KVVersionWindow
    key: IPCCacheServerKey
    # Unversioned chunk hashes of the whole lookup, from chunk 0.
    chunk_hashes: list[bytes]
    group_layout_descs: dict[int, MemoryLayoutDesc]
    attn_desc: AttnWindowDesc
    num_kv_readers: int
    # Version the in-flight round looks up.
    version: int
    # Accumulated over finished rounds.
    runs: list[KVVersionRun] = field(default_factory=list)
    l1_chunks: int = 0
    l2_chunks: int = 0
    l1_hit_keys: int = 0
    l2_hit_keys: int = 0
    l1_owners: dict[ObjectKey, int] = field(default_factory=dict)

    def add_round(
        self,
        start_chunk: int,
        end_chunk: int,
        l1_chunks: int,
        l2_chunks: int,
        result: PrefetchResult,
    ) -> None:
        """Account one finished round that hit ``[start_chunk, end_chunk)``."""
        if end_chunk > start_chunk:
            self.runs.append(KVVersionRun(start_chunk, end_chunk, self.version))
        self.l1_chunks += l1_chunks
        self.l2_chunks += l2_chunks
        self.l1_hit_keys += result.l1_hit_count
        self.l2_hit_keys += result.l2_hit_count
        self.l1_owners.update(result.l1_owners)


@dataclass
class _PrefetchJob:
    handle: PrefetchHandle
    # Sliding window of each submitted key row, in row order; empty when no
    # prefetch was submitted.
    row_windows: tuple[int, ...]
    request_id: str
    # Chunk-aligned tokens LMCache was asked to look up, past the engine-covered
    # prefix (hit-rate denominator); 0 on early exits.
    requested_tokens: int
    num_object_groups: int = 1
    attn_desc: AttnWindowDesc = DEFAULT_ATTN_WINDOW_DESC
    # First chunk of the looked-up range (``key.start``); chunks below it are
    # engine-covered and skipped, so the status offsets hits back to absolute.
    start_chunk: int = 0
    # Engine-held chunks below ``key.start``, reported as ``apc_covered_tokens``;
    # equals ``start_chunk`` except on a declined lookup, which offsets no hits.
    engine_covered_chunks: int = 0
    # Request-level view on every path: the whole prompt and the engine's own
    # prefix-cache hit from the ``engine_computed_tokens`` hint (0 without it).
    prompt_tokens: int = 0
    engine_hit_tokens: int = 0
    # Captured at lookup time so the ``MP_LOOKUP_PREFETCH_END`` event can
    # carry them as labels.  ``model_name`` lets dashboards slice hit rate
    # per model in multi-model deployments; ``cache_salt`` slices per
    # tenant / isolation domain (an empty string means no salt set).
    model_name: str = ""
    cache_salt: str = ""
    # Names the ``lookup()`` branch that returned before submitting a prefetch
    # task; empty on the normal path.
    early_exit_reason: str = ""
    # Set for a request with ``lmcache.kv_version``; ``handle`` is then the
    # in-flight round and ``start_chunk`` that round's first chunk.
    kv_probe: _KVVersionProbe | None = None


class LookupModule(DiscoverableModule):
    """Handles lookup, prefetch polling, lock release, and session lifecycle.

    Owns the prefetch-job bookkeeping (``_prefetch_jobs``) and exposes
    handlers for the LOOKUP, QUERY_PREFETCH_STATUS, FREE_LOOKUP_LOCKS, and
    END_SESSION request types.

    Args:
        ctx: Shared engine context providing storage manager, token hasher,
            session manager, event bus, layout descriptor registry, and
            chunk size.
        load_policy: Decides per lookup whether to serve the request or let
            the engine recompute it. Defaults to ``DefaultKVLoadPolicy``.
    """

    module_name = "lookup"

    @classmethod
    def create(cls, build_ctx: ModuleBuildContext) -> EngineModule | None:
        """Build the lookup module; it applies to every configuration."""
        return cls(
            build_ctx.engine_context,
            create_kv_load_policy(
                build_ctx.mp_config.kv_load_policy,
                build_ctx.mp_config.runtime_plugin_config.extra_config,
            ),
        )

    def __init__(
        self,
        ctx: MPCacheServerContext,
        load_policy: KVLoadPolicy = _DEFAULT_LOAD_POLICY,
    ) -> None:
        self._ctx = ctx
        self._load_policy = load_policy
        self._prefetch_jobs: dict[str, _PrefetchJob] = {}
        self._prefetch_job_lock = threading.Lock()
        self._kv_version_hint_warner = KVVersionHintWarner(
            KV_VERSION_HINT_WARN_INTERVAL_S
        )
        self._setup_metrics()

    @property
    def context(self) -> MPCacheServerContext:
        """Return the shared engine context. Exposed for testing only."""
        return self._ctx

    def report_status(self) -> dict[str, int]:
        """Return module-specific status information.

        Returns:
            Dictionary with the count of active prefetch jobs.
        """
        return {
            "active_prefetch_jobs": self._active_prefetch_count(),
        }

    def close(self) -> None:
        """Release resources owned by this module (no-op)."""
        pass

    # -----------------------------------------------------------------
    # Handlers
    # -----------------------------------------------------------------

    @request_handler(HandlerType.BLOCKING)
    def lookup(
        self,
        key: IPCCacheServerKey,
        tp_size: int,
    ) -> None:
        """Submit a prefix lookup.

        Hashes the key, submits a prefetch task to the storage manager,
        and registers the job under ``key.request_id`` for later polling
        via query_prefetch_status. When the KV load policy declines the
        request, no prefetch is submitted and the job reports zero hits.

        A key whose request configs carry ``lmcache.kv_version`` looks up
        version-qualified keys, newest version first: each further prefetch
        round (submitted when the previous one is consumed) continues where
        the previous hits ended, one version older, until the prompt is
        covered or the window ``lmcache.max_kv_staleness`` is exhausted. A
        chunk with no version in the window is a miss and ends the prefix.

        Args:
            key: Cache key with request_id embedded.
            tp_size: Legacy wire field; ignored (kept for payload arity).
        """
        model_name, world_size = key.model_name, key.world_size
        self._ctx.event_bus.publish(
            Event(
                event_type=EventType.MP_REQUEST_START,
                session_id=key.request_id,
            )
        )
        self._ctx.event_bus.publish(
            Event(
                event_type=EventType.MP_LOOKUP_PREFETCH_START,
                session_id=key.request_id,
            )
        )

        layout_desc = self._ctx.layout_desc_registry.find(model_name, world_size)
        if layout_desc is None:
            logger.error(
                "No GPU context found for model %s with world size %d during lookup!",
                model_name,
                world_size,
            )
            self._register_empty_prefetch_job(key, early_exit_reason="no_gpu_context")
            return

        num_kv_readers = key.require_num_kv_readers()

        chunk_hashes = self._ctx.token_hasher.compute_packed_chunk_hashes(
            key.token_bytes, end=key.end
        )
        if not chunk_hashes:
            self._register_empty_prefetch_job(
                key, early_exit_reason="empty_chunk_hashes"
            )
            return

        # Total chunk-aligned tokens submitted for lookup; surfaces as the
        # denominator of the L1+L2 token-level hit-rate via the
        # ``requested_tokens`` field on ``MP_LOOKUP_PREFETCH_END``.  Sub-chunk
        # trailing tokens are intentionally excluded — they cannot hit at
        # chunk granularity.
        requested_tokens = len(chunk_hashes) * self._ctx.chunk_size

        # Guard with has_subscribers() to avoid allocating the metadata dict
        # (including dtype/shape list comprehensions) when no subscriber is
        # listening (e.g. lookup hash logger is disabled).
        if self._ctx.event_bus.has_subscribers(EventType.MP_LOOKUP):
            self._ctx.event_bus.publish(
                Event(
                    event_type=EventType.MP_LOOKUP,
                    session_id=key.request_id,
                    metadata={
                        "request_id": key.request_id,
                        "chunk_hashes": chunk_hashes,
                        "model_name": model_name,
                        "chunk_size": self._ctx.chunk_size,
                        "seq_len": key.num_tokens,
                        "dtypes": [str(d) for d in layout_desc.dtypes],
                        "shapes": [list(s) for s in layout_desc.shapes],
                        "token_ids": unpack_token_ids(key.token_bytes),
                    },
                )
            )

        # Submit one key row per (object group, kv rank), each with its own
        # attention window.
        attn_desc = self._ctx.layout_desc_registry.find_attn_desc(
            model_name, world_size
        )
        session = self._ctx.session_manager.get_or_create(key.request_id)
        session.set_tokens(key.token_bytes)

        start_chunk = min(max(0, key.start // self._ctx.chunk_size), len(chunk_hashes))
        # The engine-covered prefix is never looked up, so it is excluded from
        # the hit-rate denominator and reported separately.
        looked_up_tokens = requested_tokens - start_chunk * self._ctx.chunk_size
        session.begin_lookup(key, tuple(attn_desc.num_chunks_in_sw))

        group_layout_descs = self._ctx.layout_desc_registry.find_group_layout_descs(
            model_name, world_size
        )
        if not group_layout_descs:
            logger.error(
                "No group layout descs found for model %s with world size %d "
                "during lookup!",
                model_name,
                world_size,
            )
            self._register_empty_prefetch_job(
                key, early_exit_reason="no_group_layout_descs"
            )
            return

        if not self._load_policy.should_load(
            KVLoadContext(
                request_id=key.request_id,
                model_name=model_name,
                chunk_size=self._ctx.chunk_size,
                num_lookup_tokens=requested_tokens,
                request_configs=key.request_configs,
            )
        ):
            # Report a miss without prefetching: nothing is fetched from L2
            # and no read lock is taken, so the engine has nothing to release.
            # requested_tokens stays non-zero so hit-rate metrics count the
            # declined tokens as unserved.
            self._register_empty_prefetch_job(
                key,
                requested_tokens=looked_up_tokens,
                num_object_groups=attn_desc.num_object_groups,
                attn_desc=attn_desc,
                engine_covered_chunks=start_chunk,
                early_exit_reason="load_policy_recompute",
            )
            return

        window = self._kv_version_hint_warner.parse(key.request_configs)
        if window is not None and window.max_staleness > 0:
            # A later round's sliding-window check would need the trailing
            # chunks of an earlier round, which its rows do not contain.
            if any(
                w != FULL_ATTENTION_WINDOW_CHUNKS for w in attn_desc.num_chunks_in_sw
            ):
                self._kv_version_hint_warner.warn(
                    (KVVersionHintProblem.STALENESS_UNSUPPORTED_LAYOUT,)
                )
                window = replace(window, max_staleness=0)

        # Keep the covered prefix's recency without locking or prefetching
        # it. After the load policy, so a declined lookup leaves no trace.
        if start_chunk:
            covered_hashes = chunk_hashes[:start_chunk]
            if window is not None:
                covered_hashes = [
                    versioned_chunk_hash(h, window.version) for h in covered_hashes
                ]
            self._ctx.storage_manager.touch_cached_keys(
                [
                    obj_key
                    for group_keys in ipc_key_to_object_keys(
                        key,
                        covered_hashes,
                        list(range(attn_desc.num_object_groups)),
                    )
                    for obj_key in group_keys
                ]
            )

        # Prefetch only [start, end); status offsets hits back to absolute.
        suffix_hashes = chunk_hashes[start_chunk:]
        if not suffix_hashes:
            # Everything is engine-covered: nothing to lock or prefetch. The
            # empty job still reports the covered prefix as the hit.
            self._register_empty_prefetch_job(
                key,
                requested_tokens=looked_up_tokens,
                num_object_groups=attn_desc.num_object_groups,
                attn_desc=attn_desc,
                start_chunk=start_chunk,
                engine_covered_chunks=start_chunk,
            )
            return

        kv_probe = None
        if window is None:
            spec = PrefetchTaskSpec(
                key_groups=ipc_key_to_grouped_object_keys(
                    key,
                    suffix_hashes,
                    list(range(attn_desc.num_object_groups)),
                    group_layout_descs,
                    attn_desc,
                ),
                num_kv_readers=num_kv_readers,
            )
        else:
            kv_probe = _KVVersionProbe(
                window=window,
                key=key,
                chunk_hashes=chunk_hashes,
                group_layout_descs=group_layout_descs,
                attn_desc=attn_desc,
                num_kv_readers=num_kv_readers,
                version=window.version,
            )
            spec = self._kv_round_spec(kv_probe, start_chunk)
        handle = self._ctx.storage_manager.submit_prefetch_task(
            spec,
            external_request_id=key.request_id,
        )
        self._register_prefetch_job(
            _PrefetchJob(
                handle=handle,
                row_windows=tuple(row.sliding_window_size for row in spec.key_groups),
                request_id=key.request_id,
                requested_tokens=looked_up_tokens,
                num_object_groups=attn_desc.num_object_groups,
                attn_desc=attn_desc,
                start_chunk=start_chunk,
                engine_covered_chunks=start_chunk,
                prompt_tokens=num_packed_tokens(key.token_bytes),
                engine_hit_tokens=int(
                    (key.request_configs or {}).get(
                        "lmcache.hint.engine_computed_tokens", 0
                    )
                ),
                model_name=model_name,
                cache_salt=key.cache_salt,
                kv_probe=kv_probe,
            )
        )

    @request_handler(HandlerType.BLOCKING)
    def query_prefetch_status(
        self,
        request_id: str,
    ) -> int | None:
        """Poll the status of a prefetch job by request_id.

        Returns the chunk count when the prefetch is complete, or None
        if it is still in progress.  The job entry is automatically
        removed once a non-None result is returned (exactly-once
        semantics).

        Args:
            request_id: The external request ID passed in the lookup key.

        Returns:
            Chunk count (int) when done, None if still in progress,
            0 if the request_id is unknown (already completed and consumed,
            or invalid).
        """
        with self._prefetch_job_lock:
            job = self._prefetch_jobs.get(request_id)
        if job is None:
            logger.warning(
                "Prefetch job for request %s not found (already completed or invalid)",
                request_id,
            )
            return 0

        return self._consume_prefetch_result(job)

    @request_handler(HandlerType.BLOCKING)
    def wait_prefetch_status(
        self,
        request_id: str,
        timeout: float,
    ) -> int | None:
        """Block until a prefetch job completes, then return its chunk count.

        Like query_prefetch_status, but waits for the daemon to publish the
        result instead of returning None while the prefetch is still in
        progress, so the caller does not have to busy-poll. The job entry is
        removed once a non-None result is returned (exactly-once semantics).

        Args:
            request_id: The external request ID passed in the lookup key.
            timeout: Maximum number of seconds to wait for the prefetch.

        Returns:
            Chunk count (int) when done, None if the wait timed out, 0 if the
            request_id is unknown (already completed and consumed, or invalid).

        Note:
            A versioned lookup may need several prefetch rounds; ``timeout``
            bounds the wait for all of them.
        """
        with self._prefetch_job_lock:
            job = self._prefetch_jobs.get(request_id)
        if job is None:
            logger.warning(
                "Prefetch job for request %s not found (already completed or invalid)",
                request_id,
            )
            return 0

        deadline = time.monotonic() + timeout
        wait_s = timeout
        while True:
            if not self._ctx.storage_manager.wait_prefetch_status(job.handle, wait_s):
                return None
            hit_chunks = self.query_prefetch_status(request_id)
            if hit_chunks is not None or job.kv_probe is None:
                return hit_chunks
            # The finished round submitted the next one; wait for that.
            with self._prefetch_job_lock:
                next_job = self._prefetch_jobs.get(request_id)
            if next_job is None:
                return 0
            job = next_job
            wait_s = max(0.0, deadline - time.monotonic())

    @request_handler(HandlerType.BLOCKING)
    def free_lookup_locks(
        self,
        key: IPCCacheServerKey,
        tp_size: int,
        extra_ranges: tuple[tuple[int, int], ...] = (),
    ) -> None:
        """Release read locks acquired during lookup.

        Hashes are computed only for chunks in ``[start, end)`` to avoid
        unnecessary work on tokens outside that range.
        ``start`` and ``end`` must be aligned to ``chunk_size``; it is the
        caller's responsibility to align the boundaries as desired.

        Only the keys the prefetch actually read-locked are released: the
        key's range is floored to the current lookup's ``start``, and each
        extra range resolves against its own boundaries.

        Releases the same per-object count the lookup reserved
        (``key.num_kv_readers``).

        The release is derived from the lookup result recorded on the
        session (hit length, lock model, L1 owners). If that result has not
        been consumed yet it is handed over here first. While the prefetch
        is still running nothing is released: its locks still belong to the
        prefetch controller and expire with the read-lock TTL.

        Args:
            key: Cache key whose read locks should be released.
            tp_size: Legacy wire field; ignored (kept for payload arity).
            extra_ranges: Lock ranges left over from this request's earlier,
                abandoned lookups, as ``(start, end)`` token pairs. Each end
                must be that lookup's hit boundary, so the range resolves to
                exactly the keys it locked (sliding-window groups translate
                from the hit).
        """
        if key.start >= key.end and not extra_ranges:
            return

        session = self._ctx.session_manager.get_or_create(key.request_id)
        if session.prefetch_hit_chunks < 0:
            # The lookup result has not reached the session yet (no
            # QUERY/WAIT_PREFETCH_STATUS consumed it). Hand it over now so
            # the release below sees the real hit length, lock model and L1
            # owners. While the prefetch is still running its locks belong
            # to the prefetch controller, which settles them itself in
            # _finish_request; releasing them here would double-decrement,
            # so leave them to the controller and the read-lock TTL.
            with self._prefetch_job_lock:
                job = self._prefetch_jobs.get(key.request_id)
            if job is not None and self._consume_prefetch_result(job) is None:
                logger.warning(
                    "free_lookup_locks for request %s while its prefetch is "
                    "still running; leaving its locks to the prefetch controller",
                    key.request_id,
                )
                return

        hit_chunks = session.prefetch_hit_chunks
        if hit_chunks < 0:
            # No lookup result exists (never looked up, or the session was
            # recreated after END_SESSION). L1 locks are anonymous refcounts:
            # guessing a range could strip a concurrent reader's lock.
            logger.warning(
                "free_lookup_locks for request %s without a lookup result; "
                "nothing to release",
                key.request_id,
            )
            return
        # Release exactly the groups the prefetch locked (std lookup: all;
        # CB prefix leg: its prefix set) -- releasing an unlocked group
        # would drop another request's lock on the shared object key.
        lookup_key = session.lookup_ipc_key
        lock_floor = lookup_key.start if lookup_key is not None else 0
        window = parse_kv_version_hints(key.request_configs).window
        obj_keys: list[ObjectKey] = []
        if key.start < key.end:
            # The lookup never locked below its own ``start``.
            obj_keys.extend(
                resolve_prefetched_obj_keys(
                    self._ctx,
                    replace(key, start=max(key.start, lock_floor)),
                    hit_chunks,
                    session.prefetch_locked_gids,
                    kv_versions=(
                        session.kv_version_assignment(window)
                        if window is not None
                        else None
                    ),
                )
            )
        chunk_size = self._ctx.chunk_size
        for range_start, range_end in extra_ranges:
            if range_start >= range_end:
                continue
            abandoned_versions = None
            if window is not None:
                abandoned_versions = session.abandoned_kv_version_assignment(
                    window, range_start // chunk_size, range_end // chunk_size
                )
                if abandoned_versions is None:
                    # Unknown versions mean unknown keys; guessing could strip
                    # another request's anonymous lock, so leave it to the TTL.
                    logger.warning(
                        "free_lookup_locks for request %s: no earlier lookup "
                        "served [%d, %d); leaving its locks to the read-lock TTL",
                        key.request_id,
                        range_start,
                        range_end,
                    )
                    continue
            # An abandoned lookup's range ends at its hit boundary, so the
            # range is its own lock model: resolve with that hit, not the
            # current lookup's.
            obj_keys.extend(
                resolve_prefetched_obj_keys(
                    self._ctx,
                    replace(key, start=range_start, end=range_end),
                    range_end // chunk_size,
                    session.prefetch_locked_gids,
                    kv_versions=abandoned_versions,
                )
            )
        if not obj_keys:
            return

        self._ctx.storage_manager.finish_read_prefetched(
            obj_keys,
            read_locks=key.require_num_kv_readers(),
            l1_owners=self._ctx.get_read_owners(key.request_id),
        )

    @request_handler(HandlerType.BLOCKING)
    def end_session(self, request_id: str) -> None:
        """Remove the session for a finished request.

        Args:
            request_id: The request ID whose session should be removed.
        """
        self._ctx.event_bus.publish(
            Event(
                event_type=EventType.MP_VLLM_END_SESSION,
                metadata={"request_id": request_id},
            )
        )
        session = self._ctx.session_manager.remove(request_id)
        self._ctx.event_bus.publish(
            Event(
                event_type=EventType.MP_REQUEST_END,
                session_id=request_id,
            )
        )
        if session is None:
            logger.warning("Session %s not found, skipping touch", request_id)
            return
        if session.lookup_ipc_key is None:
            logger.warning(
                "Session %s has no lookup ipc key, skipping touch",
                request_id,
            )
            return

        chunk_hashes = [TokenHasher.hash_to_bytes(h) for h in session.get_hashes(0)]
        window = parse_kv_version_hints(session.lookup_ipc_key.request_configs).window
        if window is not None:
            chunk_hashes = session.kv_version_assignment(window).apply(chunk_hashes, 0)
        obj_keys = self._all_object_keys(session.lookup_ipc_key, chunk_hashes)
        # unified touch of all keys, which include retrieved and stored keys
        # TODO(chunxiaozheng): when l2 is enabled, the prefetched keys from l2 are temp
        #  and will be deleted after finish_read_prefetched, when we touch all keys,
        #  these keys has been deleted and will not be touched.
        self._ctx.storage_manager.touch_cached_keys(obj_keys)

    @request_handler(HandlerType.BLOCKING)
    def query_kv_version_spans(
        self, request_id: str
    ) -> list[tuple[int, int, int]] | None:
        """Return the KV versions the request's current lookup served.

        Only requests with ``lmcache.kv_version`` record versions. Must be
        called before END_SESSION, which drops the record.

        Args:
            request_id: The request ID passed in the lookup key.

        Returns:
            ``[start_token, end_token, version]`` per served run, in token
            order, chunk-aligned and starting at the lookup's ``start``.
            Several versions mean a mixed prefix; versions never increase
            along it. Empty when the lookup served no versioned KV (no hint,
            a miss, or the lookup has not finished). None when the server has
            no session for ``request_id`` (never looked up, ended, or reaped
            after its TTL).
        """
        session = self._ctx.session_manager.get(request_id)
        if session is None:
            return None
        return session.kv_version_spans(self._ctx.chunk_size)

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    def _all_object_keys(
        self,
        key: IPCCacheServerKey,
        chunk_hashes: list[bytes],
    ) -> list[ObjectKey]:
        """Resolve every object key of a request across all object groups.

        The object-group count is read from the layout registry for
        ``key``'s ``(model_name, world_size)``. The order of the returned keys
        is unspecified.

        Args:
            key: The IPC key (model/world/worker, salt).
            chunk_hashes: Chunk hashes to resolve keys for.

        Returns:
            The object keys of all groups and kv ranks for ``chunk_hashes``.
        """
        num_groups = self._ctx.layout_desc_registry.find_attn_desc(
            key.model_name, key.world_size
        ).num_object_groups
        per_group = ipc_key_to_object_keys(key, chunk_hashes, list(range(num_groups)))
        return [obj_key for group_keys in per_group for obj_key in group_keys]

    def _register_prefetch_job(self, job: _PrefetchJob) -> None:
        with self._prefetch_job_lock:
            self._prefetch_jobs[job.request_id] = job

    def _register_empty_prefetch_job(
        self,
        key: IPCCacheServerKey,
        requested_tokens: int = 0,
        num_object_groups: int = 1,
        attn_desc: AttnWindowDesc = DEFAULT_ATTN_WINDOW_DESC,
        start_chunk: int = 0,
        engine_covered_chunks: int = 0,
        early_exit_reason: str = "",
    ) -> None:
        """Register a job that resolves immediately without a prefetch.

        Args:
            key: The lookup's cache key.
            requested_tokens: Hit-rate denominator tokens; 0 on early exits.
            num_object_groups: Object groups of the request's layout.
            attn_desc: The request's attention window descriptor.
            start_chunk: First chunk of the looked-up range; reported as the
                hit when everything below it is engine-covered.
            engine_covered_chunks: Engine-covered chunks below ``key.start``,
                reported as ``apc_covered_tokens``.
            early_exit_reason: The ``lookup()`` branch that returned before
                submitting a prefetch; empty when the lookup simply had
                nothing to fetch.
        """
        self._register_prefetch_job(
            _PrefetchJob(
                handle=PrefetchHandle(
                    prefetch_request_id=-1,
                    external_request_id=key.request_id,
                    total_requested_keys=0,
                    submit_time=time.monotonic(),
                ),
                row_windows=(),
                request_id=key.request_id,
                requested_tokens=requested_tokens,
                num_object_groups=num_object_groups,
                attn_desc=attn_desc,
                start_chunk=start_chunk,
                engine_covered_chunks=engine_covered_chunks,
                prompt_tokens=num_packed_tokens(key.token_bytes),
                engine_hit_tokens=int(
                    (key.request_configs or {}).get(
                        "lmcache.hint.engine_computed_tokens", 0
                    )
                ),
                model_name=key.model_name,
                cache_salt=key.cache_salt,
                early_exit_reason=early_exit_reason,
            )
        )

    def _consume_prefetch_result(self, job: _PrefetchJob) -> int | None:
        """Hand a finished prefetch result over to the request's session.

        Takes the result from the storage manager (each result is returned
        once), records the model-wide hit length, lock model and retained L1
        owners on the session, emits ``MP_LOOKUP_PREFETCH_END`` and drops the
        job. This is the only path that moves lookup state into the session,
        so ``query_prefetch_status`` and ``free_lookup_locks`` cannot disagree
        about what a lookup locked.

        For a versioned lookup, a finished round that leaves chunks and
        versions to probe submits the next round instead and returns None;
        only the last round hands the combined result over.

        Returns:
            The hit chunk count, or None while the prefetch is still running
            (the job is kept so a later call can resolve it).
        """
        result = self._ctx.storage_manager.query_prefetch_status(job.handle)
        if result is None:
            return None
        if job.row_windows:
            found_count, _retain = fold_unfold_grouped(
                result.hit_cells, job.row_windows
            )
            l1_found_count, _l1_retain = fold_unfold_grouped(
                result.l1_hit_cells, job.row_windows
            )
        else:
            # Nothing was submitted (early exit), so nothing can be hit.
            found_count = 0
            l1_found_count = 0

        # The fold covers only the looked-up sub-range: the tier split is
        # attributed on it, so the covered prefix never counts as an L1 hit.
        l1_chunks = min(l1_found_count, found_count)
        l2_chunks = found_count - l1_chunks
        # The engine resolves the hit in absolute chunks; shift it back.
        found_count += job.start_chunk
        l1_owners = result.l1_owners
        l1_hit_keys = result.l1_hit_count
        l2_hit_keys = result.l2_hit_count

        session = self._ctx.session_manager.get_or_create(job.request_id)
        probe = job.kv_probe
        if probe is not None:
            probe.add_round(job.start_chunk, found_count, l1_chunks, l2_chunks, result)
            if self._submit_next_kv_round(job, found_count):
                return None
            l1_chunks, l2_chunks = probe.l1_chunks, probe.l2_chunks
            l1_owners = probe.l1_owners
            l1_hit_keys, l2_hit_keys = probe.l1_hit_keys, probe.l2_hit_keys
            if len(probe.runs) > 1:
                logger.debug(
                    "Request %s served a mixed-version prefix: %s",
                    job.request_id,
                    probe.runs,
                )
            # Before the hit length, so a reader seeing the hit sees its runs.
            session.record_kv_version_runs(tuple(probe.runs))

        # Record the model-wide hit length on the session so a later
        # free_lookup_locks can reconstruct which keys the prefetch
        # read-locked (see ``unfold``: full-attention groups lock the whole
        # hit prefix, sliding-window groups only its in-window suffix).
        session.record_prefetch_result(
            found_count,
            tuple(range(job.attn_desc.num_object_groups)),
            l1_owners,
        )

        chunk_size = self._ctx.chunk_size
        covered_tokens = job.engine_covered_chunks * chunk_size
        hit_tokens = (l1_chunks + l2_chunks) * chunk_size
        # Prefix served by either cache: both ranges start at 0 and overlap in
        # the sub-chunk gap below the lookup start, so take the longer, not the sum.
        request_hit_tokens = max(job.engine_hit_tokens, covered_tokens + hit_tokens)
        self._ctx.event_bus.publish(
            Event(
                event_type=EventType.MP_LOOKUP_PREFETCH_END,
                session_id=job.request_id,
                metadata={
                    "found_count": found_count,
                    "requested_tokens": job.requested_tokens,
                    # Hit-rate counters cover only the looked-up range; the
                    # engine-covered prefix gets its own field.
                    "apc_covered_tokens": covered_tokens,
                    "hit_tokens": hit_tokens,
                    "l1_hit_tokens": l1_chunks * chunk_size,
                    "l2_hit_tokens": l2_chunks * chunk_size,
                    # Request-level view: the whole prompt as denominator and
                    # what the engine plus LMCache served as numerator.
                    "prompt_tokens": job.prompt_tokens,
                    "engine_hit_tokens": job.engine_hit_tokens,
                    "request_hit_tokens": request_hit_tokens,
                    "l1_hit_keys": l1_hit_keys,
                    "l2_hit_keys": l2_hit_keys,
                    "early_exit_reason": job.early_exit_reason,
                    "model_name": job.model_name,
                    "cache_salt": job.cache_salt,
                },
            )
        )

        with self._prefetch_job_lock:
            self._prefetch_jobs.pop(job.request_id, None)

        return found_count

    def _kv_round_spec(
        self, probe: _KVVersionProbe, start_chunk: int
    ) -> PrefetchTaskSpec:
        """Build the prefetch of chunks ``[start_chunk, n)`` at ``probe.version``."""
        return PrefetchTaskSpec(
            key_groups=ipc_key_to_grouped_object_keys(
                probe.key,
                [
                    versioned_chunk_hash(h, probe.version)
                    for h in probe.chunk_hashes[start_chunk:]
                ],
                list(range(probe.attn_desc.num_object_groups)),
                probe.group_layout_descs,
                probe.attn_desc,
            ),
            num_kv_readers=probe.num_kv_readers,
        )

    def _submit_next_kv_round(self, job: _PrefetchJob, hit_chunks: int) -> bool:
        """Submit the next, one version older, round of a versioned lookup.

        Args:
            job: The job whose round just finished; its ``kv_probe`` is set.
            hit_chunks: Absolute hit boundary after that round.

        Returns:
            True if a round was submitted (the job now tracks it), False when
            the prompt is covered or the window is exhausted.
        """
        probe = job.kv_probe
        assert probe is not None
        if (
            hit_chunks >= len(probe.chunk_hashes)
            or probe.version <= probe.window.oldest
        ):
            return False
        probe.version -= 1
        spec = self._kv_round_spec(probe, hit_chunks)
        handle = self._ctx.storage_manager.submit_prefetch_task(
            spec, external_request_id=job.request_id
        )
        self._register_prefetch_job(
            replace(
                job,
                handle=handle,
                row_windows=tuple(row.sliding_window_size for row in spec.key_groups),
                start_chunk=hit_chunks,
            )
        )
        return True

    def _active_prefetch_count(self) -> int:
        """Return the number of active prefetch jobs (thread-safe)."""
        with self._prefetch_job_lock:
            return len(self._prefetch_jobs)

    def _setup_metrics(self) -> None:
        """Register OTel observable gauges for lookup module metrics."""
        _gauge = partial(register_gauge, "lmcache.mp_server")
        _gauge(
            "lmcache_mp.active_prefetch_jobs",
            "Number of active prefetch jobs",
            self._active_prefetch_count,
        )
