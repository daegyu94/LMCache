# SPDX-License-Identifier: Apache-2.0
"""LookupModule: lookup, prefetch polling, and session lifecycle."""

# Standard
from concurrent.futures import ThreadPoolExecutor
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
from lmcache.v1.multiprocess.engine_context import MPCacheServerContext
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
    KV_VERSION_CAPABILITY,
    KVVersionAssignment,
    KVVersionHintProblem,
    KVVersioning,
    KVVersionRun,
    KVVersionWindow,
    read_kv_version_hints,
    versioned_chunk_hash,
    warn_kv_version_problem,
)
from lmcache.v1.multiprocess.request_handler import HandlerType, request_handler
from lmcache.v1.multiprocess.session import Session
from lmcache.v1.multiprocess.token_codec import num_packed_tokens, unpack_token_ids
from lmcache.v1.multiprocess.token_hasher import TokenHasher

logger = init_logger(__name__)

# Stateless, so one instance can serve every LookupModule.
_DEFAULT_LOAD_POLICY = DefaultKVLoadPolicy({})

# Concurrent multi-round versioned lookups; more queue for a free worker.
_KV_PROBE_WORKERS = 4
# How often a probe worker blocked on a round checks for shutdown.
_KV_PROBE_POLL_S = 1.0


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

    Round ``r`` prefetches chunks ``[p, n)`` under version
    ``window.version - r``, where ``p`` is where the earlier rounds' hits
    ended. A worker thread (``LookupModule._drive_kv_probe``) submits each
    round as soon as the previous one finishes, so the rounds do not wait for
    the engine to poll. A lookup runs at most ``max_staleness + 1`` rounds and
    the served versions never increase along the prompt.

    Ownership: only the driver submits rounds and writes the round fields.
    ``cancelled`` and ``recorded`` change only under ``lock``. Once
    ``finished`` is set the driver no longer touches the probe. A cancelled
    probe's driver releases every lock its rounds took; a recorded probe has
    handed its locks to the session like a plain lookup.
    """

    window: KVVersionWindow
    key: IPCCacheServerKey
    # Unversioned chunk hashes of the whole lookup, from chunk 0.
    chunk_hashes: list[bytes]
    group_layout_descs: dict[int, MemoryLayoutDesc]
    attn_desc: AttnWindowDesc
    num_kv_readers: int
    # The in-flight round: its version, first chunk, spec and handle.
    version: int
    round_start: int
    round_spec: PrefetchTaskSpec
    handle: PrefetchHandle
    # Accumulated over finished rounds.
    hit_chunks: int
    runs: list[KVVersionRun] = field(default_factory=list)
    locked: list[tuple[list[ObjectKey], dict[ObjectKey, int]]] = field(
        default_factory=list
    )
    l1_chunks: int = 0
    l2_chunks: int = 0
    l1_hit_keys: int = 0
    l2_hit_keys: int = 0
    l1_owners: dict[ObjectKey, int] = field(default_factory=dict)
    cancelled: bool = False
    recorded: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)
    finished: threading.Event = field(default_factory=threading.Event)


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
    # The version a single-round versioned lookup (``max_kv_staleness`` 0)
    # looked up; None for unversioned and multi-round lookups.
    kv_version: int | None = None
    # Set for a multi-round versioned lookup; its driver owns the rounds and
    # ``handle`` is unused.
    kv_probe: _KVVersionProbe | None = None


class LookupModule(DiscoverableModule):
    """Handles lookup, prefetch polling, lock release, and session lifecycle.

    Owns the prefetch-job bookkeeping (``_prefetch_jobs``) and exposes
    handlers for the LOOKUP, QUERY_PREFETCH_STATUS, FREE_LOOKUP_LOCKS,
    END_SESSION and QUERY_KV_VERSION_SPANS request types.

    Args:
        ctx: Shared engine context providing storage manager, token hasher,
            session manager, event bus, layout descriptor registry, and
            chunk size.
        load_policy: Decides per lookup whether to serve the request or let
            the engine recompute it. Defaults to ``DefaultKVLoadPolicy``.
        kv_versioning: Whether requests with ``lmcache.kv_version`` are
            served. Only the LMCache-driven STORE path refuses KV that
            belongs to no single version, so the server enables it only in
            that transfer mode. Defaults to ``DISABLED`` (fail closed).
    """

    module_name = "lookup"

    @classmethod
    def create(cls, build_ctx: ModuleBuildContext) -> EngineModule | None:
        """Build the lookup module; it applies to every configuration.

        KV versioning is enabled, and advertised as
        :data:`~lmcache.v1.multiprocess.kv_version.KV_VERSION_CAPABILITY`,
        only in the LMCache-driven transfer mode.
        """
        kv_versioning = KVVersioning.DISABLED
        if build_ctx.mp_config.supported_transfer_mode == "lmcache_driven":
            kv_versioning = KVVersioning.ENABLED
            build_ctx.add_experimental(KV_VERSION_CAPABILITY)
        return cls(
            build_ctx.engine_context,
            create_kv_load_policy(
                build_ctx.mp_config.kv_load_policy,
                build_ctx.mp_config.runtime_plugin_config.extra_config,
            ),
            kv_versioning=kv_versioning,
        )

    def __init__(
        self,
        ctx: MPCacheServerContext,
        load_policy: KVLoadPolicy = _DEFAULT_LOAD_POLICY,
        *,
        kv_versioning: KVVersioning = KVVersioning.DISABLED,
    ) -> None:
        self._ctx = ctx
        self._load_policy = load_policy
        self._kv_versioning = kv_versioning
        self._prefetch_jobs: dict[str, _PrefetchJob] = {}
        self._prefetch_job_lock = threading.Lock()
        # Drives multi-round versioned lookups; threads start on first use.
        self._kv_probe_pool = ThreadPoolExecutor(
            max_workers=_KV_PROBE_WORKERS, thread_name_prefix="lmcache-kv-probe"
        )
        self._closing = threading.Event()
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
        """Stop the version-probe workers.

        Probes still running stop at their next wait; their read locks are
        left to the read-lock TTL.
        """
        self._closing.set()
        self._kv_probe_pool.shutdown(wait=False, cancel_futures=True)

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
        version-qualified keys. With ``lmcache.max_kv_staleness`` 0 that is one
        ordinary prefetch. Otherwise a worker thread probes newest version
        first: each round continues where the previous hits ended, one version
        older, until the prompt is covered or the window is exhausted. A chunk
        with no version in the window is a miss and ends the prefix. A
        malformed version, or versioning disabled on this server, reports a
        miss without prefetching (fail closed).

        Args:
            key: Cache key with request_id embedded.
            tp_size: Legacy wire field; ignored (kept for payload arity).
        """
        # A previous multi-round lookup of this request is superseded.
        self._abandon_kv_probe(key.request_id)
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

        hints = read_kv_version_hints(key.request_configs)
        window = hints.window
        if hints.bypasses_lmcache or (
            window is not None and self._kv_versioning is KVVersioning.DISABLED
        ):
            if window is not None:
                warn_kv_version_problem(KVVersionHintProblem.UNSUPPORTED_TRANSFER_MODE)
            # Fail closed: a version hint the server cannot honour must never
            # be served from, or mixed into, the unversioned key space.
            self._register_empty_prefetch_job(
                key,
                requested_tokens=looked_up_tokens,
                num_object_groups=attn_desc.num_object_groups,
                attn_desc=attn_desc,
                engine_covered_chunks=start_chunk,
                early_exit_reason="kv_version_unsupported",
            )
            return
        if window is not None and window.max_staleness > 0:
            # A later round's sliding-window check would need the trailing
            # chunks of an earlier round, which its rows do not contain.
            if any(
                w != FULL_ATTENTION_WINDOW_CHUNKS for w in attn_desc.num_chunks_in_sw
            ):
                warn_kv_version_problem(
                    KVVersionHintProblem.STALENESS_UNSUPPORTED_LAYOUT
                )
                window = replace(window, max_staleness=0)

        # Keep the covered prefix's recency without locking or prefetching
        # it. After the load policy, so a declined lookup leaves no trace. A
        # versioned request skips it: LMCache did not serve the engine-held
        # prefix, so which version's copy to refresh is unknown.
        if start_chunk and window is None:
            self._ctx.storage_manager.touch_cached_keys(
                [
                    obj_key
                    for group_keys in ipc_key_to_object_keys(
                        key,
                        chunk_hashes[:start_chunk],
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

        if window is not None and window.max_staleness > 0:
            round_spec = self._kv_round_spec(
                key,
                chunk_hashes,
                start_chunk,
                window.version,
                group_layout_descs,
                attn_desc,
                num_kv_readers,
            )
            probe = _KVVersionProbe(
                window=window,
                key=key,
                chunk_hashes=chunk_hashes,
                group_layout_descs=group_layout_descs,
                attn_desc=attn_desc,
                num_kv_readers=num_kv_readers,
                version=window.version,
                round_start=start_chunk,
                round_spec=round_spec,
                handle=self._ctx.storage_manager.submit_prefetch_task(
                    round_spec, external_request_id=key.request_id
                ),
                hit_chunks=start_chunk,
            )
            job = self._new_prefetch_job(
                key, probe.handle, (), looked_up_tokens, attn_desc, start_chunk
            )
            job.kv_probe = probe
            self._register_prefetch_job(job)
            self._kv_probe_pool.submit(self._drive_kv_probe, job, probe)
            return

        kv_version = None if window is None else window.version
        if kv_version is not None:
            suffix_hashes = [versioned_chunk_hash(h, kv_version) for h in suffix_hashes]
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
        handle = self._ctx.storage_manager.submit_prefetch_task(
            spec,
            external_request_id=key.request_id,
        )
        job = self._new_prefetch_job(
            key,
            handle,
            tuple(row.sliding_window_size for row in spec.key_groups),
            looked_up_tokens,
            attn_desc,
            start_chunk,
        )
        job.kv_version = kv_version
        self._register_prefetch_job(job)

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
            A multi-round versioned lookup's rounds run on a worker thread;
            ``timeout`` bounds the wait for all of them.
        """
        with self._prefetch_job_lock:
            job = self._prefetch_jobs.get(request_id)
        if job is None:
            logger.warning(
                "Prefetch job for request %s not found (already completed or invalid)",
                request_id,
            )
            return 0

        if job.kv_probe is not None:
            # The probe's worker drives every round; ``timeout`` bounds all.
            if not job.kv_probe.finished.wait(timeout):
                return None
        elif not self._ctx.storage_manager.wait_prefetch_status(job.handle, timeout):
            return None
        return self.query_prefetch_status(request_id)

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

        A multi-round versioned lookup that is still probing is cancelled
        instead: its worker releases every lock its rounds took, and this
        call releases only ``extra_ranges``.

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
        window = read_kv_version_hints(key.request_configs).window
        if self._abandon_kv_probe(key.request_id):
            # A multi-round lookup was still running; its worker releases
            # every lock its rounds took, so only the extra ranges remain.
            self._finish_read_locks(
                key, self._extra_range_obj_keys(key, session, window, extra_ranges)
            )
            return
        with self._prefetch_job_lock:
            job = self._prefetch_jobs.get(key.request_id)
        if job is not None and job.kv_probe is not None:
            # A finished probe already recorded its result on the session;
            # report it so the job does not outlive the release.
            self._consume_prefetch_result(job)
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
        obj_keys.extend(self._extra_range_obj_keys(key, session, window, extra_ranges))
        self._finish_read_locks(key, obj_keys)

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
        # Before the session goes, so a running probe cannot record into it.
        self._abandon_kv_probe(request_id)
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
        window = read_kv_version_hints(session.lookup_ipc_key.request_configs).window
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
            along it. Empty when the finished lookup served no versioned KV
            (no hint, or a miss). None when the answer is not known yet or
            any more: the lookup's result has not been consumed, or the
            server has no session for ``request_id`` (never looked up,
            ended, or reaped after its TTL).
        """
        with self._prefetch_job_lock:
            if request_id in self._prefetch_jobs:
                return None
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

        A multi-round versioned lookup has already recorded its result on
        the session from its worker (see ``_drive_kv_probe``); this only
        reports it once the worker has finished.

        Returns:
            The hit chunk count, or None while the prefetch is still running
            (the job is kept so a later call can resolve it).
        """
        probe = job.kv_probe
        if probe is not None:
            with probe.lock:
                recorded = probe.recorded
            if not recorded and not probe.finished.is_set():
                return None
            # A probe that failed released its locks and serves nothing.
            found_count = probe.hit_chunks if recorded else job.start_chunk
            l1_chunks = probe.l1_chunks if recorded else 0
            l2_chunks = probe.l2_chunks if recorded else 0
            l1_hit_keys = probe.l1_hit_keys if recorded else 0
            l2_hit_keys = probe.l2_hit_keys if recorded else 0
            if len(probe.runs) > 1:
                logger.debug(
                    "Request %s served a mixed-version prefix: %s",
                    job.request_id,
                    probe.runs,
                )
        else:
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
            # attributed on it, so the covered prefix never counts as an L1
            # hit.
            l1_chunks = min(l1_found_count, found_count)
            l2_chunks = found_count - l1_chunks
            # The engine resolves the hit in absolute chunks; shift it back.
            found_count += job.start_chunk
            l1_hit_keys = result.l1_hit_count
            l2_hit_keys = result.l2_hit_count

            session = self._ctx.session_manager.get_or_create(job.request_id)
            if job.kv_version is not None:
                # Before the hit length, so a reader seeing the hit sees its
                # versions.
                session.record_kv_version_runs(
                    (KVVersionRun(job.start_chunk, found_count, job.kv_version),)
                    if found_count > job.start_chunk
                    else ()
                )
            # Record the model-wide hit length on the session so a later
            # free_lookup_locks can reconstruct which keys the prefetch
            # read-locked (see ``unfold``: full-attention groups lock the whole
            # hit prefix, sliding-window groups only its in-window suffix).
            session.record_prefetch_result(
                found_count,
                tuple(range(job.attn_desc.num_object_groups)),
                result.l1_owners,
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
            # A newer lookup of the same request may have replaced this job.
            if self._prefetch_jobs.get(job.request_id) is job:
                del self._prefetch_jobs[job.request_id]

        return found_count

    def _new_prefetch_job(
        self,
        key: IPCCacheServerKey,
        handle: PrefetchHandle,
        row_windows: tuple[int, ...],
        requested_tokens: int,
        attn_desc: AttnWindowDesc,
        start_chunk: int,
    ) -> _PrefetchJob:
        """Build the job of a submitted lookup (the caller registers it)."""
        return _PrefetchJob(
            handle=handle,
            row_windows=row_windows,
            request_id=key.request_id,
            requested_tokens=requested_tokens,
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
            model_name=key.model_name,
            cache_salt=key.cache_salt,
        )

    def _kv_round_spec(
        self,
        key: IPCCacheServerKey,
        chunk_hashes: list[bytes],
        start_chunk: int,
        version: int,
        group_layout_descs: dict[int, MemoryLayoutDesc],
        attn_desc: AttnWindowDesc,
        num_kv_readers: int,
    ) -> PrefetchTaskSpec:
        """Build the prefetch of chunks ``[start_chunk, n)`` under ``version``."""
        return PrefetchTaskSpec(
            key_groups=ipc_key_to_grouped_object_keys(
                key,
                [versioned_chunk_hash(h, version) for h in chunk_hashes[start_chunk:]],
                list(range(attn_desc.num_object_groups)),
                group_layout_descs,
                attn_desc,
            ),
            num_kv_readers=num_kv_readers,
        )

    def _drive_kv_probe(self, job: _PrefetchJob, probe: _KVVersionProbe) -> None:
        """Run a probe's rounds until it is done or cancelled.

        Runs on a probe worker thread. Each finished round either ends the
        probe or immediately submits the next one (``_finish_kv_round``). An
        unexpected error cancels the probe and releases its locks, so the
        lookup reports a miss.

        Args:
            job: The registered job of the lookup.
            probe: ``job.kv_probe``.
        """
        storage = self._ctx.storage_manager
        try:
            while True:
                while not storage.wait_prefetch_status(probe.handle, _KV_PROBE_POLL_S):
                    if self._closing.is_set():
                        return
                result = storage.query_prefetch_status(probe.handle)
                if result is None:
                    raise RuntimeError("probe round result was taken by another reader")
                if self._finish_kv_round(job, probe, result):
                    return
        except Exception:
            logger.exception(
                "KV version probe for request %s failed; reporting a miss",
                job.request_id,
            )
            with probe.lock:
                probe.cancelled = True
            self._release_kv_probe_locks(probe)
        finally:
            probe.finished.set()

    def _finish_kv_round(
        self, job: _PrefetchJob, probe: _KVVersionProbe, result: PrefetchResult
    ) -> bool:
        """Account one finished round, then record, release or continue.

        Returns:
            True when the probe is over: its result is recorded on the
            session, or it was cancelled and its locks are released. False
            when the next round has been submitted.
        """
        row_windows = tuple(
            row.sliding_window_size for row in probe.round_spec.key_groups
        )
        found, _retain = fold_unfold_grouped(result.hit_cells, row_windows)
        l1_found, _l1_retain = fold_unfold_grouped(result.l1_hit_cells, row_windows)
        hit_chunks = probe.round_start + found
        # Every group is full attention (checked at lookup), so each row
        # locked exactly its first ``found`` keys.
        probe.locked.append(
            (
                [k for row in probe.round_spec.key_groups for k in row.keys[:found]],
                result.l1_owners,
            )
        )
        if found:
            probe.runs.append(
                KVVersionRun(probe.round_start, hit_chunks, probe.version)
            )
        probe.hit_chunks = hit_chunks
        probe.l1_chunks += min(l1_found, found)
        probe.l2_chunks += found - min(l1_found, found)
        probe.l1_hit_keys += result.l1_hit_count
        probe.l2_hit_keys += result.l2_hit_count
        probe.l1_owners.update(result.l1_owners)
        done = (
            hit_chunks >= len(probe.chunk_hashes)
            or probe.version <= probe.window.oldest
        )

        with probe.lock:
            cancelled = probe.cancelled
            if done and not cancelled:
                session = self._ctx.session_manager.get_or_create(job.request_id)
                # Runs before the hit length, so a reader seeing the hit sees
                # its versions.
                session.record_kv_version_runs(tuple(probe.runs))
                session.record_prefetch_result(
                    hit_chunks,
                    tuple(range(probe.attn_desc.num_object_groups)),
                    dict(probe.l1_owners),
                )
                probe.recorded = True
        if cancelled:
            self._release_kv_probe_locks(probe)
            return True
        if done:
            return True

        probe.version -= 1
        probe.round_start = hit_chunks
        probe.round_spec = self._kv_round_spec(
            probe.key,
            probe.chunk_hashes,
            hit_chunks,
            probe.version,
            probe.group_layout_descs,
            probe.attn_desc,
            probe.num_kv_readers,
        )
        probe.handle = self._ctx.storage_manager.submit_prefetch_task(
            probe.round_spec, external_request_id=job.request_id
        )
        return False

    def _abandon_kv_probe(self, request_id: str) -> bool:
        """Cancel the request's multi-round lookup if it has not finished.

        A cancelled probe's job is dropped and its worker releases every
        lock its rounds took once its in-flight round ends; nothing else
        submits rounds. A probe that already recorded its result is left as
        a plain lookup result.

        Args:
            request_id: The request whose lookup is abandoned.

        Returns:
            True if a running probe was cancelled.
        """
        with self._prefetch_job_lock:
            job = self._prefetch_jobs.get(request_id)
            probe = job.kv_probe if job is not None else None
            if probe is None:
                return False
            with probe.lock:
                if probe.recorded:
                    return False
                probe.cancelled = True
            del self._prefetch_jobs[request_id]
        return True

    def _release_kv_probe_locks(self, probe: _KVVersionProbe) -> None:
        """Release every read lock a cancelled probe's rounds took."""
        storage = self._ctx.storage_manager
        for keys, owners in probe.locked:
            if keys:
                storage.finish_read_prefetched(
                    keys,
                    read_locks=probe.num_kv_readers,
                    l1_owners=owners if storage.is_multi_l1 else None,
                )
        probe.locked.clear()

    def _extra_range_obj_keys(
        self,
        key: IPCCacheServerKey,
        session: Session,
        window: KVVersionWindow | None,
        extra_ranges: tuple[tuple[int, int], ...],
    ) -> list[ObjectKey]:
        """Resolve the keys earlier, abandoned lookups of a request locked.

        Each range ends at its lookup's hit boundary, so it is its own lock
        model. A versioned range resolves under the versions that lookup
        served; one no recorded lookup served is skipped with a warning and
        left to the read-lock TTL, since guessing could strip another
        request's anonymous lock.
        """
        chunk_size = self._ctx.chunk_size
        obj_keys: list[ObjectKey] = []
        for range_start, range_end in extra_ranges:
            if range_start >= range_end:
                continue
            abandoned_versions = None
            if window is not None:
                abandoned_versions = session.abandoned_kv_version_assignment(
                    window, range_start // chunk_size, range_end // chunk_size
                )
                if abandoned_versions is None:
                    logger.warning(
                        "free_lookup_locks for request %s: no earlier lookup "
                        "served [%d, %d); leaving its locks to the read-lock TTL",
                        key.request_id,
                        range_start,
                        range_end,
                    )
                    continue
            obj_keys.extend(
                resolve_prefetched_obj_keys(
                    self._ctx,
                    replace(key, start=range_start, end=range_end),
                    range_end // chunk_size,
                    session.prefetch_locked_gids,
                    kv_versions=abandoned_versions,
                )
            )
        return obj_keys

    def _finish_read_locks(
        self, key: IPCCacheServerKey, obj_keys: list[ObjectKey]
    ) -> None:
        """Release a lookup's read locks on ``obj_keys`` (no-op when empty)."""
        if not obj_keys:
            return
        self._ctx.storage_manager.finish_read_prefetched(
            obj_keys,
            read_locks=key.require_num_kv_readers(),
            l1_owners=self._ctx.get_read_owners(key.request_id),
        )

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
