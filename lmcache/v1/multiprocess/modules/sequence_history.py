# SPDX-License-Identifier: Apache-2.0
"""SequenceHistoryModule: detect history rewrites of tagged sequences.

Built only when ``--sequence-tracker-max-chunks`` is positive. It observes
every admitted lookup through :meth:`LookupModule.add_lookup_observer`,
compares the lookup's chunk chain with the previous request of the same
``lmcache.sequence_id`` and publishes ``MP_SEQUENCE_TRANSITION`` for appends
and rewrites. With ``--sequence-demotion`` (experimental, off by default) it
also asks the storage manager to demote, in L1 eviction order, the
superseded chunks no tracked sequence still references; never a deletion.
See
``docs/design/v1/multiprocess/sequence_lifecycle.md``.
"""

# Standard
import enum
import threading
import time

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import ipc_key_to_chunk_major_object_keys
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.otel_init import register_gauge
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.engine_context import MPCacheServerContext
from lmcache.v1.multiprocess.engine_module import (
    DiscoverableModule,
    EngineModule,
    ModuleBuildContext,
)
from lmcache.v1.multiprocess.modules.lookup import LookupModule
from lmcache.v1.multiprocess.sequence_tracker import (
    SEQUENCE_ID_HINT,
    SequenceKey,
    SequenceTracker,
    TransitionKind,
    parse_sequence_id,
)

logger = init_logger(__name__)

# Throttle for client-triggered log lines (malformed hints, rewrites), so a
# misbehaving client cannot flood the server log.
_HINT_LOG_INTERVAL_SECONDS = 60.0


class _RateLimiter:
    """Admits at most one event per interval; thread-safe."""

    def __init__(self, interval_seconds: float) -> None:
        self._interval = interval_seconds
        # Monotonic time of the last admitted event; None until the first.
        self._last: float | None = None
        self._lock = threading.Lock()

    def try_acquire(self) -> bool:
        """Return whether an event may happen now, recording it if so."""
        with self._lock:
            now = time.monotonic()
            if self._last is not None and now - self._last < self._interval:
                return False
            self._last = now
            return True


class SupersededChunkAction(enum.Enum):
    """What the module does with a rewrite's superseded chunks."""

    REPORT = "report"
    """Only count them (event and metrics); no cache side effect."""

    DEMOTE_IN_L1 = "demote_in_l1"
    """Also demote the unreferenced ones in L1 eviction order."""


class SequenceHistoryModule(DiscoverableModule):
    """Tracks ``lmcache.sequence_id`` sequences, reports and demotes rewrites.

    Thread-safe as :class:`LookupObserver` requires: the tracker and the
    log rate limiters hold their own locks, and the event bus is safe to
    publish to from any thread.

    Args:
        ctx: Shared engine context providing the event bus, the layout
            registry and the storage manager.
        tracker: The bounded per-sequence chain store.
        superseded_action: Whether superseded chunks are only reported or
            also demoted.
    """

    module_name = "sequence_history"
    module_dependencies = (LookupModule,)

    @classmethod
    def create(cls, build_ctx: ModuleBuildContext) -> EngineModule | None:
        """Build the module when the server sets a tracking budget.

        Returns:
            The module, registered as an observer of the lookup module, or
            None when ``sequence_tracker_max_chunks`` is 0.
        """
        max_chunks = build_ctx.mp_config.sequence_tracker_max_chunks
        if max_chunks == 0:
            return None
        module = cls(
            build_ctx.engine_context,
            SequenceTracker(max_chunks),
            SupersededChunkAction.DEMOTE_IN_L1
            if build_ctx.mp_config.sequence_demotion
            else SupersededChunkAction.REPORT,
        )
        build_ctx.require(LookupModule).add_lookup_observer(module)
        return module

    def __init__(
        self,
        ctx: MPCacheServerContext,
        tracker: SequenceTracker,
        superseded_action: SupersededChunkAction,
    ) -> None:
        self._ctx = ctx
        self._tracker = tracker
        self._superseded_action = superseded_action
        self._malformed_hint_log_limiter = _RateLimiter(_HINT_LOG_INTERVAL_SECONDS)
        self._rewrite_log_limiter = _RateLimiter(_HINT_LOG_INTERVAL_SECONDS)
        register_gauge(
            "lmcache.mp_server",
            "lmcache_mp.sequence_tracker_chunks",
            "Chunk hashes held by the sequence tracker (bounded by "
            "--sequence-tracker-max-chunks)",
            self._tracker.num_tracked_chunks,
        )

    @property
    def context(self) -> MPCacheServerContext:
        """Return the shared engine context."""
        return self._ctx

    def report_status(self) -> dict[str, int]:
        """Return the tracker's current size.

        Returns:
            Tracked sequence and chunk counts.
        """
        return {
            "tracked_sequences": self._tracker.num_tracked_sequences(),
            "tracked_chunks": self._tracker.num_tracked_chunks(),
        }

    def close(self) -> None:
        """Release resources owned by this module (no-op)."""

    def on_lookup_admitted(
        self, key: IPCCacheServerKey, chunk_hashes: list[bytes]
    ) -> None:
        """Compare the lookup with its sequence's previous request.

        No-op for requests without ``lmcache.sequence_id``. A malformed hint
        is ignored with a rate-limited warning that omits the raw value. On
        an append or rewrite ``MP_SEQUENCE_TRANSITION`` is published. With
        ``SupersededChunkAction.DEMOTE_IN_L1``, a rewrite's superseded chunks
        that no tracked sequence references are passed to
        ``StorageManager.demote_cached_keys``, chunk-major in chain order (all
        object groups and kv ranks of a chunk together).

        Args:
            key: The lookup's cache key.
            chunk_hashes: The lookup's full chunk-hash chain from chunk 0.
        """
        try:
            sequence_id = parse_sequence_id(key.request_configs)
        except ValueError as exc:
            # Bad client input, not a server fault: no traceback, and the
            # message never carries the raw value.
            if self._malformed_hint_log_limiter.try_acquire():
                logger.warning(
                    "Ignoring malformed %s hint on request %s: %s "
                    "(further occurrences suppressed for %.0fs)",
                    SEQUENCE_ID_HINT,
                    key.request_id,
                    exc,
                    _HINT_LOG_INTERVAL_SECONDS,
                )
            return
        if sequence_id is None:
            return

        transition = self._tracker.observe(
            SequenceKey(
                model_name=key.model_name,
                world_size=key.world_size,
                cache_salt=key.cache_salt,
                sequence_id=sequence_id,
            ),
            key.request_id,
            chunk_hashes,
        )
        if transition.kind not in (TransitionKind.APPEND, TransitionKind.REWRITE):
            return
        demotable = (
            transition.unreferenced_hashes
            if self._superseded_action is SupersededChunkAction.DEMOTE_IN_L1
            else ()
        )
        if demotable:
            self._ctx.storage_manager.demote_cached_keys(
                ipc_key_to_chunk_major_object_keys(
                    key,
                    list(demotable),
                    self._object_group_ids(key),
                )
            )
        if (
            transition.kind is TransitionKind.REWRITE
            and self._rewrite_log_limiter.try_acquire()
        ):
            # An unexpected rewrite near the start often means the client's
            # chat template or tokenization changed between turns.
            logger.debug(
                "Request %s rewrote its sequence history at chunk %d of %d "
                "(further rewrites suppressed for %.0fs)",
                key.request_id,
                transition.divergence_chunk,
                transition.previous_chunks,
                _HINT_LOG_INTERVAL_SECONDS,
            )
        self._ctx.event_bus.publish(
            Event(
                event_type=EventType.MP_SEQUENCE_TRANSITION,
                session_id=key.request_id,
                metadata={
                    "kind": transition.kind.value,
                    "divergence_chunk": transition.divergence_chunk,
                    "previous_chunks": transition.previous_chunks,
                    "current_chunks": len(chunk_hashes),
                    "superseded_chunks": len(transition.superseded_hashes),
                    "demotion_requested_chunks": len(demotable),
                    "model_name": key.model_name,
                },
            )
        )

    def _object_group_ids(self, key: IPCCacheServerKey) -> list[int]:
        """Return every object group id of ``key``'s registered layout."""
        return list(
            range(
                self._ctx.layout_desc_registry.find_attn_desc(
                    key.model_name, key.world_size
                ).num_object_groups
            )
        )
