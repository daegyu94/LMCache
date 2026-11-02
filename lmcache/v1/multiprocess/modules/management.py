# SPDX-License-Identifier: Apache-2.0
"""Management and utility operations for the MPCacheServer."""

# Standard
from collections.abc import Sequence
import threading

# First Party
from lmcache.logging import init_logger
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.multiprocess.cache_generation import StaleL2KeyCollector
from lmcache.v1.multiprocess.custom_types import BlockAllocationRecord
from lmcache.v1.multiprocess.engine_context import MPCacheServerContext
from lmcache.v1.multiprocess.engine_module import (
    DiscoverableModule,
    EngineModule,
    InstanceLivenessTarget,
    ModuleBuildContext,
)
from lmcache.v1.multiprocess.request_handler import HandlerType, request_handler
from lmcache.v1.periodic_thread import (
    PeriodicThread,
    ThreadLevel,
    ThreadRunSummary,
    create_periodic_thread,
)

logger = init_logger(__name__)


class ManagementModule(DiscoverableModule):
    """Handles management and utility operations for the cache engine.

    Owns the lock used during cache clearing and provides handlers for
    ping, chunk-size queries, clear, debug, and block-allocation reporting.
    Also owns the periodic reaper that evicts workers which have gone
    silent, driving the injected liveness targets.

    The clear mode is the context's (``ctx.cache_generation.clear_mode``):
    ``"l1"`` makes ``clear`` free unlocked L1 objects only; ``"generation"``
    also bumps the cache generation and deletes the now-unreachable L2
    objects in the background (see
    ``docs/design/v1/multiprocess/cache_generation.md``).

    Args:
        ctx: The shared engine context.
        liveness_targets: Modules the reaper drives -- the transfer modules
            whose per-instance registrations are refreshed on PING and scanned
            for staleness, plus any state mirror (e.g. ``BlendModule``)
            notified via ``drop_instance_state`` when an instance is reaped.
        worker_reap_timeout_seconds: Silence budget for a ping-proven worker;
            0 disables reaping (no thread is started).
        worker_registration_grace_seconds: Silence budget for a worker that
            registered but never pinged.
        experimental: Experimental features advertised to clients through
            ``get_experimental`` (e.g. ``transfer_query``).
    """

    module_name = "management"
    # Build last so every liveness target is available; reverse close order
    # then stops the reaper before those targets release their state.
    deferred = True

    @classmethod
    def create(cls, build_ctx: ModuleBuildContext) -> EngineModule | None:
        """Build with every liveness target collected so far."""
        mp_config = build_ctx.mp_config
        return cls(
            build_ctx.engine_context,
            liveness_targets=build_ctx.liveness_targets,
            worker_reap_timeout_seconds=mp_config.worker_reap_timeout_seconds,
            worker_registration_grace_seconds=(
                mp_config.worker_registration_grace_seconds
            ),
            experimental=build_ctx.experimental,
        )

    def __init__(
        self,
        ctx: MPCacheServerContext,
        liveness_targets: Sequence[InstanceLivenessTarget] = (),
        worker_reap_timeout_seconds: float = 0.0,
        worker_registration_grace_seconds: float = 0.0,
        experimental: Sequence[str] = (),
    ) -> None:
        self._ctx = ctx
        self._clear_lock = threading.Lock()
        # Present only in generation mode: tracking L2 keys costs memory and
        # a thread, and only a generation bump makes them safe to delete.
        self._stale_l2_keys: StaleL2KeyCollector | None = (
            StaleL2KeyCollector(ctx.storage_manager)
            if ctx.cache_generation.clear_mode == "generation"
            else None
        )
        self._liveness_targets = tuple(liveness_targets)
        self._reap_timeout = worker_reap_timeout_seconds
        self._reap_grace = worker_registration_grace_seconds
        self._experimental = tuple(experimental)

        # Periodic reaper, started only when reaping is enabled and there is
        # something to scan. Scans every reap_timeout/4, so an instance is
        # reaped between timeout and timeout + interval after its last signal.
        self._reaper: PeriodicThread | None = None
        if self._reap_timeout > 0 and self._liveness_targets:
            reaper = create_periodic_thread(
                name="lmcache-mp-worker-reaper",
                interval=self._reap_timeout / 4,
                execute_fn=self._reap_cycle,
                level=ThreadLevel.MEDIUM,
            )
            reaper.start()
            self._reaper = reaper

    @property
    def context(self) -> MPCacheServerContext:
        """Return the shared engine context. Exposed for testing only."""
        return self._ctx

    def report_status(self) -> dict[str, object]:
        """Return module-specific status information.

        Returns:
            A dict with a ``worker_liveness`` summary when reaping targets
            are present, and in generation clear mode a ``cache_generation``
            entry ``{"generation": int, "stale_l2_gc": {...}}`` (the
            counters of ``StaleL2KeyCollector.report_status``); otherwise
            empty.
        """
        status: dict[str, object] = {}
        if self._liveness_targets:
            tracked = sum(t.tracked_instance_count() for t in self._liveness_targets)
            status["worker_liveness"] = {
                "enabled": self._reaper is not None,
                "reap_timeout_seconds": self._reap_timeout,
                "registration_grace_seconds": self._reap_grace,
                "tracked_instances": tracked,
            }
        if self._stale_l2_keys is not None:
            status["cache_generation"] = {
                "generation": self._ctx.cache_generation.current,
                "stale_l2_gc": self._stale_l2_keys.report_status(),
            }
        return status

    def close(self) -> None:
        """Stop the reaper and the stale-L2 collector, if running."""
        if self._reaper is not None:
            self._reaper.stop()
        if self._stale_l2_keys is not None:
            self._stale_l2_keys.close()

    @request_handler(HandlerType.BLOCKING)
    def ping(self, instance_id: int | None) -> bool:
        """Respond to a ping and refresh the sender's liveness.

        Args:
            instance_id: The sender's worker instance ID, or None for an
                untracked prober (the scheduler adapter). When not None, the
                worker's last-seen time is refreshed on every liveness target.

        Returns:
            Always True.
        """
        if instance_id is not None:
            for target in self._liveness_targets:
                target.touch_instance(instance_id)
        return True

    def _reap_cycle(self) -> ThreadRunSummary:
        """Run one reaper scan: reap stale workers, drop mirrored state.

        Each reaped instance id is passed to ``drop_instance_state`` on every
        target; it is a no-op for targets that mirror nothing for that id.

        Returns:
            A summary recording how many instances were reaped this scan.
        """
        reaped: list[int] = []
        for target in self._liveness_targets:
            reaped.extend(
                target.reap_stale_instances(self._reap_timeout, self._reap_grace)
            )
        for instance_id in reaped:
            for target in self._liveness_targets:
                target.drop_instance_state(instance_id)
        return ThreadRunSummary(success=True, message=f"reaped={len(reaped)}")

    @request_handler()
    def get_chunk_size(self) -> int:
        """Return the chunk size used for KV cache operations.

        Returns:
            The chunk size.
        """
        return self._ctx.finalize_chunk_size()

    @request_handler()
    def negotiate_chunk_size(self, required_chunk_alignment: int) -> int:
        """Negotiate the server chunk size against model geometry.

        Args:
            required_chunk_alignment: The least common multiple of cacheable
                engine group ``tokens_per_block`` values.

        Returns:
            The concrete server chunk size.

        Raises:
            ValueError: If the context cannot bind or validate the requested
                geometry.
        """
        return self._ctx.negotiate_chunk_size(required_chunk_alignment)

    @request_handler()
    def get_experimental(self) -> list[str]:
        """Return the experimental features enabled on this server."""
        return list(self._experimental)

    @request_handler(HandlerType.BLOCKING)
    def clear(self, force: bool = False) -> None:
        """Clear all stored KV cache data from the storage manager.

        In ``"l1"`` clear mode, removes L1 objects (only unlocked ones
        unless ``force``); L2 is untouched.

        In ``"generation"`` clear mode, first bumps the cache generation, so
        every object stored before the call -- in L1 or L2, including stores
        still in flight -- becomes unreachable to later lookups, then does
        the same L1 clear to free memory. L2 copies are deleted by a
        background collector; the call never waits for L2. Later lookups
        address the new generation, while a session that predates the bump
        keeps storing in its old one (see the design doc for exact
        semantics).

        Args:
            force: Also remove locked L1 objects, which may corrupt in-flight
                store/prefetch operations.
        """
        with self._clear_lock:
            self._ctx.storage_manager.memcheck()
            if self._stale_l2_keys is not None:
                # Retire before bumping: every key stored up to now hashes
                # under a generation the bump makes unreachable, while no
                # new-generation key can exist yet to be deleted by mistake.
                retired = self._stale_l2_keys.retire_tracked_keys()
                generation = self._ctx.cache_generation.bump()
                logger.info(
                    "Cache generation bumped to %d; %d L2 key(s) scheduled "
                    "for background deletion",
                    generation,
                    retired,
                )
            self._ctx.storage_manager.clear(force=force)
            self._ctx.storage_manager.memcheck()

    @request_handler(operation="noop")
    def debug(self) -> str:
        """Return a simple health-check string.

        Returns:
            The literal string ``"OK"``.
        """
        return "OK"

    @request_handler(
        HandlerType.BLOCKING,
        operation="report_block_allocation",
    )
    def report_block_allocations(
        self,
        instance_id: int,
        model_name: str,
        records: list[BlockAllocationRecord],
    ) -> None:
        """Publish vLLM block allocation records to the EventBus.

        Args:
            instance_id: The scheduler instance ID.
            model_name: The model name from the adapter.
            records: List of BlockAllocationRecord with per-request
                block and token allocation deltas.
        """
        self._ctx.event_bus.publish(
            Event(
                event_type=EventType.MP_VLLM_BLOCK_ALLOCATION,
                metadata={
                    "instance_id": instance_id,
                    "model_name": model_name,
                    "records": records,
                },
            )
        )
