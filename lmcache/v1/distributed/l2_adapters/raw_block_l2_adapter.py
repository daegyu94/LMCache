# SPDX-License-Identifier: Apache-2.0
"""
Raw-block L2 adapter for LMCache MP mode.

Uses RawBlockCore as the synchronous durable engine and adapts it to the
non-blocking L2AdapterInterface contract with separate eventfds for store,
lookup, and load.
"""

# Future
from __future__ import annotations

# Standard
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, Optional, cast
import threading

if TYPE_CHECKING:
    from lmcache.native_storage_ops import Bitmap
    from lmcache.v1.distributed.internal_api import L1MemoryDesc, L2AdapterListener

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.internal_api import L2StoreResult
from lmcache.v1.distributed.l2_adapters.base import (
    L2AdapterInterface,
    L2TaskId,
)
from lmcache.v1.distributed.l2_adapters.config import (
    L2AdapterConfigBase,
    register_l2_adapter_type,
)
from lmcache.v1.distributed.l2_adapters.factory import (
    register_l2_adapter_factory,
)
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.multiprocess.custom_types import RankPlacementInfo
from lmcache.v1.platform import EventNotifier, create_event_notifier
from lmcache.v1.storage_backend.raw_block import (
    DEFAULT_IOURING_QUEUE_DEPTH,
    RawBlockCore,
    RawBlockCoreConfig,
    decode_object_key,
    encode_object_key,
    normalize_raw_block_io_engine,
    normalize_raw_block_placement_ids,
    validate_raw_block_io_options,
)

logger = init_logger(__name__)

RawBlockStoreTaskResult = tuple[
    bool,
    list[ObjectKey],
    list[int],
]


def _normalize_fdp_placement_ids(
    placement_ids: Optional[list[int]],
    *,
    field_name: str = "fdp_placement_ids",
) -> Optional[list[int]]:
    """Validate optional FDP placement identifiers from user configuration."""
    if placement_ids is None:
        return None

    try:
        normalized_placement_ids = normalize_raw_block_placement_ids(
            placement_ids,
            len(placement_ids),
            field_name=field_name,
            allow_none=False,
        )
    except ValueError as e:
        if "placement identifier 0" in str(e):
            logger.warning(
                "raw_block FDP placement identifier 0 is reserved and cannot "
                "be configured explicitly"
            )
            raise ValueError(f"{field_name} must not contain 0") from e
        raise
    normalized = cast(list[int], normalized_placement_ids)

    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{field_name} must not contain duplicates")
    if not normalized:
        raise ValueError(f"{field_name} must not be empty")
    return normalized


def _resolve_fdp_placement_ids_from_dict(
    d: dict, *, fdp_enabled: bool
) -> list[int] | None:
    """Resolve canonical FDP placement identifier config."""
    raw_fdp_placement_ids = d.get("fdp_placement_ids")
    if raw_fdp_placement_ids is not None and not isinstance(
        raw_fdp_placement_ids, list
    ):
        raise ValueError("fdp_placement_ids must be a list")
    return raw_fdp_placement_ids if fdp_enabled else None


def _make_bitmap(size: int) -> "Bitmap":
    # First Party
    from lmcache.native_storage_ops import Bitmap

    return Bitmap(size)


@dataclass
class _PlacementCohort:
    key: str
    signature: tuple[str, str, int] | None
    local_world_size: int
    placement_ids: list[int]
    instance_ranks: dict[int, int]
    rank_instances: dict[int, int]


class _RawBlockPlacementAllocator:
    """Reserve non-overlapping FDP placement-identifier slices per LLM cohort."""

    def __init__(self, placement_ids: list[int]) -> None:
        if not placement_ids:
            raise ValueError("placement_ids must be non-empty")
        self._placement_ids = list(placement_ids)
        self._free_ranges: list[tuple[int, int]] = [(0, len(placement_ids))]
        self._cohorts: dict[str, _PlacementCohort] = {}
        self._instance_to_cohort: dict[int, str] = {}
        self._open_by_signature: dict[tuple[str, str, int], str] = {}
        self._next_auto_id = 0

    def register(
        self,
        instance_id: int,
        engine_type: str,
        model_name: str,
        placement_info: RankPlacementInfo,
    ) -> None:
        """Register one instance and reserve a cohort slice if needed."""
        local_rank = int(placement_info.local_rank)
        local_world_size = int(placement_info.local_world_size)
        if local_world_size <= 0:
            raise ValueError("local_world_size must be positive")
        if local_rank < 0 or local_rank >= local_world_size:
            raise ValueError(
                "local_rank must be in [0, local_world_size), got "
                f"{local_rank} / {local_world_size}"
            )

        existing_key = self._instance_to_cohort.get(instance_id)
        if existing_key is not None:
            cohort = self._cohorts[existing_key]
            existing_rank = cohort.instance_ranks[instance_id]
            if existing_rank != local_rank:
                raise ValueError(
                    f"instance {instance_id} already registered as rank "
                    f"{existing_rank}, got {local_rank}"
                )
            if cohort.local_world_size != local_world_size:
                raise ValueError(
                    f"instance {instance_id} already registered with "
                    f"local_world_size={cohort.local_world_size}, got "
                    f"{local_world_size}"
                )
            return

        cohort = self._get_or_create_cohort(
            engine_type,
            model_name,
            local_world_size,
            placement_info.placement_group_id,
        )
        if cohort.local_world_size != local_world_size:
            raise ValueError(
                f"cohort {cohort.key!r} has local_world_size "
                f"{cohort.local_world_size}, got {local_world_size}"
            )
        rank_owner = cohort.rank_instances.get(local_rank)
        if rank_owner is not None and rank_owner != instance_id:
            raise ValueError(
                f"cohort {cohort.key!r} rank {local_rank} is already "
                f"registered by instance {rank_owner}"
            )

        cohort.instance_ranks[instance_id] = local_rank
        cohort.rank_instances[local_rank] = instance_id
        self._instance_to_cohort[instance_id] = cohort.key
        if (
            cohort.signature is not None
            and len(cohort.rank_instances) >= cohort.local_world_size
            and self._open_by_signature.get(cohort.signature) == cohort.key
        ):
            self._open_by_signature.pop(cohort.signature, None)

    def unregister(self, instance_id: int) -> None:
        """Unregister an instance and free its cohort slice when empty."""
        cohort_key = self._instance_to_cohort.pop(instance_id, None)
        if cohort_key is None:
            return
        cohort = self._cohorts.get(cohort_key)
        if cohort is None:
            return
        local_rank = cohort.instance_ranks.pop(instance_id, None)
        if local_rank is not None:
            cohort.rank_instances.pop(local_rank, None)
        if cohort.instance_ranks:
            if (
                cohort.signature is not None
                and len(cohort.rank_instances) < cohort.local_world_size
            ):
                self._open_by_signature.setdefault(cohort.signature, cohort.key)
            return

        self._cohorts.pop(cohort.key, None)
        if cohort.signature is not None:
            self._open_by_signature.pop(cohort.signature, None)
        self._free_slice(cohort.placement_ids)

    def placement_for_instance(self, instance_id: int) -> int:
        """Return the placement identifier assigned to ``instance_id``."""
        cohort_key = self._instance_to_cohort.get(instance_id)
        if cohort_key is None:
            raise ValueError(f"instance {instance_id} has no FDP placement cohort")
        cohort = self._cohorts[cohort_key]
        local_rank = cohort.instance_ranks[instance_id]
        return cohort.placement_ids[local_rank]

    def snapshot(self) -> dict[str, Any]:
        """Return debug status for report_status."""
        return {
            "free_ids": [
                self._placement_ids[start + i]
                for start, size in self._free_ranges
                for i in range(size)
            ],
            "cohorts": {
                key: {
                    "signature": cohort.signature,
                    "local_world_size": cohort.local_world_size,
                    "placement_ids": list(cohort.placement_ids),
                    "instance_ranks": dict(cohort.instance_ranks),
                }
                for key, cohort in self._cohorts.items()
            },
        }

    def _get_or_create_cohort(
        self,
        engine_type: str,
        model_name: str,
        local_world_size: int,
        placement_group_id: str | None,
    ) -> _PlacementCohort:
        if placement_group_id is not None:
            if not placement_group_id:
                raise ValueError("placement_group_id must be non-empty when set")
            group_id = cast(str, placement_group_id)
            cohort_key = f"group:{group_id}"
            cohort = self._cohorts.get(cohort_key)
            if cohort is not None:
                return cohort
            signature = ("placement_group_id", group_id, local_world_size)
            return self._create_cohort(cohort_key, signature, local_world_size)

        signature = (engine_type, model_name, local_world_size)
        open_cohort_key = self._open_by_signature.get(signature)
        if open_cohort_key is not None:
            return self._cohorts[open_cohort_key]
        cohort_key = f"auto:{self._next_auto_id}"
        self._next_auto_id += 1
        cohort = self._create_cohort(cohort_key, signature, local_world_size)
        self._open_by_signature[signature] = cohort_key
        return cohort

    def _create_cohort(
        self,
        key: str,
        signature: tuple[str, str, int] | None,
        local_world_size: int,
    ) -> _PlacementCohort:
        placement_ids = self._allocate_slice(local_world_size)
        cohort = _PlacementCohort(
            key=key,
            signature=signature,
            local_world_size=local_world_size,
            placement_ids=placement_ids,
            instance_ranks={},
            rank_instances={},
        )
        self._cohorts[key] = cohort
        return cohort

    def _allocate_slice(self, size: int) -> list[int]:
        for idx, (start, length) in enumerate(self._free_ranges):
            if length < size:
                continue
            allocated = self._placement_ids[start : start + size]
            if length == size:
                self._free_ranges.pop(idx)
            else:
                self._free_ranges[idx] = (start + size, length - size)
            return allocated
        raise ValueError(
            f"not enough free FDP placement identifiers for local_world_size={size}"
        )

    def _free_slice(self, placement_ids: list[int]) -> None:
        if not placement_ids:
            return
        start = self._placement_ids.index(placement_ids[0])
        size = len(placement_ids)
        self._free_ranges.append((start, size))
        self._free_ranges.sort()
        merged: list[tuple[int, int]] = []
        for range_start, range_size in self._free_ranges:
            if not merged:
                merged.append((range_start, range_size))
                continue
            prev_start, prev_size = merged[-1]
            if prev_start + prev_size == range_start:
                merged[-1] = (prev_start, prev_size + range_size)
            else:
                merged.append((range_start, range_size))
        self._free_ranges = merged


class RawBlockL2AdapterConfig(L2AdapterConfigBase):
    """Configuration object for the built-in raw-block MP L2 adapter."""

    def __init__(
        self,
        *,
        device_path: str,
        slot_bytes: int,
        capacity_bytes: int = 0,
        use_odirect: bool = True,
        block_align: int = 4096,
        header_bytes: int = 4096,
        meta_total_bytes: int = 256 * 1024 * 1024,
        meta_magic: str = "LMCIDX01",
        meta_version: int = 1,
        meta_checkpoint_interval_sec: int = 60,
        meta_idle_quiet_ms: int = 100,
        meta_enable_periodic: bool = True,
        load_checkpoint_on_init: bool = True,
        meta_verify_on_load: bool = True,
        enable_zero_copy: bool = True,
        io_engine: str = "posix",
        iouring_queue_depth: int = DEFAULT_IOURING_QUEUE_DEPTH,
        use_uring_cmd: bool = False,
        max_data_transfer_size: int = 0,
        fdp_enabled: bool = False,
        fdp_placement_ids: Optional[list[int]] = None,
        num_store_workers: int = 2,
        num_lookup_workers: int = 1,
        num_load_workers: int = 4,
    ):
        """Initialize raw-block MP adapter configuration.

        Args:
            device_path: Raw device path or pre-sized file path used for L2.
            slot_bytes: Fixed data-slot size in bytes.
            capacity_bytes: Optional cap on usable bytes; zero uses device size.
            use_odirect: Whether to open the raw path with O_DIRECT.
            block_align: Required block alignment in bytes.
            header_bytes: Per-slot header reservation in bytes.
            meta_total_bytes: Reserved metadata checkpoint region size.
            meta_magic: Eight-byte ASCII metadata checkpoint magic.
            meta_version: Metadata checkpoint version.
            meta_checkpoint_interval_sec: Periodic checkpoint interval.
            meta_idle_quiet_ms: Quiet period before periodic checkpoints.
            meta_enable_periodic: Whether to run the checkpoint thread.
            load_checkpoint_on_init: Whether to load existing checkpoint metadata.
            meta_verify_on_load: Whether recovery verifies slot headers.
            enable_zero_copy: Whether to use aligned direct-buffer I/O.
            io_engine: Raw-block I/O engine: ``"posix"`` or ``"io_uring"``.
            iouring_queue_depth: Queue depth for the Rust io_uring engine.
            use_uring_cmd: Whether to use NVMe io_uring_cmd passthrough.
            max_data_transfer_size: Max data transfer size for a single request.
            fdp_enabled: Enable NVMe Flexible Data Placement discovery and
                non-zero placement-identifier registration for raw-block writes.
            fdp_placement_ids: Optional exact non-zero FDP placement identifier
                list to use for data writes. If omitted, all device-reported
                placement identifiers except 0 are used.
            num_store_workers: Number of store worker threads.
            num_lookup_workers: Number of lookup worker threads.
            num_load_workers: Number of load worker threads.
        """
        super().__init__()
        self.device_path = device_path
        self.slot_bytes = int(slot_bytes)
        self.capacity_bytes = int(capacity_bytes)
        self.use_odirect = bool(use_odirect)
        self.block_align = int(block_align)
        self.header_bytes = int(header_bytes)
        self.meta_total_bytes = int(meta_total_bytes)
        self.meta_magic = meta_magic
        self.meta_version = int(meta_version)
        self.meta_checkpoint_interval_sec = int(meta_checkpoint_interval_sec)
        self.meta_idle_quiet_ms = int(meta_idle_quiet_ms)
        self.meta_enable_periodic = bool(meta_enable_periodic)
        self.load_checkpoint_on_init = bool(load_checkpoint_on_init)
        self.meta_verify_on_load = bool(meta_verify_on_load)
        self.enable_zero_copy = bool(enable_zero_copy)
        self.io_engine = normalize_raw_block_io_engine(io_engine)
        self.iouring_queue_depth = int(iouring_queue_depth)
        validate_raw_block_io_options(
            iouring_queue_depth=self.iouring_queue_depth,
        )
        self.use_uring_cmd = bool(use_uring_cmd)
        self.max_data_transfer_size = int(max_data_transfer_size)
        self.fdp_enabled = bool(fdp_enabled)
        if self.fdp_enabled and (
            self.io_engine != "io_uring" or not self.use_uring_cmd
        ):
            raise ValueError(
                "fdp_enabled requires io_engine='io_uring' and use_uring_cmd=true"
            )
        self.fdp_placement_ids = (
            _normalize_fdp_placement_ids(fdp_placement_ids)
            if self.fdp_enabled
            else None
        )
        self.num_store_workers = int(num_store_workers)
        self.num_lookup_workers = int(num_lookup_workers)
        self.num_load_workers = int(num_load_workers)

    @classmethod
    def from_dict(cls, d: dict) -> "RawBlockL2AdapterConfig":
        """Build and validate a raw-block config from ``--l2-adapter`` JSON."""
        device_path = d.get("device_path")
        if not isinstance(device_path, str) or not device_path:
            raise ValueError("device_path must be a non-empty string")
        if "per_tp_device_paths" in d:
            raise ValueError(
                "per_tp_device_paths is not supported in MP raw_block mode"
            )
        if not bool(d.get("persist_enabled", True)):
            raise ValueError("raw_block requires persist_enabled=true")

        slot_bytes = d.get("slot_bytes")
        if not isinstance(slot_bytes, int) or slot_bytes <= 0:
            raise ValueError("slot_bytes must be a positive integer")

        block_align = int(d.get("block_align", 4096))
        header_bytes = int(d.get("header_bytes", 4096))
        meta_total_bytes = int(d.get("meta_total_bytes", 256 * 1024 * 1024))
        capacity_bytes = int(d.get("capacity_bytes", 0))
        io_engine = normalize_raw_block_io_engine(
            d.get("io_engine"),
            use_iouring=d.get("use_iouring"),
            use_uring=d.get("use_uring"),
        )
        iouring_queue_depth = int(
            d.get("iouring_queue_depth", DEFAULT_IOURING_QUEUE_DEPTH)
        )
        use_uring_cmd = bool(d.get("use_uring_cmd", False))
        max_data_transfer_size = int(d.get("max_data_transfer_size", 0))
        fdp_enabled = bool(d.get("fdp_enabled", False))
        fdp_placement_ids = _resolve_fdp_placement_ids_from_dict(
            d, fdp_enabled=fdp_enabled
        )
        if block_align <= 0:
            raise ValueError("block_align must be > 0")
        if slot_bytes % block_align != 0:
            raise ValueError("slot_bytes must be a multiple of block_align")
        if header_bytes % block_align != 0:
            raise ValueError("header_bytes must be a multiple of block_align")
        if meta_total_bytes % block_align != 0:
            raise ValueError("meta_total_bytes must be a multiple of block_align")
        if slot_bytes < header_bytes + 1:
            raise ValueError("slot_bytes must be >= header_bytes + 1")
        if capacity_bytes > 0 and capacity_bytes <= meta_total_bytes:
            raise ValueError("capacity_bytes must leave space for at least one slot")
        validate_raw_block_io_options(
            iouring_queue_depth=iouring_queue_depth,
        )
        if use_uring_cmd and io_engine != "io_uring":
            raise ValueError("use_uring_cmd requires io_uring io_engine")
        if fdp_enabled and (io_engine != "io_uring" or not use_uring_cmd):
            raise ValueError(
                "fdp_enabled requires io_engine='io_uring' and use_uring_cmd=true"
            )

        worker_defaults = {
            "num_store_workers": 2,
            "num_lookup_workers": 1,
            "num_load_workers": 4,
        }
        worker_counts: dict[str, int] = {}
        for field_name, default in worker_defaults.items():
            value = int(d.get(field_name, default))
            if value <= 0:
                raise ValueError(f"{field_name} must be > 0")
            worker_counts[field_name] = value

        return cls(
            device_path=device_path,
            slot_bytes=slot_bytes,
            capacity_bytes=capacity_bytes,
            use_odirect=bool(d.get("use_odirect", True)),
            block_align=block_align,
            header_bytes=header_bytes,
            meta_total_bytes=meta_total_bytes,
            meta_magic=str(d.get("meta_magic", "LMCIDX01")),
            meta_version=int(d.get("meta_version", 1)),
            meta_checkpoint_interval_sec=int(d.get("meta_checkpoint_interval_sec", 60)),
            meta_idle_quiet_ms=int(d.get("meta_idle_quiet_ms", 100)),
            meta_enable_periodic=bool(d.get("meta_enable_periodic", True)),
            load_checkpoint_on_init=bool(d.get("load_checkpoint_on_init", True)),
            meta_verify_on_load=bool(d.get("meta_verify_on_load", True)),
            enable_zero_copy=bool(d.get("enable_zero_copy", True)),
            io_engine=io_engine,
            iouring_queue_depth=iouring_queue_depth,
            use_uring_cmd=use_uring_cmd,
            max_data_transfer_size=max_data_transfer_size,
            fdp_enabled=fdp_enabled,
            fdp_placement_ids=fdp_placement_ids,
            num_store_workers=worker_counts["num_store_workers"],
            num_lookup_workers=worker_counts["num_lookup_workers"],
            num_load_workers=worker_counts["num_load_workers"],
        )

    @classmethod
    def help(cls) -> str:
        """Return human-readable raw-block adapter configuration help."""
        return (
            "raw_block L2 adapter config fields:\n"
            "- device_path (str): raw device or file path (required)\n"
            "- slot_bytes (int): slot size in bytes, aligned to block_align "
            "(required)\n"
            "- capacity_bytes (int): optional usable capacity cap "
            "(default 0 = device size)\n"
            "- use_odirect (bool): enable O_DIRECT raw I/O (default true)\n"
            "- block_align (int): required block alignment in bytes (default 4096)\n"
            "- header_bytes (int): per-slot header reservation (default 4096)\n"
            "- meta_total_bytes (int): reserved metadata checkpoint region "
            "(default 256MiB)\n"
            "- meta_magic (str): 8-byte metadata magic (default LMCIDX01)\n"
            "- meta_version (int): metadata version (default 1)\n"
            "- meta_checkpoint_interval_sec (int): periodic checkpoint interval "
            "(default 60)\n"
            "- meta_idle_quiet_ms (int): quiet period before checkpoint (default 100)\n"
            "- meta_enable_periodic (bool): enable periodic checkpointing "
            "(default true)\n"
            "- load_checkpoint_on_init (bool): load existing metadata checkpoint "
            "on startup (default true)\n"
            "- meta_verify_on_load (bool): validate slot headers on recovery "
            "(default true)\n"
            "- enable_zero_copy (bool): use aligned direct buffers when possible "
            "(default true)\n"
            "- io_engine (str): posix or io_uring (default posix)\n"
            "- iouring_queue_depth (int): Rust io_uring queue depth "
            f"(default {DEFAULT_IOURING_QUEUE_DEPTH})\n"
            "- use_uring_cmd (bool): enable NVMe io_uring_cmd path "
            "(default false, requires io_uring as the io_engine)\n"
            "- max_data_transfer_size (int): for a single I/O request "
            "(0: (default) auto detect limit splitting, > 0: explicit split, "
            "< 0: auto detect limit splitting)\n"
            "- fdp_enabled (bool): enable FDP discovery (default false)\n"
            "- fdp_placement_ids (list[int]): exact non-zero FDP placement "
            "identifiers to use; omitted uses all device-reported non-zero "
            "identifiers\n"
            "- num_store_workers (int): store worker threads (default 2)\n"
            "- num_lookup_workers (int): lookup worker threads (default 1)\n"
            "- num_load_workers (int): load worker threads (default 4)"
        )

    def to_core_config(self) -> RawBlockCoreConfig:
        """Convert this adapter config to the shared RawBlockCore config."""
        return RawBlockCoreConfig(
            device_path=self.device_path,
            capacity_bytes=self.capacity_bytes,
            block_align=self.block_align,
            header_bytes=self.header_bytes,
            slot_bytes=self.slot_bytes,
            use_odirect=self.use_odirect,
            enable_zero_copy=self.enable_zero_copy,
            meta_total_bytes=self.meta_total_bytes,
            meta_magic=self.meta_magic.encode("ascii"),
            meta_version=self.meta_version,
            meta_checkpoint_interval_sec=self.meta_checkpoint_interval_sec,
            meta_idle_quiet_ms=self.meta_idle_quiet_ms,
            meta_enable_periodic=self.meta_enable_periodic,
            load_checkpoint_on_init=self.load_checkpoint_on_init,
            meta_verify_on_load=self.meta_verify_on_load,
            io_engine=self.io_engine,
            iouring_queue_depth=self.iouring_queue_depth,
            use_uring_cmd=self.use_uring_cmd,
            max_data_transfer_size=self.max_data_transfer_size,
        )


class RawBlockL2Adapter(L2AdapterInterface):
    """MP L2 adapter that persists KV objects into raw-block slots."""

    def __init__(
        self,
        config: RawBlockL2AdapterConfig,
        l1_memory_desc: "Optional[L1MemoryDesc]" = None,
    ):
        """Initialize the MP raw-block L2 adapter.

        Args:
            config: Validated raw-block adapter configuration.
            l1_memory_desc: Optional L1 allocation descriptor used to validate
                O_DIRECT alignment compatibility.

        Raises:
            ValueError: If O_DIRECT is enabled and L1 alignment is insufficient.
            RuntimeError: If the shared core cannot open or recover the raw
                device.

        Notes:
            Resources created before an initialization failure are closed before
            the exception is re-raised.
        """
        super().__init__()
        if (
            (config.use_odirect or config.io_engine == "io_uring")
            and l1_memory_desc is not None
            and l1_memory_desc.align_bytes < config.block_align
        ):
            raise ValueError(
                "raw_block requires l1_align_bytes >= block_align when "
                "use_odirect=true or io_engine=io_uring"
            )

        self._closed = False
        self._core: RawBlockCore
        self._store_efd: EventNotifier | None = None
        self._lookup_efd: EventNotifier | None = None
        self._load_efd: EventNotifier | None = None
        self._store_pool: ThreadPoolExecutor
        self._lookup_pool: ThreadPoolExecutor
        self._load_pool: ThreadPoolExecutor

        try:
            self._core = RawBlockCore(config.to_core_config(), key_namespace="object")
            self._fdp_enabled = bool(config.fdp_enabled)
            self._fdp_discovered_status: list[tuple[int, int]] = []
            self._fdp_placement_ids: list[int] = []
            self._fdp_allocator: _RawBlockPlacementAllocator | None = None
            if self._fdp_enabled:
                self._configure_fdp(config.fdp_placement_ids)
                self._fdp_allocator = _RawBlockPlacementAllocator(
                    self._fdp_placement_ids
                )
            if config.io_engine == "io_uring":
                logger.warning(
                    "RawBlockL2Adapter: MP raw_block uses io_uring without "
                    "fixed-buffer registration; zero-copy fixed buffers are "
                    "disabled unless registered by a future MP allocator path"
                )
            self._max_capacity_bytes = int(
                self._core.report_status().get("usable_capacity_bytes", 0)
            )
            self._seed_usage_from_core_snapshot()

            self._store_efd = create_event_notifier()
            self._lookup_efd = create_event_notifier()
            self._load_efd = create_event_notifier()

            self._store_pool = ThreadPoolExecutor(
                max_workers=config.num_store_workers,
                thread_name_prefix="rawblk-store",
            )
            self._lookup_pool = ThreadPoolExecutor(
                max_workers=config.num_lookup_workers,
                thread_name_prefix="rawblk-lookup",
            )
            self._load_pool = ThreadPoolExecutor(
                max_workers=config.num_load_workers,
                thread_name_prefix="rawblk-load",
            )
        except Exception:
            self._cleanup_after_init_failure()
            raise

        self._lock = threading.Lock()
        self._next_task_id: L2TaskId = 0

        self._completed_store_tasks: dict[L2TaskId, L2StoreResult] = {}
        self._completed_lookup_tasks: dict[L2TaskId, Bitmap] = {}
        self._completed_load_tasks: dict[L2TaskId, Bitmap] = {}

        self._store_inflight_tasks: int = 0
        self._lookup_inflight_tasks: int = 0
        self._load_inflight_tasks: int = 0

    def get_store_event_fd(self) -> int:
        """Return the eventfd signaled when store tasks complete."""
        if self._store_efd is None:
            return -1
        return self._store_efd.fileno()

    def get_lookup_and_lock_event_fd(self) -> int:
        """Return the eventfd signaled when lookup-and-lock tasks complete."""
        if self._lookup_efd is None:
            return -1
        return self._lookup_efd.fileno()

    def get_load_event_fd(self) -> int:
        """Return the eventfd signaled when load tasks complete."""
        if self._load_efd is None:
            return -1
        return self._load_efd.fileno()

    def submit_store_task(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
    ) -> L2TaskId:
        """Submit a non-blocking raw-block store task.

        Args:
            keys: Object keys to persist.
            objects: Memory objects containing payloads for ``keys``.

        Returns:
            Task ID that can be observed through ``pop_completed_store_tasks``.

        Raises:
            ValueError: If either list is empty or the lengths differ.
        """
        return self.submit_store_task_for_instance(None, keys, objects)

    def submit_store_task_for_instance(
        self,
        instance_id: int | None,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
    ) -> L2TaskId:
        """Submit a raw-block store task with optional source-instance placement."""
        if not keys or not objects:
            raise ValueError("keys and objects must be non-empty")
        if len(keys) != len(objects):
            raise ValueError("keys and objects must have the same length")

        with self._lock:
            self._raise_if_closed_locked()
            placement_ids = self._assign_fdp_placement_ids_locked(
                instance_id, len(keys)
            )
            task_id = self._get_next_task_id_locked()
            self._store_inflight_tasks += 1
        try:
            future = self._store_pool.submit(
                self._run_store_task, list(keys), list(objects), placement_ids
            )
        except Exception:
            with self._lock:
                self._store_inflight_tasks -= 1
            raise
        future.add_done_callback(partial(self._finish_store_task, task_id))
        return task_id

    def pop_completed_store_tasks(self) -> dict[L2TaskId, L2StoreResult]:
        """Drain and return completed store task results."""
        with self._lock:
            completed = self._completed_store_tasks
            self._completed_store_tasks = {}
        return completed

    def submit_lookup_and_lock_task(
        self, keys: list[ObjectKey], layout_desc: MemoryLayoutDesc
    ) -> L2TaskId:
        """Submit a non-blocking lookup-and-lock task.

        Args:
            keys: Object keys to look up in raw-block L2.

        Returns:
            Task ID whose bitmap can be queried with
            ``query_lookup_and_lock_result``.

        Raises:
            ValueError: If ``keys`` is empty.
        """
        if not keys:
            raise ValueError("keys must be non-empty")
        with self._lock:
            self._raise_if_closed_locked()
            task_id = self._get_next_task_id_locked()
            self._lookup_inflight_tasks += 1
        try:
            future = self._lookup_pool.submit(self._run_lookup_task, list(keys))
        except Exception:
            with self._lock:
                self._lookup_inflight_tasks -= 1
            raise
        future.add_done_callback(partial(self._finish_lookup_task, task_id, len(keys)))
        return task_id

    def query_lookup_and_lock_result(self, task_id: L2TaskId) -> Bitmap | None:
        """Return and remove a completed lookup bitmap if available."""
        with self._lock:
            return self._completed_lookup_tasks.pop(task_id, None)

    def submit_unlock(self, keys: list[ObjectKey]) -> None:
        """Release L2 locks acquired by lookup-and-lock."""
        encoded_keys = [encode_object_key(key).encoded for key in keys]
        self._core.unlock_many(encoded_keys)

    def submit_load_task(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
    ) -> L2TaskId:
        """Submit a non-blocking raw-block load task.

        Args:
            keys: Object keys to load.
            objects: Caller-provided destination buffers.

        Returns:
            Task ID whose bitmap can be queried with ``query_load_result``.

        Raises:
            ValueError: If either list is empty or the lengths differ.
        """
        if not keys or not objects:
            raise ValueError("keys and objects must be non-empty")
        if len(keys) != len(objects):
            raise ValueError("keys and objects must have the same length")

        with self._lock:
            self._raise_if_closed_locked()
            task_id = self._get_next_task_id_locked()
            self._load_inflight_tasks += 1
        try:
            future = self._load_pool.submit(
                self._run_load_task, list(keys), list(objects)
            )
        except Exception:
            with self._lock:
                self._load_inflight_tasks -= 1
            raise
        future.add_done_callback(partial(self._finish_load_task, task_id, len(keys)))
        return task_id

    def query_load_result(self, task_id: L2TaskId) -> Bitmap | None:
        """Return and remove a completed load bitmap if available."""
        with self._lock:
            return self._completed_load_tasks.pop(task_id, None)

    def delete(self, keys: list[ObjectKey]) -> None:
        """Delete keys from raw-block L2 and notify listeners for removals."""
        encoded_keys = [encode_object_key(key).encoded for key in keys]
        metas = self._core.get_metadata_many(encoded_keys)
        deleted_bitmap = self._core.delete_many(encoded_keys, force=False)
        deleted_keys: list[ObjectKey] = []
        deleted_sizes: list[int] = []
        for key, meta, deleted in zip(keys, metas, deleted_bitmap, strict=False):
            if not deleted:
                continue
            deleted_keys.append(key)
            deleted_sizes.append(0 if meta is None else int(self._core.slot_bytes))
        if deleted_keys:
            try:
                self._notify_keys_deleted(deleted_keys, deleted_sizes)
            except Exception as e:
                logger.warning("RawBlockL2Adapter delete notification failed: %s", e)

    def register_listener(self, listener: "L2AdapterListener") -> None:
        """Register a listener and seed it with currently indexed keys."""
        super().register_listener(listener)
        keys = self._snapshot_indexed_object_keys()
        if not keys:
            return
        try:
            listener.on_l2_keys_stored(keys, [0] * len(keys))
        except Exception as e:
            logger.warning(
                "RawBlockL2Adapter listener recovery bootstrap failed: %s", e
            )

    def close(self) -> None:
        """Wait for worker pools, close the core, and close eventfds."""
        with self._lock:
            if self._closed:
                return
            self._closed = True

        self._store_pool.shutdown(wait=True)
        self._lookup_pool.shutdown(wait=True)
        self._load_pool.shutdown(wait=True)

        self._core.close()

        with self._lock:
            store_efd = self._store_efd
            lookup_efd = self._lookup_efd
            load_efd = self._load_efd
            self._store_efd = None
            self._lookup_efd = None
            self._load_efd = None

        if store_efd is not None:
            store_efd.close()
        if lookup_efd is not None:
            lookup_efd.close()
        if load_efd is not None:
            load_efd.close()

    def report_status(self) -> dict:
        """Return adapter health, task counters, and core status."""
        core_status = self._core.report_status()
        with self._lock:
            return {
                "is_healthy": core_status.get("is_healthy", True) and not self._closed,
                "type": "RawBlockL2Adapter",
                "store_inflight_task_count": self._store_inflight_tasks,
                "lookup_inflight_task_count": self._lookup_inflight_tasks,
                "load_inflight_task_count": self._load_inflight_tasks,
                "fdp_enabled": self._fdp_enabled,
                "fdp_discovered_status": list(self._fdp_discovered_status),
                "fdp_placement_ids": list(self._fdp_placement_ids),
                "fdp_placement_allocator": (
                    None
                    if self._fdp_allocator is None
                    else self._fdp_allocator.snapshot()
                ),
                "completed_store_task_count": len(self._completed_store_tasks),
                "completed_lookup_task_count": len(self._completed_lookup_tasks),
                "completed_load_task_count": len(self._completed_load_tasks),
                "core": core_status,
            }

    def _configure_fdp(self, configured_ids: Optional[list[int]]) -> None:
        """Fetch and register FDP placement identifiers for this adapter."""
        try:
            discovered = self._core.fetch_fdp_status()
        except Exception as e:
            raise RuntimeError("raw_block FDP status query failed") from e
        if not discovered:
            raise RuntimeError(
                "raw_block FDP enabled but device returned no placement identifiers"
            )

        self._fdp_discovered_status = [
            (int(pid), int(ruhid)) for pid, ruhid in discovered
        ]
        discovered_ids = [pid for pid, _ in self._fdp_discovered_status]
        usable_ids = [pid for pid in discovered_ids if pid != 0]
        if not usable_ids:
            raise RuntimeError(
                "raw_block FDP enabled but device returned no non-zero "
                "placement identifiers"
            )

        if configured_ids is not None:
            configured_set = set(configured_ids)
            usable_set = set(usable_ids)
            if configured_set != usable_set:
                raise RuntimeError(
                    "raw_block FDP placement identifier list does not match "
                    f"device identifiers: configured={configured_ids} "
                    f"device={usable_ids}"
                )
            self._fdp_placement_ids = list(configured_ids)
        else:
            self._fdp_placement_ids = usable_ids

        logger.info(
            "RawBlockL2Adapter registered FDP placement identifiers: %s",
            self._fdp_placement_ids,
        )

    def _raise_if_closed_locked(self) -> None:
        if self._closed:
            raise RuntimeError("RawBlockL2Adapter is closed")

    def _get_next_task_id_locked(self) -> L2TaskId:
        task_id = self._next_task_id
        self._next_task_id += 1
        return task_id

    def register_instance_placement(
        self,
        instance_id: int,
        engine_type: str,
        model_name: str,
        placement_info: RankPlacementInfo,
    ) -> None:
        """Register a serving-engine instance for FDP rank isolation."""
        if not self._fdp_enabled:
            return
        with self._lock:
            self._raise_if_closed_locked()
            if self._fdp_allocator is None:
                raise RuntimeError("raw_block FDP placement allocator is unavailable")
            self._fdp_allocator.register(
                instance_id, engine_type, model_name, placement_info
            )

    def unregister_instance_placement(self, instance_id: int) -> None:
        """Release a serving-engine instance's FDP placement registration."""
        if not self._fdp_enabled:
            return
        with self._lock:
            if self._fdp_allocator is not None:
                self._fdp_allocator.unregister(instance_id)

    def _assign_fdp_placement_ids_locked(
        self, instance_id: int | None, count: int
    ) -> list[int] | None:
        """Return one FDP placement identifier repeated for the store batch."""
        if not self._fdp_enabled:
            return None
        if not self._fdp_placement_ids or self._fdp_allocator is None:
            raise RuntimeError("raw_block FDP placement identifiers are not configured")
        if instance_id is None:
            raise RuntimeError(
                "raw_block FDP placement requires a registered origin instance"
            )
        placement_id = self._fdp_allocator.placement_for_instance(instance_id)
        return [placement_id] * count

    def _seed_usage_from_core_snapshot(self) -> None:
        """Seed byte counters for entries recovered by RawBlockCore startup."""
        recovered_keys = self._snapshot_indexed_object_keys()
        if not recovered_keys:
            return

        slot_bytes = int(self._core.slot_bytes)
        total_delta = len(recovered_keys) * slot_bytes
        by_salt: dict[str, int] = {}
        for key in recovered_keys:
            by_salt[key.cache_salt] = by_salt.get(key.cache_salt, 0) + slot_bytes

        with self._usage_lock:
            self._total_bytes_used += total_delta
            for salt, delta in by_salt.items():
                self._bytes_by_cache_salt[salt] = (
                    self._bytes_by_cache_salt.get(salt, 0) + delta
                )

    def _snapshot_indexed_object_keys(self) -> list[ObjectKey]:
        """Return decoded ObjectKeys for all indexed raw-block entries."""
        keys: list[ObjectKey] = []
        for encoded_key in self._core.snapshot_indexed_keys():
            try:
                keys.append(decode_object_key(encoded_key))
            except Exception as e:
                logger.warning(
                    "RawBlockL2Adapter could not decode indexed key %r: %s",
                    encoded_key,
                    e,
                )
        return keys

    def _run_store_task(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
        placement_ids: list[int] | None,
    ) -> RawBlockStoreTaskResult:
        """Persist one submitted store batch in the worker pool.

        Args:
            keys: Object keys submitted for storage.
            objects: Payload buffers aligned with ``keys``.

        Returns:
            A 3-tuple containing:

            - task success for the whole batch
            - newly stored object keys
            - raw-block slot byte charges aligned with the newly stored keys
        """
        specs = [encode_object_key(key) for key in keys]
        put_result = self._core.put_many(specs, objects, placement_ids=placement_ids)
        stored_encoded = set(put_result.stored_keys)
        slot_bytes = int(self._core.slot_bytes)
        stored_keys: list[ObjectKey] = []
        stored_sizes: list[int] = []
        for key, spec in zip(keys, specs, strict=False):
            if spec.encoded not in stored_encoded:
                continue
            stored_keys.append(key)
            stored_sizes.append(slot_bytes)
        return all(put_result.results), stored_keys, stored_sizes

    def _finish_store_task(
        self,
        task_id: L2TaskId,
        future: Future[RawBlockStoreTaskResult],
    ) -> None:
        success = False
        stored_keys: list[ObjectKey] = []
        stored_sizes: list[int] = []
        bytes_transferred = 0
        try:
            success, stored_keys, stored_sizes = future.result()
            bytes_transferred = sum(stored_sizes)
        except Exception as e:
            logger.error("RawBlockL2Adapter store task %d failed: %s", task_id, e)
        with self._lock:
            self._store_inflight_tasks -= 1
            self._completed_store_tasks[task_id] = L2StoreResult(
                success, bytes_transferred
            )
            event_fd = self._store_efd
        if stored_keys:
            try:
                self._notify_keys_stored(stored_keys, stored_sizes)
            except Exception as e:
                logger.warning("RawBlockL2Adapter store notification failed: %s", e)
        self._signal_event_fd(event_fd)

    def _run_lookup_task(self, keys: list[ObjectKey]) -> Bitmap:
        specs = [encode_object_key(key) for key in keys]
        exists = self._core.exists_many([spec.encoded for spec in specs], lock=True)
        bitmap = _make_bitmap(len(keys))
        for i, ok in enumerate(exists):
            if ok:
                bitmap.set(i)
        return bitmap

    def _finish_lookup_task(
        self, task_id: L2TaskId, bitmap_size: int, future: Future[Any]
    ) -> None:
        bitmap = _make_bitmap(bitmap_size)
        try:
            bitmap = future.result()
        except Exception as e:
            logger.error("RawBlockL2Adapter lookup task %d failed: %s", task_id, e)
        with self._lock:
            self._lookup_inflight_tasks -= 1
            self._completed_lookup_tasks[task_id] = bitmap
            event_fd = self._lookup_efd
        self._signal_event_fd(event_fd)

    def _run_load_task(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
    ) -> tuple[Bitmap, list[ObjectKey]]:
        specs = [encode_object_key(key) for key in keys]
        results = self._core.load_many_into([spec.encoded for spec in specs], objects)
        bitmap = _make_bitmap(len(keys))
        accessed_keys: list[ObjectKey] = []
        for i, ok in enumerate(results):
            if ok:
                bitmap.set(i)
                accessed_keys.append(keys[i])
        return bitmap, accessed_keys

    def _finish_load_task(
        self, task_id: L2TaskId, bitmap_size: int, future: Future[Any]
    ) -> None:
        bitmap = _make_bitmap(bitmap_size)
        accessed_keys: list[ObjectKey] = []
        try:
            bitmap, accessed_keys = future.result()
        except Exception as e:
            logger.error("RawBlockL2Adapter load task %d failed: %s", task_id, e)
        with self._lock:
            self._load_inflight_tasks -= 1
            self._completed_load_tasks[task_id] = bitmap
            event_fd = self._load_efd
        if accessed_keys:
            try:
                self._notify_keys_accessed(accessed_keys)
            except Exception as e:
                logger.warning("RawBlockL2Adapter access notification failed: %s", e)
        self._signal_event_fd(event_fd)

    def _signal_event_fd(self, event_fd: EventNotifier | None) -> None:
        try:
            if event_fd is not None:
                event_fd.notify()
        except OSError:
            logger.debug("event notifier was closed before signaling")

    def _cleanup_after_init_failure(self) -> None:
        for pool_name in ("_store_pool", "_lookup_pool", "_load_pool"):
            pool = getattr(self, pool_name, None)
            if pool is not None:
                pool.shutdown(wait=False, cancel_futures=True)
                setattr(self, pool_name, None)

        core = getattr(self, "_core", None)
        if core is not None:
            core.close()

        for fd_name in ("_store_efd", "_lookup_efd", "_load_efd"):
            fd = getattr(self, fd_name, None)
            if fd is not None:
                fd.close()
                setattr(self, fd_name, None)

        self._closed = True


register_l2_adapter_type("raw_block", RawBlockL2AdapterConfig)


def _create_raw_block_adapter(
    config: L2AdapterConfigBase,
    l1_memory_desc: "Optional[L1MemoryDesc]" = None,
) -> L2AdapterInterface:
    return RawBlockL2Adapter(config, l1_memory_desc)  # type: ignore[arg-type]


register_l2_adapter_factory("raw_block", _create_raw_block_adapter)
