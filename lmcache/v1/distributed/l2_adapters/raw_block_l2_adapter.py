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
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from functools import partial
from typing import TYPE_CHECKING, Any, Optional
import json
import os
import threading
import time

if TYPE_CHECKING:
    from lmcache.native_storage_ops import Bitmap
    from lmcache.v1.distributed.internal_api import L1MemoryDesc

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import ObjectKey
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
from lmcache.v1.storage_backend.raw_block import (
    RawBlockCore,
    RawBlockCoreConfig,
    decode_object_key,
    encode_object_key,
)

logger = init_logger(__name__)

RawBlockStoreTaskResult = tuple[
    bool,
    list[ObjectKey],
    list[int],
    list[ObjectKey],
    list[int],
]


class _LatencyJsonlRecorder:
    """Thread-safe JSONL recorder for raw-block L2 latency samples."""

    def __init__(self, path: str) -> None:
        absolute_path = os.path.abspath(path)
        parent = os.path.dirname(absolute_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._file = open(absolute_path, "w", buffering=1)
        self._lock = threading.Lock()

    def record(self, payload: dict[str, Any]) -> None:
        try:
            with self._lock:
                self._file.write(json.dumps(payload, separators=(",", ":")) + "\n")
        except (OSError, ValueError):
            logger.warning(
                "RawBlockL2Adapter failed to record an L2 latency sample",
                exc_info=True,
            )

    def close(self) -> None:
        with self._lock:
            self._file.close()


def _parse_fdp_ruh_ids(raw_ids: Any) -> tuple[int, ...]:
    """Parse FDP reclaim unit handle IDs from adapter config.

    Args:
        raw_ids: Sequence of IDs or comma-separated string.

    Returns:
        Parsed reclaim unit handle IDs as integers.

    Raises:
        ValueError: If the value is not a supported sequence or string.
    """
    if raw_ids is None:
        return ()
    if isinstance(raw_ids, str):
        if not raw_ids.strip():
            return ()
        return tuple(int(part.strip()) for part in raw_ids.split(","))
    if isinstance(raw_ids, Sequence):
        return tuple(int(value) for value in raw_ids)
    raise ValueError("fdp_ruh_ids must be a sequence or CSV string")


def _make_bitmap(size: int) -> "Bitmap":
    # First Party
    from lmcache.native_storage_ops import Bitmap

    return Bitmap(size)


class RawBlockL2AdapterConfig(L2AdapterConfigBase):
    """Configuration object for the built-in raw-block MP L2 adapter."""

    def __init__(
        self,
        *,
        device_path: str,
        slot_bytes: int,
        capacity_bytes: int = 0,
        base_offset_bytes: int = 0,
        use_odirect: bool = False,
        use_uring: bool = False,
        use_uring_cmd: bool = False,
        use_fdp: bool = False,
        fdp_ruh_ids: Sequence[int] | None = None,
        fdp_data_ruh_ids: Sequence[int] | None = None,
        fdp_metadata_ruh_ids: Sequence[int] | None = None,
        fdp_directive_type: int = 2,
        fdp_metadata_mode: str = "per_ruh",
        block_align: int = 4096,
        header_bytes: int = 4096,
        meta_total_bytes: int = 256 * 1024 * 1024,
        meta_magic: str = "LMCIDX01",
        meta_version: int = 1,
        meta_checkpoint_interval_sec: int = 60,
        meta_idle_quiet_ms: int = 100,
        meta_enable_periodic: bool = True,
        meta_verify_on_load: bool = True,
        enable_zero_copy: bool = True,
        num_store_workers: int = 2,
        num_lookup_workers: int = 1,
        num_load_workers: int = 4,
        internal_eviction_policy: str = "lru",
        latency_log_path: str | None = None,
    ):
        """Initialize raw-block MP adapter configuration.

        Args:
            device_path: Raw device path or pre-sized file path used for L2.
            slot_bytes: Fixed data-slot size in bytes.
            capacity_bytes: Optional byte window size; zero uses remaining size.
            base_offset_bytes: Byte offset where this raw-block window starts.
            use_odirect: Whether to open the raw path with O_DIRECT.
            use_uring: Whether to use the Rust io_uring I/O path.
            use_uring_cmd: Whether to use NVMe io_uring_cmd passthrough.
            use_fdp: Whether to set FDP directive fields on raw-command writes.
            fdp_ruh_ids: Legacy shorthand for both FDP data and metadata RUHs.
            fdp_data_ruh_ids: Reclaim unit handles used for data placement.
            fdp_metadata_ruh_ids: Reclaim unit handles used for checkpoint
                metadata placement.
            fdp_directive_type: NVMe directive type used for FDP placement.
            fdp_metadata_mode: Metadata layout for FDP. Only ``"per_ruh"`` is
                currently supported.
            block_align: Required block alignment in bytes.
            header_bytes: Per-slot header reservation in bytes.
            meta_total_bytes: Reserved metadata checkpoint region size.
            meta_magic: Eight-byte ASCII metadata checkpoint magic.
            meta_version: Metadata checkpoint version.
            meta_checkpoint_interval_sec: Periodic checkpoint interval.
            meta_idle_quiet_ms: Quiet period before periodic checkpoints.
            meta_enable_periodic: Whether to run the checkpoint thread.
            meta_verify_on_load: Whether recovery verifies slot headers.
            enable_zero_copy: Whether to use aligned direct-buffer I/O.
            num_store_workers: Number of store worker threads.
            num_lookup_workers: Number of lookup worker threads.
            num_load_workers: Number of load worker threads.
            internal_eviction_policy: Internal slot-reuse policy. PR1 supports
                only ``"lru"``.
            latency_log_path: Optional JSONL path for L2 task and raw I/O
                latency samples.
        """
        super().__init__()
        self.device_path = device_path
        self.slot_bytes = int(slot_bytes)
        self.capacity_bytes = int(capacity_bytes)
        self.base_offset_bytes = int(base_offset_bytes)
        self.use_odirect = bool(use_odirect)
        self.use_uring = bool(use_uring)
        self.use_uring_cmd = bool(use_uring_cmd)
        self.use_fdp = bool(use_fdp)
        self.fdp_ruh_ids = tuple(int(value) for value in (fdp_ruh_ids or ()))
        self.fdp_data_ruh_ids = tuple(
            int(value) for value in (fdp_data_ruh_ids or self.fdp_ruh_ids)
        )
        self.fdp_metadata_ruh_ids = tuple(
            int(value) for value in (fdp_metadata_ruh_ids or self.fdp_ruh_ids)
        )
        self.fdp_directive_type = int(fdp_directive_type)
        self.fdp_metadata_mode = str(fdp_metadata_mode)
        self.block_align = int(block_align)
        self.header_bytes = int(header_bytes)
        self.meta_total_bytes = int(meta_total_bytes)
        self.meta_magic = meta_magic
        self.meta_version = int(meta_version)
        self.meta_checkpoint_interval_sec = int(meta_checkpoint_interval_sec)
        self.meta_idle_quiet_ms = int(meta_idle_quiet_ms)
        self.meta_enable_periodic = bool(meta_enable_periodic)
        self.meta_verify_on_load = bool(meta_verify_on_load)
        self.enable_zero_copy = bool(enable_zero_copy)
        self.num_store_workers = int(num_store_workers)
        self.num_lookup_workers = int(num_lookup_workers)
        self.num_load_workers = int(num_load_workers)
        self.internal_eviction_policy = internal_eviction_policy
        self.latency_log_path = latency_log_path

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
        base_offset_bytes = int(d.get("base_offset_bytes", 0))
        use_uring = bool(d.get("use_uring", False))
        use_uring_cmd = bool(d.get("use_uring_cmd", False))
        use_fdp = bool(d.get("use_fdp", False))
        fdp_ruh_ids = _parse_fdp_ruh_ids(d.get("fdp_ruh_ids", ()))
        fdp_data_ruh_ids = _parse_fdp_ruh_ids(d.get("fdp_data_ruh_ids", fdp_ruh_ids))
        fdp_metadata_ruh_ids = _parse_fdp_ruh_ids(
            d.get("fdp_metadata_ruh_ids", fdp_ruh_ids)
        )
        fdp_directive_type = int(d.get("fdp_directive_type", 2))
        fdp_metadata_mode = str(d.get("fdp_metadata_mode", "per_ruh"))
        reserved_meta_bytes = meta_total_bytes * (
            len(fdp_metadata_ruh_ids) if use_fdp else 1
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
        if capacity_bytes > 0 and capacity_bytes <= reserved_meta_bytes:
            raise ValueError("capacity_bytes must leave space for at least one slot")
        if base_offset_bytes < 0:
            raise ValueError("base_offset_bytes must be >= 0")
        if base_offset_bytes % block_align != 0:
            raise ValueError("base_offset_bytes must be a multiple of block_align")
        if use_uring_cmd and not use_uring:
            raise ValueError("use_uring_cmd requires use_uring=true")
        if use_fdp:
            if not use_uring_cmd:
                raise ValueError("use_fdp requires use_uring_cmd=true")
            if not fdp_data_ruh_ids:
                raise ValueError("use_fdp requires non-empty fdp_data_ruh_ids")
            if not fdp_metadata_ruh_ids:
                raise ValueError("use_fdp requires non-empty fdp_metadata_ruh_ids")
            if len(set(fdp_data_ruh_ids)) != len(fdp_data_ruh_ids):
                raise ValueError("fdp_data_ruh_ids must not contain duplicates")
            if len(set(fdp_metadata_ruh_ids)) != len(fdp_metadata_ruh_ids):
                raise ValueError("fdp_metadata_ruh_ids must not contain duplicates")
            for ruh_id in fdp_data_ruh_ids + fdp_metadata_ruh_ids:
                if ruh_id < 0 or ruh_id > 0xFFFF:
                    raise ValueError("FDP RUH IDs must fit in uint16")
            if fdp_directive_type < 0 or fdp_directive_type > 0x0F:
                raise ValueError("fdp_directive_type must fit in 4 bits")
            if fdp_metadata_mode != "per_ruh":
                raise ValueError("fdp_metadata_mode must be 'per_ruh'")

        internal_eviction_policy = str(d.get("internal_eviction_policy", "lru")).lower()
        if internal_eviction_policy != "lru":
            raise ValueError("internal_eviction_policy must be 'lru' in PR1")

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
            base_offset_bytes=base_offset_bytes,
            use_odirect=bool(d.get("use_odirect", False)),
            use_uring=use_uring,
            use_uring_cmd=use_uring_cmd,
            use_fdp=use_fdp,
            fdp_ruh_ids=fdp_ruh_ids,
            fdp_data_ruh_ids=fdp_data_ruh_ids,
            fdp_metadata_ruh_ids=fdp_metadata_ruh_ids,
            fdp_directive_type=fdp_directive_type,
            fdp_metadata_mode=fdp_metadata_mode,
            block_align=block_align,
            header_bytes=header_bytes,
            meta_total_bytes=meta_total_bytes,
            meta_magic=str(d.get("meta_magic", "LMCIDX01")),
            meta_version=int(d.get("meta_version", 1)),
            meta_checkpoint_interval_sec=int(d.get("meta_checkpoint_interval_sec", 60)),
            meta_idle_quiet_ms=int(d.get("meta_idle_quiet_ms", 100)),
            meta_enable_periodic=bool(d.get("meta_enable_periodic", True)),
            meta_verify_on_load=bool(d.get("meta_verify_on_load", True)),
            enable_zero_copy=bool(d.get("enable_zero_copy", True)),
            num_store_workers=worker_counts["num_store_workers"],
            num_lookup_workers=worker_counts["num_lookup_workers"],
            num_load_workers=worker_counts["num_load_workers"],
            internal_eviction_policy=internal_eviction_policy,
            latency_log_path=(
                str(d["latency_log_path"])
                if d.get("latency_log_path") is not None
                else None
            ),
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
            "(default 0 = device size from base_offset_bytes)\n"
            "- base_offset_bytes (int): byte offset where this raw-block "
            "window starts (default 0)\n"
            "- use_odirect (bool): enable O_DIRECT raw I/O "
            "(default false)\n"
            "- use_uring (bool): enable Rust io_uring I/O (default false)\n"
            "- use_uring_cmd (bool): enable NVMe io_uring_cmd path "
            "(default false, requires use_uring)\n"
            "- use_fdp (bool): enable FDP directive fields on io_uring_cmd "
            "writes (default false, requires use_uring_cmd)\n"
            "- fdp_ruh_ids (list[int] | str): FDP reclaim unit handles used "
            "for data and metadata placement when split fields are omitted\n"
            "- fdp_data_ruh_ids (list[int] | str): FDP reclaim unit handles "
            "used for data placement; defaults to fdp_ruh_ids\n"
            "- fdp_metadata_ruh_ids (list[int] | str): FDP reclaim unit "
            "handles used for checkpoint metadata; defaults to fdp_ruh_ids\n"
            "- fdp_directive_type (int): NVMe directive type for FDP "
            "(default 2)\n"
            "- fdp_metadata_mode (str): only 'per_ruh' is supported; "
            "meta_total_bytes is reserved per metadata RUH\n"
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
            "- meta_verify_on_load (bool): validate slot headers on recovery "
            "(default true)\n"
            "- enable_zero_copy (bool): use aligned direct buffers when possible "
            "(default true)\n"
            "- num_store_workers (int): store worker threads (default 2)\n"
            "- num_lookup_workers (int): lookup worker threads (default 1)\n"
            "- num_load_workers (int): load worker threads (default 4)\n"
            "- internal_eviction_policy (str): only 'lru' is supported in PR1"
            "\n- latency_log_path (str): optional JSONL output for raw-block "
            "L2 latency samples"
        )

    def to_core_config(self) -> RawBlockCoreConfig:
        """Convert this adapter config to the shared RawBlockCore config."""
        return RawBlockCoreConfig(
            device_path=self.device_path,
            capacity_bytes=self.capacity_bytes,
            base_offset_bytes=self.base_offset_bytes,
            block_align=self.block_align,
            header_bytes=self.header_bytes,
            slot_bytes=self.slot_bytes,
            use_odirect=self.use_odirect,
            use_uring=self.use_uring,
            use_uring_cmd=self.use_uring_cmd,
            use_fdp=self.use_fdp,
            fdp_ruh_ids=self.fdp_ruh_ids,
            fdp_data_ruh_ids=self.fdp_data_ruh_ids,
            fdp_metadata_ruh_ids=self.fdp_metadata_ruh_ids,
            fdp_directive_type=self.fdp_directive_type,
            fdp_metadata_mode=self.fdp_metadata_mode,
            enable_zero_copy=self.enable_zero_copy,
            meta_total_bytes=self.meta_total_bytes,
            meta_magic=self.meta_magic.encode("ascii"),
            meta_version=self.meta_version,
            meta_checkpoint_interval_sec=self.meta_checkpoint_interval_sec,
            meta_idle_quiet_ms=self.meta_idle_quiet_ms,
            meta_enable_periodic=self.meta_enable_periodic,
            meta_verify_on_load=self.meta_verify_on_load,
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
            (config.use_odirect or config.use_uring)
            and l1_memory_desc is not None
            and l1_memory_desc.align_bytes < config.block_align
        ):
            raise ValueError(
                "raw_block requires l1_align_bytes >= block_align when "
                "use_odirect=true or use_uring=true"
            )

        self._closed = False
        self._core: RawBlockCore
        self._store_efd = -1
        self._lookup_efd = -1
        self._load_efd = -1
        self._store_pool: ThreadPoolExecutor
        self._lookup_pool: ThreadPoolExecutor
        self._load_pool: ThreadPoolExecutor
        self._latency_recorder: _LatencyJsonlRecorder | None = None

        try:
            self._core = RawBlockCore(config.to_core_config(), key_namespace="object")
            if config.latency_log_path:
                self._latency_recorder = _LatencyJsonlRecorder(config.latency_log_path)
                self._core.set_io_latency_callback(self._record_raw_io_latency)
            if config.use_uring:
                logger.warning(
                    "RawBlockL2Adapter: MP raw_block uses io_uring without "
                    "fixed-buffer registration; zero-copy fixed buffers are "
                    "disabled unless registered by a future MP allocator path"
                )
            self._max_capacity_bytes = int(
                self._core.report_status().get("usable_capacity_bytes", 0)
            )
            self._seed_usage_from_recovered_entries()

            self._store_efd = os.eventfd(0, os.EFD_NONBLOCK | os.EFD_CLOEXEC)
            self._lookup_efd = os.eventfd(0, os.EFD_NONBLOCK | os.EFD_CLOEXEC)
            self._load_efd = os.eventfd(0, os.EFD_NONBLOCK | os.EFD_CLOEXEC)

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

        self._completed_store_tasks: dict[L2TaskId, bool] = {}
        self._completed_lookup_tasks: dict[L2TaskId, Bitmap] = {}
        self._completed_load_tasks: dict[L2TaskId, Bitmap] = {}

        self._store_inflight_tasks: int = 0
        self._lookup_inflight_tasks: int = 0
        self._load_inflight_tasks: int = 0

    def _seed_usage_from_recovered_entries(self) -> None:
        """Seed base logical usage accounting from recovered raw-block metadata."""
        per_salt: dict[str, int] = {}
        total = 0
        for encoded_key, size in self._core.snapshot_accounting_entries():
            try:
                key = decode_object_key(encoded_key)
            except ValueError:
                logger.warning(
                    "RawBlockL2Adapter skipped recovered non-object key %s",
                    encoded_key,
                )
                continue
            logical_size = int(size)
            per_salt[key.cache_salt] = per_salt.get(key.cache_salt, 0) + logical_size
            total += logical_size

        with self._usage_lock:
            self._total_bytes_used = total
            self._bytes_by_cache_salt = per_salt

    def get_store_event_fd(self) -> int:
        """Return the eventfd signaled when store tasks complete."""
        return self._store_efd

    def get_lookup_and_lock_event_fd(self) -> int:
        """Return the eventfd signaled when lookup-and-lock tasks complete."""
        return self._lookup_efd

    def get_load_event_fd(self) -> int:
        """Return the eventfd signaled when load tasks complete."""
        return self._load_efd

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
        if not keys or not objects:
            raise ValueError("keys and objects must be non-empty")
        if len(keys) != len(objects):
            raise ValueError("keys and objects must have the same length")

        with self._lock:
            self._raise_if_closed_locked()
            task_id = self._get_next_task_id_locked()
            self._store_inflight_tasks += 1
        submitted_at = time.perf_counter()
        try:
            future = self._store_pool.submit(
                self._run_store_task, list(keys), list(objects)
            )
        except Exception:
            with self._lock:
                self._store_inflight_tasks -= 1
            raise
        future.add_done_callback(
            partial(self._finish_store_task, task_id, submitted_at, len(keys))
        )
        return task_id

    def pop_completed_store_tasks(self) -> dict[L2TaskId, bool]:
        """Drain and return completed store task results."""
        with self._lock:
            completed = self._completed_store_tasks
            self._completed_store_tasks = {}
        return completed

    def submit_lookup_and_lock_task(self, keys: list[ObjectKey]) -> L2TaskId:
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
        submitted_at = time.perf_counter()
        try:
            future = self._load_pool.submit(
                self._run_load_task, list(keys), list(objects)
            )
        except Exception:
            with self._lock:
                self._load_inflight_tasks -= 1
            raise
        future.add_done_callback(
            partial(self._finish_load_task, task_id, len(keys), submitted_at)
        )
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
            deleted_sizes.append(0 if meta is None else int(meta.size))
        if deleted_keys:
            self._notify_keys_deleted(deleted_keys, deleted_sizes)

    @property
    def supports_global_eviction(self) -> bool:
        """Return whether the shared L2 eviction controller should manage this adapter.

        Returns:
            ``False`` because raw-block uses its own internal slot recycling and
            should not register a second global L2 eviction policy.
        """
        return False

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
        if self._latency_recorder is not None:
            self._latency_recorder.close()
            self._latency_recorder = None

        with self._lock:
            store_efd = self._store_efd
            lookup_efd = self._lookup_efd
            load_efd = self._load_efd
            self._store_efd = -1
            self._lookup_efd = -1
            self._load_efd = -1

        if store_efd >= 0:
            os.close(store_efd)
        if lookup_efd >= 0:
            os.close(lookup_efd)
        if load_efd >= 0:
            os.close(load_efd)

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
                "completed_store_task_count": len(self._completed_store_tasks),
                "completed_lookup_task_count": len(self._completed_lookup_tasks),
                "completed_load_task_count": len(self._completed_load_tasks),
                "core": core_status,
            }

    def _raise_if_closed_locked(self) -> None:
        if self._closed:
            raise RuntimeError("RawBlockL2Adapter is closed")

    def _get_next_task_id_locked(self) -> L2TaskId:
        task_id = self._next_task_id
        self._next_task_id += 1
        return task_id

    def _run_store_task(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
    ) -> RawBlockStoreTaskResult:
        """Persist one submitted store batch in the worker pool.

        Args:
            keys: Object keys submitted for storage.
            objects: Payload buffers aligned with ``keys``.

        Returns:
            A 5-tuple containing:

            - task success for the whole batch
            - newly stored object keys
            - byte sizes aligned with the newly stored keys
            - object keys evicted by raw-block internal recycling
            - byte sizes aligned with the evicted keys
        """
        specs = [encode_object_key(key) for key in keys]
        put_result = self._core.put_many(specs, objects)
        stored_keys: list[ObjectKey] = []
        stored_sizes: list[int] = []
        for index, size in zip(
            put_result.stored_indices,
            put_result.stored_key_sizes,
            strict=True,
        ):
            stored_keys.append(keys[index])
            stored_sizes.append(int(size))
        evicted_keys: list[ObjectKey] = []
        evicted_sizes: list[int] = []
        for encoded_key, size in zip(
            put_result.evicted_keys,
            put_result.evicted_key_sizes,
            strict=False,
        ):
            try:
                evicted_keys.append(decode_object_key(encoded_key))
                evicted_sizes.append(size)
            except ValueError:
                logger.warning(
                    "RawBlockL2Adapter skipped evicted non-object key %s",
                    encoded_key,
                )
        return (
            all(put_result.results),
            stored_keys,
            stored_sizes,
            evicted_keys,
            evicted_sizes,
        )

    def _finish_store_task(
        self,
        task_id: L2TaskId,
        submitted_at: float,
        key_count: int,
        future: Future[RawBlockStoreTaskResult],
    ) -> None:
        success = False
        stored_keys: list[ObjectKey] = []
        stored_sizes: list[int] = []
        evicted_keys: list[ObjectKey] = []
        evicted_sizes: list[int] = []
        try:
            success, stored_keys, stored_sizes, evicted_keys, evicted_sizes = (
                future.result()
            )
        except Exception as e:
            logger.error("RawBlockL2Adapter store task %d failed: %s", task_id, e)
        self._record_task_latency(
            "l2_e2e_write",
            submitted_at,
            failed=not success,
            key_count=key_count,
        )
        with self._lock:
            self._store_inflight_tasks -= 1
            self._completed_store_tasks[task_id] = success
            event_fd = self._store_efd
        if stored_keys:
            try:
                self._notify_keys_stored(stored_keys, stored_sizes)
            except Exception as e:
                logger.warning("RawBlockL2Adapter store notification failed: %s", e)
        if evicted_keys:
            try:
                self._notify_keys_deleted(evicted_keys, evicted_sizes)
            except Exception as e:
                logger.warning("RawBlockL2Adapter delete notification failed: %s", e)
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
        self,
        task_id: L2TaskId,
        bitmap_size: int,
        submitted_at: float,
        future: Future[Any],
    ) -> None:
        bitmap = _make_bitmap(bitmap_size)
        accessed_keys: list[ObjectKey] = []
        try:
            bitmap, accessed_keys = future.result()
        except Exception as e:
            logger.error("RawBlockL2Adapter load task %d failed: %s", task_id, e)
        self._record_task_latency(
            "l2_e2e_read",
            submitted_at,
            failed=len(accessed_keys) != bitmap_size,
            key_count=bitmap_size,
        )
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

    def _record_task_latency(
        self,
        metric: str,
        submitted_at: float,
        *,
        failed: bool,
        key_count: int,
    ) -> None:
        recorder = self._latency_recorder
        if recorder is None:
            return
        recorder.record(
            {
                "metric": metric,
                "latency_ms": (time.perf_counter() - submitted_at) * 1000.0,
                "failed": failed,
                "key_count": key_count,
            }
        )

    def _record_raw_io_latency(
        self,
        operation: str,
        latency_s: float,
        io_class: str,
        physical_bytes: int,
        io_count: int,
        failed: bool,
    ) -> None:
        recorder = self._latency_recorder
        if recorder is None:
            return
        recorder.record(
            {
                "metric": f"raw_block_{operation}",
                "latency_ms": latency_s * 1000.0,
                "failed": failed,
                "io_class": io_class,
                "physical_bytes": physical_bytes,
                "io_count": io_count,
            }
        )

    def _signal_event_fd(self, event_fd: int) -> None:
        try:
            os.eventfd_write(event_fd, 1)
        except OSError:
            logger.debug("eventfd %d was closed before signaling", event_fd)

    def _cleanup_after_init_failure(self) -> None:
        for pool_name in ("_store_pool", "_lookup_pool", "_load_pool"):
            pool = getattr(self, pool_name, None)
            if pool is not None:
                pool.shutdown(wait=False, cancel_futures=True)
                setattr(self, pool_name, None)

        core = getattr(self, "_core", None)
        if core is not None:
            core.close()

        recorder = getattr(self, "_latency_recorder", None)
        if recorder is not None:
            recorder.close()
            self._latency_recorder = None

        for fd_name in ("_store_efd", "_lookup_efd", "_load_efd"):
            fd = int(getattr(self, fd_name, -1))
            if fd >= 0:
                os.close(fd)
                setattr(self, fd_name, -1)

        self._closed = True


register_l2_adapter_type("raw_block", RawBlockL2AdapterConfig)


def _create_raw_block_adapter(
    config: L2AdapterConfigBase,
    l1_memory_desc: "Optional[L1MemoryDesc]" = None,
) -> L2AdapterInterface:
    return RawBlockL2Adapter(config, l1_memory_desc)  # type: ignore[arg-type]


register_l2_adapter_factory("raw_block", _create_raw_block_adapter)
