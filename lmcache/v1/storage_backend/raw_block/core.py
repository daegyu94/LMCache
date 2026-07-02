# SPDX-License-Identifier: Apache-2.0

# Future
from __future__ import annotations

# Standard
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Optional
import ctypes
import hashlib
import json
import os
import stat
import struct
import threading
import time
import zlib

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.utils import (
    STR_DTYPE_TO_TORCH_DTYPE,
    TORCH_DTYPE_TO_STR_DTYPE,
    DiskCacheMetadata,
)
from lmcache.v1.memory_management import MemoryFormat, MemoryObj
from lmcache.v1.storage_backend.raw_block.key_codec import (
    RawBlockKeyNamespace,
    RawBlockKeySpec,
    decode_legacy_key,
    decode_object_key,
    slot_identity_from_encoded_key,
)

logger = init_logger(__name__)


_DEFAULT_META_MAGIC = b"LMCIDX01"
_DEFAULT_META_VERSION = 1
_META_HEADER_STRUCT = struct.Struct("<8sIQQI")
_URING_CMD_MAX_IO_BYTES = 128 * 1024


def round_up(x: int, align: int) -> int:
    """Round a value up to the nearest alignment boundary.

    Args:
        x: Value to align.
        align: Positive alignment in bytes.

    Returns:
        ``x`` rounded up to a multiple of ``align``.
    """
    return ((x + align - 1) // align) * align


@dataclass(frozen=True)
class RawBlockCoreConfig:
    """Configuration for RawBlockCore device layout, I/O, and checkpoints."""

    device_path: str
    capacity_bytes: int
    block_align: int
    header_bytes: int
    slot_bytes: int
    use_odirect: bool
    enable_zero_copy: bool
    meta_total_bytes: int
    meta_magic: bytes
    meta_version: int
    meta_checkpoint_interval_sec: int
    meta_idle_quiet_ms: int
    meta_enable_periodic: bool
    meta_verify_on_load: bool
    use_uring: bool = False
    use_uring_cmd: bool = False
    use_fdp: bool = False
    fdp_ruh_ids: tuple[int, ...] = ()
    fdp_data_ruh_ids: tuple[int, ...] = ()
    fdp_metadata_ruh_ids: tuple[int, ...] = ()
    fdp_directive_type: int = 2
    fdp_metadata_mode: str = "per_ruh"
    base_offset_bytes: int = 0


@dataclass
class _Entry:
    offset: int
    size: int
    meta: DiskCacheMetadata
    fdp_ruh_id: int | None = None


@dataclass
class _Inflight:
    offset: int
    meta: DiskCacheMetadata
    fdp_ruh_id: int | None = None
    canceled: bool = False


@dataclass(frozen=True)
class RawBlockPutManyResult:
    """Result of a RawBlockCore batched write."""

    results: list[bool]
    stored_keys: list[str]
    evicted_keys: list[str]
    evicted_key_sizes: list[int]


class RawBlockCore:
    """
    Shared raw-block storage engine used by both legacy non-MP and MP L2 paths.

    This class owns the raw-device I/O path, slot allocation, checkpoint/recovery,
    internal LRU bookkeeping, and lock refcounts that protect slots from reuse.
    """

    def __init__(
        self,
        config: RawBlockCoreConfig,
        *,
        key_namespace: RawBlockKeyNamespace,
    ):
        """Initialize the raw-block storage engine.

        Args:
            config: Raw-block device, layout, I/O, and checkpoint settings.
            key_namespace: Encoding namespace used by keys stored in this core.

        Raises:
            ValueError: If the supplied configuration is invalid.
            RuntimeError: If the raw device cannot be opened or the computed
                layout cannot fit metadata and at least one data slot.

        Notes:
            If initialization opens the device and a later recovery step fails,
            the partially opened resources are closed before the exception is
            re-raised.
        """
        self.device_path = config.device_path
        self.capacity_bytes = int(config.capacity_bytes)
        self.base_offset_bytes = int(config.base_offset_bytes)
        self.block_align = int(config.block_align)
        self.header_bytes = int(config.header_bytes)
        self.slot_bytes = int(config.slot_bytes)
        self.use_uring = bool(config.use_uring)
        self.use_uring_cmd = bool(config.use_uring_cmd)
        self.use_odirect = bool(config.use_odirect)
        if self.use_uring_cmd and self.use_odirect:
            logger.warning(
                "RawBlockCore: use_odirect is ignored for NVMe namespace "
                "character devices when use_uring_cmd=true"
            )
            self.use_odirect = False
        self.use_fdp = bool(config.use_fdp)
        self.fdp_ruh_ids = tuple(int(v) for v in config.fdp_ruh_ids)
        self.fdp_data_ruh_ids = tuple(
            int(v) for v in (config.fdp_data_ruh_ids or self.fdp_ruh_ids)
        )
        self.fdp_metadata_ruh_ids = tuple(
            int(v) for v in (config.fdp_metadata_ruh_ids or self.fdp_ruh_ids)
        )
        self.fdp_directive_type = int(config.fdp_directive_type)
        self.fdp_metadata_mode = str(config.fdp_metadata_mode)
        self.enable_zero_copy = bool(config.enable_zero_copy)

        self.meta_total_bytes = int(config.meta_total_bytes)
        self.meta_magic = bytes(config.meta_magic)
        self.meta_version = int(config.meta_version)
        self.meta_checkpoint_interval_sec = int(config.meta_checkpoint_interval_sec)
        self.meta_idle_quiet_ms = int(config.meta_idle_quiet_ms)
        self.meta_enable_periodic = bool(config.meta_enable_periodic)
        self.meta_verify_on_load = bool(config.meta_verify_on_load)
        self.key_namespace = key_namespace

        if not self.device_path:
            raise ValueError("RawBlockCore requires a non-empty device_path")
        if self.block_align <= 0:
            raise ValueError("block_align must be > 0")
        if self.base_offset_bytes < 0:
            raise ValueError("base_offset_bytes must be >= 0")
        if self.base_offset_bytes % self.block_align != 0:
            raise ValueError("base_offset_bytes must be a multiple of block_align")
        if self.header_bytes < 24:
            raise ValueError("header_bytes must be >= 24")
        if self.header_bytes % self.block_align != 0:
            raise ValueError("header_bytes must be a multiple of block_align")
        if self.slot_bytes < self.header_bytes + 1:
            raise ValueError("slot_bytes must be >= header_bytes + 1")
        if self.slot_bytes % self.block_align != 0:
            raise ValueError("slot_bytes must be a multiple of block_align")
        if self.meta_total_bytes <= self.block_align:
            raise ValueError("meta_total_bytes must provide room for metadata header")
        if self.meta_total_bytes % self.block_align != 0:
            raise ValueError("meta_total_bytes must be a multiple of block_align")
        if len(self.meta_magic) != 8:
            raise ValueError("meta_magic must be exactly 8 bytes")
        if self.meta_version <= 0:
            raise ValueError("meta_version must be > 0")
        if self.use_uring_cmd and not self.use_uring:
            raise ValueError("use_uring_cmd requires use_uring=true")
        if self.use_fdp:
            if not self.use_uring_cmd:
                raise ValueError("use_fdp requires use_uring_cmd=true")
            if not self.fdp_data_ruh_ids:
                raise ValueError("use_fdp requires non-empty fdp_data_ruh_ids")
            if not self.fdp_metadata_ruh_ids:
                raise ValueError("use_fdp requires non-empty fdp_metadata_ruh_ids")
            if len(set(self.fdp_data_ruh_ids)) != len(self.fdp_data_ruh_ids):
                raise ValueError("fdp_data_ruh_ids must not contain duplicates")
            if len(set(self.fdp_metadata_ruh_ids)) != len(self.fdp_metadata_ruh_ids):
                raise ValueError("fdp_metadata_ruh_ids must not contain duplicates")
            for ruh_id in self.fdp_data_ruh_ids + self.fdp_metadata_ruh_ids:
                if ruh_id < 0 or ruh_id > 0xFFFF:
                    raise ValueError("FDP RUH IDs must fit in uint16")
            if self.fdp_directive_type < 0 or self.fdp_directive_type > 0x0F:
                raise ValueError("fdp_directive_type must fit in 4 bits")
            if self.fdp_metadata_mode != "per_ruh":
                raise ValueError("fdp_metadata_mode must be 'per_ruh'")
        if self.use_uring_cmd:
            try:
                mode = os.stat(self.device_path).st_mode
            except OSError as e:
                raise ValueError(
                    "use_uring_cmd requires an existing NVMe namespace "
                    f"character device path, got {self.device_path!r}"
                ) from e
            if not stat.S_ISCHR(mode):
                raise ValueError(
                    "use_uring_cmd requires an NVMe namespace character device "
                    f"(for example /dev/ng0n1), got {self.device_path!r}"
                )

        try:
            self.meta_magic_text = self.meta_magic.decode("ascii")
        except UnicodeDecodeError as e:
            raise ValueError("meta_magic must be ASCII bytes") from e

        self._meta_copy_count: int = 2
        self._meta_partition_count: int = (
            len(self.fdp_metadata_ruh_ids) if self.use_fdp else 1
        )
        self._fdp_metadata_ruh_index: dict[int, int] = {
            ruh_id: idx for idx, ruh_id in enumerate(self.fdp_metadata_ruh_ids)
        }
        self._reserved_meta_bytes: int = self.meta_total_bytes * (
            self._meta_partition_count
        )
        self._meta_container_bytes: int = (
            (self.meta_total_bytes // self._meta_copy_count) // self.block_align
        ) * self.block_align
        if self._meta_container_bytes <= self.block_align:
            raise ValueError(
                "meta_total_bytes must provide room for at least two metadata copies"
            )

        self._lock = threading.Lock()
        self._index: dict[str, _Entry] = {}
        self._lock_refcnt: dict[str, int] = {}
        self._inflight: dict[str, _Inflight] = {}
        self._lru: "OrderedDict[str, None]" = OrderedDict()

        self._next_slot: int = 0
        self._free_slots: list[int] = []
        self._max_slots: int = 0
        self._effective_capacity_bytes: int = 0
        self._data_base_offset: int = 0

        self._raw = None
        self._closed = False

        self._meta_seq: int = 0
        self._meta_dirty_total: int = 0
        self._meta_persisted: int = 0
        self._inflight_io_count: int = 0
        self._last_io_ts: float = time.monotonic()
        self._meta_stop_evt = threading.Event()
        self._meta_thread: Optional[threading.Thread] = None

        try:
            self._ensure_capacity_and_layout()
            self._load_checkpoint_from_device()

            if self.meta_enable_periodic:
                self._meta_thread = threading.Thread(
                    target=self._checkpoint_loop,
                    daemon=True,
                    name="raw-block-core-checkpoint",
                )
                self._meta_thread.start()
        except Exception:
            self._cleanup_after_init_failure()
            raise

    def _rawdev(self):
        """Return the lazily opened Rust raw-block device binding."""
        if self._raw is None:
            try:
                # Third Party
                from lmcache_rust_raw_block_io import RawBlockDevice  # type: ignore
            except Exception as e:
                raise RuntimeError(
                    "Rust raw-block extension is not installed. "
                    "Install / build `rust_raw_block_io` and retry."
                ) from e
            self._raw = RawBlockDevice(
                self.device_path,
                writable=True,
                use_odirect=self.use_odirect,
                use_iouring=self.use_uring,
                use_uring_cmd=self.use_uring_cmd,
                alignment=self.block_align,
            )
        return self._raw

    def raw_device(self) -> Any:
        """Return the lazily opened Rust raw-block device.

        Returns:
            The underlying Rust ``RawBlockDevice`` object.

        Raises:
            Exception: Propagates raw-device open errors from the Rust binding.
        """
        return self._rawdev()

    def set_raw_device_for_testing(self, raw_device: Any) -> None:
        """Replace the raw device handle used by this core.

        Args:
            raw_device: Object implementing the Rust raw-device methods.
        """
        self._raw = raw_device

    def _select_fdp_ruh(
        self,
        key: RawBlockKeySpec,
        obj: MemoryObj | None = None,
    ) -> int | None:
        """Select an FDP RUH for a raw-block key.

        Args:
            key: Encoded raw-block key being written.
            obj: Optional memory object carrying a transient placement hint.

        Returns:
            A configured reclaim unit handle ID when FDP is enabled, otherwise
            None.
        """
        if not self.use_fdp:
            return None

        placement_rank = (
            getattr(obj.metadata, "fdp_placement_rank", None)
            if obj is not None
            else None
        )
        if placement_rank is not None:
            worker_index = int(placement_rank)
        elif self.key_namespace == "object":
            kv_rank = int(decode_object_key(key.encoded).kv_rank)
            # ObjectKey.ComputeKVRank packs global_rank into bits [16, 23].
            # Use that worker index directly so FDP placement is stable per
            # unique KV worker instead of hash-randomized per rank.
            worker_index = (kv_rank >> 16) & 0xFF
        else:
            parsed_key = decode_legacy_key(key.encoded)
            worker_index = int(getattr(parsed_key, "worker_id", 0))

        idx = worker_index % len(self.fdp_data_ruh_ids)
        return self.fdp_data_ruh_ids[idx]

    def _select_fdp_metadata_ruh(self, encoded_key: str) -> int | None:
        """Select an FDP RUH for checkpoint metadata that owns a key.

        Args:
            encoded_key: Durable raw-block key string.

        Returns:
            A configured metadata reclaim unit handle ID when FDP is enabled,
            otherwise None.
        """
        if not self.use_fdp:
            return None
        digest = hashlib.blake2b(encoded_key.encode("utf-8"), digest_size=8).digest()
        idx = int.from_bytes(digest, "little", signed=False) % len(
            self.fdp_metadata_ruh_ids
        )
        return self.fdp_metadata_ruh_ids[idx]

    def _write_directive_for_ruh(self, fdp_ruh_id: int | None) -> tuple[int, int]:
        """Return NVMe directive fields for an optional FDP RUH.

        Args:
            fdp_ruh_id: Reclaim unit handle selected for a write, or None.

        Returns:
            ``(dtype, dspec)`` values for the Rust io_uring_cmd write path.
        """
        if fdp_ruh_id is None:
            return (0, 0)
        if not self.use_fdp:
            return (0, 0)
        return (self.fdp_directive_type, int(fdp_ruh_id))

    def register_fixed_buffers_from_allocator(self, memory_allocator: Any) -> None:
        """Register allocator pages with io_uring when the allocator exposes them.

        Args:
            memory_allocator: Local CPU allocator that may expose
                ``get_paged_buffers()``.

        Raises:
            Exception: Propagates Rust registration errors after logging.
        """
        if not self.use_uring:
            return
        paged_buffers = getattr(memory_allocator, "get_paged_buffers", None)
        if not callable(paged_buffers):
            logger.warning(
                "RawBlockCore: allocator does not expose paged buffers; "
                "io_uring fixed-buffer zero-copy is disabled"
            )
            return
        buffers = paged_buffers()
        if not buffers:
            logger.warning(
                "RawBlockCore: allocator returned no paged buffers; "
                "io_uring fixed-buffer zero-copy is disabled"
            )
            return
        buffer_ptrs = [buf.data_ptr() for buf in buffers]
        buffer_sizes = [buf.numel() * buf.element_size() for buf in buffers]
        self._rawdev().register_fixed_buffers(buffer_ptrs, buffer_sizes)
        logger.info(
            "RawBlockCore: registered %d paged buffers for io_uring fixed I/O",
            len(buffers),
        )

    def contains_key(self, encoded_key: str, *, lock: bool = False) -> bool:
        """Return whether one encoded key is present in the raw-block index.

        Args:
            encoded_key: Encoded raw-block key string.
            lock: If true, increment the key's L2 lock refcount on hit.

        Returns:
            True when the key is indexed and available for load.
        """
        return self.exists_many([encoded_key], lock=lock)[0]

    def exists_inflight(self, encoded_key: str) -> bool:
        """Return whether a key currently has an in-flight write.

        Args:
            encoded_key: Encoded raw-block key string.

        Returns:
            True when the key is being written but not committed yet.
        """
        with self._lock:
            return encoded_key in self._inflight

    def get_metadata_many(
        self, encoded_keys: Sequence[str]
    ) -> list[DiskCacheMetadata | None]:
        """Return metadata for encoded keys without loading payload bytes.

        Args:
            encoded_keys: Ordered encoded raw-block keys to inspect.

        Returns:
            A metadata-or-None list aligned with ``encoded_keys``.
        """
        with self._lock:
            metas: list[DiskCacheMetadata | None] = []
            for encoded_key in encoded_keys:
                entry = self._index.get(encoded_key)
                metas.append(entry.meta if entry is not None else None)
            return metas

    def get_metadata_prefix(
        self,
        encoded_keys: Sequence[str],
        *,
        lock: bool = False,
        skip_locked: set[str] | None = None,
    ) -> list[DiskCacheMetadata]:
        """Return leading-hit metadata and optionally lock those entries.

        Args:
            encoded_keys: Ordered encoded raw-block keys to inspect.
            lock: If true, increment L2 lock refcounts for every returned
                metadata entry while holding the index lock.
            skip_locked: Encoded keys that are already protected by the caller
                and should not receive an additional lock refcount.

        Returns:
            Metadata for the contiguous leading hit prefix. The returned list
            stops at the first missing key.
        """
        with self._lock:
            metas: list[DiskCacheMetadata] = []
            for encoded_key in encoded_keys:
                entry = self._index.get(encoded_key)
                if entry is None:
                    break
                metas.append(entry.meta)
                if lock and (skip_locked is None or encoded_key not in skip_locked):
                    self._lock_refcnt[encoded_key] = (
                        self._lock_refcnt.get(encoded_key, 0) + 1
                    )
            return metas

    def first_encoded_key(self) -> str | None:
        """Return one indexed encoded key for diagnostics.

        Returns:
            The first indexed key according to dictionary iteration order, or
            None if the recovered/indexed metadata is empty.
        """
        with self._lock:
            return next(iter(self._index), None)

    def lock_refcount(self, encoded_key: str) -> int:
        """Return the L2 lock refcount for an encoded key.

        Args:
            encoded_key: Encoded raw-block key string.

        Returns:
            Current lock refcount, or zero when the key is unlocked or absent.
        """
        with self._lock:
            return int(self._lock_refcnt.get(encoded_key, 0))

    def inflight_io_count(self) -> int:
        """Return the number of currently active raw-device I/O operations."""
        with self._lock:
            return int(self._inflight_io_count)

    def indexed_key_count(self) -> int:
        """Return the number of entries currently present in the key index."""
        with self._lock:
            return len(self._index)

    def entry_offset(self, encoded_key: str) -> int | None:
        """Return the raw-device slot offset for an indexed key.

        Args:
            encoded_key: Encoded raw-block key string.

        Returns:
            Slot offset in bytes, or None when the key is not indexed.
        """
        with self._lock:
            entry = self._index.get(encoded_key)
            return None if entry is None else int(entry.offset)

    def metadata_container_offsets(self) -> list[int]:
        """Return checkpoint metadata container offsets in bytes."""
        return self._meta_container_offsets()

    def data_base_offset(self) -> int:
        """Return the byte offset where raw-block data slots begin."""
        return int(self._data_base_offset)

    def put_many(
        self,
        keys: Sequence[RawBlockKeySpec],
        objs: Sequence[MemoryObj],
    ) -> RawBlockPutManyResult:
        """Persist a batch of memory objects into raw-block slots.

        Args:
            keys: Ordered raw-block key specs corresponding to ``objs``.
            objs: Memory objects whose byte buffers should be written.

        Returns:
            Per-key success results, newly stored encoded keys, and encoded keys
            evicted while making room.

        Raises:
            ValueError: If either sequence is empty or the sequence lengths do
                not match.
        """
        if not keys or not objs:
            raise ValueError("keys and objs must be non-empty")
        if len(keys) != len(objs):
            raise ValueError("keys and objs must have the same length")

        results = [False] * len(keys)
        stored_keys: list[str] = []
        evicted_keys: list[str] = []
        evicted_key_sizes: list[int] = []

        for i, (key, obj) in enumerate(zip(keys, objs, strict=False)):
            if self._closed:
                break
            fdp_ruh_id = self._select_fdp_ruh(key, obj)

            with self._lock:
                if key.encoded in self._index:
                    results[i] = True
                    continue
                if key.encoded in self._inflight:
                    continue

                while True:
                    try:
                        offset = self._allocate_slot_locked()
                        break
                    except RuntimeError:
                        victim = self._evict_one_locked()
                        if victim is None:
                            logger.warning(
                                "RawBlockCore: no free slot available for key %s",
                                key.encoded,
                            )
                            offset = -1
                            break
                        victim_key, victim_size = victim
                        evicted_keys.append(victim_key)
                        evicted_key_sizes.append(victim_size)

                if offset < 0:
                    continue

                meta = DiskCacheMetadata(
                    path=f"{self.device_path}@{offset}",
                    size=len(obj.byte_array),
                    shape=obj.metadata.shape,
                    dtype=obj.metadata.dtype,
                    cached_positions=obj.metadata.cached_positions,
                    fmt=obj.metadata.fmt,
                    pin_count=0,
                )
                self._inflight[key.encoded] = _Inflight(
                    offset=offset,
                    meta=meta,
                    fdp_ruh_id=fdp_ruh_id,
                )

            success = self._write_one(key, obj, offset, fdp_ruh_id=fdp_ruh_id)

            with self._lock:
                inflight = self._inflight.pop(key.encoded, None)
                if inflight is None:
                    results[i] = False
                    continue
                if inflight.canceled or not success:
                    self._append_free_slot_locked(
                        self._offset_to_slot(int(inflight.offset))
                    )
                    self._meta_dirty_total += 1
                    results[i] = False
                    continue

                self._index[key.encoded] = _Entry(
                    offset=inflight.offset,
                    size=inflight.meta.size,
                    meta=inflight.meta,
                    fdp_ruh_id=inflight.fdp_ruh_id,
                )
                self._touch_locked(key.encoded)
                self._meta_dirty_total += 1
                results[i] = True
                stored_keys.append(key.encoded)

        return RawBlockPutManyResult(
            results=results,
            stored_keys=stored_keys,
            evicted_keys=evicted_keys,
            evicted_key_sizes=evicted_key_sizes,
        )

    def exists_many(
        self,
        encoded_keys: Sequence[str],
        *,
        lock: bool = False,
    ) -> list[bool]:
        """Return a full hit bitmap as booleans for encoded keys.

        Args:
            encoded_keys: Ordered encoded raw-block keys to check.
            lock: If true, increment L2 lock refcounts for every hit.

        Returns:
            A list of booleans aligned with ``encoded_keys``.
        """
        results: list[bool] = []
        with self._lock:
            for encoded_key in encoded_keys:
                found = encoded_key in self._index
                results.append(found)
                if found and lock:
                    self._lock_refcnt[encoded_key] = (
                        self._lock_refcnt.get(encoded_key, 0) + 1
                    )
        return results

    def load_many_into(
        self,
        encoded_keys: Sequence[str],
        objs: Sequence[MemoryObj],
        *,
        raise_on_error: bool = False,
    ) -> list[bool]:
        """Load raw-block payloads into caller-provided memory objects.

        Args:
            encoded_keys: Ordered encoded raw-block keys to load.
            objs: Destination memory objects. Buffers must remain valid until
                this method returns.
            raise_on_error: If true, re-raise the first load exception instead
                of logging it and returning ``False`` for that key.

        Returns:
            A list of per-key load success booleans aligned with
            ``encoded_keys``.

        Raises:
            ValueError: If either sequence is empty or the sequence lengths do
                not match.
            Exception: Re-raises load errors when ``raise_on_error`` is true.
        """
        if not encoded_keys or not objs:
            raise ValueError("encoded_keys and objs must be non-empty")
        if len(encoded_keys) != len(objs):
            raise ValueError("encoded_keys and objs must have the same length")

        with self._lock:
            items = [
                (encoded_key, self._index.get(encoded_key))
                for encoded_key in encoded_keys
            ]
            self._inflight_io_count += 1

        results = [False] * len(encoded_keys)
        touched: list[str] = []
        try:
            for i, (encoded_key, entry) in enumerate(items):
                if entry is None:
                    continue
                try:
                    payload_len = int(entry.size)
                    total_len = (
                        round_up(payload_len, self.block_align)
                        if self._requires_aligned_io()
                        else payload_len
                    )
                    buf = memoryview(objs[i].byte_array)
                    try:
                        buf = buf.cast("B")
                    except Exception:
                        pass

                    direct_view = self._build_direct_odirect_view(
                        memory_obj=objs[i],
                        payload_len=payload_len,
                        total_len=total_len,
                        buffer_len=len(buf),
                        zero_tail=False,
                    )
                    if direct_view is not None:
                        self._read_buffers(
                            [entry.offset + self.header_bytes],
                            [direct_view],
                            [
                                total_len
                                if len(direct_view) >= total_len
                                else payload_len
                            ],
                            [total_len],
                        )
                    else:
                        self._read_buffers(
                            [entry.offset + self.header_bytes],
                            [buf],
                            [payload_len],
                            [total_len],
                        )
                    objs[i].metadata.cached_positions = entry.meta.cached_positions
                    touched.append(encoded_key)
                    results[i] = True
                except Exception as e:
                    if raise_on_error:
                        raise
                    logger.error("RawBlockCore load failed for %s: %s", encoded_key, e)
        finally:
            with self._lock:
                for encoded_key in touched:
                    self._touch_locked(encoded_key)
                self._inflight_io_count -= 1
                self._last_io_ts = time.monotonic()
        return results

    def unlock_many(self, encoded_keys: Sequence[str]) -> None:
        """Release L2 lock references for encoded keys.

        Args:
            encoded_keys: Encoded raw-block keys whose lock refcounts should be
                decremented. Missing keys and underflow are treated as no-ops.
        """
        with self._lock:
            for encoded_key in encoded_keys:
                refcnt = self._lock_refcnt.get(encoded_key, 0)
                if refcnt <= 1:
                    self._lock_refcnt.pop(encoded_key, None)
                else:
                    self._lock_refcnt[encoded_key] = refcnt - 1

    def delete_many(
        self,
        encoded_keys: Sequence[str],
        *,
        force: bool = False,
    ) -> list[bool]:
        """Delete indexed keys and recycle their slots when allowed.

        Args:
            encoded_keys: Ordered encoded raw-block keys to delete.
            force: If true, delete locked keys as well. Normal MP eviction uses
                false so locked entries are preserved.

        Returns:
            A list of per-key deletion booleans aligned with ``encoded_keys``.
        """
        deleted: list[bool] = []
        with self._lock:
            for encoded_key in encoded_keys:
                existed = encoded_key in self._index or encoded_key in self._inflight
                entry = self._index.get(encoded_key)
                locked = self._lock_refcnt.get(encoded_key, 0) > 0
                if entry is not None and locked and not force:
                    deleted.append(False)
                    continue

                removed_entry = self._index.pop(encoded_key, None)
                inflight = self._inflight.get(encoded_key)
                if inflight is not None:
                    inflight.canceled = True
                self._lock_refcnt.pop(encoded_key, None)
                self._lru.pop(encoded_key, None)
                if removed_entry is not None:
                    self._append_free_slot_locked(
                        self._offset_to_slot(int(removed_entry.offset))
                    )
                    self._meta_dirty_total += 1
                deleted.append(
                    existed and (removed_entry is not None or inflight is not None)
                )
        return deleted

    def usage(self) -> tuple[float, float]:
        """Return current and post-eviction usage fractions.

        Returns:
            ``(current_usage, projected_usage)``. Raw-block has no separate
            projected value, so both values are identical. ``(-1.0, -1.0)``
            indicates that usable capacity is unknown.
        """
        with self._lock:
            usable_capacity = self._max_slots * self.slot_bytes
            if usable_capacity <= 0:
                return (-1.0, -1.0)
            used_slots = len(self._index) + len(self._inflight)
            usage = (used_slots * self.slot_bytes) / usable_capacity
            return (usage, usage)

    def checkpoint_now(self) -> None:
        """Synchronously write a metadata checkpoint."""
        self._checkpoint_once(force=True)

    def apply_loaded_state(self, data: dict[str, Any]) -> bool:
        """Validate and apply a recovered metadata checkpoint payload.

        Args:
            data: Decoded checkpoint dictionary.

        Returns:
            True when the payload shape and layout match this core and all
            valid entries were applied. Invalid per-entry records are skipped.
        """
        return self._apply_loaded_state(data)

    def report_status(self) -> dict:
        """Return raw-block health, layout, metadata, and in-flight counters."""
        with self._lock:
            return {
                "is_healthy": not self._closed,
                "type": "RawBlockCore",
                "key_namespace": self.key_namespace,
                "device_path": self.device_path,
                "base_offset_bytes": self.base_offset_bytes,
                "block_align": self.block_align,
                "header_bytes": self.header_bytes,
                "slot_bytes": self.slot_bytes,
                "meta_total_bytes": self.meta_total_bytes,
                "reserved_meta_bytes": self._reserved_meta_bytes,
                "meta_partition_count": self._meta_partition_count,
                "usable_capacity_bytes": self._max_slots * self.slot_bytes,
                "indexed_key_count": len(self._index),
                "inflight_key_count": len(self._inflight),
                "locked_key_count": sum(
                    1 for refcnt in self._lock_refcnt.values() if refcnt > 0
                ),
                "free_slot_count": len(self._free_slots),
                "next_slot": self._next_slot,
                "max_slots": self._max_slots,
                "metadata_seq": self._meta_seq,
                "metadata_dirty_total": self._meta_dirty_total,
                "metadata_persisted": self._meta_persisted,
                "inflight_io_count": self._inflight_io_count,
                "use_odirect": self.use_odirect,
                "use_uring": self.use_uring,
                "use_uring_cmd": self.use_uring_cmd,
                "use_fdp": self.use_fdp,
                "fdp_ruh_ids": list(self.fdp_ruh_ids),
                "fdp_data_ruh_ids": list(self.fdp_data_ruh_ids),
                "fdp_metadata_ruh_ids": list(self.fdp_metadata_ruh_ids),
                "fdp_directive_type": self.fdp_directive_type,
                "fdp_metadata_mode": self.fdp_metadata_mode,
                "enable_zero_copy": self.enable_zero_copy,
            }

    def close(self) -> None:
        """Stop checkpointing, write a final checkpoint, and close the device."""
        with self._lock:
            if self._closed:
                return
            self._closed = True

        self._meta_stop_evt.set()
        if self._meta_thread is not None:
            self._meta_thread.join(timeout=5)
            self._meta_thread = None

        try:
            self._checkpoint_once(force=True)
        except Exception as e:
            logger.warning("RawBlockCore final checkpoint failed: %s", e)

        if self._raw is not None:
            try:
                self._raw.close()
            except Exception as e:
                logger.warning(
                    "Failed to close raw block device %s: %s", self.device_path, e
                )
            finally:
                self._raw = None

    def _cleanup_after_init_failure(self) -> None:
        """Close resources that may have been opened before init failed."""
        self._meta_stop_evt.set()
        if self._meta_thread is not None:
            self._meta_thread.join(timeout=5)
            self._meta_thread = None
        if self._raw is not None:
            try:
                self._raw.close()
            except Exception as e:
                logger.warning(
                    "Failed to close raw block device %s: %s", self.device_path, e
                )
            finally:
                self._raw = None
        self._closed = True

    def _build_direct_odirect_view(
        self,
        memory_obj: MemoryObj,
        payload_len: int,
        total_len: int,
        buffer_len: int,
        *,
        zero_tail: bool,
    ) -> Optional[memoryview]:
        """Build an aligned memoryview for direct O_DIRECT I/O when possible.

        Args:
            memory_obj: Memory object whose backing allocation may be aligned.
            payload_len: Logical payload length in bytes.
            total_len: I/O length after any O_DIRECT padding.
            buffer_len: Available buffer length in bytes.
            zero_tail: Whether to zero any padded tail bytes before writing.

        Returns:
            A direct memoryview over the allocation, or None when the memory
            object is unsuitable for direct I/O.
        """
        if not self.use_odirect or not self.enable_zero_copy:
            return None

        ptr_val = getattr(memory_obj, "data_ptr", None)
        if callable(ptr_val):
            try:
                ptr_val = ptr_val()
            except Exception:
                ptr_val = None
        if ptr_val is None:
            return None
        if buffer_len <= 0:
            return None

        ptr = int(ptr_val)
        if ptr <= 0 or ptr % self.block_align != 0:
            return None
        if buffer_len < payload_len:
            return None

        view_len = min(buffer_len, total_len)
        if view_len < payload_len:
            return None

        try:
            raw = (ctypes.c_ubyte * view_len).from_address(ptr)
            view = memoryview(raw)
            if zero_tail and total_len > payload_len and view_len >= total_len:
                ctypes.memset(ptr + payload_len, 0, total_len - payload_len)
            return view
        except Exception:
            return None

    def _requires_aligned_io(self) -> bool:
        """Return whether this I/O path requires block-aligned transfers."""
        return self.use_odirect or self.use_uring_cmd

    def _prepare_write_payload(self, memory_obj: MemoryObj) -> tuple[Any, int, int]:
        """Prepare the payload buffer and lengths for a raw-block write.

        Args:
            memory_obj: Source object to persist.

        Returns:
            A tuple of ``(buffer, payload_len, total_len)`` where ``total_len``
            includes any O_DIRECT or raw-command alignment padding.

        Raises:
            RuntimeError: If the aligned payload would exceed slot capacity.
        """
        buf = memory_obj.byte_array
        if hasattr(buf, "cast"):
            buf = buf.cast("B")
        payload_len = len(memory_obj.byte_array)
        total_len = payload_len
        if self._requires_aligned_io():
            total_len = round_up(payload_len, self.block_align)
            if total_len > (self.slot_bytes - self.header_bytes):
                raise RuntimeError(f"Aligned payload {total_len} exceeds slot capacity")
            direct_view = self._build_direct_odirect_view(
                memory_obj=memory_obj,
                payload_len=payload_len,
                total_len=total_len,
                buffer_len=len(buf),
                zero_tail=True,
            )
            if direct_view is not None:
                buf = direct_view
        return buf, payload_len, total_len

    def _byte_view(self, buf: Any) -> memoryview:
        """Return a byte-addressable memoryview over a Python buffer.

        Args:
            buf: Object implementing the Python buffer protocol.

        Returns:
            A memoryview with one-byte elements.

        Raises:
            TypeError: If ``buf`` does not expose a compatible contiguous buffer.
        """
        view = buf if isinstance(buf, memoryview) else memoryview(buf)
        if view.itemsize == 1 and view.format in ("B", "b", "c"):
            return view
        return view.cast("B")

    def _validate_uring_cmd_chunk(self, offset: int, total_len: int) -> None:
        """Validate one NVMe raw-command transfer range.

        Args:
            offset: Device byte offset for the transfer.
            total_len: Transfer size in bytes.

        Raises:
            ValueError: If either value is not block aligned.
        """
        if offset % self.block_align != 0:
            raise ValueError("io_uring_cmd requires aligned offsets")
        if total_len % self.block_align != 0:
            raise ValueError("io_uring_cmd requires aligned transfer lengths")

    def _write_uring_cmd_buffers(
        self,
        offsets: Sequence[int],
        buffers: Sequence[Any],
        payload_lens: Sequence[int],
        total_lens: Sequence[int],
        *,
        fdp_ruh_id: int | None = None,
    ) -> None:
        """Write buffers as bounded NVMe raw-command chunks.

        Args:
            offsets: Device offsets for each logical write.
            buffers: Source buffers.
            payload_lens: Logical source byte counts.
            total_lens: Physical transfer sizes, including padding.
            fdp_ruh_id: Optional FDP RUH used as the NVMe directive specifier.

        Raises:
            ValueError: If lengths are inconsistent or unaligned.
            Exception: Propagates Rust raw-device write errors.
        """
        raw_dev = self._rawdev()
        chunk_offsets: list[int] = []
        chunk_buffers: list[memoryview] = []
        chunk_lens: list[int] = []
        keepalive: list[memoryview] = []

        for offset, buf, payload_len, total_len in zip(
            offsets, buffers, payload_lens, total_lens, strict=True
        ):
            offset = int(offset)
            payload_len = int(payload_len)
            total_len = int(total_len)
            self._validate_uring_cmd_chunk(offset, total_len)

            view = self._byte_view(buf)
            if len(view) < total_len:
                if len(view) < payload_len:
                    raise ValueError("input buffer shorter than payload_len")
                padded = bytearray(total_len)
                padded[:payload_len] = view[:payload_len]
                view = memoryview(padded)
            else:
                view = view[:total_len]
            keepalive.append(view)

            cursor = 0
            while cursor < total_len:
                chunk_len = min(_URING_CMD_MAX_IO_BYTES, total_len - cursor)
                self._validate_uring_cmd_chunk(offset + cursor, chunk_len)
                chunk_offsets.append(offset + cursor)
                chunk_buffers.append(view[cursor : cursor + chunk_len])
                chunk_lens.append(chunk_len)
                cursor += chunk_len

        if not chunk_offsets:
            return
        dtype, dspec = self._write_directive_for_ruh(fdp_ruh_id)
        batch_id = raw_dev.batched_write(
            chunk_offsets,
            chunk_buffers,
            chunk_lens,
            dtype,
            dspec,
        )
        raw_dev.wait_iouring(batch_id)
        keepalive.clear()

    def _read_uring_cmd_buffers(
        self,
        offsets: Sequence[int],
        buffers: Sequence[Any],
        payload_lens: Sequence[int],
        total_lens: Sequence[int],
    ) -> None:
        """Read buffers as bounded NVMe raw-command chunks.

        Args:
            offsets: Device offsets for each logical read.
            buffers: Destination buffers.
            payload_lens: Logical bytes to expose to callers.
            total_lens: Physical transfer sizes, including padding.

        Raises:
            ValueError: If lengths are inconsistent or unaligned.
            Exception: Propagates Rust raw-device read errors.
        """
        raw_dev = self._rawdev()
        read_uring = raw_dev.read_uring

        for offset, buf, payload_len, total_len in zip(
            offsets, buffers, payload_lens, total_lens, strict=True
        ):
            offset = int(offset)
            payload_len = int(payload_len)
            total_len = int(total_len)
            self._validate_uring_cmd_chunk(offset, total_len)

            dst = self._byte_view(buf)
            if len(dst) < total_len:
                if len(dst) < payload_len:
                    raise ValueError("output buffer shorter than payload_len")
                target = memoryview(bytearray(total_len))
                copy_back = True
            else:
                target = dst[:total_len]
                copy_back = False

            cursor = 0
            while cursor < total_len:
                chunk_len = min(_URING_CMD_MAX_IO_BYTES, total_len - cursor)
                self._validate_uring_cmd_chunk(offset + cursor, chunk_len)
                read_uring(
                    offset + cursor,
                    target[cursor : cursor + chunk_len],
                    chunk_len,
                    chunk_len,
                )
                cursor += chunk_len

            if copy_back:
                dst[:payload_len] = target[:payload_len]

    def _write_buffers(
        self,
        offsets: Sequence[int],
        buffers: Sequence[Any],
        payload_lens: Sequence[int],
        total_lens: Sequence[int],
        *,
        fdp_ruh_id: int | None = None,
    ) -> None:
        """Write one or more buffers through the configured Rust I/O path.

        Args:
            offsets: Device offsets for each write.
            buffers: Python buffers to write.
            payload_lens: Logical payload lengths for each buffer.
            total_lens: Physical I/O lengths for each buffer.
            fdp_ruh_id: Optional FDP RUH used for NVMe raw-command writes.

        Raises:
            RuntimeError: If the requested io_uring mode is unavailable.
            Exception: Propagates Rust raw-device write errors.
        """
        raw_dev = self._rawdev()
        if not self.use_uring:
            for offset, buf, payload_len, total_len in zip(
                offsets, buffers, payload_lens, total_lens, strict=True
            ):
                raw_dev.pwrite_from_buffer(offset, buf, payload_len, total_len)
            return

        if self.use_uring_cmd:
            self._write_uring_cmd_buffers(
                offsets,
                buffers,
                payload_lens,
                total_lens,
                fdp_ruh_id=fdp_ruh_id,
            )
            return

        can_batch = all(
            int(payload_len) == int(total_len)
            for payload_len, total_len in zip(payload_lens, total_lens, strict=True)
        )
        if can_batch:
            batch_id = raw_dev.batched_write(
                [int(offset) for offset in offsets],
                list(buffers),
                [int(total_len) for total_len in total_lens],
            )
            raw_dev.wait_iouring(batch_id)
            return

        write_uring = getattr(raw_dev, "write_uring", None)
        if not callable(write_uring):
            if self.use_uring_cmd:
                raise RuntimeError("io_uring_cmd write path requires write_uring")
            logger.warning(
                "RawBlockCore falling back to pwrite for padded io_uring writes "
                "because write_uring is unavailable"
            )
            for offset, buf, payload_len, total_len in zip(
                offsets, buffers, payload_lens, total_lens, strict=True
            ):
                raw_dev.pwrite_from_buffer(offset, buf, payload_len, total_len)
            return

        for offset, buf, payload_len, total_len in zip(
            offsets, buffers, payload_lens, total_lens, strict=True
        ):
            write_uring(int(offset), buf, int(payload_len), int(total_len))

    def _read_buffers(
        self,
        offsets: Sequence[int],
        buffers: Sequence[Any],
        payload_lens: Sequence[int],
        total_lens: Sequence[int],
    ) -> None:
        """Read one or more buffers through the configured Rust I/O path.

        Args:
            offsets: Device offsets for each read.
            buffers: Destination Python buffers.
            payload_lens: Logical payload lengths to expose to callers.
            total_lens: Physical I/O lengths for each read.

        Raises:
            RuntimeError: If the requested io_uring mode is unavailable.
            Exception: Propagates Rust raw-device read errors.
        """
        raw_dev = self._rawdev()
        if not self.use_uring:
            for offset, buf, payload_len, total_len in zip(
                offsets, buffers, payload_lens, total_lens, strict=True
            ):
                raw_dev.pread_into(offset, buf, payload_len, total_len)
            return

        if self.use_uring_cmd:
            self._read_uring_cmd_buffers(offsets, buffers, payload_lens, total_lens)
            return

        can_batch = all(
            int(payload_len) == int(total_len)
            for payload_len, total_len in zip(payload_lens, total_lens, strict=True)
        )
        if can_batch:
            batch_id = raw_dev.batched_read(
                [int(offset) for offset in offsets],
                list(buffers),
                [int(total_len) for total_len in total_lens],
            )
            raw_dev.wait_iouring(batch_id)
            return

        for offset, buf, payload_len, total_len in zip(
            offsets, buffers, payload_lens, total_lens, strict=True
        ):
            raw_dev.read_uring(int(offset), buf, int(payload_len), int(total_len))

    def _write_one(
        self,
        key: RawBlockKeySpec,
        memory_obj: MemoryObj,
        offset: int,
        *,
        fdp_ruh_id: int | None = None,
    ) -> bool:
        """Write one object header and payload into a raw-block slot.

        Args:
            key: Raw-block key spec with the slot-header identity.
            memory_obj: Source object to write.
            offset: Slot byte offset on the raw device.
            fdp_ruh_id: Optional FDP RUH used as the NVMe directive specifier.

        Returns:
            True when both header and payload writes complete; false otherwise.
        """
        try:
            header = self._encode_header(key.slot_identity, len(memory_obj.byte_array))
            buf, payload_len, total_len = self._prepare_write_payload(memory_obj)

            with self._lock:
                self._inflight_io_count += 1
            try:
                hdr_total = (
                    round_up(len(header), self.block_align)
                    if self._requires_aligned_io()
                    else len(header)
                )
                header_buf: Any = header
                if self.use_uring and len(header) < hdr_total:
                    padded_header = bytearray(header)
                    padded_header.extend(b"\x00" * (hdr_total - len(header)))
                    header_buf = padded_header
                self._write_buffers(
                    [offset, offset + self.header_bytes],
                    [header_buf, buf],
                    [
                        hdr_total if self.use_uring else len(header),
                        payload_len,
                    ],
                    [hdr_total, total_len],
                    fdp_ruh_id=fdp_ruh_id,
                )
            finally:
                with self._lock:
                    self._inflight_io_count -= 1
                    self._last_io_ts = time.monotonic()
            return True
        except Exception as e:
            logger.error("RawBlockCore write failed for %s: %s", key.encoded, e)
            return False

    def _encode_header(self, slot_identity: int, payload_len: int) -> bytes:
        """Encode a fixed-size raw-block slot header."""
        hdr = bytearray(self.header_bytes)
        hdr[0:8] = b"LMCBLK01"
        hdr[8:16] = int(slot_identity & ((1 << 64) - 1)).to_bytes(
            8,
            "little",
            signed=False,
        )
        hdr[16:24] = int(payload_len).to_bytes(8, "little", signed=False)
        return bytes(hdr)

    def _decode_slot_header(self, hdr: bytes) -> Optional[tuple[int, int]]:
        """Decode a raw-block slot header into identity and payload length."""
        if len(hdr) < 24 or hdr[0:8] != b"LMCBLK01":
            return None
        slot_identity = int.from_bytes(hdr[8:16], "little", signed=False)
        payload_len = int.from_bytes(hdr[16:24], "little", signed=False)
        return slot_identity, payload_len

    def _read_slot_header(self, offset: int) -> Optional[tuple[int, int]]:
        """Read and decode the slot header at a raw-device offset."""
        buf = bytearray(self.header_bytes)
        try:
            with self._lock:
                self._inflight_io_count += 1
            self._read_buffers(
                [offset],
                [buf],
                [self.header_bytes],
                [self.header_bytes],
            )
            return self._decode_slot_header(buf)
        except Exception:
            return None
        finally:
            with self._lock:
                self._inflight_io_count -= 1
                self._last_io_ts = time.monotonic()

    def _ensure_capacity_and_layout(self) -> None:
        """Open the device if needed and compute metadata/data layout."""
        if self._effective_capacity_bytes > 0 and self._max_slots > 0:
            return

        device_size = int(self._rawdev().size_bytes())
        if self.base_offset_bytes >= device_size:
            raise RuntimeError("base_offset_bytes exceeds raw device capacity")

        available_bytes = device_size - self.base_offset_bytes
        requested = self.capacity_bytes if self.capacity_bytes > 0 else available_bytes
        self._effective_capacity_bytes = min(requested, available_bytes)
        self.capacity_bytes = self._effective_capacity_bytes

        if self._reserved_meta_bytes >= self._effective_capacity_bytes:
            raise RuntimeError("metadata region exceeds usable device capacity")

        self._data_base_offset = self.base_offset_bytes + self._reserved_meta_bytes
        data_bytes = self._effective_capacity_bytes - self._reserved_meta_bytes
        self._max_slots = data_bytes // self.slot_bytes
        if self._max_slots <= 0:
            raise RuntimeError(
                "raw block capacity too small for slot size after metadata"
            )

    def _slot_to_offset(self, slot: int) -> int:
        """Convert a data-slot index to its byte offset."""
        return self._data_base_offset + slot * self.slot_bytes

    def _offset_to_slot(self, offset: int) -> int:
        """Convert a data-slot byte offset to its slot index."""
        return (offset - self._data_base_offset) // self.slot_bytes

    def _allocate_slot_locked(self) -> int:
        """Allocate a slot offset while ``self._lock`` is held."""
        self._ensure_capacity_and_layout()
        if self._free_slots:
            return self._slot_to_offset(self._free_slots.pop())
        if self._next_slot < self._max_slots:
            slot = self._next_slot
            self._next_slot += 1
            return self._slot_to_offset(slot)
        raise RuntimeError("No free slots available; eviction required")

    def _touch_locked(self, encoded_key: str) -> None:
        """Mark a key as most recently used while ``self._lock`` is held."""
        self._lru.pop(encoded_key, None)
        self._lru[encoded_key] = None

    def _append_free_slot_locked(self, slot: int) -> None:
        """Add a slot to the free list while ``self._lock`` is held."""
        if slot < 0 or slot >= self._max_slots:
            return
        if slot in self._free_slots:
            return
        self._free_slots.append(slot)

    def _evict_one_locked(self) -> Optional[tuple[str, int]]:
        """Evict one unlocked LRU entry while ``self._lock`` is held."""
        for victim in list(self._lru.keys()):
            if self._lock_refcnt.get(victim, 0) > 0:
                continue
            if victim in self._inflight:
                continue
            entry = self._index.pop(victim, None)
            if entry is None:
                self._lru.pop(victim, None)
                continue
            self._lru.pop(victim, None)
            self._lock_refcnt.pop(victim, None)
            self._append_free_slot_locked(self._offset_to_slot(int(entry.offset)))
            self._meta_dirty_total += 1
            return victim, int(entry.size)
        return None

    def _checkpoint_loop(self) -> None:
        """Periodically checkpoint dirty metadata until shutdown."""
        interval = max(1, self.meta_checkpoint_interval_sec)
        while not self._meta_stop_evt.wait(interval):
            try:
                self._checkpoint_once(force=False)
            except Exception as e:
                logger.warning("Periodic raw-block metadata checkpoint failed: %s", e)

    def _meta_payload_capacity(self) -> int:
        """Return usable bytes in one metadata checkpoint payload area."""
        return self._meta_container_bytes - self.block_align

    def _meta_container_offsets(self, fdp_ruh_id: int | None = None) -> list[int]:
        """Return byte offsets for mirrored metadata checkpoint containers."""
        if not self.use_fdp:
            return [
                self.base_offset_bytes + idx * self._meta_container_bytes
                for idx in range(self._meta_copy_count)
            ]

        if fdp_ruh_id is None:
            offsets: list[int] = []
            for ruh_id in self.fdp_metadata_ruh_ids:
                offsets.extend(self._meta_container_offsets(ruh_id))
            return offsets

        if fdp_ruh_id not in self._fdp_metadata_ruh_index:
            raise ValueError(f"Unknown FDP metadata RUH ID {fdp_ruh_id}")
        partition_base = (
            self._fdp_metadata_ruh_index[fdp_ruh_id] * self.meta_total_bytes
        )
        return [
            self.base_offset_bytes + partition_base + idx * self._meta_container_bytes
            for idx in range(self._meta_copy_count)
        ]

    def _read_meta_header(self, container_offset: int) -> Optional[dict[str, int]]:
        """Read and validate a metadata checkpoint header."""
        buf = bytearray(self.block_align)
        try:
            self._read_buffers(
                [container_offset],
                [buf],
                [self.block_align],
                [self.block_align],
            )
        except Exception:
            return None

        hdr = bytes(buf[: _META_HEADER_STRUCT.size])
        magic, version, seq, payload_len, crc = _META_HEADER_STRUCT.unpack(hdr)
        if magic != self.meta_magic or version != self.meta_version:
            return None

        payload_cap = self._meta_payload_capacity()
        if payload_len <= 0 or payload_len > payload_cap:
            return None
        return {
            "seq": int(seq),
            "payload_len": int(payload_len),
            "crc": int(crc),
            "container_offset": int(container_offset),
        }

    def _load_meta_payload(self, header: dict[str, int]) -> Optional[bytes]:
        """Load and CRC-validate a checkpoint payload for a metadata header."""
        payload_len = int(header["payload_len"])
        payload_off = int(header["container_offset"]) + self.block_align
        total_len = round_up(payload_len, self.block_align)
        buf = bytearray(total_len)
        try:
            self._read_buffers([payload_off], [buf], [payload_len], [total_len])
        except Exception:
            return None

        payload = bytes(buf[:payload_len])
        crc = zlib.crc32(payload) & 0xFFFFFFFF
        if crc != int(header["crc"]):
            return None
        return payload

    def _select_latest_checkpoint(
        self,
        fdp_ruh_id: int | None = None,
    ) -> tuple[Optional[dict[str, int]], Optional[bytes]]:
        """Return the newest valid checkpoint header and payload."""
        best_header: Optional[dict[str, int]] = None
        best_payload: Optional[bytes] = None
        for offset in self._meta_container_offsets(fdp_ruh_id):
            header = self._read_meta_header(offset)
            if header is None:
                continue
            payload = self._load_meta_payload(header)
            if payload is None:
                continue
            if best_header is None or int(header["seq"]) > int(best_header["seq"]):
                best_header = header
                best_payload = payload
        return best_header, best_payload

    def _snapshot_state(
        self,
        *,
        fdp_ruh_id: int | None = None,
    ) -> tuple[dict[str, Any], int]:
        """Build a JSON-serializable checkpoint state snapshot."""
        if self.use_fdp and fdp_ruh_id is None:
            raise ValueError("FDP checkpoints require a specific RUH ID")
        with self._lock:
            dirty_total = self._meta_dirty_total
            checkpoint_entries = {
                encoded_key: entry
                for encoded_key, entry in self._index.items()
                if not self.use_fdp
                or self._select_fdp_metadata_ruh(encoded_key) == fdp_ruh_id
            }
            snapshot = {
                "version": 1,
                "device_path": self.device_path,
                "capacity_bytes": self.capacity_bytes,
                "base_offset_bytes": self.base_offset_bytes,
                "block_align": self.block_align,
                "header_bytes": self.header_bytes,
                "slot_bytes": self.slot_bytes,
                "meta_total_bytes": self.meta_total_bytes,
                "meta_magic": self.meta_magic_text,
                "meta_version": self.meta_version,
                "use_fdp": self.use_fdp,
                "fdp_ruh_ids": list(self.fdp_ruh_ids),
                "fdp_data_ruh_ids": list(self.fdp_data_ruh_ids),
                "fdp_metadata_ruh_ids": list(self.fdp_metadata_ruh_ids),
                "fdp_ruh_id": fdp_ruh_id,
                "fdp_metadata_ruh_id": fdp_ruh_id,
                "fdp_directive_type": self.fdp_directive_type,
                "fdp_metadata_mode": self.fdp_metadata_mode,
                "reserved_meta_bytes": self._reserved_meta_bytes,
                "data_base_offset": self._data_base_offset,
                "next_slot": self._next_slot,
                "free_slots": list(self._free_slots),
                "lru_keys": [
                    encoded_key
                    for encoded_key in self._lru.keys()
                    if encoded_key in checkpoint_entries
                ],
                "entries": {
                    encoded_key: {
                        "offset": entry.offset,
                        "size": entry.meta.size,
                        "fdp_ruh_id": entry.fdp_ruh_id,
                        "shape": list(entry.meta.shape)
                        if entry.meta.shape is not None
                        else None,
                        "dtype": self._checkpoint_dtype_name(entry.meta.dtype),
                        "fmt": (
                            entry.meta.fmt.name
                            if entry.meta.fmt is not None
                            and hasattr(entry.meta.fmt, "name")
                            else str(entry.meta.fmt)
                            if entry.meta.fmt is not None
                            else None
                        ),
                        "cached_positions": (
                            entry.meta.cached_positions.tolist()
                            if entry.meta.cached_positions is not None
                            and hasattr(entry.meta.cached_positions, "tolist")
                            else None
                        ),
                    }
                    for encoded_key, entry in checkpoint_entries.items()
                },
            }
        return snapshot, dirty_total

    def _checkpoint_dtype_name(self, dtype: torch.dtype | None) -> str | None:
        """Return a durable checkpoint string for a torch dtype.

        Args:
            dtype: Torch dtype from recovered or live memory metadata.

        Returns:
            Stable LMCache dtype name when known, ``str(dtype)`` for unknown
            torch dtypes, or None when no dtype is available.
        """
        if dtype is None:
            return None
        return TORCH_DTYPE_TO_STR_DTYPE.get(dtype, str(dtype))

    def _write_checkpoint(
        self,
        payload: bytes,
        dirty_total_snapshot: int,
        *,
        fdp_ruh_id: int | None = None,
        next_seq: int | None = None,
        update_counters: bool = True,
    ) -> bool:
        """Write one checkpoint copy and advance persisted metadata counters."""
        payload_cap = self._meta_payload_capacity()
        if len(payload) > payload_cap:
            logger.warning(
                "RawBlockCore metadata payload too large (%d > %d), "
                "skipping checkpoint",
                len(payload),
                payload_cap,
            )
            return False

        if next_seq is None:
            next_seq = self._meta_seq + 1
        target_idx = int((next_seq - 1) % self._meta_copy_count)
        target = self._meta_container_offsets(fdp_ruh_id)[target_idx]

        payload_len = len(payload)
        payload_total_len = round_up(payload_len, self.block_align)
        payload_off = target + self.block_align
        crc = zlib.crc32(payload) & 0xFFFFFFFF

        header_block = bytearray(self.block_align)
        header_block[: _META_HEADER_STRUCT.size] = _META_HEADER_STRUCT.pack(
            self.meta_magic,
            self.meta_version,
            int(next_seq),
            int(payload_len),
            int(crc),
        )

        self._write_buffers(
            [payload_off, target],
            [payload, header_block],
            [payload_len, self.block_align],
            [payload_total_len, self.block_align],
            fdp_ruh_id=fdp_ruh_id,
        )

        if update_counters:
            with self._lock:
                self._meta_seq = int(next_seq)
                self._meta_persisted = max(
                    self._meta_persisted, int(dirty_total_snapshot)
                )
        return True

    def _checkpoint_once(self, force: bool) -> bool:
        """Write a metadata checkpoint when dirty and sufficiently idle."""
        with self._lock:
            dirty = self._meta_dirty_total > self._meta_persisted
            idle_ok = self._inflight_io_count == 0 and (
                time.monotonic() - self._last_io_ts
            ) >= (self.meta_idle_quiet_ms / 1000.0)

        if not dirty:
            return False
        if not force and not idle_ok:
            return False

        if not self.use_fdp:
            snapshot, dirty_total_snapshot = self._snapshot_state()
            payload = json.dumps(
                snapshot, separators=(",", ":"), ensure_ascii=True
            ).encode("utf-8")
            return self._write_checkpoint(payload, dirty_total_snapshot)

        next_seq = self._meta_seq + 1
        dirty_total_snapshot = 0
        for ruh_id in self.fdp_metadata_ruh_ids:
            snapshot, dirty_total = self._snapshot_state(fdp_ruh_id=ruh_id)
            dirty_total_snapshot = max(dirty_total_snapshot, dirty_total)
            payload = json.dumps(
                snapshot, separators=(",", ":"), ensure_ascii=True
            ).encode("utf-8")
            if not self._write_checkpoint(
                payload,
                dirty_total,
                fdp_ruh_id=ruh_id,
                next_seq=next_seq,
                update_counters=False,
            ):
                return False

        with self._lock:
            self._meta_seq = int(next_seq)
            self._meta_persisted = max(self._meta_persisted, dirty_total_snapshot)
        return True

    def _is_valid_checkpoint_entry(self, offset: int, size: int) -> bool:
        """Return whether a checkpoint entry references a valid data slot."""
        if offset < self._data_base_offset:
            return False
        rel = offset - self._data_base_offset
        if rel % self.slot_bytes != 0:
            return False
        slot = rel // self.slot_bytes
        if slot < 0 or slot >= self._max_slots:
            return False
        return 0 < size <= (self.slot_bytes - self.header_bytes)

    def _apply_loaded_state(
        self,
        data: dict[str, Any],
        *,
        merge: bool = False,
        expected_fdp_ruh_id: int | None = None,
    ) -> bool:
        """Apply decoded checkpoint state after validating layout fields."""
        if not isinstance(data, dict):
            return False
        if int(data.get("version", 0)) != 1:
            return False
        if data.get("device_path") and data.get("device_path") != self.device_path:
            logger.warning("Device metadata device_path mismatch; ignoring metadata")
            return False
        if int(data.get("base_offset_bytes", 0)) != self.base_offset_bytes:
            logger.warning(
                "Device metadata base_offset_bytes mismatch; ignoring metadata"
            )
            return False
        if int(data.get("slot_bytes", self.slot_bytes)) != self.slot_bytes:
            logger.warning("Device metadata slot_bytes mismatch; ignoring metadata")
            return False
        if (
            int(data.get("meta_total_bytes", self.meta_total_bytes))
            != self.meta_total_bytes
        ):
            logger.warning(
                "Device metadata meta_total_bytes mismatch; ignoring metadata"
            )
            return False
        if str(data.get("meta_magic", self.meta_magic_text)) != self.meta_magic_text:
            logger.warning("Device metadata meta_magic mismatch; ignoring metadata")
            return False
        if int(data.get("meta_version", self.meta_version)) != self.meta_version:
            logger.warning("Device metadata meta_version mismatch; ignoring metadata")
            return False

        data_use_fdp = bool(data.get("use_fdp", False))
        if data_use_fdp != self.use_fdp:
            logger.warning("Device metadata FDP mode mismatch; ignoring metadata")
            return False
        if self.use_fdp:
            if str(data.get("fdp_metadata_mode", "per_ruh")) != self.fdp_metadata_mode:
                logger.warning(
                    "Device metadata FDP metadata mode mismatch; ignoring metadata"
                )
                return False
            raw_fdp_ruh_ids = data.get("fdp_ruh_ids", [])
            if raw_fdp_ruh_ids is None:
                raw_fdp_ruh_ids = []
            raw_fdp_data_ruh_ids = data.get(
                "fdp_data_ruh_ids",
                raw_fdp_ruh_ids,
            )
            if raw_fdp_data_ruh_ids is None:
                raw_fdp_data_ruh_ids = []
            raw_fdp_metadata_ruh_ids = data.get(
                "fdp_metadata_ruh_ids",
                raw_fdp_ruh_ids,
            )
            if raw_fdp_metadata_ruh_ids is None:
                raw_fdp_metadata_ruh_ids = []
            data_fdp_data_ruh_ids = tuple(int(v) for v in raw_fdp_data_ruh_ids)
            data_fdp_metadata_ruh_ids = tuple(int(v) for v in raw_fdp_metadata_ruh_ids)
            if data_fdp_data_ruh_ids != self.fdp_data_ruh_ids:
                logger.warning(
                    "Device metadata FDP data RUH list mismatch; ignoring metadata"
                )
                return False
            if data_fdp_metadata_ruh_ids != self.fdp_metadata_ruh_ids:
                logger.warning(
                    "Device metadata FDP metadata RUH list mismatch; ignoring metadata"
                )
                return False
            try:
                raw_data_fdp_ruh_id = data.get(
                    "fdp_metadata_ruh_id",
                    data.get("fdp_ruh_id"),
                )
                if raw_data_fdp_ruh_id is None:
                    raise ValueError
                data_fdp_ruh_id = int(raw_data_fdp_ruh_id)
            except Exception:
                logger.warning(
                    "Device metadata FDP metadata RUH ID is invalid; ignoring metadata"
                )
                return False
            if (
                expected_fdp_ruh_id is not None
                and data_fdp_ruh_id != expected_fdp_ruh_id
            ):
                logger.warning("Device metadata FDP metadata RUH partition mismatch")
                return False
            if data_fdp_ruh_id not in self._fdp_metadata_ruh_index:
                logger.warning("Device metadata FDP metadata RUH ID is not configured")
                return False
        else:
            data_fdp_ruh_id = None

        try:
            next_slot = int(data.get("next_slot", 0))
        except Exception:
            logger.warning("Device metadata next_slot is invalid; ignoring metadata")
            return False
        if next_slot < 0 or next_slot > self._max_slots:
            logger.warning(
                "Device metadata next_slot out of range (%d); ignoring metadata",
                next_slot,
            )
            return False

        raw_free_slots = data.get("free_slots", [])
        if not isinstance(raw_free_slots, list):
            logger.warning("Device metadata free_slots is invalid; ignoring metadata")
            return False
        free_slots: list[int] = []
        seen_slots: set[int] = set()
        for raw_slot in raw_free_slots:
            try:
                slot = int(raw_slot)
            except Exception:
                logger.warning(
                    "Device metadata free_slots contains non-integer; ignoring metadata"
                )
                return False
            if slot < 0 or slot >= self._max_slots:
                logger.warning(
                    "Device metadata free_slots contains out-of-range slot %d; "
                    "ignoring metadata",
                    slot,
                )
                return False
            if slot in seen_slots:
                continue
            seen_slots.add(slot)
            free_slots.append(slot)

        with self._lock:
            if not merge:
                self._next_slot = next_slot
                self._free_slots = free_slots
                self._index.clear()
                self._lru.clear()
                self._lock_refcnt.clear()

            entries = data.get("entries", {})
            if isinstance(entries, dict):
                for encoded_key, entry in entries.items():
                    if not isinstance(entry, dict):
                        continue

                    offset = int(entry.get("offset", 0))
                    size = int(entry.get("size", 0))
                    shape_list = entry.get("shape")
                    fmt_name = entry.get("fmt")
                    cached_positions_list = entry.get("cached_positions")
                    dtype_name = entry.get("dtype")
                    entry_fdp_ruh_id = None
                    if self.use_fdp:
                        raw_entry_fdp_ruh_id = entry.get(
                            "fdp_ruh_id",
                            data_fdp_ruh_id,
                        )
                        if raw_entry_fdp_ruh_id is None:
                            continue
                        entry_fdp_ruh_id = int(raw_entry_fdp_ruh_id)
                    if self.use_fdp and entry_fdp_ruh_id not in self.fdp_data_ruh_ids:
                        continue

                    if not self._is_valid_checkpoint_entry(offset, size):
                        continue

                    shape = (
                        torch.Size(list(shape_list)) if shape_list is not None else None
                    )
                    fmt = (
                        MemoryFormat[fmt_name]
                        if isinstance(fmt_name, str)
                        and fmt_name in MemoryFormat.__members__
                        else MemoryFormat.UNDEFINED
                    )
                    cached_positions = (
                        torch.tensor(cached_positions_list, dtype=torch.long)
                        if cached_positions_list is not None
                        else None
                    )
                    dtype = self._recover_checkpoint_dtype(
                        str(encoded_key),
                        dtype_name,
                    )

                    meta = DiskCacheMetadata(
                        path=f"{self.device_path}@{offset}",
                        size=size,
                        shape=shape,
                        dtype=dtype,
                        cached_positions=cached_positions,
                        fmt=fmt,
                        pin_count=0,
                    )
                    self._index[encoded_key] = _Entry(
                        offset=offset,
                        size=size,
                        meta=meta,
                        fdp_ruh_id=entry_fdp_ruh_id,
                    )

            if not merge:
                used_slots = {
                    self._offset_to_slot(int(entry.offset))
                    for entry in self._index.values()
                }
                self._free_slots = [
                    slot for slot in self._free_slots if slot not in used_slots
                ]

            lru_keys = data.get("lru_keys", [])
            if isinstance(lru_keys, list) and lru_keys:
                for encoded_key in lru_keys:
                    if encoded_key in self._index:
                        self._lru[encoded_key] = None
            else:
                for encoded_key in self._index:
                    self._lru[encoded_key] = None

            if not merge:
                self._meta_dirty_total = 0
                self._meta_persisted = 0

        if self.meta_verify_on_load and not merge:
            self._validate_loaded_entries()
        return True

    def _rebuild_slot_allocator_from_index_locked(self) -> None:
        """Rebuild slot allocator state from the recovered index."""
        used_slots = {
            self._offset_to_slot(int(entry.offset)) for entry in self._index.values()
        }
        if not used_slots:
            self._next_slot = 0
            self._free_slots = []
            return

        self._next_slot = max(used_slots) + 1
        self._free_slots = [
            slot for slot in range(self._next_slot) if slot not in used_slots
        ]

    def _recover_checkpoint_dtype(
        self,
        encoded_key: str,
        dtype_name: Any,
    ) -> torch.dtype | None:
        """Recover checkpoint dtype from entry metadata or legacy key strings.

        Args:
            encoded_key: Encoded raw-block key from the checkpoint entry.
            dtype_name: Raw dtype value stored in the checkpoint entry.

        Returns:
            A torch dtype when recovery succeeds, otherwise None.
        """
        if isinstance(dtype_name, str):
            dtype = STR_DTYPE_TO_TORCH_DTYPE.get(dtype_name)
            if dtype is not None:
                return dtype

            torch_prefix = "torch."
            if dtype_name.startswith(torch_prefix):
                dtype_attr = dtype_name.removeprefix(torch_prefix)
                dtype = STR_DTYPE_TO_TORCH_DTYPE.get(dtype_attr)
                if dtype is not None:
                    return dtype
                torch_dtype = getattr(torch, dtype_attr, None)
                if isinstance(torch_dtype, torch.dtype):
                    return torch_dtype

        if self.key_namespace != "legacy":
            return None

        try:
            return decode_legacy_key(encoded_key).dtype
        except Exception:
            logger.debug(
                "Unable to recover dtype from legacy raw-block key %s",
                encoded_key,
                exc_info=True,
            )
            return None

    def _validate_loaded_entries(self) -> None:
        """Drop recovered entries whose slot headers do not match metadata."""
        to_drop: list[str] = []
        with self._lock:
            items = list(self._index.items())

        for encoded_key, entry in items:
            slot_hdr = self._read_slot_header(int(entry.offset))
            if slot_hdr is None:
                to_drop.append(encoded_key)
                continue
            try:
                expected_identity = slot_identity_from_encoded_key(
                    encoded_key,
                    self.key_namespace,
                )
            except Exception:
                to_drop.append(encoded_key)
                continue
            slot_identity, payload_len = slot_hdr
            if int(slot_identity) != int(expected_identity):
                to_drop.append(encoded_key)
                continue
            if int(payload_len) != int(entry.size):
                to_drop.append(encoded_key)

        if not to_drop:
            return

        with self._lock:
            for encoded_key in to_drop:
                removed_entry = self._index.pop(encoded_key, None)
                self._lru.pop(encoded_key, None)
                self._lock_refcnt.pop(encoded_key, None)
                if removed_entry is not None:
                    self._append_free_slot_locked(
                        self._offset_to_slot(int(removed_entry.offset))
                    )
            self._meta_dirty_total += 1

        logger.warning(
            "RawBlockCore dropped %d stale metadata entries after "
            "slot-header validation",
            len(to_drop),
        )

    def _load_checkpoint_from_device(self) -> None:
        """Load the newest valid checkpoint from the raw device if present."""
        if self.use_fdp:
            self._load_fdp_checkpoints_from_device()
            return

        header, payload = self._select_latest_checkpoint()
        if header is None:
            logger.info("RawBlockCore: no valid on-device metadata checkpoint found")
            return
        if payload is None:
            logger.warning("RawBlockCore: checkpoint header had no payload")
            return
        try:
            data = json.loads(payload.decode("utf-8"))
        except Exception:
            logger.warning("RawBlockCore: failed to decode metadata payload")
            return
        if not self.apply_loaded_state(data):
            logger.warning("RawBlockCore: metadata payload rejected by checks")
            return
        self._meta_seq = int(header["seq"])
        logger.info(
            "RawBlockCore loaded checkpoint (entries=%d next_slot=%d seq=%d device=%s)",
            len(self._index),
            self._next_slot,
            self._meta_seq,
            self.device_path,
        )

    def _load_fdp_checkpoints_from_device(self) -> None:
        """Load and merge per-RUH FDP metadata checkpoints from the raw device."""
        loaded_partitions = 0
        max_seq = 0
        with self._lock:
            self._index.clear()
            self._lru.clear()
            self._lock_refcnt.clear()
            self._free_slots.clear()
            self._next_slot = 0

        for ruh_id in self.fdp_metadata_ruh_ids:
            header, payload = self._select_latest_checkpoint(ruh_id)
            if header is None or payload is None:
                continue
            try:
                data = json.loads(payload.decode("utf-8"))
            except Exception:
                logger.warning(
                    "RawBlockCore: failed to decode FDP metadata payload for RUH %d",
                    ruh_id,
                )
                continue
            if not self._apply_loaded_state(
                data,
                merge=True,
                expected_fdp_ruh_id=ruh_id,
            ):
                logger.warning(
                    "RawBlockCore: FDP metadata payload rejected for RUH %d",
                    ruh_id,
                )
                continue
            loaded_partitions += 1
            max_seq = max(max_seq, int(header["seq"]))

        if loaded_partitions == 0:
            logger.info("RawBlockCore: no valid FDP metadata checkpoint found")
            return

        with self._lock:
            self._rebuild_slot_allocator_from_index_locked()
            self._meta_dirty_total = 0
            self._meta_persisted = 0
            self._meta_seq = max_seq

        if self.meta_verify_on_load:
            self._validate_loaded_entries()

        logger.info(
            "RawBlockCore loaded FDP checkpoints "
            "(partitions=%d entries=%d next_slot=%d seq=%d device=%s)",
            loaded_partitions,
            len(self._index),
            self._next_slot,
            self._meta_seq,
            self.device_path,
        )
