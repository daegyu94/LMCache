# SPDX-License-Identifier: Apache-2.0
"""
Filesystem native L2 adapter config and factory.

Backed by the native C++ filesystem connector wrapped with
``NativeConnectorL2Adapter``.
"""

# Future
from __future__ import annotations

# Standard
from typing import TYPE_CHECKING, Any, Optional
import math
import threading
import time

if TYPE_CHECKING:
    from lmcache.v1.distributed.internal_api import (
        L1MemoryDesc,
    )

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.l2_adapters.base import (
    L2AdapterInterface,
)
from lmcache.v1.distributed.l2_adapters.config import (
    L2AdapterConfigBase,
    register_l2_adapter_type,
)
from lmcache.v1.distributed.l2_adapters.factory import (
    register_l2_adapter_factory,
)
from lmcache.v1.distributed.l2_adapters.native_connector_l2_adapter import (
    NativeConnectorL2Adapter,
)

logger = init_logger(__name__)


class FSNativeL2AdapterConfig(L2AdapterConfigBase):
    """
    Config for an L2 adapter backed by the native C++
    filesystem connector.

    Fields:
    - base_path: directory for storing KV cache files.
    - num_workers: C++ worker threads for I/O (default 4).
    - relative_tmp_dir: relative sub-dir for temp files.
    - use_odirect: bypass page cache via O_DIRECT.
    - read_ahead_size: trigger filesystem readahead by
      reading this many bytes first (optional).
    - io_log_interval_sec: emit interval I/O throughput logs at
      this interval; zero disables progress logs.
    """

    def __init__(
        self,
        base_path: str,
        num_workers: int = 4,
        relative_tmp_dir: str = "",
        use_odirect: bool = False,
        read_ahead_size: Optional[int] = None,
        max_capacity_gb: float = 0,
        io_log_interval_sec: float = 0,
    ):
        self.base_path = base_path
        self.num_workers = num_workers
        self.relative_tmp_dir = relative_tmp_dir
        self.use_odirect = use_odirect
        self.read_ahead_size = read_ahead_size
        self.max_capacity_gb = max_capacity_gb
        self.io_log_interval_sec = io_log_interval_sec

    @classmethod
    def from_dict(cls, d: dict) -> "FSNativeL2AdapterConfig":
        base_path = d.get("base_path")
        if not isinstance(base_path, str) or not base_path:
            raise ValueError("base_path must be a non-empty string")

        num_workers = d.get("num_workers", 4)
        if not isinstance(num_workers, int) or num_workers <= 0:
            raise ValueError("num_workers must be a positive integer")

        relative_tmp_dir = d.get("relative_tmp_dir", "")
        if not isinstance(relative_tmp_dir, str):
            raise ValueError("relative_tmp_dir must be a string")

        use_odirect = d.get("use_odirect", False)
        if not isinstance(use_odirect, bool):
            raise ValueError("use_odirect must be a boolean")

        read_ahead_size = d.get("read_ahead_size", None)
        if read_ahead_size is not None:
            if not isinstance(read_ahead_size, int) or read_ahead_size <= 0:
                raise ValueError("read_ahead_size must be a positive integer")

        max_capacity_gb = d.get("max_capacity_gb", 0)
        if not isinstance(max_capacity_gb, (int, float)) or max_capacity_gb < 0:
            raise ValueError("max_capacity_gb must be a non-negative number")

        io_log_interval_sec = d.get("io_log_interval_sec", 0)
        if (
            isinstance(io_log_interval_sec, bool)
            or not isinstance(io_log_interval_sec, (int, float))
            or not math.isfinite(io_log_interval_sec)
            or io_log_interval_sec < 0
        ):
            raise ValueError("io_log_interval_sec must be a finite non-negative number")

        return cls(
            base_path=base_path,
            num_workers=num_workers,
            relative_tmp_dir=str(relative_tmp_dir),
            use_odirect=use_odirect,
            read_ahead_size=read_ahead_size,
            max_capacity_gb=float(max_capacity_gb),
            io_log_interval_sec=float(io_log_interval_sec),
        )

    @classmethod
    def help(cls) -> str:
        return (
            "FS native L2 adapter config fields:\n"
            "- base_path (str): directory for KV "
            "cache files (required)\n"
            "- num_workers (int): C++ worker threads "
            "for I/O (default 4, >0)\n"
            "- relative_tmp_dir (str): relative "
            "sub-dir for temp files (default empty)\n"
            "- use_odirect (bool): bypass page cache "
            "via O_DIRECT (default false)\n"
            "- read_ahead_size (int): trigger fs "
            "readahead by reading this many bytes "
            "first (optional)\n"
            "- max_capacity_gb (float): max L2 capacity "
            "in GB for usage tracking / eviction "
            "(default 0 = disabled)\n"
            "- io_log_interval_sec (float): interval I/O "
            "throughput-log interval in seconds (default 0 = disabled)"
        )


class FSNativeL2Adapter(NativeConnectorL2Adapter):
    """Native FS adapter with optional interval I/O logs."""

    def __init__(
        self,
        native_client: Any,
        max_capacity_gb: float = 0,
        type_name: str = "",
        extra_status: dict[str, Any] | None = None,
        io_log_interval_sec: float = 0,
    ) -> None:
        self._io_log_interval_sec = io_log_interval_sec
        self._io_log_started_at = time.monotonic()
        self._io_log_last_at = self._io_log_started_at
        self._io_log_last_stats = {
            "read_ops": 0,
            "read_bytes": 0,
            "write_ops": 0,
            "write_bytes": 0,
        }
        self._io_log_stop = threading.Event()
        self._io_log_thread: threading.Thread | None = None
        self._io_log_close_lock = threading.Lock()
        self._io_log_closed = False

        super().__init__(
            native_client,
            max_capacity_gb=max_capacity_gb,
            type_name=type_name,
            extra_status=extra_status,
        )

        if self._io_log_interval_sec > 0:
            self._io_log_thread = threading.Thread(
                target=self._io_progress_loop,
                daemon=True,
                name="fs-native-io-progress",
            )
            self._io_log_thread.start()

    def close(self) -> None:
        """Stop logging, emit the final interval, and close the connector."""
        with self._io_log_close_lock:
            if self._io_log_closed:
                return
            self._io_log_closed = True

        self._io_log_stop.set()
        if self._io_log_thread is not None:
            self._io_log_thread.join(timeout=5.0)

        self._log_io_interval()
        super().close()

    def _io_progress_loop(self) -> None:
        while not self._io_log_stop.wait(self._io_log_interval_sec):
            self._log_io_interval()

    def _log_io_interval(self) -> None:
        get_io_stats = getattr(self._client, "get_io_stats", None)
        if not callable(get_io_stats):
            return

        try:
            raw_stats = get_io_stats()
            stats = {key: int(value) for key, value in raw_stats.items()}
        except Exception:
            logger.exception("Failed to collect FS native I/O statistics")
            return

        now = time.monotonic()
        elapsed = max(now - self._io_log_started_at, 1e-9)
        interval = max(now - self._io_log_last_at, 1e-9)
        deltas = {
            key: max(0, stats.get(key, 0) - self._io_log_last_stats[key])
            for key in self._io_log_last_stats
        }
        self._io_log_last_at = now
        self._io_log_last_stats = {
            key: stats.get(key, 0) for key in self._io_log_last_stats
        }

        read_ops = deltas["read_ops"]
        read_bytes = deltas["read_bytes"]
        write_ops = deltas["write_ops"]
        write_bytes = deltas["write_bytes"]
        total_ops = read_ops + write_ops
        total_bytes = read_bytes + write_bytes
        gib = 1024**3

        logger.info(
            "FS native I/O interval: elapsed=%.3fs interval=%.3fs "
            "total_ops=%d total_bytes=%d total_GiB/s=%.3f "
            "read_ops=%d read_bytes=%d read_GiB/s=%.3f "
            "write_ops=%d write_bytes=%d write_GiB/s=%.3f",
            elapsed,
            interval,
            total_ops,
            total_bytes,
            total_bytes / interval / gib,
            read_ops,
            read_bytes,
            read_bytes / interval / gib,
            write_ops,
            write_bytes,
            write_bytes / interval / gib,
        )


def _create_fs_native_l2_adapter(
    config: L2AdapterConfigBase,
    l1_memory_desc: "Optional[L1MemoryDesc]" = None,
) -> L2AdapterInterface:
    """Create a NativeConnectorL2Adapter backed by the
    C++ filesystem connector."""
    try:
        # First Party
        from lmcache.lmcache_fs import (
            LMCacheFSClient,
        )
    except ImportError as e:
        raise RuntimeError(
            "FS native L2 adapter requires the C++ FS "
            "extension. Build with: pip install -e ."
        ) from e

    assert isinstance(config, FSNativeL2AdapterConfig)
    native_client = LMCacheFSClient(
        config.base_path,
        config.num_workers,
        config.relative_tmp_dir,
        config.use_odirect,
        config.read_ahead_size or 0,
    )
    logger.info(
        "Created FS native L2 adapter: %s (workers=%d, odirect=%s, "
        "read_ahead=%s, io_log_interval_sec=%s)",
        config.base_path,
        config.num_workers,
        config.use_odirect,
        config.read_ahead_size,
        config.io_log_interval_sec,
    )
    return FSNativeL2Adapter(
        native_client,
        max_capacity_gb=config.max_capacity_gb,
        type_name="FSNativeL2Adapter",
        extra_status={
            "base_path": config.base_path,
            "use_odirect": config.use_odirect,
            "num_workers": config.num_workers,
            "read_ahead_size": config.read_ahead_size,
            "io_log_interval_sec": config.io_log_interval_sec,
        },
        io_log_interval_sec=config.io_log_interval_sec,
    )


register_l2_adapter_type("fs_native", FSNativeL2AdapterConfig)
register_l2_adapter_factory("fs_native", _create_fs_native_l2_adapter)
