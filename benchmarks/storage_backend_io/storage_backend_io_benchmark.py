# SPDX-License-Identifier: Apache-2.0
"""Benchmark LocalDiskBackend vs RustRawBlockBackend for put/get I/O."""

# Future
from __future__ import annotations

# Standard
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional
import argparse
import asyncio
import collections
import json
import os
import stat
import tempfile
import threading
import time

# Third Party
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import (
    AdHocMemoryAllocator,
    MemoryFormat,
    MemoryObjMetadata,
    TensorMemoryObj,
)
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
from lmcache.v1.storage_backend.local_disk_backend import LocalDiskBackend
from lmcache.v1.storage_backend.plugins.rust_raw_block_backend import (
    RustRawBlockBackend,
)

DEFAULT_PAYLOAD_SHAPE = torch.Size([2, 16, 8, 128])
DEFAULT_DTYPE = torch.bfloat16


@dataclass
class _ThreadLatencyStats:
    ops: int = 0
    total_ns: int = 0


@dataclass
class _LatencyBreakdownCollector:
    e2e_total_ns: dict[str, int] = field(
        default_factory=lambda: collections.defaultdict(int)
    )
    e2e_ops: dict[str, int] = field(
        default_factory=lambda: collections.defaultdict(int)
    )
    e2e_by_thread: dict[str, dict[int, _ThreadLatencyStats]] = field(
        default_factory=lambda: collections.defaultdict(
            lambda: collections.defaultdict(_ThreadLatencyStats)
        )
    )
    io_total_ns: dict[str, int] = field(
        default_factory=lambda: collections.defaultdict(int)
    )
    io_ops: dict[str, int] = field(default_factory=lambda: collections.defaultdict(int))
    io_by_thread: dict[str, dict[int, _ThreadLatencyStats]] = field(
        default_factory=lambda: collections.defaultdict(
            lambda: collections.defaultdict(_ThreadLatencyStats)
        )
    )
    stage_total_ns: dict[str, dict[str, int]] = field(
        default_factory=lambda: collections.defaultdict(
            lambda: collections.defaultdict(int)
        )
    )
    stage_ops: dict[str, dict[str, int]] = field(
        default_factory=lambda: collections.defaultdict(
            lambda: collections.defaultdict(int)
        )
    )
    stage_by_thread: dict[str, dict[str, dict[int, _ThreadLatencyStats]]] = field(
        default_factory=lambda: collections.defaultdict(
            lambda: collections.defaultdict(
                lambda: collections.defaultdict(_ThreadLatencyStats)
            )
        )
    )
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def record_e2e(self, op: str, latency_ns: int, thread_id: int) -> None:
        with self._lock:
            self.e2e_total_ns[op] += latency_ns
            self.e2e_ops[op] += 1
            thread_stats = self.e2e_by_thread[op][thread_id]
            thread_stats.ops += 1
            thread_stats.total_ns += latency_ns

    def record_io(self, op: str, latency_ns: int, thread_id: int) -> None:
        with self._lock:
            self.io_total_ns[op] += latency_ns
            self.io_ops[op] += 1
            thread_stats = self.io_by_thread[op][thread_id]
            thread_stats.ops += 1
            thread_stats.total_ns += latency_ns

    def record_stage(
        self, op: str, stage: str, latency_ns: int, thread_id: int
    ) -> None:
        with self._lock:
            self.stage_total_ns[op][stage] += latency_ns
            self.stage_ops[op][stage] += 1
            thread_stats = self.stage_by_thread[op][stage][thread_id]
            thread_stats.ops += 1
            thread_stats.total_ns += latency_ns

    def to_result_dict(self) -> dict[str, dict]:
        result: dict[str, dict] = {}
        for op in sorted(set(self.e2e_ops.keys()) | set(self.io_ops.keys())):
            e2e_ops = int(self.e2e_ops.get(op, 0))
            io_ops = int(self.io_ops.get(op, 0))
            e2e_total_ns = int(self.e2e_total_ns.get(op, 0))
            io_total_ns = int(self.io_total_ns.get(op, 0))
            lmcache_total_ns = max(0, e2e_total_ns - io_total_ns)
            ref_ops = e2e_ops if e2e_ops > 0 else io_ops

            result[op] = {
                "samples_e2e": e2e_ops,
                "samples_io_syscall": io_ops,
                "e2e_total_us": e2e_total_ns / 1e3,
                "e2e_avg_us": (e2e_total_ns / e2e_ops / 1e3) if e2e_ops > 0 else 0.0,
                "io_syscall_total_us": io_total_ns / 1e3,
                "io_syscall_avg_us": (io_total_ns / io_ops / 1e3)
                if io_ops > 0
                else 0.0,
                "lmcache_total_us": lmcache_total_ns / 1e3,
                "lmcache_avg_us": (lmcache_total_ns / ref_ops / 1e3)
                if ref_ops > 0
                else 0.0,
                "e2e_by_thread": self._serialize_thread_stats(self.e2e_by_thread[op]),
                "io_syscall_by_thread": self._serialize_thread_stats(
                    self.io_by_thread[op]
                ),
                "stages": self._serialize_stage_stats(op),
            }
        return result

    def _serialize_stage_stats(self, op: str) -> dict[str, dict]:
        serialized: dict[str, dict] = {}
        for stage, total_ns in sorted(self.stage_total_ns[op].items()):
            stage_ops = int(self.stage_ops[op][stage])
            serialized[stage] = {
                "samples": stage_ops,
                "total_us": total_ns / 1e3,
                "avg_us": (total_ns / stage_ops / 1e3) if stage_ops > 0 else 0.0,
                "by_thread": self._serialize_thread_stats(
                    self.stage_by_thread[op][stage]
                ),
            }
        return serialized

    @staticmethod
    def _serialize_thread_stats(
        thread_stats: dict[int, _ThreadLatencyStats],
    ) -> dict[str, dict[str, float | int]]:
        serialized: dict[str, dict[str, float | int]] = {}
        for tid, stats in sorted(thread_stats.items()):
            serialized[str(tid)] = {
                "ops": int(stats.ops),
                "total_us": stats.total_ns / 1e3,
                "avg_us": (stats.total_ns / stats.ops / 1e3)
                if stats.ops > 0
                else 0.0,
            }
        return serialized


def _start_loop() -> tuple[asyncio.AbstractEventLoop, threading.Thread]:
    loop = asyncio.new_event_loop()
    t = threading.Thread(target=loop.run_forever, name="bench-loop", daemon=True)
    t.start()
    return loop, t


def _stop_loop(loop: asyncio.AbstractEventLoop, t: threading.Thread) -> None:
    loop.call_soon_threadsafe(loop.stop)
    t.join(timeout=5)
    loop.close()


def _start_loops(
    num_loops: int,
) -> list[tuple[asyncio.AbstractEventLoop, threading.Thread]]:
    return [_start_loop() for _ in range(max(1, num_loops))]


def _stop_loops(
    loops: list[tuple[asyncio.AbstractEventLoop, threading.Thread]],
) -> None:
    for loop, t in loops:
        _stop_loop(loop, t)


def _build_metadata() -> LMCacheMetadata:
    return LMCacheMetadata(
        model_name="benchmark_model",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=DEFAULT_DTYPE,
        kv_shape=(4, 2, 256, 8, 128),
    )


def _resolve_payload_shape(payload_size_kb: float) -> torch.Size:
    payload_bytes = int(payload_size_kb * 1024)
    if payload_bytes <= 0:
        raise ValueError("payload_size_kb must be > 0")

    itemsize = DEFAULT_DTYPE.itemsize
    if payload_bytes % itemsize != 0:
        raise ValueError(
            "payload_size_kb must produce a whole number of dtype elements"
        )
    numel = payload_bytes // itemsize

    # Keep the historical tensor layout [2, 16, 8, X].
    base = 2 * 16 * 8
    if numel % base != 0:
        raise ValueError(
            "payload_size_kb must be a multiple of 0.5KB "
            "(for bfloat16 with layout [2,16,8,X])"
        )
    return torch.Size([2, 16, 8, numel // base])


def _payload_mb_per_op(payload_size_kb: float) -> float:
    return payload_size_kb / 1024.0


def _make_memory_objs(
    num_ops: int,
    use_aligned: bool,
    alignment: int,
    keepalive: list[torch.Tensor],
    payload_shape: torch.Size,
) -> list:
    allocator = AdHocMemoryAllocator(device="cpu")
    objs = []
    for _ in range(num_ops):
        if use_aligned:
            num_bytes = payload_shape.numel() * DEFAULT_DTYPE.itemsize
            base = torch.empty(
                torch.Size([num_bytes + alignment]),
                dtype=torch.uint8,
                device="cpu",
            )
            offset = (-base.data_ptr()) % alignment
            aligned = base[offset : offset + num_bytes]
            keepalive.append(base)
            obj = TensorMemoryObj(
                raw_data=aligned,
                metadata=MemoryObjMetadata(
                    shape=payload_shape,
                    dtype=DEFAULT_DTYPE,
                    address=0,
                    phy_size=0,
                    ref_count=1,
                    pin_count=0,
                    fmt=MemoryFormat.KV_T2D,
                    shapes=[payload_shape],
                    dtypes=[DEFAULT_DTYPE],
                ),
                parent_allocator=allocator,
            )
        else:
            obj = allocator.allocate(
                [payload_shape],
                [DEFAULT_DTYPE],
                fmt=MemoryFormat.KV_T2D,
            )
            assert obj is not None
        assert obj.tensor is not None
        obj.tensor.fill_(7)
        objs.append(obj)
    return objs


def _release_memory_objs(objs: list) -> None:
    for obj in objs:
        try:
            obj.ref_count_down()
        except Exception:
            # Best effort for benchmark cleanup.
            pass


def _make_keys(num_ops: int) -> list[CacheEngineKey]:
    return [
        CacheEngineKey("benchmark_model", 1, 0, i, DEFAULT_DTYPE)
        for i in range(num_ops)
    ]


def _bench_get_phase(
    backend: LocalDiskBackend | RustRawBlockBackend,
    keys: list[CacheEngineKey],
    concurrency: int,
    latency: Optional[_LatencyBreakdownCollector] = None,
) -> float:
    def read_slice(start: int, end: int) -> None:
        for key in keys[start:end]:
            e2e_start_ns = time.perf_counter_ns() if latency is not None else 0
            obj = backend.get_blocking(key)
            if obj is None:
                raise RuntimeError(f"get miss for key={key}")
            if latency is not None:
                latency.record_e2e(
                    "read",
                    time.perf_counter_ns() - e2e_start_ns,
                    threading.get_native_id(),
                )
            obj.ref_count_down()

    slice_size = max(1, len(keys) // concurrency)
    slices = []
    for i in range(0, len(keys), slice_size):
        slices.append((i, min(i + slice_size, len(keys))))

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = [ex.submit(read_slice, s[0], s[1]) for s in slices]
        timeout_sec = max(300.0, float(len(keys)) / 100.0)
        for fut in futures:
            fut.result(timeout=timeout_sec)
    return time.perf_counter() - start


def _bench_local_disk(
    num_ops: int,
    concurrency: int,
    local_disk_dir: str,
    max_disk_gb: float,
    use_odirect: bool,
    alignment: int,
    operation: str,
    payload_shape: torch.Size,
    payload_size_kb: float,
    measure_latency_breakdown: bool,
    submit_mode: str,
) -> dict:
    loop, t = _start_loop()
    metadata = _build_metadata()
    config = LMCacheEngineConfig.from_defaults(
        chunk_size=256,
        local_cpu=True,
        max_local_cpu_size=0.1,
        lmcache_instance_id="bench_local_disk",
    )
    config.local_disk = local_disk_dir
    config.max_local_disk_size = max_disk_gb
    config.extra_config = {"use_odirect": use_odirect}

    local_cpu = LocalCPUBackend(
        config=config,
        metadata=metadata,
        dst_device="cpu",
        memory_allocator=AdHocMemoryAllocator(device="cpu"),
    )
    backend = LocalDiskBackend(
        config=config,
        loop=loop,
        local_cpu_backend=local_cpu,
        dst_device="cpu",
        metadata=metadata,
    )
    latency = _LatencyBreakdownCollector() if measure_latency_breakdown else None
    if latency is not None:
        backend.set_io_latency_callback(latency.record_io)
        backend.set_put_stage_latency_callback(latency.record_stage)

    keys = _make_keys(num_ops)
    keepalive: list[torch.Tensor] = []
    objs = _make_memory_objs(
        num_ops, use_odirect, alignment, keepalive, payload_shape
    )

    completed = 0
    lock = threading.Lock()
    done = threading.Event()
    submit_start_ns: dict[CacheEngineKey, int] = {}

    def on_complete(_key: CacheEngineKey) -> None:
        nonlocal completed
        end_ns = time.perf_counter_ns() if latency is not None else 0
        with lock:
            started = submit_start_ns.pop(_key, None) if latency is not None else None
            if started is not None and latency is not None:
                latency.record_e2e(
                    "write", end_ns - started, threading.get_native_id()
                )
            completed += 1
            if completed >= num_ops:
                done.set()

    def submit_slice(start: int, end: int) -> None:
        if submit_mode == "single_key":
            for key, obj in zip(keys[start:end], objs[start:end], strict=False):
                if latency is not None:
                    with lock:
                        submit_start_ns[key] = time.perf_counter_ns()
                backend.batched_submit_put_task(
                    [key],
                    [obj],
                    on_complete_callback=on_complete,
                )
            return

        if latency is not None:
            now_ns = time.perf_counter_ns()
            with lock:
                for key in keys[start:end]:
                    submit_start_ns[key] = now_ns
        backend.batched_submit_put_task(
            keys[start:end],
            objs[start:end],
            on_complete_callback=on_complete,
        )

    slice_size = max(1, num_ops // concurrency)
    slices = []
    for i in range(0, num_ops, slice_size):
        slices.append((i, min(i + slice_size, num_ops)))

    def run_put_phase(measure: bool) -> float:
        nonlocal completed
        completed = 0
        done.clear()

        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            for s in slices:
                ex.submit(submit_slice, s[0], s[1])

        # Keep a floor for normal runs but scale for large-op runs.
        # This avoids premature timeout for long single-shot benchmarks.
        timeout_sec = max(300.0, float(num_ops) / 100.0)
        while not done.wait(timeout=1.0):
            if completed >= num_ops:
                break
            if (time.perf_counter() - start) >= timeout_sec:
                raise TimeoutError(
                    "LocalDisk benchmark timed out: "
                    f"completed={completed}, expected={num_ops}"
                )
        elapsed = time.perf_counter() - start
        return elapsed if measure else 0.0

    put_elapsed: Optional[float] = None
    get_phase_elapsed: Optional[float] = None
    get_with_buffered_io = False

    if operation in ("put", "both"):
        put_elapsed = run_put_phase(measure=True)

    _release_memory_objs(objs)

    if operation in ("both"):
        # LocalDisk get path may hit EINVAL with O_DIRECT if read buffer
        # alignment is not compatible with O_DIRECT requirements.
        get_with_buffered_io = backend.use_odirect
        if get_with_buffered_io:
            backend.use_odirect = False
        get_phase_elapsed = _bench_get_phase(backend, keys, concurrency, latency)

    backend.disk_worker.close()
    _stop_loop(loop, t)

    result = {
        "backend": "local_disk",
        "num_ops": num_ops,
        "concurrency": concurrency,
        "use_odirect": use_odirect,
        "local_disk_dir": local_disk_dir,
        "operation": operation,
        "payload_size_kb": payload_size_kb,
        "submit_mode": submit_mode,
    }
    if put_elapsed is not None:
        result["put_elapsed_sec"] = put_elapsed
        result["put_ops_per_sec"] = num_ops / put_elapsed if put_elapsed > 0 else 0.0
        result["put_mb_per_sec"] = (
            num_ops * _payload_mb_per_op(payload_size_kb) / put_elapsed
            if put_elapsed > 0
            else 0.0
        )
    if get_phase_elapsed is not None:
        result["get_elapsed_sec"] = get_phase_elapsed
        result["get_ops_per_sec"] = (
            num_ops / get_phase_elapsed if get_phase_elapsed > 0 else 0.0
        )
        result["get_mb_per_sec"] = (
            num_ops * _payload_mb_per_op(payload_size_kb) / get_phase_elapsed
            if get_phase_elapsed > 0
            else 0.0
        )
        result["get_buffered_io_fallback"] = get_with_buffered_io
    if latency is not None:
        result["latency_breakdown"] = latency.to_result_dict()
    return result


def _bench_rust_raw_block(
    num_ops: int,
    concurrency: int,
    raw_device: str,
    raw_device_size_gb: float,
    use_odirect: bool,
    alignment: int,
    cleanup_raw_device: bool,
    use_callback: bool,
    raw_slot_bytes: int,
    operation: str,
    payload_shape: torch.Size,
    payload_size_kb: float,
    measure_latency_breakdown: bool,
    submit_mode: str,
    raw_submit_mode: str,
    raw_submit_workers: int,
    raw_submit_loops: int,
    raw_submit_pools: int,
) -> dict:
    loop_pairs: list[tuple[asyncio.AbstractEventLoop, threading.Thread]] = []
    loop: Optional[asyncio.AbstractEventLoop] = None
    submit_loops: Optional[list[asyncio.AbstractEventLoop]] = None
    if raw_submit_mode == "async_loop":
        loop_pairs = _start_loops(raw_submit_loops)
        submit_loops = [lp[0] for lp in loop_pairs]
        loop = submit_loops[0]
    metadata = _build_metadata()
    config = LMCacheEngineConfig.from_defaults(
        chunk_size=256,
        local_cpu=True,
        max_local_cpu_size=0.1,
        lmcache_instance_id="bench_rust_raw_block",
    )
    # Create a backing file if raw_device is not provided. For a real block
    # device path (e.g. /dev/nvme*), do not truncate.
    temp_dir: Optional[str] = None
    is_block_device = False
    if not raw_device:
        temp_dir = tempfile.mkdtemp(prefix="raw_block_bench_")
        raw_device = os.path.join(temp_dir, "raw_block.bin")
    else:
        try:
            st_mode = os.stat(raw_device).st_mode
            is_block_device = stat.S_ISBLK(st_mode)
        except FileNotFoundError:
            is_block_device = False

    if raw_device and not is_block_device:
        with open(raw_device, "wb") as f:
            f.truncate(int(raw_device_size_gb * 1024**3))

    manifest_path = os.path.join(
        tempfile.gettempdir(),
        f"lmcache_rust_raw_block_bench_{os.getpid()}_{time.time_ns()}.manifest.json",
    )
    config.extra_config = {
        "rust_raw_block.device_path": raw_device,
        "rust_raw_block.block_align": alignment,
        "rust_raw_block.header_bytes": alignment,
        "rust_raw_block.use_odirect": use_odirect,
        "rust_raw_block.manifest_path": manifest_path,
        "rust_raw_block.manifest_write_interval": 0,
        "rust_raw_block.submit_mode": raw_submit_mode,
        "rust_raw_block.submit_workers": raw_submit_workers,
        "rust_raw_block.submit_pools": raw_submit_pools,
    }
    if raw_slot_bytes > 0:
        config.extra_config["rust_raw_block.slot_bytes"] = raw_slot_bytes

    local_cpu = LocalCPUBackend(
        config=config,
        metadata=metadata,
        dst_device="cpu",
        memory_allocator=AdHocMemoryAllocator(device="cpu"),
    )
    backend = RustRawBlockBackend(
        config=config,
        metadata=metadata,
        local_cpu_backend=local_cpu,
        loop=loop,
        submit_loops=submit_loops,
        dst_device="cpu",
    )
    latency = _LatencyBreakdownCollector() if measure_latency_breakdown else None
    if latency is not None:
        backend.set_io_latency_callback(latency.record_io)
        backend.set_put_stage_latency_callback(latency.record_stage)

    keys = _make_keys(num_ops)
    keepalive: list[torch.Tensor] = []
    objs = _make_memory_objs(
        num_ops, use_odirect, alignment, keepalive, payload_shape
    )

    completed = 0
    lock = threading.Lock()
    done = threading.Event()
    submit_start_ns: dict[CacheEngineKey, int] = {}

    futures: list[tuple[Future, CacheEngineKey]] = []
    fut_lock = threading.Lock()

    def on_complete(_key: CacheEngineKey) -> None:
        nonlocal completed
        end_ns = time.perf_counter_ns() if latency is not None else 0
        with lock:
            started = submit_start_ns.pop(_key, None) if latency is not None else None
            if started is not None and latency is not None:
                latency.record_e2e(
                    "write", end_ns - started, threading.get_native_id()
                )
            completed += 1
            if completed >= num_ops:
                done.set()

    def submit_slice(start: int, end: int) -> None:
        if submit_mode == "single_key":
            for key, obj in zip(keys[start:end], objs[start:end], strict=False):
                if latency is not None:
                    with lock:
                        submit_start_ns[key] = time.perf_counter_ns()
                if use_callback:
                    backend.batched_submit_put_task(
                        [key],
                        [obj],
                        on_complete_callback=on_complete,
                    )
                else:
                    futs = backend.batched_submit_put_task([key], [obj])
                    if futs:
                        with fut_lock:
                            for fut in futs:
                                futures.append((fut, key))
            return

        if latency is not None:
            now_ns = time.perf_counter_ns()
            with lock:
                for key in keys[start:end]:
                    submit_start_ns[key] = now_ns
        if use_callback:
            backend.batched_submit_put_task(
                keys[start:end],
                objs[start:end],
                on_complete_callback=on_complete,
            )
        else:
            futs = backend.batched_submit_put_task(keys[start:end], objs[start:end])
            if futs:
                with fut_lock:
                    for key, fut in zip(keys[start:end], futs, strict=False):
                        futures.append((fut, key))

    slice_size = max(1, num_ops // concurrency)
    slices = []
    for i in range(0, num_ops, slice_size):
        slices.append((i, min(i + slice_size, num_ops)))

    def run_put_phase(measure: bool) -> float:
        nonlocal completed
        completed = 0
        done.clear()
        with fut_lock:
            futures.clear()

        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            for s in slices:
                ex.submit(submit_slice, s[0], s[1])

        if use_callback:
            timeout_sec = max(300.0, float(num_ops) / 100.0)
            while not done.wait(timeout=1.0):
                if completed >= num_ops:
                    break
                if (time.perf_counter() - start) >= timeout_sec:
                    raise TimeoutError(
                        "RustRaw benchmark timed out: "
                        f"completed={completed}, expected={num_ops}"
                    )
        else:
            for fut, key in futures:
                fut.result(timeout=120)
                end_ns = time.perf_counter_ns() if latency is not None else 0
                with lock:
                    started = (
                        submit_start_ns.pop(key, None) if latency is not None else None
                    )
                if started is not None and latency is not None:
                    latency.record_e2e(
                        "write", end_ns - started, threading.get_native_id()
                    )
        elapsed = time.perf_counter() - start
        return elapsed if measure else 0.0

    put_elapsed: Optional[float] = None
    get_phase_elapsed: Optional[float] = None

    if operation in ("put", "both"):
        put_elapsed = run_put_phase(measure=True)

    _release_memory_objs(objs)

    if operation in ("both"):
        get_phase_elapsed = _bench_get_phase(backend, keys, concurrency, latency)

    backend.close()
    if loop_pairs:
        _stop_loops(loop_pairs)

    # Best-effort cleanup for temp file or requested cleanup.
    if cleanup_raw_device or temp_dir:
        try:
            os.remove(raw_device)
        except Exception:
            pass
        if temp_dir:
            try:
                os.rmdir(temp_dir)
            except Exception:
                pass
    try:
        os.remove(manifest_path)
    except Exception:
        pass

    result = {
        "backend": "rust_raw_block",
        "num_ops": num_ops,
        "concurrency": concurrency,
        "use_odirect": use_odirect,
        "raw_device": raw_device,
        "raw_slot_bytes": raw_slot_bytes,
        "operation": operation,
        "payload_size_kb": payload_size_kb,
        "submit_mode": submit_mode,
        "raw_submit_mode": raw_submit_mode,
        "raw_submit_workers": raw_submit_workers,
        "raw_submit_loops": raw_submit_loops,
        "raw_submit_pools": raw_submit_pools,
    }
    if put_elapsed is not None:
        result["put_elapsed_sec"] = put_elapsed
        result["put_ops_per_sec"] = num_ops / put_elapsed if put_elapsed > 0 else 0.0
        result["put_mb_per_sec"] = (
            num_ops * _payload_mb_per_op(payload_size_kb) / put_elapsed
            if put_elapsed > 0
            else 0.0
        )
    if get_phase_elapsed is not None:
        result["get_elapsed_sec"] = get_phase_elapsed
        result["get_ops_per_sec"] = (
            num_ops / get_phase_elapsed if get_phase_elapsed > 0 else 0.0
        )
        result["get_mb_per_sec"] = (
            num_ops * _payload_mb_per_op(payload_size_kb) / get_phase_elapsed
            if get_phase_elapsed > 0
            else 0.0
        )
    if latency is not None:
        result["latency_breakdown"] = latency.to_result_dict()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark LocalDiskBackend vs RustRawBlockBackend "
            "for put/get workloads."
        )
    )
    parser.add_argument("--num-ops", type=int, default=256, help="Total KV ops")
    parser.add_argument(
        "--concurrency", type=int, default=16, help="Number of submit threads"
    )
    parser.add_argument(
        "--backend",
        choices=["local_disk", "rust_raw_block", "both"],
        default="both",
    )
    parser.add_argument(
        "--local-disk-dir",
        type=str,
        default="/tmp/lmcache_local_disk_bench",
    )
    parser.add_argument("--max-local-disk-gb", type=float, default=2.0)
    parser.add_argument(
        "--local-disk-odirect",
        action="store_true",
        help="Enable O_DIRECT for local disk backend",
    )
    parser.add_argument(
        "--raw-device",
        type=str,
        default="",
        help="Raw block device path (if empty, uses a temp file)",
    )
    parser.add_argument("--raw-device-size-gb", type=float, default=1.0)
    parser.add_argument(
        "--raw-odirect",
        action="store_true",
        help="Enable O_DIRECT for raw block backend",
    )
    parser.add_argument(
        "--raw-slot-bytes",
        type=int,
        default=0,
        help=(
            "Override rust_raw_block slot size in bytes. "
            "0 means backend default sizing."
        ),
    )
    parser.add_argument("--alignment", type=int, default=4096)
    parser.add_argument(
        "--payload-size-kb",
        type=float,
        default=64.0,
        help=(
            "Payload size per I/O operation in KB. Must be a multiple of 0.5KB "
            "for bfloat16 layout [2,16,8,X]."
        ),
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="",
        help="Output JSON file path or directory",
    )
    parser.add_argument("--raw-use-callback", action="store_true")
    parser.add_argument(
        "--raw-submit-mode",
        choices=["async_loop", "threadpool", "sync"],
        default="async_loop",
        help=(
            "Rust raw-block submit path: "
            "'async_loop' uses run_coroutine_threadsafe()+asyncio.to_thread "
            "and shards keys across --raw-submit-loops by chunk_hash %% num_loops, "
            "'threadpool' submits directly to sharded backend "
            "ThreadPoolExecutors by chunk_hash %% num_pools, "
            "'sync' executes writes on the caller thread."
        ),
    )
    parser.add_argument(
        "--raw-submit-workers",
        type=int,
        default=1,
        help=(
            "Worker count per pool for rust raw-block 'threadpool' submit mode. "
            "Ignored by other modes."
        ),
    )
    parser.add_argument(
        "--raw-submit-pools",
        type=int,
        default=1,
        help=(
            "Number of ThreadPoolExecutor shards for rust raw-block "
            "'threadpool' submit mode. Keys are assigned by chunk_hash %% "
            "num_pools. Ignored by other modes."
        ),
    )
    parser.add_argument(
        "--raw-submit-loops",
        type=int,
        default=1,
        help=(
            "Number of asyncio loops for rust raw-block 'async_loop' submit mode. "
            "Keys are assigned by chunk_hash %% num_loops. Ignored by other modes."
        ),
    )
    parser.add_argument(
        "--latency-breakdown",
        action="store_true",
        help=(
            "Enable latency breakdown instrumentation "
            "(E2E / LMCache / syscall I/O, unit=us)."
        ),
    )
    parser.add_argument(
        "--operation",
        choices=["put", "both"],
        default="put",
        help=(
            "Benchmark operation: "
            "'put' writes only, "
            "'both' measures put then get."
        ),
    )
    parser.add_argument(
        "--submit-mode",
        choices=["single_key", "batch_slice"],
        default="single_key",
        help=(
            "Write submission mode: "
            "'single_key' submits one key at a time, "
            "'batch_slice' submits one slice per worker thread."
        ),
    )

    args = parser.parse_args()
    payload_shape = _resolve_payload_shape(args.payload_size_kb)

    results = []
    if args.backend in ("local_disk", "both"):
        results.append(
            _bench_local_disk(
                num_ops=args.num_ops,
                concurrency=args.concurrency,
                local_disk_dir=args.local_disk_dir,
                max_disk_gb=args.max_local_disk_gb,
                use_odirect=args.local_disk_odirect,
                alignment=args.alignment,
                operation=args.operation,
                payload_shape=payload_shape,
                payload_size_kb=args.payload_size_kb,
                measure_latency_breakdown=args.latency_breakdown,
                submit_mode=args.submit_mode,
            )
        )

    if args.backend in ("rust_raw_block", "both"):
        raw_device = args.raw_device
        cleanup_raw_device = False
        if not raw_device:
            # Use the same filesystem as local disk backend for apples-to-apples.
            raw_device = os.path.join(args.local_disk_dir, "raw_block.bin")
            cleanup_raw_device = True
        results.append(
            _bench_rust_raw_block(
                num_ops=args.num_ops,
                concurrency=args.concurrency,
                raw_device=raw_device,
                raw_device_size_gb=args.raw_device_size_gb,
                use_odirect=args.raw_odirect,
                alignment=args.alignment,
                cleanup_raw_device=cleanup_raw_device,
                use_callback=args.raw_use_callback,
                raw_slot_bytes=args.raw_slot_bytes,
                operation=args.operation,
                payload_shape=payload_shape,
                payload_size_kb=args.payload_size_kb,
                measure_latency_breakdown=args.latency_breakdown,
                submit_mode=args.submit_mode,
                raw_submit_mode=args.raw_submit_mode,
                raw_submit_workers=args.raw_submit_workers,
                raw_submit_loops=args.raw_submit_loops,
                raw_submit_pools=args.raw_submit_pools,
            )
        )

    for result in results:
        if "put_elapsed_sec" in result:
            print(
                f"{result['backend']} [put]: ops={result['num_ops']} "
                f"concurrency={result['concurrency']} "
                f"payload={result['payload_size_kb']}KB "
                f"elapsed={result['put_elapsed_sec']:.3f}s "
                f"ops/sec={result['put_ops_per_sec']:.2f} "
                f"MB/s={result['put_mb_per_sec']:.2f}"
            )
        if "get_elapsed_sec" in result:
            print(
                f"{result['backend']} [get]: ops={result['num_ops']} "
                f"concurrency={result['concurrency']} "
                f"payload={result['payload_size_kb']}KB "
                f"elapsed={result['get_elapsed_sec']:.3f}s "
                f"ops/sec={result['get_ops_per_sec']:.2f} "
                f"MB/s={result['get_mb_per_sec']:.2f}"
            )
        latency_breakdown = result.get("latency_breakdown", {})
        for op in ("write", "read"):
            op_stats = latency_breakdown.get(op)
            if op_stats is None:
                continue
            print(
                f"{result['backend']} [{op}-latency]: "
                f"e2e_avg={op_stats['e2e_avg_us']:.3f}us "
                f"lmcache_avg={op_stats['lmcache_avg_us']:.3f}us "
                f"io_syscall_avg={op_stats['io_syscall_avg_us']:.3f}us "
                f"(samples_e2e={op_stats['samples_e2e']}, "
                f"samples_io={op_stats['samples_io_syscall']})"
            )
            stage_stats = op_stats.get("stages", {})
            if stage_stats:
                stage_summary = ", ".join(
                    f"{stage}={stats['avg_us']:.3f}us"
                    for stage, stats in sorted(stage_stats.items())
                )
                print(
                    f"{result['backend']} [{op}-stages]: "
                    f"{stage_summary}"
                )

    if args.output_json:
        output_path = args.output_json
        if output_path.endswith(os.sep) or os.path.isdir(output_path):
            ts = time.strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(output_path, f"storage_backend_io_{ts}.json")
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Wrote results to {output_path}")


if __name__ == "__main__":
    main()
