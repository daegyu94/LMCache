# SPDX-License-Identifier: Apache-2.0
"""Benchmark LocalDiskBackend vs RustRawBlockBackend for put/get I/O."""

# Future
from __future__ import annotations

# Standard
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
import argparse
import asyncio
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


def _start_loop() -> tuple[asyncio.AbstractEventLoop, threading.Thread]:
    loop = asyncio.new_event_loop()
    t = threading.Thread(target=loop.run_forever, name="bench-loop", daemon=True)
    t.start()
    return loop, t


def _stop_loop(loop: asyncio.AbstractEventLoop, t: threading.Thread) -> None:
    loop.call_soon_threadsafe(loop.stop)
    t.join(timeout=5)
    loop.close()


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
) -> float:
    def read_slice(start: int, end: int) -> None:
        for key in keys[start:end]:
            obj = backend.get_blocking(key)
            if obj is None:
                raise RuntimeError(f"get miss for key={key}")
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
    config.extra_config = {"use_odirect": use_odirect,
                           "local_cpu.pinned_align_bytes": 4096,
                           }

    local_cpu = LocalCPUBackend(
        config=config,
        metadata=metadata,
        dst_device="cpu",
        # memory_allocator=AdHocMemoryAllocator(device="cpu"),
    )
    backend = LocalDiskBackend(
        config=config,
        loop=loop,
        local_cpu_backend=local_cpu,
        dst_device="cpu",
        metadata=metadata,
    )

    keys = _make_keys(num_ops)
    keepalive: list[torch.Tensor] = []
    objs = _make_memory_objs(
        num_ops, use_odirect, alignment, keepalive, payload_shape
    )

    completed = 0
    lock = threading.Lock()
    done = threading.Event()

    def on_complete(_key: CacheEngineKey) -> None:
        nonlocal completed
        with lock:
            completed += 1
            if completed >= num_ops:
                done.set()

    def submit_slice(start: int, end: int) -> None:
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
        get_phase_elapsed = _bench_get_phase(backend, keys, concurrency)

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
    }
    if put_elapsed is not None:
        result["put_elapsed_sec"] = put_elapsed
        result["put_ops_per_sec"] = num_ops / put_elapsed if put_elapsed > 0 else 0.0
    if get_phase_elapsed is not None:
        result["get_elapsed_sec"] = get_phase_elapsed
        result["get_ops_per_sec"] = (
            num_ops / get_phase_elapsed if get_phase_elapsed > 0 else 0.0
        )
        result["get_buffered_io_fallback"] = get_with_buffered_io
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
    operation: str,
    payload_shape: torch.Size,
    payload_size_kb: float,
) -> dict:
    loop, t = _start_loop()
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
    }

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
        dst_device="cpu",
    )

    keys = _make_keys(num_ops)
    keepalive: list[torch.Tensor] = []
    objs = _make_memory_objs(
        num_ops, use_odirect, alignment, keepalive, payload_shape
    )

    completed = 0
    lock = threading.Lock()
    done = threading.Event()

    futures = []
    fut_lock = threading.Lock()

    def on_complete(_key: CacheEngineKey) -> None:
        nonlocal completed
        with lock:
            completed += 1
            if completed >= num_ops:
                done.set()

    def submit_slice(start: int, end: int) -> None:
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
                    futures.extend(futs)

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
            for fut in futures:
                fut.result(timeout=120)
        elapsed = time.perf_counter() - start
        return elapsed if measure else 0.0

    put_elapsed: Optional[float] = None
    get_phase_elapsed: Optional[float] = None

    if operation in ("put", "both"):
        put_elapsed = run_put_phase(measure=True)

    _release_memory_objs(objs)

    if operation in ("both"):
        get_phase_elapsed = _bench_get_phase(backend, keys, concurrency)

    backend.close()
    _stop_loop(loop, t)

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
        "operation": operation,
        "payload_size_kb": payload_size_kb,
    }
    if put_elapsed is not None:
        result["put_elapsed_sec"] = put_elapsed
        result["put_ops_per_sec"] = num_ops / put_elapsed if put_elapsed > 0 else 0.0
    if get_phase_elapsed is not None:
        result["get_elapsed_sec"] = get_phase_elapsed
        result["get_ops_per_sec"] = (
            num_ops / get_phase_elapsed if get_phase_elapsed > 0 else 0.0
        )
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
        "--operation",
        choices=["put", "both"],
        default="put",
        help=(
            "Benchmark operation: "
            "'put' writes only, "
            "'both' measures put then get."
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
                operation=args.operation,
                payload_shape=payload_shape,
                payload_size_kb=args.payload_size_kb,
            )
        )

    for result in results:
        if "put_elapsed_sec" in result:
            print(
                f"{result['backend']} [put]: ops={result['num_ops']} "
                f"concurrency={result['concurrency']} "
                f"payload={result['payload_size_kb']}KB "
                f"elapsed={result['put_elapsed_sec']:.3f}s "
                f"ops/sec={result['put_ops_per_sec']:.2f}"
            )
        if "get_elapsed_sec" in result:
            print(
                f"{result['backend']} [get]: ops={result['num_ops']} "
                f"concurrency={result['concurrency']} "
                f"payload={result['payload_size_kb']}KB "
                f"elapsed={result['get_elapsed_sec']:.3f}s "
                f"ops/sec={result['get_ops_per_sec']:.2f}"
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
