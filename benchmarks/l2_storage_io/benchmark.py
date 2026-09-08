# SPDX-License-Identifier: Apache-2.0
"""CPU-only synthetic KV I/O benchmark using LMCache's real L2 adapter API.

This is an I/O microbenchmark, not a model-serving benchmark: E2E means
request admission/scheduling through adapter completion, never TTFT.
"""

# Future
from __future__ import annotations

# Standard
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import argparse
import ctypes
import json
import math
import multiprocessing
import os
import subprocess
import sys
import threading
import time
import uuid

# Local
from .metrics import LatencySample, Metrics
from .workload import SyntheticChunk, SyntheticRequest, WorkloadSpec, build_workload


@dataclass(frozen=True)
class Profile:
    """A named adapter/backend label; mount identity is supplied by the user."""

    name: str
    adapter: str
    filesystem: str
    backend: str | None = None
    use_direct_io: bool = True
    use_uring: bool = False
    mount_point: str | None = None


PROFILES = (
    Profile("xfs-fs-native", "fs_native", "xfs"),
    Profile("pnfs-fs-native", "fs_native", "pnfs"),
    Profile("pnfs-nixl-posix", "nixl_store_dynamic", "pnfs", "POSIX"),
    Profile("3fs-nixl-hf3fs", "nixl_store_dynamic", "3fs", "HF3FS"),
)


class CpuArena:
    """A 4096-aligned CPU arena with stable writable slices for async I/O."""

    def __init__(self, size: int, alignment: int = 4096) -> None:
        if size <= 0 or alignment <= 0:
            raise ValueError("arena size and alignment must be positive")
        self._raw = ctypes.create_string_buffer(size + alignment)
        raw_address = ctypes.addressof(self._raw)
        self.offset = (-raw_address) % alignment
        self.size = size
        self.alignment = alignment
        self.ptr = raw_address + self.offset
        self._view = memoryview(self._raw).cast("B")

    def view(self, offset: int, size: int) -> memoryview:
        """Return an in-bounds writable byte slice with a stable lifetime."""
        if offset < 0 or size < 0 or offset + size > self.size:
            raise ValueError("arena slice exceeds bounded buffer")
        return self._view[self.offset + offset : self.offset + offset + size]


class ArenaSlice:
    """A non-overlapping view of a parent arena, suitable for one request."""

    def __init__(self, parent: CpuArena, offset: int, size: int) -> None:
        self._parent = parent
        self._offset = offset
        self.size = size
        self.alignment = parent.alignment
        self.ptr = parent.ptr + offset

    def view(self, offset: int, size: int) -> memoryview:
        """Return a slice relative to this request's reserved region."""
        if offset < 0 or size < 0 or offset + size > self.size:
            raise ValueError("arena slice exceeds request slot")
        return self._parent.view(self._offset + offset, size)


def _payload(chunk: SyntheticChunk, size: int) -> bytes:
    repetitions = (size + len(chunk.payload_seed) - 1) // len(chunk.payload_seed)
    return (chunk.payload_seed * repetitions)[:size]


def _objects(
    chunks: tuple[SyntheticChunk, ...],
    arena: CpuArena | ArenaSlice,
    chunk_bytes: int,
    poison: bool,
) -> list[Any]:
    """Build real ``MemoryObj`` instances over a caller-owned CPU arena."""
    # Third Party
    import torch

    # First Party
    from lmcache.v1.memory_management import BytesBufferMemoryObj, MemoryObjMetadata

    objects: list[Any] = []
    for index, chunk in enumerate(chunks):
        raw = arena.view(index * chunk_bytes, chunk_bytes)
        raw[:] = b"\xa5" * chunk_bytes if poison else _payload(chunk, chunk_bytes)
        metadata = MemoryObjMetadata(
            shape=torch.Size([chunk_bytes]),
            dtype=torch.uint8,
            address=arena.ptr + index * chunk_bytes,
            phy_size=chunk_bytes,
            ref_count=1,
        )
        objects.append(BytesBufferMemoryObj(raw, metadata))
    return objects


def _keys(chunks: tuple[SyntheticChunk, ...]) -> list[Any]:
    """Make stable L2 keys for synthetic chunks."""
    # First Party
    from lmcache.v1.distributed.api import ObjectKey

    return [ObjectKey(chunk.payload_seed, "synthetic-l2", 0) for chunk in chunks]


def _layout(chunk_bytes: int) -> dict[int, Any]:
    # Third Party
    import torch

    # First Party
    from lmcache.v1.distributed.api import MemoryLayoutDesc

    return {0: MemoryLayoutDesc([torch.Size([chunk_bytes])], [torch.uint8])}


class StoreCompletionDispatcher:
    """Retain store completions claimed while another worker is polling."""

    def __init__(self, adapter: Any) -> None:
        self._adapter = adapter
        self._completed: dict[int, Any] = {}
        self._lock = threading.Lock()

    def wait(self, task_id: int, timeout: float) -> tuple[Any | None, bool]:
        """Wait for one task without discarding another request's completion."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                result = self._completed.pop(task_id, None)
                if result is not None:
                    return result, False
                self._completed.update(self._adapter.pop_completed_store_tasks())
                result = self._completed.pop(task_id, None)
                if result is not None:
                    return result, False
            time.sleep(0.001)
        return None, True


def _wait_bitmap(
    adapter: Any, task_id: int, lookup: bool, timeout: float
) -> tuple[Any | None, bool]:
    query = (
        adapter.query_lookup_and_lock_result if lookup else adapter.query_load_result
    )
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = query(task_id)
        if result is not None:
            return result, False
        time.sleep(0.001)
    return None, True


def _revision(checkout: Path) -> dict[str, object]:
    """Return commit and all dirty/untracked state relevant to reproducibility."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=checkout, text=True
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=checkout, text=True
        ).splitlines()
        return {"commit": commit, "dirty_or_untracked": bool(status), "status": status}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty_or_untracked": None, "status": []}


def _profile(name: str | None) -> list[Profile]:
    if name is None:
        return list(PROFILES)
    selected = [entry for entry in PROFILES if entry.name == name]
    if not selected:
        raise ValueError(f"unknown profile {name!r}")
    return selected


def _config(profile: Profile, path: Path, capacity_gb: float) -> Any:
    if profile.adapter == "fs_native":
        # First Party
        from lmcache.v1.distributed.l2_adapters.fs_native_l2_adapter import (
            FSNativeL2AdapterConfig,
        )

        return FSNativeL2AdapterConfig(
            str(path), num_workers=4, use_odirect=profile.use_direct_io
        )
    # First Party
    from lmcache.v1.distributed.l2_adapters.nixl_store_dynamic_l2_adapter import (
        DynamicNixlStoreL2AdapterConfig,
    )

    return DynamicNixlStoreL2AdapterConfig(
        profile.backend or "POSIX",
        {
            "file_path": str(path),
            "use_direct_io": str(profile.use_direct_io).lower(),
            "use_uring": str(profile.use_uring).lower(),
            **({"mount_point": profile.mount_point} if profile.mount_point else {}),
            "max_capacity_gb": str(capacity_gb),
        },
    )


def _fail_stop_timeout() -> None:
    """Exit the worker without unlocking, freeing, or reusing in-flight buffers."""
    os._exit(124)


def _request(
    adapter: Any,
    request: SyntheticRequest,
    arena: CpuArena | ArenaSlice,
    spec: WorkloadSpec,
    timeout: float,
    queue_seconds: float,
    store_completions: StoreCompletionDispatcher,
) -> tuple[SyntheticRequest, LatencySample, dict[str, int | bool]]:
    """Execute one request while retaining its arena until all tasks settle."""
    started = time.monotonic() - queue_seconds
    read_chunks = tuple(request.chunks[i] for i in request.read_indices)
    write_chunks = tuple(request.chunks[i] for i in request.write_indices)
    lookup_seconds = submit_seconds = completion_seconds = 0.0
    read_bytes = write_bytes = hits = misses = 0
    error = timed_out = integrity = False
    locked: list[Any] = []
    try:
        if read_chunks:
            keys = _keys(read_chunks)
            at = time.monotonic()
            lookup_task = adapter.submit_lookup_and_lock_task(
                keys, _layout(spec.chunk_bytes)
            )
            submit_seconds += time.monotonic() - at
            at = time.monotonic()
            bitmap, timed_out = _wait_bitmap(adapter, lookup_task, True, timeout)
            lookup_seconds += time.monotonic() - at
            if timed_out:
                _fail_stop_timeout()
            if bitmap is None:
                error = True
            else:
                hit_indices = [
                    index for index in range(len(keys)) if bitmap.test(index)
                ]
                hits = len(hit_indices)
                misses = len(read_chunks) - hits
                locked = [keys[index] for index in hit_indices]
                hit_chunks = tuple(read_chunks[index] for index in hit_indices)
                objects = _objects(hit_chunks, arena, spec.chunk_bytes, poison=True)
                at = time.monotonic()
                load_task = adapter.submit_load_task(locked, objects)
                submit_seconds += time.monotonic() - at
                at = time.monotonic()
                loaded, load_timeout = _wait_bitmap(adapter, load_task, False, timeout)
                completion_seconds += time.monotonic() - at
                timed_out = timed_out or load_timeout
                if load_timeout:
                    _fail_stop_timeout()
                if loaded is None:
                    error = True
                else:
                    loaded_hits = loaded.popcount()
                    hits = min(hits, loaded_hits)
                    misses = len(read_chunks) - hits
                    read_bytes = hits * spec.chunk_bytes
                    for index, chunk in enumerate(hit_chunks):
                        if loaded.test(index) and objects[
                            index
                        ].byte_array.tobytes() != _payload(chunk, spec.chunk_bytes):
                            integrity = True
                    if integrity:
                        read_bytes = 0
                        error = True
        if write_chunks and not error:
            objects = _objects(write_chunks, arena, spec.chunk_bytes, poison=False)
            at = time.monotonic()
            task = adapter.submit_store_task(_keys(write_chunks), objects)
            submit_seconds += time.monotonic() - at
            at = time.monotonic()
            result, timed_out = store_completions.wait(task, timeout)
            completion_seconds += time.monotonic() - at
            if timed_out:
                _fail_stop_timeout()
            if result is None or not result.is_successful():
                error = True
            else:
                write_bytes = result.bytes_transferred()
                if write_bytes != len(write_chunks) * spec.chunk_bytes:
                    error = True
    except Exception as exc:
        print(
            f"L2 request failed with uncertain in-flight I/O: {exc!r}", file=sys.stderr
        )
        os._exit(125)
    finally:
        if locked:
            adapter.submit_unlock(locked)
    return (
        request,
        LatencySample(
            queue_seconds,
            lookup_seconds,
            submit_seconds,
            completion_seconds,
            time.monotonic() - started,
        ),
        {
            "read_bytes": 0 if error else read_bytes,
            "write_bytes": 0 if error else write_bytes,
            "hits": hits,
            "misses": misses,
            "error": error,
            "timeout": timed_out,
            "integrity": integrity,
        },
    )


def _worker_entry(
    connection: Any,
    profile: Profile,
    spec: WorkloadSpec,
    storage_root: Path,
    timeout: float,
    concurrency: int,
    checkout: Path,
) -> None:
    """Run one profile in its owning child process and return JSON-safe output."""
    try:
        connection.send(
            _run_profile_worker(
                profile, spec, storage_root, timeout, concurrency, checkout
            )
        )
    except Exception as exc:
        connection.send(
            {
                "profile": profile.name,
                "failed": True,
                "errors": 1,
                "failure": repr(exc),
                "source": _revision(checkout),
            }
        )
    finally:
        connection.close()


def _run_profile_isolated(
    profile: Profile,
    spec: WorkloadSpec,
    storage_root: Path,
    timeout: float,
    run_timeout: float,
    concurrency: int,
    checkout: Path,
) -> dict[str, object]:
    """Run a worker while draining its pipe before join can deadlock on payload."""  # noqa: E501
    if (
        not math.isfinite(timeout)
        or not math.isfinite(run_timeout)
        or timeout <= 0
        or run_timeout <= 0
    ):
        raise ValueError("timeout values must be positive")
    context = multiprocessing.get_context("spawn")
    parent, child_connection = context.Pipe(duplex=False)
    process = context.Process(
        target=_worker_entry,
        args=(
            child_connection,
            profile,
            spec,
            storage_root,
            timeout,
            concurrency,
            checkout,
        ),
    )
    process.start()
    child_connection.close()
    deadline = time.monotonic() + run_timeout
    result: dict[str, object] | None = None
    try:
        while time.monotonic() < deadline:
            if parent.poll(min(0.1, max(0.0, deadline - time.monotonic()))):
                try:
                    result = parent.recv()
                except EOFError:
                    result = None
                break
            if not process.is_alive():
                break
        if result is not None:
            process.join(5)
            return result
        if process.is_alive():
            process.terminate()
            process.join(5)
            if process.is_alive():
                process.kill()
                process.join(5)
            return {
                "profile": profile.name,
                "failed": True,
                "errors": 1,
                "timeouts": 1,
                "failure": f"run timeout after {run_timeout} seconds",
                "source": _revision(checkout),
            }
        process.join()
        return {
            "profile": profile.name,
            "failed": True,
            "errors": 1,
            "timeouts": 1 if process.exitcode == 124 else 0,
            "failure": f"worker exited without a result (exitcode={process.exitcode})",
            "source": _revision(checkout),
        }
    finally:
        parent.close()


def _run_profile_worker(
    profile: Profile,
    spec: WorkloadSpec,
    storage_root: Path,
    timeout: float,
    concurrency: int,
    checkout: Path,
) -> dict[str, object]:
    """Run one profile with a bounded number of non-aliasing CPU arenas."""
    if spec.chunk_bytes % 4096:
        raise ValueError("chunk_bytes must be a multiple of 4096 for direct I/O")
    if concurrency < 1:
        raise ValueError("concurrency must be positive")
    warm_chunks, requests = build_workload(spec)
    if not warm_chunks and any(request.read_indices for request in requests):
        raise ValueError("read and mixed workloads require working_set_chunks >= 1")
    run_path = storage_root / profile.name / f"run-{uuid.uuid4().hex}"
    run_path.mkdir(parents=True, exist_ok=False)
    slot_bytes = max(spec.chunks_per_request, 1) * spec.chunk_bytes
    parent_arena = CpuArena(slot_bytes * concurrency)
    arenas = [
        ArenaSlice(parent_arena, slot * slot_bytes, slot_bytes)
        for slot in range(concurrency)
    ]
    # First Party
    from lmcache.v1.distributed.internal_api import L1MemoryDesc
    from lmcache.v1.distributed.l2_adapters.factory import (
        create_l2_adapter_from_registry,
    )
    from lmcache.v1.distributed.l2_adapters.instrumentation import (
        L2AdapterInstrumentation,
    )

    capacity_gb = max(
        1.0,
        (
            spec.working_set_chunks
            + (spec.warmup_requests + spec.requests) * spec.chunks_per_request
        )
        * spec.chunk_bytes
        / 2**30,
    )
    l1_desc = L1MemoryDesc(parent_arena.ptr, parent_arena.size, parent_arena.alignment)
    adapter = create_l2_adapter_from_registry(
        _config(profile, run_path, capacity_gb),
        l1_desc if profile.adapter != "fs_native" else None,
    )
    collector = L2AdapterInstrumentation()
    set_instrumentation = getattr(adapter, "set_instrumentation", None)
    metrics = Metrics()
    store_completions = StoreCompletionDispatcher(adapter)
    try:
        # Prepopulate every reusable key once in bounded batches; excluded from metrics.
        for start in range(0, len(warm_chunks), spec.chunks_per_request):
            batch = tuple(warm_chunks[start : start + spec.chunks_per_request])
            result, timed_out = store_completions.wait(
                adapter.submit_store_task(
                    _keys(batch), _objects(batch, arenas[0], spec.chunk_bytes, False)
                ),
                timeout,
            )
            if timed_out:
                _fail_stop_timeout()
            if (
                result is None
                or not result.is_successful()
                or result.bytes_transferred() != len(batch) * spec.chunk_bytes
            ):
                raise RuntimeError("prepopulation failed or was deduplicated")
        for warmup in (request for request in requests if request.warmup):
            _, _, warmup_values = _request(
                adapter,
                warmup,
                arenas[0],
                spec,
                timeout,
                0.0,
                store_completions,
            )
            if warmup_values["error"]:
                raise RuntimeError("warmup request failed")
        if callable(set_instrumentation):
            set_instrumentation(collector)
        measured_started = time.monotonic()
        slots = list(range(concurrency))
        slots_lock = threading.Lock()

        def execute(
            request: SyntheticRequest, admitted_at: float
        ) -> tuple[SyntheticRequest, LatencySample, dict[str, int | bool]]:
            queued = admitted_at
            with slots_lock:
                slot = slots.pop()
            try:
                return _request(
                    adapter,
                    request,
                    arenas[slot],
                    spec,
                    timeout,
                    time.monotonic() - queued,
                    store_completions,
                )
            finally:
                with slots_lock:
                    slots.append(slot)

        measured = [request for request in requests if not request.warmup]
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = []
            for request in measured:
                admitted_at = time.monotonic()
                futures.append(executor.submit(execute, request, admitted_at))
            for future in as_completed(futures):
                request, sample, values = future.result()
                metrics.add(
                    request.operation,
                    sample,
                    read_bytes=int(values["read_bytes"]),
                    write_bytes=int(values["write_bytes"]),
                    hits=int(values["hits"]),
                    misses=int(values["misses"]),
                    error=bool(values["error"]),
                    timeout=bool(values["timeout"]),
                    integrity=bool(values["integrity"]),
                )
        elapsed = time.monotonic() - measured_started
        result = metrics.summary(elapsed)
        result.update(
            {
                "profile": profile.name,
                "effective_profile": asdict(profile),
                "filesystem_label": profile.filesystem,
                "adapter": profile.adapter,
                "backend": profile.backend,
                "storage_path": str(run_path),
                "source": _revision(checkout),
                "workload": asdict(spec),
                "concurrency": concurrency,
                "mount_metadata": {
                    "provided_root": str(storage_root),
                    "verified_by_benchmark": False,
                },
                "internal_latency_seconds": collector.snapshot(),
                "timing_semantics": {
                    "e2e": "request admission/scheduling through task completion; not TTFT",  # noqa: E501
                    "request_spans": "lookup, submit, and completion are inclusive/overlapping and are not additive",  # noqa: E501
                    "native.queue_io_completion": "native connector queue + I/O + completion dispatch; opaque C++ interval",  # noqa: E501
                    "nixl.transfer_wait": "NIXL transfer plus the adapter's 10 ms polling; not pure storage-service latency",  # noqa: E501
                    "internal_spans": "per-file spans may overlap under concurrency and must not be summed",  # noqa: E501
                },
            }
        )
        result["failed"] = bool(
            result["errors"]
            or result["timeouts"]
            or result["integrity_failures"]
            or result["read_misses"]
        )
        return result
    finally:
        adapter.close()


def main() -> None:
    """Parse CLI arguments and write one JSON result per requested profile."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lmcache-checkout", type=Path, required=True)
    parser.add_argument("--storage-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--profile", choices=[entry.name for entry in PROFILES], required=True
    )
    parser.add_argument(
        "--operation", choices=("read", "write", "mixed"), default="mixed"
    )
    parser.add_argument("--requests", type=int, default=32)
    parser.add_argument("--chunks-per-request", type=int, default=8)
    parser.add_argument("--chunk-bytes", type=int, default=1024 * 1024)
    parser.add_argument("--working-set-chunks", type=int, default=128)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--run-timeout-seconds", type=float, default=900.0)
    parser.add_argument("--warmup-requests", type=int, default=1)
    parser.add_argument("--mount-point")
    parser.add_argument(
        "--use-uring", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument(
        "--direct-io", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--read-ratio", type=float, default=0.5)
    parser.add_argument("--prefix-reuse", type=float, default=0.8)
    parser.add_argument("--hot-set-fraction", type=float, default=0.25)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    spec = WorkloadSpec(
        operation=args.operation,
        requests=args.requests,
        chunks_per_request=args.chunks_per_request,
        chunk_bytes=args.chunk_bytes,
        working_set_chunks=args.working_set_chunks,
        warmup_requests=args.warmup_requests,
        read_ratio=args.read_ratio,
        prefix_reuse=args.prefix_reuse,
        hot_set_fraction=args.hot_set_fraction,
        seed=args.seed,
    )
    if (
        not math.isfinite(args.timeout_seconds)
        or not math.isfinite(args.run_timeout_seconds)
        or args.timeout_seconds <= 0
        or args.run_timeout_seconds <= 0
    ):
        parser.error("timeout values must be positive")
    selected = _profile(args.profile)[0]
    profile = Profile(
        selected.name,
        selected.adapter,
        selected.filesystem,
        selected.backend,
        selected.use_direct_io if args.direct_io is None else args.direct_io,
        selected.use_uring if args.use_uring is None else args.use_uring,
        args.mount_point,
    )
    if profile.adapter == "fs_native" and (
        args.mount_point or args.use_uring is not None
    ):
        parser.error("--mount-point and --use-uring apply only to NIXL profiles")
    if profile.backend == "HF3FS" and not profile.mount_point:
        parser.error("HF3FS requires --mount-point")
    profiles = [profile]
    if args.dry_run:
        warm, requests = build_workload(spec)
        print(
            json.dumps(
                {
                    "profiles": [asdict(item) for item in profiles],
                    "warmup_chunks": len(warm),
                    "requests": len(requests),
                    "workload": asdict(spec),
                    "source": _revision(args.lmcache_checkout),
                },
                indent=2,
            )
        )
        return
    args.output.mkdir(parents=True, exist_ok=True)
    for profile in profiles:
        existing_result = args.output / f"{profile.name}.json"
        if existing_result.exists():
            raise FileExistsError(
                f"refusing to overwrite existing result: {existing_result}"
            )
    exit_code = 0
    for profile in profiles:
        try:
            result = _run_profile_isolated(
                profile,
                spec,
                args.storage_root,
                args.timeout_seconds,
                args.run_timeout_seconds,
                args.concurrency,
                args.lmcache_checkout,
            )
        except BaseException as exc:
            result = {
                "profile": profile.name,
                "failed": True,
                "errors": 1,
                "failure": repr(exc),
                "source": _revision(args.lmcache_checkout),
            }
            exit_code = 1
        result_path = args.output / f"{profile.name}.json"
        result_path.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if result.get("failed"):
            exit_code = 1
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
