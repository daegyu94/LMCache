#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Microbenchmark for ``FSL2Adapter`` lookup latency.

Submits lookup tasks of ``--keys`` random (absent) keys -- the common case on a
prefetch, where most suffix keys are not in L2 -- and reports the time until
the result is visible, with ``--busy-threads`` CPU-bound Python threads
competing for the GIL as an MP server's other threads do. Run with::

    python benchmarks/microbenchmark/fs_l2_lookup_benchmark.py --base-path /tmp/l2

The adapter checks the whole batch in one executor call; per-key
``aiofiles.os.path.exists`` awaits each paid a thread-pool round trip and a GIL
re-acquisition, which is what dominated the lookup under load.
"""

# Standard
import argparse
import os
import statistics
import threading
import time

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.l2_adapters.fs_l2_adapter import (
    FSL2Adapter,
    FSL2AdapterConfig,
)


def _spin(stop: threading.Event) -> None:
    """Hold the GIL in a busy loop until ``stop`` is set."""
    while not stop.is_set():
        pass


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--base-path", required=True)
    p.add_argument("--keys", type=int, default=32)
    p.add_argument("--busy-threads", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--iterations", type=int, default=60)
    args = p.parse_args()
    os.makedirs(args.base_path, exist_ok=True)
    adapter = FSL2Adapter(FSL2AdapterConfig(base_path=args.base_path))
    try:
        for busy in args.busy_threads:
            stop = threading.Event()

            threads = [
                threading.Thread(target=_spin, args=(stop,), daemon=True)
                for _ in range(busy)
            ]
            for t in threads:
                t.start()
            latencies = []
            for _ in range(args.iterations):
                keys = [
                    ObjectKey(chunk_hash=os.urandom(16), model_name="m", kv_rank=0)
                    for _ in range(args.keys)
                ]
                t0 = time.perf_counter()
                task_id = adapter.submit_lookup_and_lock_task(keys, {})
                while adapter.query_lookup_and_lock_result(task_id) is None:
                    time.sleep(0.0001)
                latencies.append((time.perf_counter() - t0) * 1e3)
            stop.set()
            for t in threads:
                t.join()
            latencies.sort()
            n = len(latencies)
            print(
                f"keys={args.keys} busy_threads={busy} "
                f"p50={latencies[n // 2]:.2f}ms p90={latencies[int(n * 0.9)]:.2f}ms "
                f"mean={statistics.mean(latencies):.2f}ms"
            )
    finally:
        adapter.close()


if __name__ == "__main__":
    main()
