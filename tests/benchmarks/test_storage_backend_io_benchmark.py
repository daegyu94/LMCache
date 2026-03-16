# SPDX-License-Identifier: Apache-2.0

# Standard
import importlib.util
from pathlib import Path
import sys

# First Party
from lmcache.utils import CacheEngineKey


_MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "benchmarks"
    / "storage_backend_io"
    / "storage_backend_io_benchmark.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "storage_backend_io_benchmark",
    _MODULE_PATH,
)
assert _SPEC is not None
assert _SPEC.loader is not None
storage_backend_io_benchmark = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = storage_backend_io_benchmark
_SPEC.loader.exec_module(storage_backend_io_benchmark)


def test_is_device_mounted_matches_exact_device(tmp_path) -> None:
    """Mounted block devices should be detected from /proc/self/mounts style data."""
    device_path = tmp_path / "nvme0n1"
    device_path.touch()
    mounts_path = tmp_path / "mounts"
    mounts_path.write_text(f"{device_path} /mnt/test ext4 rw 0 0\n")

    assert (
        storage_backend_io_benchmark._is_device_mounted(
            str(device_path),
            mounts_path=str(mounts_path),
        )
        is True
    )


def test_is_device_mounted_handles_missing_mounts_file(tmp_path) -> None:
    """Missing mounts files should fail open instead of crashing the benchmark."""
    device_path = tmp_path / "nvme0n1"
    device_path.touch()

    assert (
        storage_backend_io_benchmark._is_device_mounted(
            str(device_path),
            mounts_path=str(tmp_path / "missing-mounts"),
        )
        is False
    )


def test_is_device_mounted_ignores_non_device_sources(tmp_path) -> None:
    """Non-/dev mount sources should not trip the raw-device safety check."""
    device_path = tmp_path / "nvme0n1"
    device_path.touch()
    mounts_path = tmp_path / "mounts"
    mounts_path.write_text("tmpfs /tmp tmpfs rw,nosuid,nodev 0 0\n")

    assert (
        storage_backend_io_benchmark._is_device_mounted(
            str(device_path),
            mounts_path=str(mounts_path),
        )
        is False
    )


def test_latency_breakdown_collector_summarizes_per_key_samples() -> None:
    """Latency breakdown collector should summarize per-key put/get samples."""
    collector = storage_backend_io_benchmark._LatencyBreakdownCollector()
    put_key0 = CacheEngineKey(
        "benchmark_model",
        1,
        0,
        0,
        storage_backend_io_benchmark.DEFAULT_DTYPE,
    )
    put_key1 = CacheEngineKey(
        "benchmark_model",
        1,
        0,
        1,
        storage_backend_io_benchmark.DEFAULT_DTYPE,
    )
    get_key0 = CacheEngineKey(
        "benchmark_model",
        1,
        0,
        2,
        storage_backend_io_benchmark.DEFAULT_DTYPE,
    )
    get_key1 = CacheEngineKey(
        "benchmark_model",
        1,
        0,
        3,
        storage_backend_io_benchmark.DEFAULT_DTYPE,
    )

    collector.record_stage("put", put_key0, "queue_wait", 10_000, 1)
    collector.record_io("put", put_key0, 30_000, 1)
    collector.record_e2e("put", put_key0, 100_000, 1)
    collector.record_stage("put", put_key1, "queue_wait", 20_000, 1)
    collector.record_io("put", put_key1, 40_000, 1)
    collector.record_e2e("put", put_key1, 110_000, 1)

    collector.record_io("get", get_key0, 50_000, 1)
    collector.record_e2e("get", get_key0, 150_000, 1)
    collector.record_io("get", get_key1, 60_000, 1)
    collector.record_e2e("get", get_key1, 170_000, 1)

    result = collector.to_result_dict()

    assert result["put"]["samples"] == 2
    assert result["put"]["e2e_us"]["avg"] == 105.0
    assert result["put"]["queue_wait_us"]["avg"] == 15.0
    assert result["put"]["io_us"]["avg"] == 35.0
    assert result["put"]["other_us"]["avg"] == 55.0

    assert result["get"]["samples"] == 2
    assert result["get"]["queue_wait_us"]["samples"] == 0
    assert result["get"]["io_us"]["avg"] == 55.0
    assert result["get"]["other_us"]["avg"] == 105.0

    put_key0_stats = result["put"]["per_key"][put_key0.to_string()]
    assert put_key0_stats["queue_wait_us"] == 10.0
    assert put_key0_stats["io_us"] == 30.0
    assert put_key0_stats["other_us"] == 60.0
