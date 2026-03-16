# SPDX-License-Identifier: Apache-2.0

# Standard
import importlib.util
from pathlib import Path
import sys


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
