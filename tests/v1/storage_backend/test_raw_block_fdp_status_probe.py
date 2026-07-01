# SPDX-License-Identifier: Apache-2.0
"""Hardware-gated FDP status probe for the Rust raw-block device binding."""

# Standard
import errno
import os
import stat

# Third Party
import pytest


def _get_fdp_char_device_path() -> str | None:
    """Return the explicitly configured FDP-capable NVMe character device path."""
    return os.environ.get("LMCACHE_TEST_FDP_CHAR_DEVICE")


def _message_contains(exc: BaseException, fragments: tuple[str, ...]) -> bool:
    """Return whether an exception message contains any expected fragment."""
    msg = str(exc).lower()
    return any(fragment in msg for fragment in fragments)


def _is_skip_safe_device_setup_error(exc: BaseException) -> bool:
    """Return whether raw-device setup failed for an external HW/OS reason."""
    if getattr(exc, "errno", None) in {
        errno.EACCES,
        errno.ENOSYS,
        errno.ENOTTY,
        errno.EPERM,
    }:
        return True
    return _message_contains(
        exc,
        (
            "function not implemented",
            "inappropriate ioctl",
            "nvme identify namespace ioctl failed",
            "operation not permitted",
            "permission denied",
            "requires an nvme namespace character device",
        ),
    )


def _is_skip_safe_fdp_status_error(exc: BaseException) -> bool:
    """Return whether the FDP status ioctl failed for missing HW/OS support."""
    if "nvme fdp reclaim unit handle status ioctl failed" not in str(exc).lower():
        return False
    if getattr(exc, "errno", None) in {
        errno.EINVAL,
        errno.ENOSYS,
        errno.ENOTTY,
        errno.EPERM,
    }:
        return True
    return _message_contains(
        exc,
        (
            "function not implemented",
            "inappropriate ioctl",
            "invalid argument",
            "not supported",
            "operation not permitted",
            "permission denied",
            "unsupported",
        ),
    )


def test_uring_cmd_fetch_fdp_status_status_only_probe() -> None:
    """Probe FDP status on an explicitly configured NVMe character device.

    This hardware-gated test is status-only. It does not write to the device,
    initialize the raw-block adapter layout, or verify KV placement policy.
    Set ``LMCACHE_TEST_FDP_CHAR_DEVICE`` to an FDP-capable NVMe namespace
    character device such as ``/dev/ng0n1``.
    """
    configured_device_path = _get_fdp_char_device_path()
    if configured_device_path is None or configured_device_path == "":
        pytest.skip("Set LMCACHE_TEST_FDP_CHAR_DEVICE to run the FDP status probe.")
        return
    device_path: str = configured_device_path
    if not os.path.exists(device_path):
        pytest.skip(f"FDP test device {device_path} not found.")
    if not stat.S_ISCHR(os.stat(device_path).st_mode):
        pytest.skip(f"FDP test device {device_path} is not a character device.")

    lmcache_rust_raw_block_io = pytest.importorskip("lmcache_rust_raw_block_io")
    raw_device = None
    try:
        raw_device = lmcache_rust_raw_block_io.RawBlockDevice(
            device_path,
            writable=False,
            use_odirect=False,
            alignment=4096,
            io_engine="io_uring",
            use_uring_cmd=True,
            iouring_queue_depth=256,
        )
    except Exception as e:
        if _is_skip_safe_device_setup_error(e):
            pytest.skip(f"FDP status probe setup is unavailable on {device_path}: {e}")
        raise

    try:
        status = raw_device.fetch_fdp_status()
    except Exception as e:
        if _is_skip_safe_fdp_status_error(e):
            pytest.skip(f"FDP status probe is unavailable on {device_path}: {e}")
        raise
    finally:
        if raw_device is not None:
            raw_device.close()

    assert isinstance(status, list)
    assert status, f"Expected FDP status entries from {device_path}, got none"
    for placement_id, ruh_id in status:
        assert isinstance(placement_id, int)
        assert isinstance(ruh_id, int)
        assert 0 <= placement_id <= 0xFFFF
        assert 0 <= ruh_id <= 0xFFFF
