# SPDX-License-Identifier: Apache-2.0
"""Tests for the native C++ filesystem connector."""

# Standard
from array import array
from pathlib import Path
from typing import Any
import ctypes
import os
import select
import time

# Third Party
import pytest


def _import_fs_client() -> type:
    try:
        # First Party
        from lmcache.lmcache_fs import LMCacheFSClient
    except ImportError as exc:
        pytest.skip(f"native FS extension is not available: {exc}")
    return LMCacheFSClient


def _buffer_addr(buf: memoryview) -> int:
    return ctypes.addressof(ctypes.c_char.from_buffer(buf))


def _aligned_memoryview(
    size: int,
    alignment: int,
) -> tuple[bytearray, memoryview]:
    raw = bytearray(size + alignment)
    raw_addr = ctypes.addressof(ctypes.c_char.from_buffer(raw))
    offset = (-raw_addr) % alignment
    view = memoryview(raw)[offset : offset + size]
    assert _buffer_addr(view) % alignment == 0
    return raw, view


def _misaligned_memoryview(
    size: int,
    alignment: int,
) -> tuple[bytearray, memoryview]:
    raw = bytearray(size + alignment + 1)
    for offset in range(1, alignment + 1):
        view = memoryview(raw)[offset : offset + size]
        if _buffer_addr(view) % alignment != 0:
            return raw, view
    raise AssertionError("failed to create misaligned memoryview")


def _fill(view: memoryview) -> None:
    view[:] = bytes(i % 251 for i in range(len(view)))


def _wait_for_completion(
    client: Any,
    future_id: int,
    timeout: float = 5.0,
) -> tuple[int, bool, str, list[bool] | None]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for completion in client.drain_completions():
            if completion[0] == future_id:
                return completion

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        select.select([client.event_fd()], [], [], min(remaining, 0.1))

    raise TimeoutError(f"timed out waiting for future {future_id}")


def _submit_and_wait(
    client: Any,
    method_name: str,
    key: str,
    view: memoryview,
) -> tuple[int, bool, str, list[bool] | None]:
    future_id = getattr(client, method_name)([key], [view])
    return _wait_for_completion(client, future_id)


def test_odirect_read_does_not_split_for_read_ahead(tmp_path) -> None:
    """O_DIRECT reads should ignore read_ahead_size and use one aligned read."""
    if not hasattr(os, "O_DIRECT"):
        pytest.skip("O_DIRECT is not available on this platform")

    LMCacheFSClient = _import_fs_client()
    block_size = os.statvfs(tmp_path).f_bsize
    if block_size <= 0:
        pytest.skip("filesystem block size is unavailable")

    size = block_size * 2
    key = "test_model@00000000@0@0123456789abcdef"
    _source_raw, source = _aligned_memoryview(size, block_size)
    _dest_raw, dest = _aligned_memoryview(size, block_size)
    _fill(source)

    client = LMCacheFSClient(str(tmp_path), 1, "", True, 1)
    try:
        store = _submit_and_wait(client, "submit_batch_set", key, source)
        if not store[1]:
            pytest.skip(f"O_DIRECT is unsupported on this filesystem: {store[2]}")

        load = _submit_and_wait(client, "submit_batch_get", key, dest)
        assert load[1], load[2]
        assert bytes(dest) == bytes(source)
    finally:
        client.close()


def test_odirect_fails_for_misaligned_buffer(tmp_path) -> None:
    """Misaligned buffers should fail instead of hiding O_DIRECT misuse."""
    LMCacheFSClient = _import_fs_client()
    block_size = os.statvfs(tmp_path).f_bsize
    if block_size <= 0:
        pytest.skip("filesystem block size is unavailable")

    size = block_size * 2
    key = "test_model@00000000@0@fedcba9876543210"
    _source_raw, source = _misaligned_memoryview(size, block_size)
    _fill(source)

    client = LMCacheFSClient(str(tmp_path), 1, "", True, 1)
    try:
        store = _submit_and_wait(client, "submit_batch_set", key, source)
        assert not store[1]
        assert "O_DIRECT buffer address is not aligned" in store[2]
    finally:
        client.close()


@pytest.mark.parametrize("shape", [None, (2, 2), (1, 4)])
def test_typed_buffers_round_trip(
    tmp_path: Path, shape: tuple[int, ...] | None
) -> None:
    """Transfer every byte of contiguous typed and multidimensional views."""
    client = _import_fs_client()(str(tmp_path), 1)
    source = memoryview(array("I", [1, 256, 65536, 16777216]))
    destination = memoryview(array("I", [0] * 4))
    if shape is not None:
        source = source.cast("B").cast("I", shape)
        destination = destination.cast("B").cast("I", shape)
    key = "test_model@00000000@0@0123456789abcdef"
    try:
        store = _submit_and_wait(client, "submit_batch_set", key, source)
        assert store[1], store[2]
        load = _submit_and_wait(client, "submit_batch_get", key, destination)
        assert load[1], load[2]
        assert destination.tobytes() == source.tobytes()
        assert next(tmp_path.glob("*.data")).stat().st_size == source.nbytes
    finally:
        client.close()


@pytest.mark.parametrize("operation", ["submit_batch_get", "submit_batch_set"])
@pytest.mark.parametrize("step", [2, -1])
def test_strided_buffers_are_rejected(
    tmp_path: Path, operation: str, step: int
) -> None:
    """Reject a noncontiguous batch before submitting any of its buffers."""
    client = _import_fs_client()(str(tmp_path), 1)
    contiguous = memoryview(bytearray(b"abcdefgh"))
    strided = memoryview(bytearray(b"0123456789abcdef"))[::step]
    keys = ["model@00000000@0@00000001", "model@00000000@0@00000002"]
    try:
        with pytest.raises(ValueError, match="C-contiguous"):
            getattr(client, operation)(keys, [contiguous, strided])
    finally:
        client.close()
    assert contiguous.tobytes() == b"abcdefgh"
    assert not list(tmp_path.glob("*.data"))


def test_readonly_load_buffer_is_rejected(tmp_path: Path) -> None:
    """Reject immutable load destinations before modifying their bytes."""
    client = _import_fs_client()(str(tmp_path), 1)
    source = memoryview(b"abcdefgh")
    destination = memoryview(b"XXXXXXXX")
    key = "test_model@00000000@0@0123456789abcdef"
    try:
        store = _submit_and_wait(client, "submit_batch_set", key, source)
        assert store[1], store[2]
        with pytest.raises(BufferError):
            client.submit_batch_get([key], [destination])
        assert destination.tobytes() == b"XXXXXXXX"
    finally:
        client.close()


def test_empty_buffers_round_trip(tmp_path: Path) -> None:
    """Continue accepting contiguous zero-byte stores and loads."""
    client = _import_fs_client()(str(tmp_path), 1)
    key = "test_model@00000000@0@0123456789abcdef"
    source = memoryview(b"")
    destination = memoryview(bytearray())
    try:
        assert _submit_and_wait(client, "submit_batch_set", key, source)[1]
        assert _submit_and_wait(client, "submit_batch_get", key, destination)[1]
    finally:
        client.close()
