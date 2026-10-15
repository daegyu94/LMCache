# SPDX-License-Identifier: Apache-2.0
"""Tests for the native C++ filesystem connector."""

# Standard
from concurrent.futures import ThreadPoolExecutor
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


def _collect_completions(
    client: Any,
    future_ids: set[int],
    timeout: float = 10.0,
) -> dict[int, tuple[int, bool, str, list[bool] | None]]:
    """Collect each expected future exactly once through the public eventfd.

    Args:
        client: Native connector exposing event_fd and drain_completions.
        future_ids: The complete set of submitted future IDs.
        timeout: Maximum time to wait for all completions, in seconds.

    Returns:
        Completions indexed by their future IDs.
    """
    completed: dict[int, tuple[int, bool, str, list[bool] | None]] = {}
    deadline = time.monotonic() + timeout
    while len(completed) < len(future_ids):
        remaining = deadline - time.monotonic()
        assert remaining > 0, f"missing completions: {future_ids - completed.keys()}"
        assert select.select([client.event_fd()], [], [], remaining)[0]
        for completion in client.drain_completions():
            future_id = completion[0]
            assert future_id in future_ids
            assert future_id not in completed
            completed[future_id] = completion
    assert client.drain_completions() == []
    return completed


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


def test_completion_bursts_rearm_eventfd(tmp_path: Path) -> None:
    """Every lookup completes once and a later burst wakes the consumer again."""
    client = _import_fs_client()(str(tmp_path), 4)
    try:
        assert client.drain_completions() == []
        for _ in range(3):
            future_ids = {
                client.submit_batch_exists([f"model@00000000@0@{i:016x}"])
                for i in range(128)
            }
            completed = _collect_completions(client, future_ids)
            assert len(completed) == 128
            assert all(c[1:] == (True, "", [False]) for c in completed.values())
    finally:
        client.close()


def test_mixed_completion_results_survive_draining(tmp_path: Path) -> None:
    """Draining preserves IDs, errors, per-key results, and loaded payloads."""
    client = _import_fs_client()(str(tmp_path), 4)
    source = memoryview(bytearray(b"cache payload"))
    loaded = [memoryview(bytearray(len(source))) for _ in range(64)]
    store_ids = set()
    lookup_ids = set()
    load_ids = set()
    delete_ids = set()
    try:
        for i in range(64):
            key = f"model@00000000@0@{i:016x}"
            (tmp_path / f"model@0x00000000@0@{i:016x}.data").write_bytes(source)
            delete_key = f"delete@00000000@0@{i:016x}"
            (tmp_path / f"delete@0x00000000@0@{i:016x}.data").write_bytes(source)
            store_ids.add(
                client.submit_batch_set([f"store@00000000@0@{i:016x}"], [source])
            )
            lookup_ids.add(client.submit_batch_exists([key, "missing@0@0@0"]))
            load_ids.add(client.submit_batch_get([key], [loaded[i]]))
            delete_ids.add(client.submit_batch_delete([delete_key]))

        failed_id = client.submit_batch_set(["malformed"], [source])
        missing_id = client.submit_batch_get(["missing@0@0@0"], [loaded[0]])
        completed = _collect_completions(
            client,
            store_ids | lookup_ids | load_ids | delete_ids | {failed_id, missing_id},
        )
        assert all(completed[fid][1:] == (True, "", None) for fid in store_ids)
        assert all(
            completed[fid][1:] == (True, "", [True, False]) for fid in lookup_ids
        )
        assert all(
            completed[fid][1:] == (True, "", [True]) for fid in load_ids | delete_ids
        )
        assert not completed[failed_id][1]
        assert "malformed key" in completed[failed_id][2]
        assert completed[missing_id][1:] == (True, "", [False])
        assert all(bytes(view) == bytes(source) for view in loaded)
    finally:
        client.close()


def test_completion_producer_races_eventfd_draining(tmp_path: Path) -> None:
    """Concurrent submission and polling deliver every completion exactly once."""
    client = _import_fs_client()(str(tmp_path), 4)
    key = "model@00000000@0@0123456789abcdef"
    (tmp_path / "model@0x00000000@0@0123456789abcdef.data").write_bytes(b"cache")
    completed: dict[int, tuple[int, bool, str, list[bool] | None]] = {}

    def submit_lookups() -> set[int]:
        return {client.submit_batch_exists([key]) for _ in range(1024)}

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            submitted = executor.submit(submit_lookups)
            deadline = time.monotonic() + 10
            while len(completed) < 1024:
                remaining = deadline - time.monotonic()
                assert remaining > 0, f"received {len(completed)} completions"
                assert select.select([client.event_fd()], [], [], remaining)[0]
                for completion in client.drain_completions():
                    assert completion[0] not in completed
                    assert completion[1:] == (True, "", [True])
                    completed[completion[0]] = completion
            assert set(completed) == submitted.result()
        assert client.drain_completions() == []
        next_id = client.submit_batch_exists([key])
        assert _collect_completions(client, {next_id})[next_id][1:] == (
            True,
            "",
            [True],
        )
    finally:
        client.close()
