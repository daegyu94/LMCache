# SPDX-License-Identifier: Apache-2.0
"""Integration tests for the native filesystem connector."""

# Standard
import os
import platform
import select

# Third Party
import pytest

lmcache_fs = pytest.importorskip("lmcache.lmcache_fs")
LMCacheFSClient = lmcache_fs.LMCacheFSClient


def _wait_for_completion(client, future_id: int, timeout_ms: int = 5000) -> None:
    poller = select.poll()
    poller.register(client.event_fd(), select.POLLIN)
    assert poller.poll(timeout_ms), f"native operation {future_id} timed out"

    completions = client.drain_completions()
    completion = next((item for item in completions if item[0] == future_id), None)
    assert completion is not None, f"completion {future_id} was not returned"
    assert completion[1], completion[2]


@pytest.mark.skipif(
    platform.system() != "Linux" or os.getenv("LMCACHE_RUN_ODIRECT_SMOKE") != "1",
    reason="O_DIRECT smoke is Linux-only and opt-in",
)
def test_odirect_handles_unaligned_buffer_and_length(tmp_path):
    client = LMCacheFSClient(str(tmp_path), 1, "", True, 0)
    try:
        for payload_size in (4097, 8192, 8193):
            payload = bytes((index % 251 for index in range(payload_size)))
            key = f"test-model@00000000@0@{payload_size}"

            write_storage = bytearray(payload_size + 1)
            write_storage[1:] = payload
            write_view = memoryview(write_storage)[1:]
            write_future = client.submit_batch_set([key], [write_view])
            _wait_for_completion(client, write_future)

            read_storage = bytearray(payload_size + 1)
            read_view = memoryview(read_storage)[1:]
            read_future = client.submit_batch_get([key], [read_view])
            _wait_for_completion(client, read_future)

            assert bytes(read_view) == payload
            data_path = tmp_path / f"test-model@0x00000000@0@{payload_size}.data"
            assert data_path.stat().st_size >= payload_size

        stats = client.get_io_stats()
        assert stats["write_ops"] == 3
        assert stats["read_ops"] == 3
        assert stats["write_direct_ops"] == 3
        assert stats["read_direct_ops"] == 3
        assert stats["write_buffered_ops"] == 0
        assert stats["read_buffered_ops"] == 0
        assert stats["write_bytes"] == 4097 + 8192 + 8193
        assert stats["read_bytes"] == 4097 + 8192 + 8193
        assert stats["write_direct_bytes"] >= stats["write_bytes"]
        assert stats["read_direct_bytes"] >= stats["read_bytes"]
        assert stats["write_errors"] == 0
        assert stats["read_errors"] == 0
    finally:
        client.close()
