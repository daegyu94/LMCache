# SPDX-License-Identifier: Apache-2.0
"""Native storage buffer contracts, using real local filesystem I/O."""

# Standard
from array import array
from collections.abc import Iterator
from pathlib import Path
import select
import time

# Third Party
import pytest

# First Party
from lmcache.lmcache_fs import LMCacheFSClient


def wait_result(
    client: LMCacheFSClient, task_id: int
) -> tuple[int, bool, str, list[bool] | None]:
    """Wait for the submitted native filesystem operation."""
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        select.select([client.event_fd()], [], [], 0.1)
        results = client.drain_completions()
        if results:
            assert len(results) == 1 and results[0][0] == task_id
            return results[0]
    raise TimeoutError("Native operation timed out")


@pytest.fixture
def client(tmp_path: Path) -> Iterator[LMCacheFSClient]:
    instance = LMCacheFSClient(str(tmp_path), 1)
    yield instance
    instance.close()


@pytest.mark.parametrize("multidimensional", [False, True])
def test_formatted_buffers_transfer_all_bytes(
    client: LMCacheFSClient, multidimensional: bool
) -> None:
    source = array("I", [1, 2, 3, 4])
    destination = array("I", [0, 0, 0, 0])
    src, dst = memoryview(source), memoryview(destination)
    if multidimensional:
        src = src.cast("B").cast("I", shape=[2, 2])
        dst = dst.cast("B").cast("I", shape=[2, 2])
    key = "model@00000000@0@12345678"
    assert wait_result(client, client.submit_batch_set([key], [src]))[1]
    result = wait_result(client, client.submit_batch_get([key], [dst]))
    assert result[1] and result[3] == [True]
    assert source == destination


def test_get_rejects_readonly_destination(client: LMCacheFSClient) -> None:
    with pytest.raises(BufferError):
        client.submit_batch_get(["model@00000000@0@12345678"], [memoryview(b"abc")])


@pytest.mark.parametrize("operation", ["submit_batch_get", "submit_batch_set"])
def test_rejects_strided_buffer(client: LMCacheFSClient, operation: str) -> None:
    view = memoryview(bytearray(16))[::2]
    with pytest.raises(ValueError, match="C-contiguous"):
        getattr(client, operation)(["model@00000000@0@12345678"], [view])


def test_set_accepts_readonly_contiguous_source(client: LMCacheFSClient) -> None:
    source = memoryview(b"unchanged byte buffer")
    destination = memoryview(bytearray(source.nbytes))
    key = "model@00000000@0@12345678"
    assert wait_result(client, client.submit_batch_set([key], [source]))[1]
    result = wait_result(client, client.submit_batch_get([key], [destination]))
    assert result[1] and result[3] == [True]
    assert destination == source


def test_zero_length_contiguous_buffer(client: LMCacheFSClient) -> None:
    key = "model@00000000@0@12345678"
    source = memoryview(bytearray())
    assert wait_result(client, client.submit_batch_set([key], [source]))[1]
    result = wait_result(client, client.submit_batch_get([key], [source]))
    assert result[1] and result[3] == [True]
