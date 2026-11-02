# SPDX-License-Identifier: Apache-2.0
"""FSL2Adapter loads an object only when its file has the buffer's exact size.

Keys do not encode the KV dtype or layout, so a cache directory reused across
configurations can hold an object of another size under the same key.
"""

# Standard
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import TypeVar, cast
import ctypes
import time

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.l2_adapters.fs_l2_adapter import (
    FSL2Adapter,
    FSL2AdapterConfig,
)
from lmcache.v1.memory_management import MemoryObj

_ALIGN = 4096
_T = TypeVar("_T")


class _Buf:
    """MemoryObj stand-in whose ``byte_array`` is 4096-aligned for O_DIRECT."""

    def __init__(self, data: bytes) -> None:
        self._raw = bytearray(len(data) + _ALIGN)
        base = ctypes.addressof((ctypes.c_byte * 1).from_buffer(self._raw))
        start = (-base) % _ALIGN
        self._view = memoryview(self._raw)[start : start + len(data)]
        self._view[:] = data

    @property
    def byte_array(self) -> memoryview:
        return self._view


def _wait(poll: Callable[[], _T | None], what: str) -> _T:
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        result = poll()
        if result is not None:
            return result
        time.sleep(0.01)
    raise AssertionError(f"{what} did not complete within 5s")


@pytest.fixture(params=[False, True], ids=["buffered", "odirect"])
def adapter(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[FSL2Adapter]:
    adp = FSL2Adapter(
        FSL2AdapterConfig(base_path=str(tmp_path), use_odirect=request.param)
    )
    try:
        yield adp
    finally:
        adp.close()


@pytest.mark.parametrize("stored_blocks", [3, 5])
def test_object_of_another_size_is_not_loaded(
    adapter: FSL2Adapter, stored_blocks: int
) -> None:
    """A larger or smaller object is a miss; a larger one must not be served
    by its prefix."""
    key = ObjectKey(chunk_hash=b"\xde\xad\xbe\xef", model_name="llama", kv_rank=0)
    stored = _Buf(bytes(range(256)) * (stored_blocks * _ALIGN // 256))
    task_id = adapter.submit_store_task([key], cast("list[MemoryObj]", [stored]))
    assert _wait(lambda: adapter.pop_completed_store_tasks().get(task_id), "store")

    dest = _Buf(b"\x00" * (4 * _ALIGN))
    task_id = adapter.submit_load_task([key], cast("list[MemoryObj]", [dest]))
    bitmap = _wait(lambda: adapter.query_load_result(task_id), "load")
    assert not bitmap.test(0)


def test_object_of_the_same_size_is_loaded(adapter: FSL2Adapter) -> None:
    """The size check does not reject an exact match."""
    key = ObjectKey(chunk_hash=b"\xca\xfe", model_name="llama", kv_rank=0)
    payload = bytes(range(256)) * (4 * _ALIGN // 256)
    task_id = adapter.submit_store_task([key], cast("list[MemoryObj]", [_Buf(payload)]))
    assert _wait(lambda: adapter.pop_completed_store_tasks().get(task_id), "store")

    dest = _Buf(b"\x00" * len(payload))
    task_id = adapter.submit_load_task([key], cast("list[MemoryObj]", [dest]))
    bitmap = _wait(lambda: adapter.query_load_result(task_id), "load")
    assert bitmap.test(0)
    assert bytes(dest.byte_array) == payload
