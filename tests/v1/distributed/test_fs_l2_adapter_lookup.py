# SPDX-License-Identifier: Apache-2.0
"""FSL2Adapter lookup: one batched existence check per lookup task."""

# Standard
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import TypeVar, cast
import time

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.l2_adapters.fs_l2_adapter import (
    FSL2Adapter,
    FSL2AdapterConfig,
)
from lmcache.v1.memory_management import MemoryObj

_LAYOUT = {0: MemoryLayoutDesc(shapes=[], dtypes=[])}


class _Buf:
    """Minimal MemoryObj stand-in for a store."""

    def __init__(self, data: bytes) -> None:
        self.byte_array = memoryview(bytearray(data))


def _key(i: int) -> ObjectKey:
    return ObjectKey(chunk_hash=ObjectKey.IntHash2Bytes(i), model_name="m", kv_rank=0)


_T = TypeVar("_T")


def _wait(poll: Callable[[], _T | None], what: str) -> _T:
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        result = poll()
        if result is not None:
            return result
        time.sleep(0.01)
    raise AssertionError(f"{what} did not complete within 5s")


@pytest.fixture
def adapter(tmp_path: Path) -> Iterator[FSL2Adapter]:
    adp = FSL2Adapter(FSL2AdapterConfig(base_path=str(tmp_path)))
    try:
        yield adp
    finally:
        adp.close()


def _store(adp: FSL2Adapter, keys: list[ObjectKey]) -> None:
    bufs = [_Buf(b"x" * 64) for _ in keys]
    task_id = adp.submit_store_task(keys, cast("list[MemoryObj]", bufs))
    _wait(lambda: adp.pop_completed_store_tasks().get(task_id), "store")


def _lookup(adp: FSL2Adapter, keys: list[ObjectKey]) -> list[bool]:
    task_id = adp.submit_lookup_and_lock_task(keys, _LAYOUT)
    bitmap = _wait(lambda: adp.query_lookup_and_lock_result(task_id), "lookup")
    return [bitmap.test(i) for i in range(len(keys))]


def test_lookup_reports_each_key_in_order(adapter: FSL2Adapter) -> None:
    keys = [_key(i) for i in range(40)]
    stored = keys[::3]
    _store(adapter, stored)

    assert _lookup(adapter, keys) == [k in stored for k in keys]


def test_lookup_reflects_files_removed_behind_the_adapter(
    adapter: FSL2Adapter, tmp_path: Path
) -> None:
    """No caching: a file another node deleted is a miss on the next lookup."""
    keys = [_key(i) for i in range(4)]
    _store(adapter, keys)
    assert _lookup(adapter, keys) == [True] * 4

    for path in tmp_path.iterdir():
        if path.is_file():
            path.unlink()

    assert _lookup(adapter, keys) == [False] * 4


def test_empty_lookup_completes(adapter: FSL2Adapter) -> None:
    assert _lookup(adapter, []) == []
