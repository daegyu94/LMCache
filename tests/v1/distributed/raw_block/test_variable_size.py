# SPDX-License-Identifier: Apache-2.0
"""Per-object space allocation in RawBlockCore, on temp-file devices."""

# Standard
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
import importlib.util
import json
import os
import random
import struct
import threading
import zlib

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.l2_adapters.raw_block import (
    RawBlockCore,
    encode_object_key,
)
from lmcache.v1.distributed.l2_adapters.raw_block.io_engine import (
    IoUringBlockIO,
    PosixBlockIO,
)
from tests.v1.distributed.raw_block.test_utils import (
    RAW_BLOCK_CI_BLOCK_ALIGN,
    RAW_BLOCK_CI_HEADER_BYTES,
    RAW_BLOCK_CI_META_TOTAL_BYTES,
    make_aligned_memory_obj,
    make_empty_memory_obj,
    make_memory_obj,
    make_object_key,
    make_raw_block_core_config,
    make_raw_block_file,
    memory_obj_bytes,
)

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("lmcache_rust_raw_block_io") is None,
    reason="lmcache_rust_raw_block_io is not installed",
)

ALIGN = RAW_BLOCK_CI_BLOCK_ALIGN
HEADER = RAW_BLOCK_CI_HEADER_BYTES
_META_HEADER = struct.Struct("<8sIQQI")

# (io_engine, use_odirect, enable_zero_copy)
ENGINES = [
    ("posix", False, False),
    ("posix", True, True),
    ("io_uring", False, False),
    ("io_uring", True, False),
    ("io_uring", True, True),
]
ENGINE_IDS = ["posix", "posix-odirect", "uring", "uring-odirect", "uring-zc"]


def _allocated(core: RawBlockCore, encoded_keys) -> list[int]:
    """Device bytes each key holds, 0 when it is not indexed."""
    held = dict(core.snapshot_indexed_allocations())
    return [held.get(k, 0) for k in encoded_keys]


def _need(payload_len: int) -> int:
    return (HEADER + payload_len + ALIGN - 1) // ALIGN * ALIGN


def _spec(i: int):
    return encode_object_key(make_object_key(i))


def _payload(i: int, size: int) -> bytes:
    return random.Random(i).randbytes(size)


def _config(tmp_path: Path, engine=ENGINES[0], data_bytes: int = 64 << 20, **kw):
    io_engine, odirect, zero_copy = engine
    path = make_raw_block_file(tmp_path, RAW_BLOCK_CI_META_TOTAL_BYTES + data_bytes)
    return replace(
        make_raw_block_core_config(path, RAW_BLOCK_CI_META_TOTAL_BYTES + data_bytes),
        io_engine=io_engine,
        use_odirect=odirect,
        enable_zero_copy=zero_copy,
        **kw,
    )


def _obj(payload: bytes, aligned: bool):
    return make_aligned_memory_obj(payload) if aligned else make_memory_obj(payload)


def _put(core: RawBlockCore, i: int, payload: bytes, aligned: bool = False) -> bool:
    return core.put_many([_spec(i)], [_obj(payload, aligned)]).results == [True]


def _load(core: RawBlockCore, i: int, size: int) -> bytes | None:
    obj = make_empty_memory_obj(size)
    if core.load_many_into([_spec(i).encoded], [obj]) != [True]:
        return None
    return memory_obj_bytes(obj)


def _fragment(core: RawBlockCore, count: int, payload_len: int) -> list[int]:
    """Fill with ``count`` objects and delete every other one.

    Returns:
        Keys still stored. Free space is ``count // 2`` holes of one object.
    """
    for i in range(count):
        assert _put(core, i, _payload(i, payload_len))
    assert core.delete_many([_spec(i).encoded for i in range(0, count, 2)]) == [
        True
    ] * len(range(0, count, 2))
    return list(range(1, count, 2))


def _check_accounting(core: RawBlockCore) -> None:
    """With no write in flight, the index's objects (summed per key) must be
    exactly the space the allocator does not have free: a leak or double free
    breaks it."""
    status = core.report_status()
    assert status["inflight_key_count"] == 0
    held = sum(_allocated(core, core.snapshot_indexed_keys()))
    assert held == status["usable_capacity_bytes"] - status["free_bytes"]
    assert held == status["allocated_bytes"]


def _first_object_offset(core: RawBlockCore) -> int:
    """Objects start after the data region's format-marker block."""
    return core.data_base_offset() + ALIGN


@pytest.mark.parametrize("engine", ENGINES, ids=ENGINE_IDS)
def test_objects_of_different_sizes_round_trip(tmp_path, engine):
    core = RawBlockCore(_config(tmp_path, engine))
    sizes = [1, 7, ALIGN - 1, ALIGN, ALIGN + 1, 3 * ALIGN + 17, 1 << 20, (2 << 20) + 5]
    try:
        for i, size in enumerate(sizes):
            assert _put(core, i, _payload(i, size), aligned=engine[2])
            assert _allocated(core, [_spec(i).encoded]) == [_need(size)]
        for i, size in enumerate(sizes):
            assert _load(core, i, size) == _payload(i, size)
        _check_accounting(core)
        status = core.report_status()
        assert status["allocated_bytes"] == sum(map(_need, sizes))
        fraction = status["allocated_bytes"] / status["usable_capacity_bytes"]
        assert core.usage() == (fraction, fraction)
    finally:
        core.close()


@pytest.mark.parametrize("engine", ENGINES, ids=ENGINE_IDS)
def test_fragmented_space_fails_cleanly_until_a_hole_fits(tmp_path, engine):
    payload_len = 3 * ALIGN  # 16 KiB objects
    count = 16
    core = RawBlockCore(
        _config(tmp_path, engine, data_bytes=ALIGN + count * _need(payload_len))
    )
    try:
        kept = _fragment(core, count, payload_len)
        before = core.report_status()
        assert before["largest_free_extent_bytes"] == _need(payload_len)
        assert before["free_bytes"] == count // 2 * _need(payload_len)
        # Larger than any hole, smaller than their sum.
        assert not _put(core, 100, _payload(100, payload_len + 1), aligned=engine[2])
        after = core.report_status()
        assert after["free_bytes"] == before["free_bytes"]
        assert after["alloc_failure_count"] == before["alloc_failure_count"] + 1
        assert after["inflight_key_count"] == 0
        # A smaller object takes the head of a hole.
        assert _put(core, 101, _payload(101, 5), aligned=engine[2])
        offset = core.entry_offset(_spec(101).encoded)
        assert offset is not None and offset % ALIGN == 0
        assert _load(core, 101, 5) == _payload(101, 5)
        for i in kept:
            assert _load(core, i, payload_len) == _payload(i, payload_len)
        _check_accounting(core)
    finally:
        core.close()


def test_a_store_that_finds_no_long_enough_range_reports_the_device_full(tmp_path):
    """Free space split into short ranges can sit below the eviction
    watermark; the core must say so until space is freed."""
    payload_len = 3 * ALIGN
    count = 16
    core = RawBlockCore(
        _config(tmp_path, data_bytes=ALIGN + count * _need(payload_len))
    )
    try:
        kept = _fragment(core, count, payload_len)
        assert core.space_exhausted_count() == 0
        assert not _put(core, 100, _payload(100, 2 * _need(payload_len)))
        assert core.space_exhausted_count() == 1
        assert core.report_status()["space_exhausted_count"] == 1
        assert core.delete_many([_spec(kept[0]).encoded]) == [True]
        assert core.space_exhausted_count() == 1
        # An object larger than the device never fits: evicting cannot help.
        assert not _put(core, 101, b"x" * (count * _need(payload_len)))
        assert core.space_exhausted_count() == 1
    finally:
        core.close()


def test_variable_size_entries_survive_restart(tmp_path):
    payload_len = 3 * ALIGN
    count = 16
    config = _config(tmp_path, data_bytes=ALIGN + count * _need(payload_len))
    small = 100
    core = RawBlockCore(config)
    try:
        kept = _fragment(core, count, payload_len)
        assert _put(core, 100, _payload(100, small))
        offsets = {k: core.entry_offset(_spec(k).encoded) for k in [*kept, 100]}
    finally:
        core.close()

    reopened = RawBlockCore(config)
    try:
        for k, offset in offsets.items():
            assert reopened.entry_offset(_spec(k).encoded) == offset
        assert _load(reopened, 100, small) == _payload(100, small)
        for i in kept:
            assert _load(reopened, i, payload_len) == _payload(i, payload_len)
        status = reopened.report_status()
        assert status["free_bytes"] == count // 2 * _need(payload_len) - _need(small)
        _check_accounting(reopened)
    finally:
        reopened.close()


def test_entries_dropped_by_header_validation_free_their_space(tmp_path):
    config = _config(tmp_path)
    core = RawBlockCore(config)
    try:
        assert _put(core, 1, _payload(1, 3 * ALIGN))
        assert _put(core, 2, _payload(2, 100))
        stale = core.entry_offset(_spec(1).encoded)
    finally:
        core.close()
    with open(config.device_path, "r+b") as f:
        f.seek(stale)
        f.write(b"\0" * 8)  # key 1's header no longer matches

    reopened = RawBlockCore(config)
    try:
        assert reopened.snapshot_indexed_keys() == [_spec(2).encoded]
        _check_accounting(reopened)
        status = reopened.report_status()
        assert status["free_bytes"] == status["usable_capacity_bytes"] - _need(100)
    finally:
        reopened.close()


def test_zero_copy_padding_is_zeroed_only_after_the_payload(tmp_path):
    """The aligned L1 allocation's tail is zeroed in place, so bytes of the
    object's last block past the payload reach the device as zeros."""
    engine = ("io_uring", True, True)
    config = _config(tmp_path, engine)
    size = 3 * ALIGN - 100  # 100 bytes of padding
    core = RawBlockCore(config)
    try:
        source = make_aligned_memory_obj(_payload(1, size), tail_fill=0xA5)
        assert core.put_many([_spec(1)], [source]).results == [True]
        offset = core.entry_offset(_spec(1).encoded)
        assert offset is not None
        assert _load(core, 1, size) == _payload(1, size)
    finally:
        core.close()
    with open(config.device_path, "rb") as f:
        f.seek(offset + _need(size) - 100)
        assert f.read(100) == b"\0" * 100


def test_unaligned_source_buffer_is_bounced(tmp_path):
    engine = ("io_uring", True, True)
    core = RawBlockCore(_config(tmp_path, engine))
    try:
        size = 2 * ALIGN - 3
        obj = make_memory_obj(_payload(1, size))  # bytearray: not block-aligned
        assert core.put_many([_spec(1)], [obj]).results == [True]
        assert _load(core, 1, size) == _payload(1, size)
    finally:
        core.close()


def test_load_into_a_short_destination_fails_without_touching_others(tmp_path):
    core = RawBlockCore(_config(tmp_path))
    try:
        assert _put(core, 1, b"a" * 9000)
        assert _put(core, 2, b"b" * 10)
        short = make_empty_memory_obj(8999)
        ok = make_empty_memory_obj(10)
        assert core.load_many_into(
            [_spec(1).encoded, _spec(2).encoded], [short, ok]
        ) == [False, True]
        assert memory_obj_bytes(ok) == b"b" * 10
    finally:
        core.close()


def test_delete_reports_the_bytes_each_key_held(tmp_path):
    core = RawBlockCore(_config(tmp_path))
    try:
        assert _put(core, 1, b"a" * 9000)
        assert _put(core, 2, b"b" * 10)
        assert core.exists_many([_spec(2).encoded], lock=True) == [True]
        assert core.delete_many_with_sizes(
            [_spec(1).encoded, _spec(2).encoded, _spec(3).encoded]
        ) == [_need(9000), None, None]
        assert core.delete_many_with_sizes([_spec(2).encoded], force=True) == [
            _need(10)
        ]
        _check_accounting(core)
        assert core.report_status()["allocated_bytes"] == 0
    finally:
        core.close()


def test_a_delete_writes_nothing(tmp_path):
    """A deleted object is left as it is on the device: a restart that still
    finds it in its checkpoint gets its own bytes back."""
    config = _config(tmp_path)
    old = _payload(1, 3 * ALIGN)
    core = RawBlockCore(config)
    try:
        assert _put(core, 1, old)
        core.checkpoint_now()  # the device lists key 1
        with patch.object(PosixBlockIO, "write_buffers") as write:
            assert core.delete_many([_spec(1).encoded]) == [True]
        write.assert_not_called()
        assert core.report_status()["reuse_hazard_count"] == 1
        crashed = RawBlockCore(replace(config, meta_enable_periodic=False))
        try:
            assert _load(crashed, 1, len(old)) == old
        finally:
            crashed.close()
    finally:
        core.close()


def test_an_object_deleted_before_any_checkpoint_leaves_no_hazard(tmp_path):
    core = RawBlockCore(replace(_config(tmp_path), meta_enable_periodic=False))
    try:
        assert _put(core, 1, _payload(1, 3 * ALIGN))
        assert core.delete_many([_spec(1).encoded]) == [True]
        assert core.report_status()["reuse_hazard_count"] == 0
    finally:
        core.close()


def test_hazards_end_once_both_checkpoint_copies_are_rewritten(tmp_path):
    core = RawBlockCore(replace(_config(tmp_path), meta_enable_periodic=False))
    try:
        assert _put(core, 1, _payload(1, 3 * ALIGN))
        assert _put(core, 2, _payload(2, 3 * ALIGN))
        core.checkpoint_now()  # one copy lists keys 1 and 2
        assert core.delete_many([_spec(1).encoded]) == [True]
        assert core.report_status()["reuse_hazard_count"] == 1
        # Stores too long for the freed space give each checkpoint something
        # to write without overwriting the deleted objects.
        assert _put(core, 3, _payload(3, 8 * ALIGN))
        core.checkpoint_now()  # the other copy, which never listed key 1
        assert core.report_status()["reuse_hazard_count"] == 1
        assert _put(core, 4, _payload(4, 8 * ALIGN))
        core.checkpoint_now()  # rewrites the copy that listed key 1
        assert core.report_status()["reuse_hazard_count"] == 0
        assert core.delete_many([_spec(2).encoded]) == [True]  # in both copies
        assert _put(core, 5, _payload(5, 8 * ALIGN))
        core.checkpoint_now()
        assert core.report_status()["reuse_hazard_count"] == 1
        assert _put(core, 6, _payload(6, 8 * ALIGN))
        core.checkpoint_now()
        assert core.report_status()["reuse_hazard_count"] == 0
    finally:
        core.close()


def test_an_object_deleted_while_a_checkpoint_lists_it_stays_guarded(tmp_path):
    """The checkpoint snapshot lists the key; it is deleted before that
    checkpoint reaches the device, which then lists it after all."""
    config = replace(_config(tmp_path), meta_enable_periodic=False)
    old = _payload(1, 5 * ALIGN)
    core = RawBlockCore(config)
    real_write = PosixBlockIO.write_buffers
    deleted: list[int] = []
    meta_end = core.data_base_offset()

    def delete_during_checkpoint(self, offsets, *args, **kwargs):
        if offsets[0] < meta_end and not deleted:
            deleted.append(1)
            assert core.delete_many([_spec(1).encoded]) == [True]
        return real_write(self, offsets, *args, **kwargs)

    try:
        assert _put(core, 1, old)
        offset = core.entry_offset(_spec(1).encoded)
        with patch.object(PosixBlockIO, "write_buffers", delete_during_checkpoint):
            core.checkpoint_now()
        assert deleted
        assert core.report_status()["reuse_hazard_count"] == 1
        _reuse_with_a_delayed_head(core, config, offset, 1, old)
    finally:
        core.close()


def test_an_object_loaded_at_startup_and_deleted_stays_guarded(tmp_path):
    """Both checkpoint copies may list what a restart loads."""
    config = replace(_config(tmp_path), meta_enable_periodic=False)
    old = _payload(1, 5 * ALIGN)
    first = RawBlockCore(config)
    try:
        assert _put(first, 1, old)
        offset = first.entry_offset(_spec(1).encoded)
        assert _put(first, 3, b"after")  # bounds the space key 1 frees
    finally:
        first.close()
    core = RawBlockCore(config)
    try:
        assert core.delete_many([_spec(1).encoded]) == [True]
        # Too long for the freed space, so it does not overwrite key 1.
        assert _put(core, 2, _payload(2, 8 * ALIGN))
        core.checkpoint_now()  # one copy no longer lists it
        assert core.report_status()["reuse_hazard_count"] == 1
        _reuse_with_a_delayed_head(core, config, offset, 1, old)
    finally:
        core.close()


def test_a_failed_write_over_a_deleted_object_keeps_it_guarded(tmp_path):
    config = replace(_config(tmp_path), meta_enable_periodic=False)
    old = _payload(1, 5 * ALIGN)
    core = RawBlockCore(config)
    real_write = PosixBlockIO.write_buffers

    def failing(self, offsets, *args, **kwargs):
        if offsets[0] == offset:
            raise OSError("injected")
        return real_write(self, offsets, *args, **kwargs)

    try:
        assert _put(core, 1, old)
        offset = core.entry_offset(_spec(1).encoded)
        core.checkpoint_now()
        assert core.delete_many([_spec(1).encoded]) == [True]
        with patch.object(PosixBlockIO, "write_buffers", failing):
            assert not _put(core, 2, _payload(2, ALIGN))
        assert core.report_status()["reuse_hazard_count"] == 1
        _check_accounting(core)
        _reuse_with_a_delayed_head(core, config, offset, 1, old)
    finally:
        core.close()


@pytest.mark.parametrize("batched", [False, True])
def test_io_uring_writes_the_header_over_a_deleted_object_first(tmp_path, batched):
    """io_uring requests of one batch land in any order: a store whose header
    covers a deleted object's first byte must not submit its payload, which
    overwrites the rest of that object, before the header is written."""
    config = replace(
        _config(tmp_path, ("io_uring", False, False)), meta_enable_periodic=False
    )
    core = RawBlockCore(config)
    real_write = IoUringBlockIO.write_buffers
    calls: list[list[int]] = []

    def record(self, offsets, *args, **kwargs):
        calls.append(list(offsets))
        return real_write(self, offsets, *args, **kwargs)

    try:
        assert _put(core, 1, _payload(1, 5 * ALIGN))
        offset = core.entry_offset(_spec(1).encoded)
        core.checkpoint_now()
        assert core.delete_many([_spec(1).encoded]) == [True]
        with patch.object(IoUringBlockIO, "write_buffers", record):
            if batched:
                result = core.put_many(
                    [_spec(2), _spec(3)],
                    [_obj(_payload(2, ALIGN), False), _obj(_payload(3, ALIGN), False)],
                )
                assert result.results == [True, True]
            else:
                assert _put(core, 2, _payload(2, ALIGN))
        assert core.entry_offset(_spec(2).encoded) == offset
        header_call = next(i for i, c in enumerate(calls) if offset in c)
        payload_call = next(i for i, c in enumerate(calls) if offset + HEADER in c)
        assert header_call < payload_call
        assert core.report_status()["reuse_hazard_count"] == 0
    finally:
        core.close()


def test_reuse_before_the_head_write_lands_cannot_resurrect_a_deleted_object(
    tmp_path,
):
    """Deleted space is reused by two stores: the first, at the deleted
    object's start, has not reached the device when the second, inside its old
    payload, has. A crash then leaves the old header intact unless the delete
    invalidated it, and recovery would return the second store's bytes for
    the deleted key."""
    config = _config(tmp_path)
    old = _payload(1, 5 * ALIGN)
    core = RawBlockCore(config)
    real_write = PosixBlockIO.write_buffers
    head_started, release_head = threading.Event(), threading.Event()
    head_offset: list[int] = []

    def write_buffers(self, offsets, *args, **kwargs):
        if len(offsets) > 1 and head_offset and offsets[0] == head_offset[0]:
            head_started.set()
            assert release_head.wait(10)
        return real_write(self, offsets, *args, **kwargs)

    threads: list[threading.Thread] = []
    try:
        assert _put(core, 1, old)
        head_offset.append(core.entry_offset(_spec(1).encoded))
        core.checkpoint_now()  # the device lists key 1
        assert core.delete_many([_spec(1).encoded]) == [True]
        with patch.object(PosixBlockIO, "write_buffers", write_buffers):
            head = threading.Thread(target=_put, args=(core, 2, _payload(2, ALIGN)))
            head.start()
            threads.append(head)
            assert head_started.wait(10)
            assert _put(core, 3, _payload(3, 2 * ALIGN))
            # Held off the deleted object until the head write lands.
            third = core.entry_offset(_spec(3).encoded)
            assert not head_offset[0] <= third < head_offset[0] + _need(len(old))
            crashed = RawBlockCore(replace(config, meta_enable_periodic=False))
            try:
                assert _load(crashed, 1, len(old)) in (None, old)
            finally:
                crashed.close()
            release_head.set()
            head.join(10)
    finally:
        release_head.set()
        for t in threads:
            t.join(10)
        core.close()


def _reuse_with_a_delayed_head(core, config, head_offset, old_key, old):
    """Reuse the space at ``head_offset`` with two stores: the first, at
    ``head_offset``, is held before its write; the second, behind it, lands.
    Then check what a crash at that point recovers for ``old_key``."""
    real_write = PosixBlockIO.write_buffers
    head_started, release = threading.Event(), threading.Event()

    def write_buffers(self, offsets, *args, **kwargs):
        if len(offsets) > 1 and offsets[0] == head_offset:
            head_started.set()
            assert release.wait(10)
        return real_write(self, offsets, *args, **kwargs)

    with patch.object(PosixBlockIO, "write_buffers", write_buffers):
        head = threading.Thread(target=_put, args=(core, 9001, _payload(9001, ALIGN)))
        head.start()
        try:
            assert head_started.wait(10)
            assert _put(core, 9002, _payload(9002, 2 * ALIGN))
            crashed = RawBlockCore(replace(config, meta_enable_periodic=False))
            try:
                assert _load(crashed, old_key, len(old)) in (None, old)
            finally:
                crashed.close()
        finally:
            release.set()
            head.join(10)


def test_entry_whose_header_cannot_be_read_at_startup_keeps_its_space(tmp_path):
    """A header read that fails says nothing about the header on the device,
    which may still be valid: freeing the space would let a crash bring the
    key back over reused bytes."""
    config = replace(_config(tmp_path), meta_enable_periodic=False)
    old = _payload(1, 5 * ALIGN)
    core = RawBlockCore(config)
    try:
        assert _put(core, 1, old)
        offset = core.entry_offset(_spec(1).encoded)
    finally:
        core.close()  # the checkpoint lists key 1

    real_read = PosixBlockIO.read_buffers
    failed: list[int] = []

    def flaky_read(self, offsets, *args, **kwargs):
        if list(offsets) == [offset] and not failed:
            failed.append(1)
            return [False]
        return real_read(self, offsets, *args, **kwargs)

    with patch.object(PosixBlockIO, "read_buffers", flaky_read):
        run = RawBlockCore(config)
    try:
        assert run.indexed_key_count() == 0
        assert run.report_status()["quarantined_bytes"] == _need(len(old))
        _check_accounting(run)
        assert _put(run, 9001, _payload(9001, ALIGN))
        new = run.entry_offset(_spec(9001).encoded)
        assert not offset <= new < offset + _need(len(old))
    finally:
        run.close()


def test_failed_rewrite_of_a_key_in_its_old_place_cannot_resurrect_it(tmp_path):
    """A re-stored key lands where it was (best fit) and the checkpoint still
    lists it there; if the write fails after its header, freeing the space
    without invalidating that header lets a crash bring the key back over
    reused bytes."""
    config = replace(_config(tmp_path), meta_enable_periodic=False)
    old = _payload(1, 5 * ALIGN)
    core = RawBlockCore(config)
    real_write = PosixBlockIO.write_buffers

    def header_then_fail(self, offsets, buffers, *args, **kwargs):
        if len(offsets) == 2 and offsets[0] == offset:
            real_write(self, offsets[:1], buffers[:1], *[a[:1] for a in args], **kwargs)
            raise OSError("injected payload write failure")
        return real_write(self, offsets, buffers, *args, **kwargs)

    try:
        assert _put(core, 1, old)
        offset = core.entry_offset(_spec(1).encoded)
        core.checkpoint_now()  # lists key 1 at offset
        assert core.delete_many([_spec(1).encoded]) == [True]
        with patch.object(PosixBlockIO, "write_buffers", header_then_fail):
            assert not _put(core, 1, old)
        _check_accounting(core)
        _reuse_with_a_delayed_head(core, config, offset, 1, old)
    finally:
        core.close()


@pytest.mark.parametrize("engine", ENGINES, ids=ENGINE_IDS)
def test_reusing_many_deleted_objects_never_resurrects_them(tmp_path, engine):
    """Deleted objects listed by the checkpoint are reused by objects of
    another size, so new objects start inside old ones; a restart that still
    finds the old keys must get their own bytes or nothing."""
    config = replace(_config(tmp_path, engine), meta_enable_periodic=False)
    keys = list(range(100))
    old = {k: _payload(k, 3 * ALIGN) for k in keys}
    core = RawBlockCore(config)
    try:
        for k in keys:
            assert _put(core, k, old[k], aligned=engine[2])
        core.checkpoint_now()
        assert core.delete_many([_spec(k).encoded for k in keys]) == [True] * 100
        assert core.report_status()["reuse_hazard_count"] == 100
        for k in keys:
            assert _put(
                core, 1000 + k, _payload(1000 + k, 5 * ALIGN), aligned=engine[2]
            )
        _check_accounting(core)
        crashed = RawBlockCore(config)
        try:
            for k in keys:
                assert _load(crashed, k, len(old[k])) in (None, old[k])
        finally:
            crashed.close()
    finally:
        core.close()


def test_a_run_without_verification_still_protects_later_verified_runs(tmp_path):
    """meta_verify_on_load is a per-run setting: a run without it must still
    guard deleted objects, or a later run with verification trusts an old
    header while another object reused the payload."""
    config = replace(_config(tmp_path), meta_verify_on_load=False)
    old = _payload(1, 5 * ALIGN)
    core = RawBlockCore(config)
    try:
        assert _put(core, 1, old)
        offset = core.entry_offset(_spec(1).encoded)
        core.checkpoint_now()  # lists key 1
        assert core.delete_many([_spec(1).encoded]) == [True]
        _reuse_with_a_delayed_head(
            core, replace(config, meta_verify_on_load=True), offset, 1, old
        )
    finally:
        core.close()


def test_a_run_with_another_meta_magic_voids_the_old_checkpoint(tmp_path):
    """A run with another meta_magic (or meta_version) cannot read the old
    checkpoint, so it must void it before storing: switching back must not
    load it over bytes that run reused."""
    config = replace(_config(tmp_path), meta_enable_periodic=False)
    old = _payload(1, 5 * ALIGN)
    core = RawBlockCore(config)
    try:
        assert _put(core, 1, old)
        offset = core.entry_offset(_spec(1).encoded)
    finally:
        core.close()  # its checkpoint lists key 1
    other = RawBlockCore(replace(config, meta_magic=b"LMCIDX02"))
    try:
        _reuse_with_a_delayed_head(other, config, offset, 1, old)
    finally:
        other.close()


def test_an_import_that_outgrows_the_container_starts_empty(tmp_path):
    """Format 2 entries are longer: a fixed-slot checkpoint that nearly fills
    a metadata container no longer fits once rewritten. Startup then goes on
    with an empty index instead of failing."""
    config = replace(
        _config(tmp_path), meta_enable_periodic=False, meta_verify_on_load=False
    )
    probe = RawBlockCore(replace(config, load_checkpoint_on_init=False))
    base = probe.data_base_offset()
    offsets = probe.metadata_container_offsets()
    cap = offsets[1] - offsets[0] - ALIGN
    slot_bytes = 2 * ALIGN

    def state(n: int) -> dict:
        entries = {
            _spec(i).encoded: {
                "offset": base + (i + 1) * slot_bytes,
                **_entry_meta(100),
            }
            for i in range(n)
        }
        return _v1_state(probe, slot_bytes, entries)

    def fits(n: int) -> bool:
        return len(json.dumps(state(n), separators=(",", ":"))) <= cap

    lo, hi = 1, 2
    while fits(hi):
        lo, hi = hi, hi * 2
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (mid, hi) if fits(mid) else (lo, mid)
    v1 = state(lo)
    probe.close()
    _write_container(config.device_path, offsets[0], 10, v1)
    _write_container(config.device_path, offsets[1], 9, v1)
    core = RawBlockCore(config)
    try:
        assert core.indexed_key_count() == 0
        assert _put(core, 1, _payload(1, 100))
    finally:
        core.close()
    again = RawBlockCore(config)  # the fixed-slot checkpoint is gone
    try:
        assert again.snapshot_indexed_keys() == [_spec(1).encoded]
    finally:
        again.close()


def test_io_uring_batch_key_without_space_fails_alone(tmp_path):
    """In one io_uring batch, a key that finds no long enough range fails
    alone and the others commit."""
    payload_len = 3 * ALIGN
    engine = ENGINES[4]  # io_uring, zero copy
    core = RawBlockCore(
        _config(tmp_path, engine, data_bytes=ALIGN + 8 * _need(payload_len))
    )
    try:
        sizes = [100, 2 * ALIGN, 8 * _need(payload_len), 3 * ALIGN, 5]
        result = core.put_many(
            [_spec(i) for i in range(len(sizes))],
            [_obj(_payload(i, n), True) for i, n in enumerate(sizes)],
        )
        assert result.results == [True, True, False, True, True]
        assert core.report_status()["inflight_key_count"] == 0
        for i, n in enumerate(sizes):
            if i != 2:
                assert _load(core, i, n) == _payload(i, n)
    finally:
        core.close()


def test_forked_writers_use_different_generations(tmp_path):
    """Generations must differ across processes forked from one parent, or a
    restarted child could repeat the generation of the run before it."""
    headers = []
    for n in range(2):
        (tmp_path / str(n)).mkdir()
        config = _config(tmp_path / str(n))
        read_end, write_end = os.pipe()
        pid = os.fork()
        if pid == 0:  # child: one store, then report its header bytes
            code = 1
            try:
                core = RawBlockCore(config)
                assert _put(core, 1, _payload(1, 100))
                offset = core.entry_offset(_spec(1).encoded)
                core.close()
                with open(config.device_path, "rb") as f:
                    f.seek(offset)
                    os.write(write_end, f.read(32))
                code = 0
            finally:
                os._exit(code)
        os.close(write_end)
        _, status = os.waitpid(pid, 0)
        assert status == 0
        headers.append(os.read(read_end, 32))
        os.close(read_end)
    assert headers[0][:24] == headers[1][:24]  # same key and size
    assert headers[0][24:32] != headers[1][24:32]


def test_load_racing_a_forced_delete_and_reuse_is_a_miss(tmp_path):
    """A load reads without the lock; if the key is force-deleted and its
    space reused meanwhile, the load must report a miss, not the new bytes."""
    config = _config(tmp_path)
    core = RawBlockCore(config)
    real_read = PosixBlockIO.read_buffers

    def read_after_reuse(self, offsets, *args, **kwargs):
        if not reused:
            reused.append(1)
            assert core.delete_many([_spec(1).encoded], force=True) == [True]
            assert _put(core, 2, _payload(2, ALIGN))
            assert core.entry_offset(_spec(2).encoded) == offset
        return real_read(self, offsets, *args, **kwargs)

    reused: list[int] = []
    try:
        assert _put(core, 1, _payload(1, ALIGN))
        offset = core.entry_offset(_spec(1).encoded)
        with patch.object(PosixBlockIO, "read_buffers", read_after_reuse):
            assert _load(core, 1, ALIGN) is None
        assert _load(core, 2, ALIGN) == _payload(2, ALIGN)
    finally:
        core.close()


def _write_container(path: str, offset: int, seq: int, data: dict) -> None:
    payload = json.dumps(data, separators=(",", ":")).encode()
    with open(path, "r+b") as f:
        f.seek(offset + ALIGN)
        f.write(payload)
        f.seek(offset)
        f.write(
            _META_HEADER.pack(
                b"LMCIDX01", 1, seq, len(payload), zlib.crc32(payload) & 0xFFFFFFFF
            )
        )


def _containers(path: str, offsets: list[int]) -> list[tuple[int, int]]:
    """Return ``(seq, format)`` of each metadata container."""
    out = []
    with open(path, "rb") as f:
        for offset in offsets:
            f.seek(offset)
            _, _, seq, length, _ = _META_HEADER.unpack(f.read(_META_HEADER.size))
            f.seek(offset + ALIGN)
            out.append((seq, json.loads(f.read(length))["version"]))
    return out


def _v1_state(core: RawBlockCore, slot_bytes: int, entries: dict) -> dict:
    return {
        "version": 1,
        "device_path": core.device_path,
        "capacity_bytes": core.capacity_bytes,
        "block_align": core.block_align,
        "header_bytes": core.header_bytes,
        "slot_bytes": slot_bytes,
        "meta_total_bytes": core.meta_total_bytes,
        "meta_magic": core.meta_magic_text,
        "meta_version": core.meta_version,
        "data_base_offset": core.data_base_offset(),
        "next_slot": len(entries),
        "entries": entries,
    }


def _v2_state(core: RawBlockCore, entries: dict) -> dict:
    return {
        "version": 2,
        "device_path": core.device_path,
        "block_align": ALIGN,
        "header_bytes": HEADER,
        "meta_total_bytes": core.meta_total_bytes,
        "meta_magic": core.meta_magic_text,
        "meta_version": core.meta_version,
        "entries": entries,
    }


def _entry_meta(size: int) -> dict:
    return {
        "size": size,
        "shape": [size],
        "dtype": "uint8",
        "fmt": "BINARY",
        "cached_positions": None,
    }


def _write_slot_object(path: str, offset: int, key: int, size: int) -> None:
    """Write an object as the fixed-slot layout did: header, then payload."""
    with open(path, "r+b") as f:
        f.seek(offset)
        f.write(
            b"LMCBLK01"
            + (_spec(key).slot_identity & ((1 << 64) - 1)).to_bytes(8, "little")
            + size.to_bytes(8, "little")
        )
        f.seek(offset + HEADER)
        f.write(_payload(key, size))


@pytest.mark.parametrize("periodic", [False, True])
def test_fixed_slot_checkpoint_is_imported_and_rewritten(tmp_path, periodic):
    """A device written by the fixed-slot layout keeps its cache: each object
    keeps only the space it needs and the slot tails become free. The object
    in slot 0 is dropped, since that space now holds the format marker."""
    config = replace(_config(tmp_path), meta_enable_periodic=periodic)
    slot_bytes = 64 * 1024
    sizes = [100, 100, 30000, slot_bytes - HEADER]
    probe = RawBlockCore(replace(config, load_checkpoint_on_init=False))
    base = probe.data_base_offset()
    offsets = probe.metadata_container_offsets()
    state = _v1_state(
        probe,
        slot_bytes,
        {
            _spec(i).encoded: {"offset": base + i * slot_bytes, **_entry_meta(size)}
            for i, size in enumerate(sizes)
        },
    )
    probe.close()
    for i, size in enumerate(sizes):
        _write_slot_object(config.device_path, base + i * slot_bytes, i, size)
    _write_container(config.device_path, offsets[0], 1000, state)
    _write_container(config.device_path, offsets[1], 999, state)

    core = RawBlockCore(config)
    try:
        assert not core.contains_key(_spec(0).encoded)
        for i, size in list(enumerate(sizes))[1:]:
            assert _load(core, i, size) == _payload(i, size)
            assert core.entry_offset(_spec(i).encoded) == base + i * slot_bytes
            assert _allocated(core, [_spec(i).encoded]) == [_need(size)]
        _check_accounting(core)
        # The newest container is in the new format.
        assert max(_containers(config.device_path, offsets)) == (1001, 2)
        # Freed slot tails are reusable; best fit takes the smallest (slot 2).
        assert _put(core, 50, b"z" * 100)
        assert core.entry_offset(_spec(50).encoded) == base + 2 * slot_bytes + _need(
            sizes[2]
        )
    finally:
        core.close()
    with open(config.device_path, "rb") as f:
        f.seek(base)
        assert f.read(8) == b"LMCVAR01"


def test_checkpoint_from_before_a_fixed_slot_run_is_not_trusted(tmp_path):
    """A fixed-slot binary that runs on the device rejects the new format,
    starts empty and writes its first object at the start of the data region.
    If it stops before writing a checkpoint, the next run must not trust the
    older checkpoint, whose objects that run may have overwritten."""
    config = _config(tmp_path)
    core = RawBlockCore(config)
    try:
        assert _put(core, 1, _payload(1, 100))
        assert _put(core, 2, _payload(2, 5 * ALIGN))
        core.checkpoint_now()
        base = core.data_base_offset()
        second = core.entry_offset(_spec(2).encoded)
    finally:
        core.close()
    # The fixed-slot run writes slot 0 (over the marker) and a slot that starts
    # inside key 2's payload, leaving key 2's header intact.
    slot_bytes = second + HEADER + ALIGN - base
    _write_slot_object(config.device_path, base, 9, 100)
    _write_slot_object(config.device_path, base + slot_bytes, 10, 100)

    reopened = RawBlockCore(config)
    try:
        assert reopened.indexed_key_count() == 0
        _check_accounting(reopened)
        assert reopened.report_status()["allocated_bytes"] == 0
    finally:
        reopened.close()
    # The marker is back, so a restart now trusts this run's checkpoints.
    again = RawBlockCore(config)
    try:
        assert _put(again, 3, b"c" * 10)
    finally:
        again.close()
    last = RawBlockCore(config)
    try:
        assert last.snapshot_indexed_keys() == [_spec(3).encoded]
    finally:
        last.close()


def test_checkpoint_after_loading_past_a_torn_copy_keeps_the_loaded_copy(tmp_path):
    """When the newest copy is torn and the older one is loaded, the next
    checkpoint must replace the torn copy, not the only good one."""
    config = _config(tmp_path)
    probe = RawBlockCore(replace(config, load_checkpoint_on_init=False))
    offsets = probe.metadata_container_offsets()
    state = _v2_state(probe, {})
    probe.close()
    with open(config.device_path, "r+b") as f:
        f.seek(probe.data_base_offset())
        f.write(b"LMCVAR01")
    # Copy n goes to container (n - 1) % 2.
    _write_container(config.device_path, offsets[1], 10, state)
    _write_container(config.device_path, offsets[0], 11, state)
    with open(config.device_path, "r+b") as f:
        f.seek(offsets[0] + ALIGN)
        f.write(b"torn")  # the payload no longer matches its CRC

    core = RawBlockCore(config)
    try:
        assert _put(core, 1, b"a" * 10)
        core.checkpoint_now()
    finally:
        core.close()
    assert _containers(config.device_path, offsets)[1] == (10, 2)


def test_crash_while_restoring_a_key_in_its_listed_place_drops_it(tmp_path):
    """K is listed at X. K is deleted, B reuses X and is deleted, then K is
    stored again and best fit puts it back at X. A crash after K's new header
    landed but before its payload did must not serve B's bytes as K: the
    header comes from another write than the one the checkpoint lists."""
    config = replace(_config(tmp_path), meta_enable_periodic=False)
    old = _payload(1, 5 * ALIGN)
    core = RawBlockCore(config)
    real_write = PosixBlockIO.write_buffers
    gate, header_done = threading.Event(), threading.Event()

    def header_then_hang(self, offsets, buffers, *args, **kwargs):
        if len(offsets) == 2 and offsets[0] == x:
            real_write(self, offsets[:1], buffers[:1], *[a[:1] for a in args])
            header_done.set()
            assert gate.wait(10)  # the payload is still in flight
        return real_write(self, offsets, buffers, *args, **kwargs)

    restore = threading.Thread(target=_put, args=(core, 1, old))
    try:
        assert _put(core, 1, old)
        x = core.entry_offset(_spec(1).encoded)
        core.checkpoint_now()  # lists key 1 at x
        assert core.delete_many([_spec(1).encoded]) == [True]
        assert _put(core, 2, _payload(2, 5 * ALIGN))
        assert core.entry_offset(_spec(2).encoded) == x
        assert core.delete_many([_spec(2).encoded]) == [True]
        with patch.object(PosixBlockIO, "write_buffers", header_then_hang):
            restore.start()
            assert header_done.wait(10)
            crashed = RawBlockCore(config)  # the process died here
            try:
                assert _load(crashed, 1, len(old)) in (None, old)
                assert crashed.indexed_key_count() == 0
            finally:
                crashed.close()
    finally:
        gate.set()
        if restore.is_alive():
            restore.join(10)
        core.close()


def test_imported_entries_adopt_the_generation_of_their_header(tmp_path):
    """Fixed-slot headers never had a generation, so an imported entry takes
    the bytes its header holds there; a later write of the key at the same
    place then no longer matches the checkpoint entry."""
    config = _config(tmp_path)
    slot_bytes = 64 * 1024
    probe = RawBlockCore(replace(config, load_checkpoint_on_init=False))
    base = probe.data_base_offset()
    offsets = probe.metadata_container_offsets()
    state = _v1_state(
        probe,
        slot_bytes,
        {_spec(1).encoded: {"offset": base + slot_bytes, **_entry_meta(100)}},
    )
    probe.close()
    _write_slot_object(config.device_path, base + slot_bytes, 1, 100)
    with open(config.device_path, "r+b") as f:
        f.seek(base + slot_bytes + 24)
        f.write((0xA5A5).to_bytes(8, "little"))  # left over in the old header
    _write_container(config.device_path, offsets[0], 7, state)

    core = RawBlockCore(config)
    try:
        assert _load(core, 1, 100) == _payload(1, 100)
        core.checkpoint_now()
    finally:
        core.close()
    newest: tuple[int, dict] = (0, {})
    with open(config.device_path, "rb") as f:
        for offset in offsets:
            f.seek(offset)
            _, _, seq, length, _ = _META_HEADER.unpack(f.read(_META_HEADER.size))
            f.seek(offset + ALIGN)
            if seq > newest[0]:
                newest = (seq, json.loads(f.read(length))["entries"])
    assert newest[1][_spec(1).encoded]["gen"] == 0xA5A5


@pytest.mark.parametrize("retired", [1, 4])  # the longest range after it, or before
def test_an_object_longer_than_every_unretired_range_does_not_count(tmp_path, retired):
    """A retired range in the middle splits the data region: an object that
    fits the usable capacity but no range between retired bytes never fits,
    so evicting for it would only empty the cache."""
    payload_len = 3 * ALIGN
    count = 8
    config = _config(tmp_path, data_bytes=ALIGN + count * _need(payload_len))
    first = RawBlockCore(config)
    try:
        for i in range(count):
            assert _put(first, i, _payload(i, payload_len))
        bad = first.entry_offset(_spec(retired).encoded)
    finally:
        first.close()
    real_read = PosixBlockIO.read_buffers

    def unreadable(self, offsets, *args, **kwargs):
        if list(offsets) == [bad]:
            return [False]
        return real_read(self, offsets, *args, **kwargs)

    with patch.object(PosixBlockIO, "read_buffers", unreadable):
        core = RawBlockCore(config)  # retires the object at bad
    try:
        core.delete_many(core.snapshot_indexed_keys())
        status = core.report_status()
        span = status["largest_free_extent_bytes"]
        too_long = span + _need(payload_len)
        assert too_long <= status["usable_capacity_bytes"]
        assert not _put(core, 100, b"x" * (too_long - HEADER))
        assert core.space_exhausted_count() == 0
        # One that fits a range counts when it fails.
        for i in range(count):
            _put(core, 200 + i, _payload(i, payload_len))
        counted = core.space_exhausted_count()
        assert not _put(core, 300, b"x" * (span - HEADER))
        assert core.space_exhausted_count() == counted + 1
    finally:
        core.close()


def test_exhausted_space_is_counted_once_per_failed_store(tmp_path):
    """Each store that finds no long enough range counts once; reading the
    count changes nothing, so several readers see the same value."""
    payload_len = 3 * ALIGN
    count = 16
    core = RawBlockCore(
        _config(tmp_path, data_bytes=ALIGN + count * _need(payload_len))
    )
    try:
        _fragment(core, count, payload_len)
        assert not _put(core, 100, _payload(100, 2 * _need(payload_len)))
        assert core.space_exhausted_count() == 1
        assert core.space_exhausted_count() == 1
        assert not _put(core, 101, _payload(101, 2 * _need(payload_len)))
        assert core.space_exhausted_count() == 2
        assert _put(core, 102, b"s")  # a store that fits counts nothing
        assert core.space_exhausted_count() == 2
    finally:
        core.close()


def test_imported_entry_with_zero_generation_bytes_reused_in_place(tmp_path):
    """Fixed-slot writers left header bytes 24-32 at 0 on a zeroed device, so
    an imported entry adopts generation 0. A later write of the key at the
    same place has an odd generation and must not match it: delete K, reuse
    and free its place, re-store K there with only its header landing, crash."""
    config = replace(_config(tmp_path), meta_enable_periodic=False)
    slot = 64 * 1024
    full = slot - HEADER
    probe = RawBlockCore(replace(config, load_checkpoint_on_init=False))
    base = probe.data_base_offset()
    offsets = probe.metadata_container_offsets()
    entries = {
        _spec(i).encoded: {"offset": base + i * slot, **_entry_meta(full)}
        for i in (2, 3, 4)
    }
    state = _v1_state(probe, slot, entries)
    probe.close()
    for i in (2, 3, 4):
        _write_slot_object(config.device_path, base + i * slot, i, full)
    _write_container(config.device_path, offsets[0], 5, state)

    core = RawBlockCore(config)
    real_write = PosixBlockIO.write_buffers

    def header_only(self, offsets, buffers, *args, **kwargs):
        if len(offsets) == 2 and offsets[0] == x:
            return real_write(self, offsets[:1], buffers[:1], *[a[:1] for a in args])
        return real_write(self, offsets, buffers, *args, **kwargs)

    try:
        x = base + 3 * slot
        assert core.entry_offset(_spec(3).encoded) == x
        assert core.delete_many([_spec(3).encoded]) == [True]
        assert _put(core, 9, _payload(9, full))
        assert core.entry_offset(_spec(9).encoded) == x
        assert core.delete_many([_spec(9).encoded]) == [True]
        with patch.object(PosixBlockIO, "write_buffers", header_only):
            assert _put(core, 3, _payload(3, full))
        crashed = RawBlockCore(config)
        try:
            assert _load(crashed, 3, full) in (None, _payload(3, full))
        finally:
            crashed.close()
    finally:
        core.close()


def test_load_racing_a_delete_and_rewrite_of_the_same_key_is_a_miss(tmp_path):
    """The key is still indexed after the race, but by another write: a load
    that read the old place must not report the new entry's hit."""
    config = _config(tmp_path)
    core = RawBlockCore(config)
    real_read = PosixBlockIO.read_buffers
    raced: list[int] = []

    def read_after_rewrite(self, offsets, *args, **kwargs):
        if not raced:
            raced.append(1)
            assert core.delete_many([_spec(1).encoded], force=True) == [True]
            assert _put(core, 7, _payload(7, ALIGN))  # takes the old place
            assert _put(core, 1, _payload(1, ALIGN))  # the key, elsewhere
        return real_read(self, offsets, *args, **kwargs)

    try:
        assert _put(core, 1, _payload(1, ALIGN))
        with patch.object(PosixBlockIO, "read_buffers", read_after_rewrite):
            assert _load(core, 1, ALIGN) is None
        assert _load(core, 1, ALIGN) == _payload(1, ALIGN)
    finally:
        core.close()


def test_a_delete_alone_makes_the_next_checkpoint_write(tmp_path):
    core = RawBlockCore(_config(tmp_path))
    try:
        assert _put(core, 1, _payload(1, 100))
        core.checkpoint_now()
        seq = core.report_status()["metadata_seq"]
        core.checkpoint_now()  # nothing changed: no write
        assert core.report_status()["metadata_seq"] == seq
        assert core.delete_many([_spec(1).encoded]) == [True]
        core.checkpoint_now()
        assert core.report_status()["metadata_seq"] == seq + 1
    finally:
        core.close()


def test_header_bytes_must_hold_the_write_generation(tmp_path):
    with pytest.raises(ValueError, match="header_bytes"):
        RawBlockCore(replace(_config(tmp_path), block_align=8, header_bytes=24))


def test_unreadable_marker_next_to_a_new_format_checkpoint_stops_startup(tmp_path):
    config = _config(tmp_path)
    core = RawBlockCore(config)
    try:
        assert _put(core, 1, _payload(1, 100))
        marker = core.data_base_offset()
    finally:
        core.close()
    real_read = PosixBlockIO.read_buffers

    def failing_marker(self, offsets, *args, **kwargs):
        if list(offsets) == [marker]:
            return [False]
        return real_read(self, offsets, *args, **kwargs)

    with patch.object(PosixBlockIO, "read_buffers", failing_marker):
        with pytest.raises(RuntimeError, match="format marker"):
            RawBlockCore(config)
    again = RawBlockCore(config)
    try:
        assert _load(again, 1, 100) == _payload(1, 100)
    finally:
        again.close()


def test_import_keeps_the_cache_if_the_marker_write_fails(tmp_path):
    """The marker is written before the imported checkpoint is rewritten: if
    that write fails, the next start imports the fixed-slot checkpoint again
    instead of finding a new-format one without its marker."""
    config = _config(tmp_path)
    slot = 64 * 1024
    probe = RawBlockCore(replace(config, load_checkpoint_on_init=False))
    base = probe.data_base_offset()
    offsets = probe.metadata_container_offsets()
    state = _v1_state(
        probe, slot, {_spec(1).encoded: {"offset": base + slot, **_entry_meta(100)}}
    )
    probe.close()
    with open(config.device_path, "r+b") as f:
        f.seek(base)
        f.write(b"\0" * ALIGN)  # as a fixed-slot device has it
    _write_slot_object(config.device_path, base + slot, 1, 100)
    _write_container(config.device_path, offsets[0], 9, state)
    real_write = PosixBlockIO.write_buffers

    def failing_marker(self, offsets, *args, **kwargs):
        if list(offsets) == [base]:
            raise OSError("injected")
        return real_write(self, offsets, *args, **kwargs)

    with patch.object(PosixBlockIO, "write_buffers", failing_marker):
        with pytest.raises(OSError):
            RawBlockCore(config)
    again = RawBlockCore(config)
    try:
        assert _load(again, 1, 100) == _payload(1, 100)
    finally:
        again.close()


@pytest.mark.parametrize("how", ["skipped", "rejected"])
def test_unloaded_checkpoint_is_superseded_before_the_first_store(tmp_path, how):
    """A run that does not load the checkpoint will reuse space it lists,
    possibly without overwriting those objects' headers, so a crash before
    its first checkpoint must not bring them back."""
    config = _config(tmp_path)
    core = RawBlockCore(config)
    try:
        assert _put(core, 1, _payload(1, 5 * ALIGN))
        core.checkpoint_now()
    finally:
        core.close()
    if how == "skipped":
        run = RawBlockCore(replace(config, load_checkpoint_on_init=False))
    else:
        run = RawBlockCore(replace(config, header_bytes=2 * HEADER))
    try:
        crashed = RawBlockCore(replace(config, meta_enable_periodic=False))
        try:
            assert crashed.indexed_key_count() == 0
        finally:
            crashed.close()
    finally:
        run.close()


@pytest.mark.parametrize(
    "bad",
    [
        {"offset": 0},  # inside the metadata region
        {"offset": 1 << 40},  # past the end
        "marker",  # the format-marker block
        "misaligned",
        "tail",  # runs past the end
    ],
)
def test_invalid_checkpoint_entries_are_dropped(tmp_path, bad):
    config = replace(_config(tmp_path), meta_verify_on_load=False)
    core = RawBlockCore(config)
    try:
        first = _first_object_offset(core)
        end = (
            core.data_base_offset()
            + core.report_status()["usable_capacity_bytes"]
            + ALIGN
        )
        if isinstance(bad, str):
            bad = {
                "marker": {"offset": core.data_base_offset()},
                "misaligned": {"offset": first + 512},
                "tail": {"offset": end - ALIGN},
            }[bad]
        state = _v2_state(
            core,
            {
                _spec(1).encoded: {**bad, **_entry_meta(100)},
                _spec(2).encoded: {"offset": first + 65536, **_entry_meta(100)},
            },
        )
        assert core.apply_loaded_state(state) is True
        assert core.snapshot_indexed_keys() == [_spec(2).encoded]
        _check_accounting(core)
    finally:
        core.close()


def test_overlapping_checkpoint_entries_are_both_dropped(tmp_path):
    config = replace(_config(tmp_path), meta_verify_on_load=False)
    core = RawBlockCore(config)
    try:
        first = _first_object_offset(core)
        state = _v2_state(
            core,
            {
                _spec(1).encoded: {"offset": first, **_entry_meta(5000)},  # 12 KiB
                _spec(2).encoded: {"offset": first + 8192, **_entry_meta(100)},
                _spec(3).encoded: {"offset": first + 65536, **_entry_meta(100)},
            },
        )
        assert core.apply_loaded_state(state) is True
        assert core.snapshot_indexed_keys() == [_spec(3).encoded]
        _check_accounting(core)
    finally:
        core.close()


def test_fixed_slot_entries_off_the_slot_grid_are_dropped(tmp_path):
    config = replace(_config(tmp_path), meta_verify_on_load=False)
    core = RawBlockCore(config)
    try:
        base = core.data_base_offset()
        slot = 64 * 1024
        state = _v1_state(
            core,
            slot,
            {
                _spec(1).encoded: {"offset": base + slot + ALIGN, **_entry_meta(100)},
                _spec(2).encoded: {"offset": base + slot, **_entry_meta(slot)},
                _spec(3).encoded: {"offset": base + 2 * slot, **_entry_meta(100)},
            },
        )
        assert core.apply_loaded_state(state) is True
        # Off the slot grid, and larger than a slot, respectively.
        assert core.snapshot_indexed_keys() == [_spec(3).encoded]
    finally:
        core.close()


@pytest.mark.parametrize("periodic", [False, True])
@pytest.mark.parametrize("engine", [ENGINES[0], ENGINES[4]], ids=["posix", "uring-zc"])
@pytest.mark.parametrize("seed", range(4))
def test_randomized_mixed_sizes_never_return_wrong_bytes(
    tmp_path, engine, periodic, seed
):
    """Random stores, deletes, checkpoints and crash-reopens of a nearly full
    device with mixed sizes: every load returns the key's own bytes or a miss,
    accounting balances, and live objects never overlap."""
    rng = random.Random(seed)
    data_bytes = 4 << 20
    config = replace(
        _config(tmp_path, engine, data_bytes=data_bytes),
        meta_enable_periodic=periodic,
        meta_checkpoint_interval_sec=3600,
        meta_idle_quiet_ms=3_600_000,
    )
    sizes = [1, 100, ALIGN, 5 * ALIGN + 3, 40_000, 200_000]
    stored: dict[int, int] = {}
    core = RawBlockCore(config)
    crashed: list[RawBlockCore] = []
    try:
        for step in range(1500):
            r = rng.random()
            if r < 0.55:
                key = rng.randrange(10_000)
                size = rng.choice(sizes)
                if key not in stored and _put(
                    core, key, _payload(key, size), engine[2]
                ):
                    stored[key] = size
            elif r < 0.85 and stored:
                victims = rng.sample(
                    sorted(stored), min(len(stored), rng.randint(1, 8))
                )
                core.delete_many([_spec(k).encoded for k in victims])
                for k in victims:
                    del stored[k]
            elif r < 0.9:
                core.checkpoint_now()
            elif r < 0.93:
                # Crash: what a restart would recover from the device now. Its
                # entries may be older than the live index, but each must hold
                # that key's own bytes at its recorded size.
                view = RawBlockCore(replace(config, meta_enable_periodic=False))
                crashed.append(view)
                keys = {_spec(k).encoded: k for k in range(10_000)}
                for encoded in view.snapshot_indexed_keys():
                    meta = view.get_metadata_many([encoded])[0]
                    assert meta is not None
                    k = keys[encoded]
                    got = _load(view, k, meta.size)
                    assert got in (None, _payload(k, meta.size))
            if step % 100 == 0:
                _check_accounting(core)
                spans = sorted(
                    (core.entry_offset(_spec(k).encoded), _need(n))
                    for k, n in stored.items()
                )
                for (a, n), (b, _) in zip(spans, spans[1:], strict=False):
                    assert a + n <= b
                sample = rng.sample(sorted(stored), min(10, len(stored)))
                for k in sample:
                    assert _load(core, k, stored[k]) == _payload(k, stored[k])
        for k, size in stored.items():
            assert _load(core, k, size) == _payload(k, size)
    finally:
        for view in crashed:
            view.close()
        core.close()


def test_concurrent_store_load_delete_keep_bytes_intact(tmp_path):
    engine = ("io_uring", True, True)
    config = replace(
        _config(tmp_path, engine, data_bytes=8 << 20),
        meta_enable_periodic=True,
        meta_checkpoint_interval_sec=1,
        meta_idle_quiet_ms=0,
    )
    core = RawBlockCore(config)
    errors: list[BaseException] = []
    sizes = [100, 7000, 33_000, 150_000]

    def worker(wid: int) -> None:
        rng = random.Random(wid)
        mine: dict[int, int] = {}
        try:
            for _ in range(400):
                r = rng.random()
                if r < 0.5:
                    key = wid * 100_000 + rng.randrange(1000)
                    size = rng.choice(sizes)
                    if key not in mine and _put(core, key, _payload(key, size), True):
                        mine[key] = size
                elif r < 0.75 and mine:
                    key = rng.choice(sorted(mine))
                    assert core.delete_many([_spec(key).encoded]) == [True]
                    del mine[key]
                elif mine:
                    key = rng.choice(sorted(mine))
                    assert _load(core, key, mine[key]) == _payload(key, mine[key])
        except BaseException as e:  # noqa: BLE001 - surfaced in the main thread
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(w,)) for w in range(6)]
    try:
        for t in threads:
            t.start()
        for t in threads:
            t.join(120)
        assert not errors, errors[0]
        _check_accounting(core)
    finally:
        core.close()


@pytest.mark.parametrize("pattern", ["alternating", "phase"])
@pytest.mark.parametrize("engine", [ENGINES[0], ENGINES[4]], ids=["posix", "uring-zc"])
def test_alternating_small_and_large_objects_under_heavy_churn(
    tmp_path, engine, pattern
):
    """Small and large objects of varying size (as unfull chunks are) that
    alternate or come in phases, evicted in an order unrelated to their
    placement (as LRU hits reorder keys), fragment the device; every live
    object must keep its bytes, accounting must balance, and a restart must
    recover them all."""
    data_bytes = 8 << 20
    config = replace(
        _config(tmp_path, engine, data_bytes=data_bytes),
        meta_checkpoint_interval_sec=3600,
        meta_idle_quiet_ms=3_600_000,
    )
    small = 3000
    rng = random.Random(0)
    core = RawBlockCore(config)
    stored: dict[int, int] = {}
    try:
        for key in range(1500):
            if pattern == "alternating":
                is_small = key % 2 == 0
            else:
                is_small = key // 120 % 2 == 1
            size = small if is_small else rng.randrange(100_000, 400_000)
            while not _put(core, key, _payload(key, size), engine[2]):
                # Evict a tenth.
                victims = rng.sample(sorted(stored), max(1, len(stored) // 10))
                core.delete_many([_spec(k).encoded for k in victims])
                for k in victims:
                    del stored[k]
            stored[key] = size
            if key % 100 == 0:
                _check_accounting(core)
        status = core.report_status()
        assert status["alloc_failure_count"] > 0
        assert (
            sum(_need(n) for n in stored.values()) / status["usable_capacity_bytes"]
            > 0.5
        )
        for k, size in stored.items():
            assert _load(core, k, size) == _payload(k, size)
    finally:
        core.close()
    reopened = RawBlockCore(config)
    try:
        assert sorted(reopened.snapshot_indexed_keys()) == sorted(
            _spec(k).encoded for k in stored
        )
        for k, size in stored.items():
            assert _load(reopened, k, size) == _payload(k, size)
    finally:
        reopened.close()
