# SPDX-License-Identifier: Apache-2.0

# Future
from __future__ import annotations

# Standard
import stat
import sys
import types

# Third Party
import pytest

# First Party
from lmcache.v1.storage_backend.raw_block import (
    RawBlockCore,
    RawBlockCoreConfig,
    encode_object_key,
)
from tests.v1.storage_backend.raw_block_test_utils import (
    RAW_BLOCK_CI_BLOCK_ALIGN,
    RAW_BLOCK_CI_CAPACITY_BYTES,
    RAW_BLOCK_CI_HEADER_BYTES,
    RAW_BLOCK_CI_META_TOTAL_BYTES,
    RAW_BLOCK_CI_SLOT_BYTES,
    make_empty_memory_obj,
    make_memory_obj,
    make_object_key,
    make_raw_block_core_config,
    make_raw_block_file,
    memory_obj_bytes,
)
import lmcache.v1.storage_backend.raw_block.core as raw_block_core

pytest.importorskip("lmcache_rust_raw_block_io")


def test_raw_block_core_store_load_and_exists(tmp_path):
    path = make_raw_block_file(tmp_path)
    config = make_raw_block_core_config(path)
    core = RawBlockCore(config, key_namespace="object")

    try:
        keys = [make_object_key(i) for i in range(3)]
        specs = [encode_object_key(key) for key in keys]
        payloads = [
            bytes([1]) * 1024,
            bytes([2]) * 2048,
            bytes([3]) * 3072,
        ]
        objects = [make_memory_obj(payload) for payload in payloads]

        put_result = core.put_many(specs, objects)

        assert put_result.results == [True, True, True]
        assert put_result.stored_keys == [spec.encoded for spec in specs]
        assert core.exists_many([spec.encoded for spec in specs]) == [
            True,
            True,
            True,
        ]

        loaded = [make_empty_memory_obj(len(payload)) for payload in payloads]
        load_result = core.load_many_into([spec.encoded for spec in specs], loaded)

        assert load_result == [True, True, True]
        assert [memory_obj_bytes(obj) for obj in loaded] == payloads
    finally:
        core.close()


def test_raw_block_core_duplicate_put_keeps_original_payload(tmp_path):
    path = make_raw_block_file(tmp_path)
    config = make_raw_block_core_config(path)
    core = RawBlockCore(config, key_namespace="object")

    try:
        spec = encode_object_key(make_object_key(11))
        original = b"original"
        duplicate = b"mutated!"

        first_result = core.put_many([spec], [make_memory_obj(original)])
        duplicate_result = core.put_many([spec], [make_memory_obj(duplicate)])

        assert first_result.results == [True]
        assert first_result.stored_keys == [spec.encoded]
        assert duplicate_result.results == [True]
        assert duplicate_result.stored_keys == []

        loaded = make_empty_memory_obj(len(original))
        assert core.load_many_into([spec.encoded], [loaded]) == [True]
        assert memory_obj_bytes(loaded) == original
    finally:
        core.close()


def test_raw_block_core_delete_and_missing_load(tmp_path):
    path = make_raw_block_file(tmp_path)
    config = make_raw_block_core_config(path)
    core = RawBlockCore(config, key_namespace="object")

    try:
        existing = encode_object_key(make_object_key(21))
        missing = encode_object_key(make_object_key(22))

        put_result = core.put_many([existing], [make_memory_obj(b"delete-me")])
        assert put_result.results == [True]
        assert core.contains_key(existing.encoded) is True

        assert core.delete_many([existing.encoded, missing.encoded]) == [True, False]
        assert core.exists_many([existing.encoded, missing.encoded]) == [False, False]

        loaded = make_empty_memory_obj(len(b"delete-me"))
        assert core.load_many_into([existing.encoded], [loaded]) == [False]
    finally:
        core.close()


def test_raw_block_core_recovers_checkpoint_from_temp_file(tmp_path):
    path = make_raw_block_file(tmp_path)
    config = make_raw_block_core_config(path)
    spec = encode_object_key(make_object_key(31))
    payload = b"recoverable-raw-block-payload"

    core = RawBlockCore(config, key_namespace="object")
    try:
        put_result = core.put_many([spec], [make_memory_obj(payload)])
        assert put_result.results == [True]
        core.checkpoint_now()
    finally:
        core.close()

    recovered = RawBlockCore(config, key_namespace="object")
    try:
        assert recovered.contains_key(spec.encoded) is True
        loaded = make_empty_memory_obj(len(payload))
        assert recovered.load_many_into([spec.encoded], [loaded]) == [True]
        assert memory_obj_bytes(loaded) == payload
    finally:
        recovered.close()


class _FakeRawDevice:
    def __init__(self, size_bytes: int = RAW_BLOCK_CI_CAPACITY_BYTES) -> None:
        self._size_bytes = int(size_bytes)
        self.batched_write_calls: list[
            tuple[list[int], list[int], list[int | None] | None]
        ] = []

    def size_bytes(self) -> int:
        return self._size_bytes

    def pread_into(self, offset, out, payload_len, total_len=None):
        del offset, total_len
        out[:payload_len] = b"\x00" * payload_len

    def pwrite_from_buffer(self, offset, data, payload_len=None, total_len=None):
        del offset, data, payload_len, total_len

    def batched_write(
        self,
        offsets: list[int],
        buffers: list[bytearray],
        total_lens: list[int],
        placement_ids: list[int | None] | None = None,
    ) -> int:
        del buffers
        self.batched_write_calls.append((offsets, total_lens, placement_ids))
        return 123

    def wait_iouring(self, batch_id: int) -> None:
        assert batch_id == 123

    def close(self) -> None:
        return None


def _make_fake_io_uring_core(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    use_uring_cmd: bool = False,
    max_data_transfer_size: int = 0,
) -> tuple[RawBlockCore, _FakeRawDevice]:
    raw_devices: list[_FakeRawDevice] = []

    def create_fake_device(path: str, **kwargs):
        del path, kwargs
        raw_device = _FakeRawDevice()
        raw_devices.append(raw_device)
        return raw_device

    monkeypatch.setitem(
        sys.modules,
        "lmcache_rust_raw_block_io",
        types.SimpleNamespace(RawBlockDevice=create_fake_device),
    )
    if use_uring_cmd:
        monkeypatch.setattr(
            raw_block_core.os,
            "stat",
            lambda path: types.SimpleNamespace(st_mode=stat.S_IFCHR),
        )

    device_path = tmp_path / "ng0n1"
    core = RawBlockCore(
        RawBlockCoreConfig(
            device_path=str(device_path),
            capacity_bytes=RAW_BLOCK_CI_CAPACITY_BYTES,
            block_align=RAW_BLOCK_CI_BLOCK_ALIGN,
            header_bytes=RAW_BLOCK_CI_HEADER_BYTES,
            slot_bytes=RAW_BLOCK_CI_SLOT_BYTES,
            use_odirect=False,
            enable_zero_copy=False,
            meta_total_bytes=RAW_BLOCK_CI_META_TOTAL_BYTES,
            meta_magic=b"LMCIDX01",
            meta_version=1,
            meta_checkpoint_interval_sec=60,
            meta_idle_quiet_ms=0,
            meta_enable_periodic=False,
            load_checkpoint_on_init=True,
            meta_verify_on_load=True,
            io_engine="io_uring",
            iouring_queue_depth=8,
            use_uring_cmd=use_uring_cmd,
            max_data_transfer_size=max_data_transfer_size,
        ),
        key_namespace="object",
    )
    return core, raw_devices[0]


def test_raw_block_core_put_many_preserves_none_and_positive_placement(
    tmp_path, monkeypatch
):
    core, raw_device = _make_fake_io_uring_core(tmp_path, monkeypatch)
    try:
        specs = [encode_object_key(make_object_key(500 + i)) for i in range(2)]
        put_result = core.put_many(
            specs,
            [make_memory_obj(b"a"), make_memory_obj(b"b")],
            placement_ids=[None, 1],
        )

        assert put_result.results == [True, True]
        assert [call[2] for call in raw_device.batched_write_calls] == [
            [None, None],
            [1, 1],
        ]
    finally:
        core.close()


def test_raw_block_core_put_many_rejects_zero_placement_before_io(
    tmp_path, monkeypatch
):
    core, raw_device = _make_fake_io_uring_core(tmp_path, monkeypatch)
    spec = encode_object_key(make_object_key(502))

    try:
        with pytest.raises(ValueError, match="placement identifier 0"):
            core.put_many([spec], [make_memory_obj(b"data")], placement_ids=[0])

        assert raw_device.batched_write_calls == []
    finally:
        core.close()


def test_raw_block_core_put_many_sets_same_placement_for_header_and_payload(
    tmp_path, monkeypatch
):
    core, raw_device = _make_fake_io_uring_core(tmp_path, monkeypatch)
    spec = encode_object_key(make_object_key(503))

    try:
        assert core.put_many(
            [spec], [make_memory_obj(b"data")], placement_ids=[1]
        ).results == [True]
        assert [call[2] for call in raw_device.batched_write_calls] == [[1, 1]]
    finally:
        core.close()


def test_raw_block_core_put_many_chunks_uring_cmd_with_placement_ids(
    tmp_path, monkeypatch
):
    core, raw_device = _make_fake_io_uring_core(
        tmp_path,
        monkeypatch,
        use_uring_cmd=True,
        max_data_transfer_size=RAW_BLOCK_CI_BLOCK_ALIGN,
    )
    spec = encode_object_key(make_object_key(504))

    try:
        assert core.put_many(
            [spec],
            [make_memory_obj(b"x" * (RAW_BLOCK_CI_BLOCK_ALIGN * 2))],
            placement_ids=[7],
        ).results == [True]

        assert len(raw_device.batched_write_calls) == 1
        _, total_lens, placement_ids = raw_device.batched_write_calls[0]
        assert total_lens == [
            RAW_BLOCK_CI_BLOCK_ALIGN,
            RAW_BLOCK_CI_BLOCK_ALIGN,
            RAW_BLOCK_CI_BLOCK_ALIGN,
        ]
        assert placement_ids == [7, 7, 7]
    finally:
        core.close()
