# SPDX-License-Identifier: Apache-2.0

# Future
from __future__ import annotations

# Standard
from dataclasses import replace

# Third Party
import pytest

# First Party
from lmcache.v1.storage_backend.raw_block import RawBlockCore, encode_object_key
from tests.v1.storage_backend.raw_block_test_utils import (
    RAW_BLOCK_CI_META_TOTAL_BYTES,
    make_empty_memory_obj,
    make_memory_obj,
    make_object_key,
    make_raw_block_core_config,
    make_raw_block_file,
    memory_obj_bytes,
)

pytest.importorskip("lmcache_rust_raw_block_io")


def _make_buddy_core_config(
    path,
    *,
    min_subslot_bytes: int = 8 * 1024,
    slot_bytes: int = 64 * 1024,
    capacity_bytes: int | None = None,
):
    if capacity_bytes is None:
        capacity_bytes = RAW_BLOCK_CI_META_TOTAL_BYTES + (4 * slot_bytes)
    base_config = make_raw_block_core_config(path, capacity_bytes=capacity_bytes)
    return replace(
        base_config,
        allocator="buddy",
        min_subslot_bytes=min_subslot_bytes,
        slot_bytes=slot_bytes,
    )


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


def test_raw_block_core_buddy_store_loads_variable_payload_blocks(tmp_path):
    path = make_raw_block_file(tmp_path)
    config = _make_buddy_core_config(path)
    core = RawBlockCore(config, key_namespace="object")

    try:
        specs = [encode_object_key(make_object_key(i)) for i in range(3)]
        payloads = [
            bytes([1]) * 1024,
            bytes([2]) * 9000,
            bytes([3]) * 30000,
        ]
        put_result = core.put_many(
            specs,
            [make_memory_obj(payload) for payload in payloads],
        )

        assert put_result.results == [True, True, True]
        assert put_result.stored_keys == [spec.encoded for spec in specs]
        assert core.allocated_bytes_many([spec.encoded for spec in specs]) == [
            8 * 1024,
            16 * 1024,
            64 * 1024,
        ]

        loaded = [make_empty_memory_obj(len(payload)) for payload in payloads]
        assert core.load_many_into([spec.encoded for spec in specs], loaded) == [
            True,
            True,
            True,
        ]
        assert [memory_obj_bytes(obj) for obj in loaded] == payloads
    finally:
        core.close()


def test_raw_block_core_buddy_status_uses_slot_buddy_counters(tmp_path):
    path = make_raw_block_file(tmp_path)
    config = _make_buddy_core_config(path)
    core = RawBlockCore(config, key_namespace="object")

    try:
        status = core.report_status()

        assert status["allocator"] == "buddy"
        assert status["slot_bytes"] == 64 * 1024
        assert status["min_subslot_bytes"] == 8 * 1024
        assert status["slot_count"] == 4
        assert status["max_subslot_count"] == 32
        assert status["free_block_count"] == 4
        assert "free_slot_count" not in status
        assert "next_slot" not in status
        assert "max_slots" not in status
    finally:
        core.close()


def test_raw_block_core_buddy_recovers_extended_checkpoint(tmp_path):
    path = make_raw_block_file(tmp_path)
    config = _make_buddy_core_config(path)
    keep_spec = encode_object_key(make_object_key(100))
    drop_spec = encode_object_key(make_object_key(101))
    payload = b"recoverable-buddy" * 512

    core = RawBlockCore(config, key_namespace="object")
    try:
        put_result = core.put_many(
            [keep_spec, drop_spec],
            [make_memory_obj(payload), make_memory_obj(b"drop")],
        )
        assert put_result.results == [True, True]
        assert core.delete_many([drop_spec.encoded]) == [True]
        core.checkpoint_now()
    finally:
        core.close()

    recovered = RawBlockCore(config, key_namespace="object")
    try:
        assert recovered.contains_key(keep_spec.encoded) is True
        assert recovered.contains_key(drop_spec.encoded) is False
        assert recovered.allocated_bytes_many([keep_spec.encoded]) == [16 * 1024]

        loaded = make_empty_memory_obj(len(payload))
        assert recovered.load_many_into([keep_spec.encoded], [loaded]) == [True]
        assert memory_obj_bytes(loaded) == payload
    finally:
        recovered.close()


def test_raw_block_core_buddy_recovery_frees_stale_entries(tmp_path):
    path = make_raw_block_file(tmp_path)
    config = _make_buddy_core_config(
        path,
        min_subslot_bytes=8 * 1024,
        slot_bytes=32 * 1024,
        capacity_bytes=RAW_BLOCK_CI_META_TOTAL_BYTES + (32 * 1024),
    )
    specs = [encode_object_key(make_object_key(130 + i)) for i in range(4)]
    stale_spec = specs[1]

    core = RawBlockCore(config, key_namespace="object")
    try:
        put_result = core.put_many(
            specs,
            [make_memory_obj(bytes([i + 1]) * 1024) for i in range(4)],
        )
        assert put_result.results == [True, True, True, True]
        stale_offset = core.entry_offset(stale_spec.encoded)
        assert stale_offset is not None
        core.checkpoint_now()
    finally:
        core.close()

    with open(path, "r+b") as f:
        f.seek(stale_offset)
        f.write(b"\x00" * config.header_bytes)

    recovered = RawBlockCore(config, key_namespace="object")
    try:
        assert recovered.contains_key(stale_spec.encoded) is False
        assert recovered.allocated_bytes_many([stale_spec.encoded]) == [0]

        replacement = encode_object_key(make_object_key(200))
        replacement_result = recovered.put_many(
            [replacement],
            [make_memory_obj(b"replacement")],
        )

        assert replacement_result.results == [True]
        assert replacement_result.stored_keys == [replacement.encoded]
        assert recovered.report_status()["indexed_key_count"] == 4
    finally:
        recovered.close()


def test_raw_block_core_buddy_rejects_checkpoint_with_larger_managed_span(tmp_path):
    path = make_raw_block_file(tmp_path)
    large_config = _make_buddy_core_config(path)
    small_config = _make_buddy_core_config(
        path,
        capacity_bytes=RAW_BLOCK_CI_META_TOTAL_BYTES + (64 * 1024),
    )
    old_spec = encode_object_key(make_object_key(110))

    core = RawBlockCore(large_config, key_namespace="object")
    try:
        assert core.put_many([old_spec], [make_memory_obj(b"old")]).results == [True]
        core.checkpoint_now()
    finally:
        core.close()

    recovered = RawBlockCore(small_config, key_namespace="object")
    try:
        assert recovered.contains_key(old_spec.encoded) is False
        status = recovered.report_status()
        assert status["usable_capacity_bytes"] == 64 * 1024
        assert status["indexed_key_count"] == 0

        specs = [encode_object_key(make_object_key(111 + i)) for i in range(2)]
        result = recovered.put_many(
            specs,
            [make_memory_obj(b"x" * (50 * 1024)) for _ in specs],
        )

        assert result.results == [True, False]
    finally:
        recovered.close()


def test_raw_block_core_buddy_respects_locks_before_delete(tmp_path):
    path = make_raw_block_file(tmp_path)
    config = _make_buddy_core_config(path)
    spec = encode_object_key(make_object_key(120))
    core = RawBlockCore(config, key_namespace="object")

    try:
        assert core.put_many([spec], [make_memory_obj(b"locked")]).results == [True]
        assert core.exists_many([spec.encoded], lock=True) == [True]

        assert core.delete_many([spec.encoded]) == [False]
        assert core.contains_key(spec.encoded) is True

        core.unlock_many([spec.encoded])
        assert core.delete_many([spec.encoded]) == [True]
        assert core.contains_key(spec.encoded) is False
    finally:
        core.close()


def test_raw_block_core_buddy_fails_when_fragmented(tmp_path):
    path = make_raw_block_file(tmp_path)
    config = _make_buddy_core_config(
        path,
        min_subslot_bytes=8 * 1024,
        slot_bytes=32 * 1024,
        capacity_bytes=RAW_BLOCK_CI_META_TOTAL_BYTES + (32 * 1024),
    )
    core = RawBlockCore(config, key_namespace="object")

    try:
        specs = [encode_object_key(make_object_key(i + 200)) for i in range(4)]
        result = core.put_many(specs, [make_memory_obj(b"x" * 1024) for _ in specs])
        assert result.results == [True, True, True, True]
        assert core.delete_many([specs[0].encoded, specs[2].encoded]) == [True, True]

        needs_merged_block = encode_object_key(make_object_key(300))
        failed = core.put_many(
            [needs_merged_block],
            [make_memory_obj(b"z" * 9000)],
        )

        assert failed.results == [False]
        assert failed.stored_keys == []
        assert core.contains_key(needs_merged_block.encoded) is False
    finally:
        core.close()
