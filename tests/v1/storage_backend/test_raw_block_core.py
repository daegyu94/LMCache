# SPDX-License-Identifier: Apache-2.0

# Future
from __future__ import annotations

# Third Party
import pytest

# First Party
from lmcache.v1.storage_backend.raw_block import RawBlockCore, encode_object_key
from tests.v1.storage_backend.raw_block_test_utils import (
    make_empty_memory_obj,
    make_memory_obj,
    make_object_key,
    make_raw_block_core_config,
    make_raw_block_file,
    memory_obj_bytes,
)

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


def test_raw_block_core_apply_loaded_state_skips_invalid_entry(tmp_path):
    path = make_raw_block_file(tmp_path)
    config = make_raw_block_core_config(path)

    seed_core = RawBlockCore(config, key_namespace="object")
    try:
        valid_spec = encode_object_key(make_object_key(41))
        payload = b"checkpoint-survivor"
        put_result = seed_core.put_many([valid_spec], [make_memory_obj(payload)])
        assert put_result.results == [True]

        valid_offset = seed_core.entry_offset(valid_spec.encoded)
        assert valid_offset is not None
    finally:
        seed_core.close()

    recovered = RawBlockCore(config, key_namespace="object")
    try:
        checkpoint_state = {
            "version": 1,
            "device_path": str(path),
            "capacity_bytes": config.capacity_bytes,
            "block_align": config.block_align,
            "header_bytes": config.header_bytes,
            "slot_bytes": config.slot_bytes,
            "meta_total_bytes": config.meta_total_bytes,
            "meta_magic": config.meta_magic.decode("ascii"),
            "meta_version": config.meta_version,
            "data_base_offset": config.meta_total_bytes,
            "next_slot": 2,
            "free_slots": [],
            "entries": {
                valid_spec.encoded: {
                    "offset": valid_offset,
                    "size": len(payload),
                    "shape": [len(payload)],
                    "dtype": "uint8",
                    "fmt": "BINARY",
                    "cached_positions": None,
                },
                "bad-offset-entry": {
                    # Invalid offset: int("not-an-int") raises ValueError.
                    "offset": "not-an-int",
                    "size": 8,
                    "shape": [8],
                    "dtype": "uint8",
                    "fmt": "BINARY",
                    "cached_positions": None,
                },
                "bad-shape-entry": {
                    "offset": valid_offset + config.slot_bytes,
                    "size": 8,
                    # Invalid shape: object() is not iterable, so TypeError is raised.
                    "shape": object(),
                    "dtype": "uint8",
                    "fmt": "BINARY",
                    "cached_positions": None,
                },
            },
        }

        assert recovered.apply_loaded_state(checkpoint_state) is True
        assert recovered.contains_key(valid_spec.encoded) is True
        assert recovered.contains_key("bad-offset-entry") is False
        assert recovered.contains_key("bad-shape-entry") is False

        loaded = make_empty_memory_obj(len(payload))
        assert recovered.load_many_into([valid_spec.encoded], [loaded]) == [True]
        assert memory_obj_bytes(loaded) == payload
    finally:
        recovered.close()
