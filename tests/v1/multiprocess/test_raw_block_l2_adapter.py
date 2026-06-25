# SPDX-License-Identifier: Apache-2.0

# Future
from __future__ import annotations

# Standard
from pathlib import Path
from typing import Any
from unittest.mock import patch

# Third Party
import pytest

# First Party
from tests.v1.storage_backend.raw_block_test_utils import (
    RAW_BLOCK_CI_BLOCK_ALIGN,
    RAW_BLOCK_CI_CAPACITY_BYTES,
    RAW_BLOCK_CI_HEADER_BYTES,
    RAW_BLOCK_CI_META_TOTAL_BYTES,
    RAW_BLOCK_CI_SLOT_BYTES,
    install_native_storage_ops_fallback,
    make_empty_memory_obj,
    make_memory_obj,
    make_object_key,
    make_raw_block_file,
    memory_obj_bytes,
    wait_for_event_fd,
)

install_native_storage_ops_fallback()
pytest.importorskip("lmcache_rust_raw_block_io")


@pytest.fixture(autouse=True)
def isolate_raw_block_fdp_claim_dir(tmp_path, monkeypatch):
    """Keep FDP handle claim locks isolated per test."""
    monkeypatch.setenv("LMCACHE_RAW_BLOCK_FDP_CLAIM_DIR", str(tmp_path / "fdp_claims"))


# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey  # noqa: E402
from lmcache.v1.distributed.l2_adapters.raw_block_l2_adapter import (  # noqa: E402
    RawBlockL2Adapter,
    RawBlockL2AdapterConfig,
)
from lmcache.v1.storage_backend.raw_block import RawBlockPutManyResult  # noqa: E402

_EMPTY_LAYOUT = MemoryLayoutDesc(shapes=[], dtypes=[])


def _make_adapter(tmp_path: Path) -> RawBlockL2Adapter:
    path = make_raw_block_file(tmp_path)
    config = RawBlockL2AdapterConfig(
        device_path=str(path),
        capacity_bytes=RAW_BLOCK_CI_CAPACITY_BYTES,
        block_align=RAW_BLOCK_CI_BLOCK_ALIGN,
        header_bytes=RAW_BLOCK_CI_HEADER_BYTES,
        slot_bytes=RAW_BLOCK_CI_SLOT_BYTES,
        meta_total_bytes=RAW_BLOCK_CI_META_TOTAL_BYTES,
        use_odirect=False,
        enable_zero_copy=False,
        meta_enable_periodic=False,
        meta_idle_quiet_ms=0,
        io_engine="posix",
        iouring_queue_depth=8,
        num_store_workers=1,
        num_lookup_workers=1,
        num_load_workers=1,
    )
    return RawBlockL2Adapter(config)


def test_raw_block_l2_adapter_store_lookup_load_roundtrip(tmp_path):
    adapter = _make_adapter(tmp_path)
    try:
        key = make_object_key(1)
        missing_key = make_object_key(999)
        payload = b"raw-block-l2-adapter-payload"

        store_task_id = adapter.submit_store_task([key], [make_memory_obj(payload)])
        assert wait_for_event_fd(adapter.get_store_event_fd())
        store_result = adapter.pop_completed_store_tasks()[store_task_id]
        assert store_result.is_successful()
        assert store_result.bytes_transferred() == RAW_BLOCK_CI_SLOT_BYTES

        lookup_task_id = adapter.submit_lookup_and_lock_task(
            [key, missing_key], _EMPTY_LAYOUT
        )
        assert wait_for_event_fd(adapter.get_lookup_and_lock_event_fd())
        lookup_bitmap = adapter.query_lookup_and_lock_result(lookup_task_id)
        assert lookup_bitmap is not None
        assert lookup_bitmap.test(0) is True
        assert lookup_bitmap.test(1) is False

        loaded = make_empty_memory_obj(len(payload))
        missing = make_empty_memory_obj(len(payload))
        load_task_id = adapter.submit_load_task([key, missing_key], [loaded, missing])
        assert wait_for_event_fd(adapter.get_load_event_fd())
        load_bitmap = adapter.query_load_result(load_task_id)
        assert load_bitmap is not None
        assert load_bitmap.test(0) is True
        assert load_bitmap.test(1) is False
        assert memory_obj_bytes(loaded) == payload

        adapter.submit_unlock([key])
    finally:
        adapter.close()


def test_raw_block_l2_adapter_delete_makes_key_miss(tmp_path):
    adapter = _make_adapter(tmp_path)
    try:
        key = make_object_key(2)
        payload = b"delete-from-raw-block-l2"

        store_task_id = adapter.submit_store_task([key], [make_memory_obj(payload)])
        assert wait_for_event_fd(adapter.get_store_event_fd())
        store_result = adapter.pop_completed_store_tasks()[store_task_id]
        assert store_result.is_successful()
        assert store_result.bytes_transferred() == RAW_BLOCK_CI_SLOT_BYTES

        adapter.delete([key])

        lookup_task_id = adapter.submit_lookup_and_lock_task([key], _EMPTY_LAYOUT)
        assert wait_for_event_fd(adapter.get_lookup_and_lock_event_fd())
        lookup_bitmap = adapter.query_lookup_and_lock_result(lookup_task_id)
        assert lookup_bitmap is not None
        assert lookup_bitmap.test(0) is False
    finally:
        adapter.close()


class _FakeFdpCore:
    def __init__(self, status: list[tuple[int, int]] | None = None) -> None:
        self.status = status if status is not None else [(0, 10), (7, 17)]
        self.slot_bytes = RAW_BLOCK_CI_SLOT_BYTES
        self.put_many_calls: list[tuple[list[str], list[int | None] | None]] = []

    def fetch_fdp_status(self, max_ruhs: int = 256) -> list[tuple[int, int]]:
        return self.status

    def put_many(
        self,
        keys: list[Any],
        objs: list[Any],
        placement_ids: list[int | None] | None = None,
    ) -> RawBlockPutManyResult:
        del objs
        self.put_many_calls.append(([spec.encoded for spec in keys], placement_ids))
        return RawBlockPutManyResult(
            results=[True] * len(keys),
            stored_keys=[spec.encoded for spec in keys],
        )

    def report_status(self) -> dict:
        return {
            "is_healthy": True,
            "usable_capacity_bytes": RAW_BLOCK_CI_SLOT_BYTES * 8,
        }

    def snapshot_indexed_keys(self) -> list[str]:
        return []

    def close(self) -> None:
        pass


def _make_fdp_config(
    handles: list[int] | None = None,
    *,
    fdp_policy: str = "rank_isolation",
) -> RawBlockL2AdapterConfig:
    return RawBlockL2AdapterConfig(
        device_path="/dev/ng0n1",
        capacity_bytes=RAW_BLOCK_CI_CAPACITY_BYTES,
        block_align=RAW_BLOCK_CI_BLOCK_ALIGN,
        header_bytes=RAW_BLOCK_CI_HEADER_BYTES,
        slot_bytes=RAW_BLOCK_CI_SLOT_BYTES,
        meta_total_bytes=RAW_BLOCK_CI_META_TOTAL_BYTES,
        use_odirect=False,
        enable_zero_copy=False,
        meta_enable_periodic=False,
        meta_idle_quiet_ms=0,
        io_engine="io_uring",
        use_uring_cmd=True,
        iouring_queue_depth=8,
        fdp_enabled=True,
        fdp_policy=fdp_policy,  # type: ignore[arg-type]
        fdp_placement_handles=handles,
        num_store_workers=1,
        num_lookup_workers=1,
        num_load_workers=1,
    )


def _make_fdp_adapter(fake_core: _FakeFdpCore, config: RawBlockL2AdapterConfig):
    with patch(
        "lmcache.v1.distributed.l2_adapters.raw_block_l2_adapter.RawBlockCore",
        return_value=fake_core,
    ):
        return RawBlockL2Adapter(config)


def _with_cache_salt(key: ObjectKey, cache_salt: str) -> ObjectKey:
    return key.__class__(
        chunk_hash=key.chunk_hash,
        model_name=key.model_name,
        kv_rank=key.kv_rank,
        object_group_id=key.object_group_id,
        cache_salt=cache_salt,
    )


def test_raw_block_fdp_requires_uring_cmd_config():
    with pytest.raises(ValueError, match="fdp_enabled requires"):
        RawBlockL2AdapterConfig(
            device_path="/dev/ng0n1",
            slot_bytes=RAW_BLOCK_CI_SLOT_BYTES,
            io_engine="posix",
            use_uring_cmd=False,
            fdp_enabled=True,
        )


def test_raw_block_fdp_enabled_requires_policy():
    with pytest.raises(ValueError, match="fdp_enabled requires fdp_policy"):
        RawBlockL2AdapterConfig(
            device_path="/dev/ng0n1",
            slot_bytes=RAW_BLOCK_CI_SLOT_BYTES,
            io_engine="io_uring",
            use_uring_cmd=True,
            fdp_enabled=True,
        )


def test_raw_block_fdp_explicit_handle_validation_rejects_missing_handle():
    fake_core = _FakeFdpCore(status=[(1, 11)])
    config = _make_fdp_config(handles=[1, 2])

    with pytest.raises(ValueError, match="not reported"):
        _make_fdp_adapter(fake_core, config)


def test_raw_block_fdp_explicit_handle_validation_rejects_duplicate_handle():
    fake_core = _FakeFdpCore(status=[(0, 10), (7, 17)])
    config = _make_fdp_config(handles=[0, 0])

    with pytest.raises(ValueError, match="duplicate handles"):
        _make_fdp_adapter(fake_core, config)


def test_raw_block_fdp_empty_status_fails_startup():
    fake_core = _FakeFdpCore(status=[])
    config = _make_fdp_config()

    with pytest.raises(RuntimeError, match="no handles"):
        _make_fdp_adapter(fake_core, config)


def test_raw_block_fdp_status_is_reported():
    fake_core = _FakeFdpCore(status=[(0, 10), (7, 17)])
    adapter = _make_fdp_adapter(
        fake_core,
        _make_fdp_config(handles=[0]),
    )
    try:
        status = adapter.report_status()
        assert status["fdp_enabled"] is True
        assert status["fdp_discovered_status"] == [(0, 10), (7, 17)]
        assert status["fdp_placement_handles"] == [0]
    finally:
        adapter.close()


def test_raw_block_fdp_rank_isolation_requires_fdp_enabled():
    with pytest.raises(ValueError, match="fdp_policy requires"):
        RawBlockL2AdapterConfig(
            device_path="/dev/ng0n1",
            slot_bytes=RAW_BLOCK_CI_SLOT_BYTES,
            fdp_policy="rank_isolation",  # type: ignore[arg-type]
        )


def test_raw_block_fdp_rank_isolation_claims_handles_exclusively():
    status = [(handle, 100 + handle) for handle in range(8)]
    first_adapter = _make_fdp_adapter(
        _FakeFdpCore(status=status),
        _make_fdp_config(
            handles=[0, 1],
            fdp_policy="rank_isolation",
        ),
    )
    try:
        with pytest.raises(RuntimeError, match="already claimed"):
            _make_fdp_adapter(
                _FakeFdpCore(status=status),
                _make_fdp_config(
                    handles=[0, 1],
                    fdp_policy="rank_isolation",
                ),
            )

        second_adapter = _make_fdp_adapter(
            _FakeFdpCore(status=status),
            _make_fdp_config(
                handles=[2, 3],
                fdp_policy="rank_isolation",
            ),
        )
        second_adapter.close()
    finally:
        first_adapter.close()

    released_adapter = _make_fdp_adapter(
        _FakeFdpCore(status=status),
        _make_fdp_config(
            handles=[0, 1],
            fdp_policy="rank_isolation",
        ),
    )
    released_adapter.close()


def test_raw_block_fdp_rank_isolation_passes_per_key_placements():
    fake_core = _FakeFdpCore(status=[(0, 10), (5, 15)])
    config = _make_fdp_config(fdp_policy="rank_isolation")
    adapter = _make_fdp_adapter(fake_core, config)
    try:

        def with_rank(chunk_id: int, local_rank: int) -> ObjectKey:
            key = make_object_key(chunk_id)
            return key.__class__(
                chunk_hash=key.chunk_hash,
                model_name=key.model_name,
                kv_rank=ObjectKey.ComputeKVRank(4, local_rank, 4, local_rank),
                object_group_id=key.object_group_id,
                cache_salt=key.cache_salt,
            )

        keys = [with_rank(1, 1), with_rank(2, 0), with_rank(3, 1)]
        objects = [
            make_memory_obj(b"a"),
            make_memory_obj(b"b"),
            make_memory_obj(b"c"),
        ]
        task_id = adapter.submit_store_task(keys, objects)

        assert wait_for_event_fd(adapter.get_store_event_fd())
        assert adapter.pop_completed_store_tasks()[task_id].is_successful()
        assert fake_core.put_many_calls[0][1] == [5, 0, 5]
        status = adapter.report_status()
        assert status["fdp_policy"] == "rank_isolation"
        assert status["fdp_local_rank_to_placement"] == {0: 0, 1: 5}
    finally:
        adapter.close()


def test_raw_block_fdp_rank_isolation_fails_at_store_for_missing_handle():
    fake_core = _FakeFdpCore(status=[(0, 10)])
    adapter = _make_fdp_adapter(
        fake_core,
        _make_fdp_config(handles=[0], fdp_policy="rank_isolation"),
    )
    try:
        key = make_object_key(1)
        key = key.__class__(
            chunk_hash=key.chunk_hash,
            model_name=key.model_name,
            kv_rank=ObjectKey.ComputeKVRank(4, 1, 4, 1),
            object_group_id=key.object_group_id,
            cache_salt=key.cache_salt,
        )
        task_id = adapter.submit_store_task([key], [make_memory_obj(b"a")])

        assert wait_for_event_fd(adapter.get_store_event_fd())
        assert not adapter.pop_completed_store_tasks()[task_id].is_successful()
        assert fake_core.put_many_calls == []
        assert adapter.report_status()["fdp_local_rank_to_placement"] == {}
    finally:
        adapter.close()


def test_raw_block_fdp_domain_isolation_assigns_and_reuses_cache_salts():
    fake_core = _FakeFdpCore(status=[(0, 10), (7, 17)])
    config = _make_fdp_config(
        handles=[0, 7],
        fdp_policy="domain_isolation",
    )
    adapter = _make_fdp_adapter(fake_core, config)
    try:
        keys = [
            _with_cache_salt(make_object_key(1), "tenant-a"),
            _with_cache_salt(make_object_key(2), "tenant-b"),
            _with_cache_salt(make_object_key(3), "tenant-a"),
        ]
        objects = [make_memory_obj(b"a"), make_memory_obj(b"b"), make_memory_obj(b"c")]
        task_id = adapter.submit_store_task(keys, objects)

        assert wait_for_event_fd(adapter.get_store_event_fd())
        assert adapter.pop_completed_store_tasks()[task_id].is_successful()
        assert fake_core.put_many_calls[0][1] == [0, 7, 0]
        status = adapter.report_status()
        assert status["fdp_policy"] == "domain_isolation"
        assert status["fdp_cache_salt_to_placement"] == {
            "tenant-a": 0,
            "tenant-b": 7,
        }
    finally:
        adapter.close()


def test_raw_block_fdp_domain_isolation_exhaustion_uses_default_once():
    fake_core = _FakeFdpCore(status=[(0, 10)])
    config = _make_fdp_config(
        handles=[0],
        fdp_policy="domain_isolation",
    )
    adapter = _make_fdp_adapter(fake_core, config)
    warning_patch = patch(
        "lmcache.v1.distributed.l2_adapters.raw_block_l2_adapter.logger.warning"
    )
    warning_mock = warning_patch.start()
    try:
        keys = [
            _with_cache_salt(make_object_key(1), "tenant-a"),
            _with_cache_salt(make_object_key(2), "tenant-b"),
        ]
        task_id = adapter.submit_store_task(
            keys,
            [make_memory_obj(b"a"), make_memory_obj(b"b")],
        )
        assert wait_for_event_fd(adapter.get_store_event_fd())
        assert adapter.pop_completed_store_tasks()[task_id].is_successful()

        keys = [
            _with_cache_salt(make_object_key(3), "tenant-c"),
            _with_cache_salt(make_object_key(4), "tenant-b"),
        ]
        task_id = adapter.submit_store_task(
            keys,
            [make_memory_obj(b"c"), make_memory_obj(b"d")],
        )
        assert wait_for_event_fd(adapter.get_store_event_fd())
        assert adapter.pop_completed_store_tasks()[task_id].is_successful()

        assert fake_core.put_many_calls[0][1] == [0, None]
        assert fake_core.put_many_calls[1][1] == [None, None]
        status = adapter.report_status()
        assert status["fdp_cache_salt_to_placement"] == {
            "tenant-a": 0,
            "tenant-b": None,
            "tenant-c": None,
        }
        assert warning_mock.call_count == 1
        assert "domain_isolation exhausted placement handles" in str(
            warning_mock.call_args.args[0]
        )
    finally:
        warning_patch.stop()
        adapter.close()
