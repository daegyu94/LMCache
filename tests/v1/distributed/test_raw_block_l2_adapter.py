# SPDX-License-Identifier: Apache-2.0

# Standard
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch
import json
import os
import select
import tempfile
import time

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.internal_api import L2AdapterListener
from lmcache.v1.distributed.l2_adapters.raw_block_l2_adapter import (
    RawBlockL2Adapter,
    RawBlockL2AdapterConfig,
)
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObjMetadata,
    TensorMemoryObj,
)
from lmcache.v1.storage_backend.raw_block import encode_object_key


def _has_ext() -> bool:
    try:
        # Third Party
        import lmcache_rust_raw_block_io  # noqa: F401

        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _has_ext(), reason="lmcache_rust_raw_block_io extension not installed"
)


class _RecordingListener(L2AdapterListener):
    def __init__(self):
        self.stored: list[list[ObjectKey]] = []
        self.stored_sizes: list[list[int] | None] = []
        self.accessed: list[list[ObjectKey]] = []
        self.deleted: list[list[ObjectKey]] = []
        self.deleted_sizes: list[list[int] | None] = []

    def on_l2_keys_stored(
        self,
        keys: list[ObjectKey],
        object_sizes: list[int] | None = None,
    ):
        self.stored.append(list(keys))
        self.stored_sizes.append(None if object_sizes is None else list(object_sizes))

    def on_l2_keys_accessed(self, keys: list[ObjectKey]):
        self.accessed.append(list(keys))

    def on_l2_keys_deleted(
        self,
        keys: list[ObjectKey],
        object_sizes: list[int] | None = None,
    ):
        self.deleted.append(list(keys))
        self.deleted_sizes.append(None if object_sizes is None else list(object_sizes))


class _FailingListener(L2AdapterListener):
    def on_l2_keys_stored(
        self,
        keys: list[ObjectKey],
        object_sizes: list[int] | None = None,
    ):
        del object_sizes
        raise RuntimeError("store listener failed")

    def on_l2_keys_accessed(self, keys: list[ObjectKey]):
        raise RuntimeError("access listener failed")

    def on_l2_keys_deleted(
        self,
        keys: list[ObjectKey],
        object_sizes: list[int] | None = None,
    ):
        del object_sizes
        raise RuntimeError("delete listener failed")


def _create_object_key(chunk_id: int, model_name: str = "test_model") -> ObjectKey:
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(chunk_id),
        model_name=model_name,
        kv_rank=0,
    )


def _create_memory_obj(size: int = 1024, fill_value: float = 0.0) -> TensorMemoryObj:
    raw_data = torch.empty(size, dtype=torch.float32)
    raw_data.fill_(fill_value)
    metadata = MemoryObjMetadata(
        shape=torch.Size([size]),
        dtype=torch.float32,
        address=0,
        phy_size=size * 4,
        fmt=MemoryFormat.KV_2LTD,
        ref_count=1,
    )
    return TensorMemoryObj(raw_data, metadata, parent_allocator=None)


def _create_complex_memory_obj(
    size: int = 1024,
    fill_value: complex = 0j,
) -> TensorMemoryObj:
    raw_data = torch.empty(size, dtype=torch.complex64)
    raw_data.fill_(fill_value)
    metadata = MemoryObjMetadata(
        shape=torch.Size([size]),
        dtype=torch.complex64,
        address=0,
        phy_size=raw_data.numel() * raw_data.element_size(),
        fmt=MemoryFormat.KV_2LTD,
        ref_count=1,
    )
    return TensorMemoryObj(raw_data, metadata, parent_allocator=None)


def _wait_event_fd(event_fd: int, timeout: float = 5.0) -> bool:
    poll = select.poll()
    poll.register(event_fd, select.POLLIN)
    events = poll.poll(timeout * 1000)
    if events:
        try:
            os.eventfd_read(event_fd)
        except BlockingIOError:
            pass
        return True
    return False


def _wait_for_condition(
    predicate,
    timeout: float = 5.0,
    poll_interval: float = 0.05,
) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(poll_interval)
    return False


def _make_config(
    device_path: str,
    *,
    slot_bytes: int = 64 * 1024,
    capacity_bytes: int = 0,
    base_offset_bytes: int = 0,
    use_uring: bool = False,
    use_uring_cmd: bool = False,
    use_fdp: bool = False,
    fdp_ruh_ids: tuple[int, ...] = (),
    fdp_data_ruh_ids: tuple[int, ...] = (),
    fdp_metadata_ruh_ids: tuple[int, ...] = (),
    meta_total_bytes: int = 1 * 1024 * 1024,
    meta_magic: str = "LMCIDX01",
    latency_log_path: str | None = None,
) -> RawBlockL2AdapterConfig:
    return RawBlockL2AdapterConfig(
        device_path=device_path,
        slot_bytes=slot_bytes,
        capacity_bytes=capacity_bytes,
        base_offset_bytes=base_offset_bytes,
        use_odirect=False,
        use_uring=use_uring,
        use_uring_cmd=use_uring_cmd,
        use_fdp=use_fdp,
        fdp_ruh_ids=fdp_ruh_ids,
        fdp_data_ruh_ids=fdp_data_ruh_ids,
        fdp_metadata_ruh_ids=fdp_metadata_ruh_ids,
        block_align=4096,
        header_bytes=4096,
        meta_total_bytes=meta_total_bytes,
        meta_magic=meta_magic,
        meta_enable_periodic=False,
        num_store_workers=2,
        num_lookup_workers=1,
        num_load_workers=2,
        latency_log_path=latency_log_path,
    )


def _run_store(adapter: RawBlockL2Adapter, keys, objects) -> bool:
    task_id = adapter.submit_store_task(keys, objects)
    assert _wait_event_fd(adapter.get_store_event_fd())
    completed = adapter.pop_completed_store_tasks()
    assert task_id in completed
    return completed[task_id]


def test_raw_block_l2_adapter_config_parses_uring_flags():
    default_cfg = RawBlockL2AdapterConfig.from_dict(
        {
            "type": "raw_block",
            "device_path": "/tmp/raw-block-dev",
            "slot_bytes": 64 * 1024,
        }
    )
    assert default_cfg.use_odirect is False

    cfg = RawBlockL2AdapterConfig.from_dict(
        {
            "type": "raw_block",
            "device_path": "/tmp/raw-block-dev",
            "slot_bytes": 64 * 1024,
            "use_odirect": False,
            "use_uring": True,
        }
    )

    assert cfg.use_uring is True
    assert cfg.use_uring_cmd is False
    assert cfg.to_core_config().use_uring is True

    with pytest.raises(ValueError, match="use_uring_cmd requires use_uring"):
        RawBlockL2AdapterConfig.from_dict(
            {
                "type": "raw_block",
                "device_path": "/tmp/raw-block-dev",
                "slot_bytes": 64 * 1024,
                "use_uring_cmd": True,
            }
        )

    fdp_cfg = RawBlockL2AdapterConfig.from_dict(
        {
            "type": "raw_block",
            "device_path": "/dev/ng0n1",
            "slot_bytes": 64 * 1024,
            "capacity_bytes": 8 * 1024 * 1024,
            "base_offset_bytes": 4 * 1024 * 1024,
            "meta_total_bytes": 1 * 1024 * 1024,
            "use_uring": True,
            "use_uring_cmd": True,
            "use_fdp": True,
            "fdp_ruh_ids": [3, 4],
        }
    )
    assert fdp_cfg.use_fdp is True
    assert fdp_cfg.fdp_ruh_ids == (3, 4)
    assert fdp_cfg.fdp_data_ruh_ids == (3, 4)
    assert fdp_cfg.fdp_metadata_ruh_ids == (3, 4)
    assert fdp_cfg.base_offset_bytes == 4 * 1024 * 1024
    core_cfg = fdp_cfg.to_core_config()
    assert core_cfg.use_fdp is True
    assert core_cfg.fdp_ruh_ids == (3, 4)
    assert core_cfg.fdp_data_ruh_ids == (3, 4)
    assert core_cfg.fdp_metadata_ruh_ids == (3, 4)
    assert core_cfg.base_offset_bytes == 4 * 1024 * 1024

    split_cfg = RawBlockL2AdapterConfig.from_dict(
        {
            "type": "raw_block",
            "device_path": "/dev/ng0n1",
            "slot_bytes": 64 * 1024,
            "capacity_bytes": 8 * 1024 * 1024,
            "meta_total_bytes": 1 * 1024 * 1024,
            "use_uring": True,
            "use_uring_cmd": True,
            "use_fdp": True,
            "fdp_data_ruh_ids": [0, 1, 5, 6],
            "fdp_metadata_ruh_ids": [2],
        }
    )
    assert split_cfg.fdp_ruh_ids == ()
    assert split_cfg.fdp_data_ruh_ids == (0, 1, 5, 6)
    assert split_cfg.fdp_metadata_ruh_ids == (2,)
    split_core_cfg = split_cfg.to_core_config()
    assert split_core_cfg.fdp_data_ruh_ids == (0, 1, 5, 6)
    assert split_core_cfg.fdp_metadata_ruh_ids == (2,)

    with pytest.raises(ValueError, match="base_offset_bytes"):
        RawBlockL2AdapterConfig.from_dict(
            {
                "type": "raw_block",
                "device_path": "/tmp/raw-block-dev",
                "slot_bytes": 64 * 1024,
                "base_offset_bytes": 1,
            }
        )

    with pytest.raises(ValueError, match="use_fdp requires use_uring_cmd"):
        RawBlockL2AdapterConfig.from_dict(
            {
                "type": "raw_block",
                "device_path": "/dev/ng0n1",
                "slot_bytes": 64 * 1024,
                "use_uring": True,
                "use_fdp": True,
                "fdp_ruh_ids": [1],
            }
        )
    with pytest.raises(ValueError, match="non-empty fdp_data_ruh_ids"):
        RawBlockL2AdapterConfig.from_dict(
            {
                "type": "raw_block",
                "device_path": "/dev/ng0n1",
                "slot_bytes": 64 * 1024,
                "use_uring": True,
                "use_uring_cmd": True,
                "use_fdp": True,
            }
        )
    with pytest.raises(ValueError, match="non-empty fdp_metadata_ruh_ids"):
        RawBlockL2AdapterConfig.from_dict(
            {
                "type": "raw_block",
                "device_path": "/dev/ng0n1",
                "slot_bytes": 64 * 1024,
                "use_uring": True,
                "use_uring_cmd": True,
                "use_fdp": True,
                "fdp_data_ruh_ids": [1],
            }
        )
    with pytest.raises(ValueError, match="leave space"):
        RawBlockL2AdapterConfig.from_dict(
            {
                "type": "raw_block",
                "device_path": "/dev/ng0n1",
                "slot_bytes": 64 * 1024,
                "capacity_bytes": 2 * 1024 * 1024,
                "meta_total_bytes": 1 * 1024 * 1024,
                "use_uring": True,
                "use_uring_cmd": True,
                "use_fdp": True,
                "fdp_ruh_ids": [1, 2],
            }
        )


def test_raw_block_l2_adapter_uring_cmd_rejects_regular_file():
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dev.bin")
        with open(dev_path, "wb") as f:
            f.truncate(8 * 1024 * 1024)

        with pytest.raises(ValueError, match="NVMe namespace character device"):
            RawBlockL2Adapter(
                _make_config(
                    dev_path,
                    use_uring=True,
                    use_uring_cmd=True,
                )
            )


def _run_lookup(adapter: RawBlockL2Adapter, keys):
    task_id = adapter.submit_lookup_and_lock_task(keys)
    assert _wait_event_fd(adapter.get_lookup_and_lock_event_fd())
    return task_id, adapter.query_lookup_and_lock_result(task_id)


def _run_load(adapter: RawBlockL2Adapter, keys, objects):
    task_id = adapter.submit_load_task(keys, objects)
    assert _wait_event_fd(adapter.get_load_event_fd())
    return task_id, adapter.query_load_result(task_id)


def test_raw_block_l2_adapter_store_lookup_load_roundtrip():
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dev.bin")
        with open(dev_path, "wb") as f:
            f.truncate(8 * 1024 * 1024)

        adapter = RawBlockL2Adapter(_make_config(dev_path))
        try:
            key1 = _create_object_key(1)
            key_miss = _create_object_key(2)
            key3 = _create_object_key(3)
            obj1 = _create_memory_obj(fill_value=1.0)
            obj3 = _create_memory_obj(fill_value=3.0)

            assert _run_store(adapter, [key1, key3], [obj1, obj3]) is True

            lookup_task_id, lookup_bitmap = _run_lookup(
                adapter,
                [key1, key_miss, key3],
            )
            assert lookup_bitmap is not None
            assert lookup_bitmap.get_indices_list() == [0, 2]
            assert adapter.query_lookup_and_lock_result(lookup_task_id) is None

            load_buffers = [
                _create_memory_obj(fill_value=0.0),
                _create_memory_obj(fill_value=0.0),
                _create_memory_obj(fill_value=0.0),
            ]
            load_task_id, load_bitmap = _run_load(
                adapter,
                [key1, key_miss, key3],
                load_buffers,
            )
            assert load_bitmap is not None
            assert load_bitmap.get_indices_list() == [0, 2]
            assert adapter.query_load_result(load_task_id) is None
            assert torch.equal(load_buffers[0].tensor, obj1.tensor)
            assert torch.equal(load_buffers[2].tensor, obj3.tensor)
            assert torch.count_nonzero(load_buffers[1].tensor) == 0

            adapter.submit_unlock([key1, key_miss, key3])
        finally:
            adapter.close()


def test_raw_block_l2_adapter_records_e2e_and_device_io_latency():
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dev.bin")
        latency_path = os.path.join(td, "l2_latency.jsonl")
        with open(dev_path, "wb") as file_obj:
            file_obj.truncate(8 * 1024 * 1024)

        adapter = RawBlockL2Adapter(
            _make_config(dev_path, latency_log_path=latency_path)
        )
        try:
            key = _create_object_key(101)
            stored = _create_memory_obj(fill_value=101.0)
            assert _run_store(adapter, [key], [stored]) is True

            _, lookup = _run_lookup(adapter, [key])
            assert lookup is not None
            loaded = _create_memory_obj(fill_value=0.0)
            _, load_result = _run_load(adapter, [key], [loaded])
            assert load_result is not None
            assert torch.equal(loaded.tensor, stored.tensor)
            adapter.submit_unlock([key])
        finally:
            adapter.close()

        with open(latency_path) as file_obj:
            records = [json.loads(line) for line in file_obj if line.strip()]

        metrics = {record["metric"] for record in records}
        assert {
            "l2_e2e_write",
            "l2_e2e_read",
            "raw_block_write",
            "raw_block_read",
        }.issubset(metrics)
        assert any(
            record["metric"] == "raw_block_write"
            and record["io_class"] == "data"
            and record["latency_ms"] >= 0
            for record in records
        )


def test_raw_block_l2_adapter_base_offset_separates_same_device_windows():
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dev.bin")
        with open(dev_path, "wb") as f:
            f.truncate(8 * 1024 * 1024)

        meta_total_bytes = 1 * 1024 * 1024
        capacity_bytes = 2 * 1024 * 1024
        second_base = 4 * 1024 * 1024
        adapter1 = RawBlockL2Adapter(
            _make_config(
                dev_path,
                capacity_bytes=capacity_bytes,
                meta_total_bytes=meta_total_bytes,
                meta_magic="WIN00001",
            )
        )
        adapter2 = RawBlockL2Adapter(
            _make_config(
                dev_path,
                capacity_bytes=capacity_bytes,
                base_offset_bytes=second_base,
                meta_total_bytes=meta_total_bytes,
                meta_magic="WIN00002",
            )
        )

        try:
            key1 = _create_object_key(51, model_name="window-1")
            key2 = _create_object_key(52, model_name="window-2")
            obj1 = _create_memory_obj(fill_value=51.0)
            obj2 = _create_memory_obj(fill_value=52.0)

            with ThreadPoolExecutor(max_workers=2) as executor:
                fut1 = executor.submit(_run_store, adapter1, [key1], [obj1])
                fut2 = executor.submit(_run_store, adapter2, [key2], [obj2])
                assert fut1.result(timeout=10) is True
                assert fut2.result(timeout=10) is True

            offset1 = adapter1._core.entry_offset(encode_object_key(key1).encoded)
            offset2 = adapter2._core.entry_offset(encode_object_key(key2).encoded)
            assert offset1 is not None
            assert offset2 is not None
            assert meta_total_bytes <= offset1 < capacity_bytes
            assert (
                second_base + meta_total_bytes
                <= offset2
                < (second_base + capacity_bytes)
            )

            load1 = _create_memory_obj(fill_value=0.0)
            load2 = _create_memory_obj(fill_value=0.0)
            assert adapter1._core.load_many_into(
                [encode_object_key(key1).encoded], [load1]
            ) == [True]
            assert adapter2._core.load_many_into(
                [encode_object_key(key2).encoded], [load2]
            ) == [True]
            assert torch.equal(load1.tensor, obj1.tensor)
            assert torch.equal(load2.tensor, obj2.tensor)
        finally:
            adapter1.close()
            adapter2.close()


def test_raw_block_l2_adapter_uring_store_lookup_load_roundtrip():
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dev.bin")
        with open(dev_path, "wb") as f:
            f.truncate(8 * 1024 * 1024)

        adapter = RawBlockL2Adapter(_make_config(dev_path, use_uring=True))
        try:
            key = _create_object_key(5)
            obj = _create_memory_obj(fill_value=5.0)
            assert _run_store(adapter, [key], [obj]) is True

            _, lookup = _run_lookup(adapter, [key])
            assert lookup is not None
            assert lookup.get_indices_list() == [0]

            load_buffer = _create_memory_obj(fill_value=0.0)
            _, loaded = _run_load(adapter, [key], [load_buffer])
            assert loaded is not None
            assert loaded.get_indices_list() == [0]
            assert torch.equal(load_buffer.tensor, obj.tensor)
            adapter.submit_unlock([key])
        finally:
            adapter.close()


def test_raw_block_l2_adapter_delete_respects_lock_until_unlock():
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dev.bin")
        with open(dev_path, "wb") as f:
            f.truncate(8 * 1024 * 1024)

        adapter = RawBlockL2Adapter(_make_config(dev_path))
        try:
            key = _create_object_key(11)
            obj = _create_memory_obj(fill_value=11.0)
            assert _run_store(adapter, [key], [obj]) is True

            _, bitmap = _run_lookup(adapter, [key])
            assert bitmap is not None
            assert bitmap.get_indices_list() == [0]

            adapter.delete([key])
            _, still_present = _run_lookup(adapter, [key])
            assert still_present is not None
            assert still_present.get_indices_list() == [0]
            adapter.submit_unlock([key, key])

            adapter.delete([key])
            _, after_delete = _run_lookup(adapter, [key])
            assert after_delete is not None
            assert after_delete.get_indices_list() == []
            adapter.submit_unlock([key])
        finally:
            adapter.close()


def test_raw_block_l2_adapter_listeners_usage_and_internal_eviction():
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dev.bin")
        with open(dev_path, "wb") as f:
            f.truncate(8 * 1024 * 1024)

        slot_bytes = 64 * 1024
        capacity_bytes = (1 * 1024 * 1024) + slot_bytes
        adapter = RawBlockL2Adapter(
            _make_config(
                dev_path,
                slot_bytes=slot_bytes,
                capacity_bytes=capacity_bytes,
            )
        )
        listener = _RecordingListener()
        adapter.register_listener(listener)

        try:
            key1 = _create_object_key(21)
            key2 = _create_object_key(22)
            obj1 = _create_memory_obj(fill_value=21.0)
            obj2 = _create_memory_obj(fill_value=22.0)

            assert _run_store(adapter, [key1], [obj1]) is True
            assert _run_store(adapter, [key2], [obj2]) is True

            assert listener.stored[0] == [key1]
            assert listener.stored_sizes[0] == [obj1.get_size()]
            assert listener.deleted[-1] == [key1]
            assert listener.deleted_sizes[-1] == [obj1.get_size()]
            assert listener.stored[-1] == [key2]
            assert listener.stored_sizes[-1] == [obj2.get_size()]

            usage = adapter.get_usage()
            assert usage.total_bytes_used == obj2.get_size()
            assert usage.total_capacity_bytes == slot_bytes
            assert 0.0 < usage.usage_fraction <= 1.0
            assert adapter.supports_global_eviction is False

            status = adapter.report_status()
            assert status["is_healthy"] is True
            assert status["type"] == "RawBlockL2Adapter"
            assert status["core"]["usable_capacity_bytes"] == slot_bytes
            accounting = status["core"]["io_accounting"]
            assert accounting["eviction_count"] == 1
            assert accounting["eviction_logical_bytes"] == obj1.get_size()

            _, bitmap1 = _run_lookup(adapter, [key1])
            assert bitmap1 is not None
            assert bitmap1.get_indices_list() == []
            _, bitmap2 = _run_lookup(adapter, [key2])
            assert bitmap2 is not None
            assert bitmap2.get_indices_list() == [0]
            adapter.submit_unlock([key1, key2])
        finally:
            adapter.close()


def test_raw_block_l2_adapter_full_locked_store_fails_then_evicts_after_unlock():
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dev.bin")
        with open(dev_path, "wb") as f:
            f.truncate(8 * 1024 * 1024)

        slot_bytes = 64 * 1024
        capacity_bytes = (1 * 1024 * 1024) + slot_bytes
        adapter = RawBlockL2Adapter(
            _make_config(
                dev_path,
                slot_bytes=slot_bytes,
                capacity_bytes=capacity_bytes,
            )
        )
        listener = _RecordingListener()
        adapter.register_listener(listener)

        try:
            key1 = _create_object_key(23)
            key2 = _create_object_key(24)
            obj1 = _create_memory_obj(fill_value=23.0)
            obj2 = _create_memory_obj(fill_value=24.0)

            assert _run_store(adapter, [key1], [obj1]) is True
            _, locked = _run_lookup(adapter, [key1])
            assert locked is not None
            assert locked.get_indices_list() == [0]

            assert _run_store(adapter, [key2], [obj2]) is False
            _, still_locked = _run_lookup(adapter, [key1, key2])
            assert still_locked is not None
            assert still_locked.get_indices_list() == [0]
            adapter.submit_unlock([key1, key1])

            assert listener.stored == [[key1]]
            assert listener.deleted == []
            before_eviction = adapter.report_status()["core"]["io_accounting"]
            assert before_eviction["eviction_count"] == 0
            assert before_eviction["eviction_logical_bytes"] == 0

            assert _run_store(adapter, [key2], [obj2]) is True
            assert listener.deleted == [[key1]]
            assert listener.deleted_sizes == [[obj1.get_size()]]
            assert listener.stored[-1] == [key2]
            assert listener.stored_sizes[-1] == [obj2.get_size()]

            _, after_eviction = _run_lookup(adapter, [key1, key2])
            assert after_eviction is not None
            assert after_eviction.get_indices_list() == [1]
            after_accounting = adapter.report_status()["core"]["io_accounting"]
            assert after_accounting["eviction_count"] == 1
            assert after_accounting["eviction_logical_bytes"] == obj1.get_size()
            adapter.submit_unlock([key1, key2])
        finally:
            adapter.close()


def test_raw_block_l2_adapter_fdp_full_locked_store_fails_then_evicts_after_unlock():
    dev_path = os.environ.get("LMCACHE_RAW_BLOCK_FDP_TEST_DEVICE")
    if not dev_path:
        pytest.skip("LMCACHE_RAW_BLOCK_FDP_TEST_DEVICE is not set")
    ruh_ids = tuple(
        int(item)
        for item in os.environ.get("LMCACHE_RAW_BLOCK_FDP_RUH_IDS", "0,1,5,6").split(
            ","
        )
        if item.strip()
    )
    if not ruh_ids:
        pytest.skip("LMCACHE_RAW_BLOCK_FDP_RUH_IDS is empty")

    slot_bytes = 64 * 1024
    meta_total_bytes = 1 * 1024 * 1024
    capacity_bytes = meta_total_bytes * len(ruh_ids) + slot_bytes
    adapter = RawBlockL2Adapter(
        _make_config(
            dev_path,
            slot_bytes=slot_bytes,
            capacity_bytes=capacity_bytes,
            use_uring=True,
            use_uring_cmd=True,
            use_fdp=True,
            fdp_ruh_ids=ruh_ids,
            meta_total_bytes=meta_total_bytes,
            meta_magic="TSTFDP01",
        )
    )

    try:
        key1 = _create_object_key(2300, model_name="fdp-evict")
        key2 = _create_object_key(2400, model_name="fdp-evict")
        obj1 = _create_memory_obj(fill_value=23.0)
        obj2 = _create_memory_obj(fill_value=24.0)

        assert _run_store(adapter, [key1], [obj1]) is True
        _, locked = _run_lookup(adapter, [key1])
        assert locked is not None
        assert locked.get_indices_list() == [0]

        assert _run_store(adapter, [key2], [obj2]) is False
        adapter.submit_unlock([key1])

        assert _run_store(adapter, [key2], [obj2]) is True
        _, after_eviction = _run_lookup(adapter, [key1, key2])
        assert after_eviction is not None
        assert after_eviction.get_indices_list() == [1]
        adapter.submit_unlock([key1, key2])
    finally:
        adapter.close()


def test_raw_block_l2_adapter_evicts_lru_not_recently_loaded_key():
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dev.bin")
        with open(dev_path, "wb") as f:
            f.truncate(8 * 1024 * 1024)

        slot_bytes = 64 * 1024
        capacity_bytes = (1 * 1024 * 1024) + 2 * slot_bytes
        adapter = RawBlockL2Adapter(
            _make_config(
                dev_path,
                slot_bytes=slot_bytes,
                capacity_bytes=capacity_bytes,
            )
        )

        try:
            key1 = _create_object_key(26)
            key2 = _create_object_key(27)
            key3 = _create_object_key(28)
            obj1 = _create_memory_obj(fill_value=26.0)
            obj2 = _create_memory_obj(fill_value=27.0)
            obj3 = _create_memory_obj(fill_value=28.0)

            assert _run_store(adapter, [key1, key2], [obj1, obj2]) is True

            _, lookup = _run_lookup(adapter, [key1])
            assert lookup is not None
            assert lookup.get_indices_list() == [0]
            load_buffer = _create_memory_obj(fill_value=0.0)
            _, loaded = _run_load(adapter, [key1], [load_buffer])
            assert loaded is not None
            assert loaded.get_indices_list() == [0]
            assert torch.equal(load_buffer.tensor, obj1.tensor)
            adapter.submit_unlock([key1])

            assert _run_store(adapter, [key3], [obj3]) is True

            _, bitmap = _run_lookup(adapter, [key1, key2, key3])
            assert bitmap is not None
            assert bitmap.get_indices_list() == [0, 2]
            adapter.submit_unlock([key1, key2, key3])
        finally:
            adapter.close()


def test_raw_block_l2_adapter_duplicate_store_batch_counts_once():
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dev.bin")
        with open(dev_path, "wb") as f:
            f.truncate(8 * 1024 * 1024)

        adapter = RawBlockL2Adapter(_make_config(dev_path))
        listener = _RecordingListener()
        adapter.register_listener(listener)

        try:
            key = _create_object_key(25)
            obj_a = _create_memory_obj(size=512, fill_value=25.0)
            obj_b = _create_memory_obj(size=1024, fill_value=26.0)

            assert _run_store(adapter, [key, key], [obj_a, obj_b]) is True
            assert _run_store(adapter, [key], [obj_b]) is True

            assert listener.stored == [[key]]
            assert listener.stored_sizes == [[obj_a.get_size()]]
            assert adapter.get_usage().total_bytes_used == obj_a.get_size()
            assert adapter._core.indexed_key_count() == 1

            accounting = adapter.report_status()["core"]["io_accounting"]
            assert accounting["store_attempted_count"] == 3
            assert accounting["store_attempted_logical_bytes"] == (
                obj_a.get_size() + 2 * obj_b.get_size()
            )
            assert accounting["store_committed_count"] == 1
            assert accounting["store_committed_logical_bytes"] == obj_a.get_size()
            assert accounting["eviction_count"] == 0
            assert accounting["eviction_logical_bytes"] == 0
            assert accounting["store_existing_hit_count"] == 2
            assert accounting["store_existing_hit_logical_bytes"] == (
                2 * obj_a.get_size()
            )
            assert accounting["media_write_logical_bytes"] == obj_a.get_size()
            assert accounting["data_write_logical_bytes"] == obj_a.get_size()
            assert accounting["data_write_payload_physical_bytes"] == obj_a.get_size()
            assert accounting["data_write_header_physical_bytes"] > 0
            assert accounting["data_write_physical_bytes"] == (
                accounting["data_write_payload_physical_bytes"]
                + accounting["data_write_header_physical_bytes"]
            )
            assert (
                accounting["total_write_physical_bytes"]
                == accounting["data_write_physical_bytes"]
            )

            adapter._core.checkpoint_now()
            after_checkpoint = adapter.report_status()["core"]["io_accounting"]
            assert after_checkpoint["metadata_write_physical_bytes"] > 0
            assert after_checkpoint["total_write_physical_bytes"] == (
                after_checkpoint["data_write_physical_bytes"]
                + after_checkpoint["metadata_write_physical_bytes"]
            )
            assert (
                after_checkpoint["media_write_physical_bytes"]
                == after_checkpoint["total_write_physical_bytes"]
            )
        finally:
            adapter.close()


def test_raw_block_l2_adapter_delete_subtracts_exact_metadata_size():
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dev.bin")
        with open(dev_path, "wb") as f:
            f.truncate(8 * 1024 * 1024)

        adapter = RawBlockL2Adapter(_make_config(dev_path))
        try:
            key_small = _create_object_key(261)
            key_large = _create_object_key(262)
            obj_small = _create_memory_obj(size=512, fill_value=26.1)
            obj_large = _create_memory_obj(size=1024, fill_value=26.2)
            assert (
                _run_store(
                    adapter,
                    [key_small, key_large],
                    [obj_small, obj_large],
                )
                is True
            )

            assert adapter.get_usage().total_bytes_used == (
                obj_small.get_size() + obj_large.get_size()
            )

            adapter.delete([key_small])
            assert adapter.get_usage().total_bytes_used == obj_large.get_size()

            adapter.delete([key_large])
            assert adapter.get_usage().total_bytes_used == 0
            assert adapter._core.indexed_key_count() == 0
        finally:
            adapter.close()


def test_raw_block_l2_adapter_load_hit_miss_io_accounting():
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dev.bin")
        with open(dev_path, "wb") as f:
            f.truncate(8 * 1024 * 1024)

        adapter = RawBlockL2Adapter(_make_config(dev_path))
        try:
            key = _create_object_key(251)
            missing_key = _create_object_key(252)
            obj = _create_memory_obj(size=512, fill_value=25.1)
            assert _run_store(adapter, [key], [obj]) is True

            before = adapter.report_status()["core"]["io_accounting"]
            load_buffers = [
                _create_memory_obj(size=512, fill_value=0.0),
                _create_memory_obj(size=512, fill_value=0.0),
            ]
            _, loaded = _run_load(adapter, [key, missing_key], load_buffers)
            assert loaded is not None
            assert loaded.get_indices_list() == [0]
            assert torch.equal(load_buffers[0].tensor, obj.tensor)
            assert torch.count_nonzero(load_buffers[1].tensor) == 0

            after = adapter.report_status()["core"]["io_accounting"]
            assert after["load_attempted_count"] - before["load_attempted_count"] == 2
            assert after["load_index_hit_count"] - before["load_index_hit_count"] == 1
            assert (
                after["load_index_hit_logical_bytes"]
                - before["load_index_hit_logical_bytes"]
                == obj.get_size()
            )
            assert (
                after["media_read_logical_bytes"] - before["media_read_logical_bytes"]
                == obj.get_size()
            )
            assert adapter.get_usage().total_bytes_used == obj.get_size()
        finally:
            adapter.close()


def test_raw_block_l2_adapter_l1_hit_does_not_touch_raw_block_accounting():
    # This exercises the full StorageManager tiering path. A DRAM/L1 prefix hit
    # should short-circuit before raw-block lookup/load, so raw-block counters
    # stay unchanged.
    # First Party
    from lmcache.v1.distributed.api import MemoryLayoutDesc
    from lmcache.v1.distributed.config import (
        EvictionConfig,
        L1ManagerConfig,
        L1MemoryManagerConfig,
        StorageManagerConfig,
    )
    from lmcache.v1.distributed.l2_adapters.config import L2AdaptersConfig
    from lmcache.v1.distributed.storage_manager import StorageManager

    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dev.bin")
        with open(dev_path, "wb") as f:
            f.truncate(16 * 1024 * 1024)

        config = StorageManagerConfig(
            l1_manager_config=L1ManagerConfig(
                memory_config=L1MemoryManagerConfig(
                    size_in_bytes=16 * 1024 * 1024,
                    use_lazy=False,
                    init_size_in_bytes=16 * 1024 * 1024,
                    align_bytes=4096,
                ),
            ),
            eviction_config=EvictionConfig(eviction_policy="noop"),
            l2_adapter_config=L2AdaptersConfig(
                adapters=[
                    _make_config(
                        dev_path,
                        slot_bytes=64 * 1024,
                        meta_total_bytes=1 * 1024 * 1024,
                    )
                ],
            ),
        )
        storage_manager = StorageManager(config)
        try:
            adapter = storage_manager._l2_adapters[0]
            key = _create_object_key(253)
            layout = MemoryLayoutDesc(
                shapes=[torch.Size([128])],
                dtypes=[torch.float32],
            )

            reserved = storage_manager.reserve_write([key], layout, mode="new")
            assert list(reserved) == [key]
            reserved[key].tensor.fill_(25.3)
            storage_manager.finish_write([key])

            assert _wait_for_condition(lambda: adapter._core.indexed_key_count() == 1)
            before = adapter.report_status()["core"]["io_accounting"]
            assert before["store_committed_count"] == 1
            assert before["media_write_logical_bytes"] == reserved[key].get_size()

            handle = storage_manager.submit_prefetch_task([key], layout)
            assert handle.prefetch_request_id == -1
            assert storage_manager.query_prefetch_status(handle) == 1

            after_prefetch = adapter.report_status()["core"]["io_accounting"]
            assert after_prefetch == before

            with storage_manager.read_prefetched_results([key]) as objs:
                assert objs is not None
                assert torch.equal(objs[0].tensor, reserved[key].tensor)
            storage_manager.finish_read_prefetched([key])

            after_read = adapter.report_status()["core"]["io_accounting"]
            assert after_read == before
        finally:
            storage_manager.close()


def test_raw_block_l2_adapter_metadata_full_skips_checkpoint_but_live_data_works():
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dev.bin")
        with open(dev_path, "wb") as f:
            f.truncate(8 * 1024 * 1024)

        slot_bytes = 64 * 1024
        meta_total_bytes = 16 * 1024
        capacity_bytes = meta_total_bytes + 32 * slot_bytes
        config = _make_config(
            dev_path,
            slot_bytes=slot_bytes,
            capacity_bytes=capacity_bytes,
            meta_total_bytes=meta_total_bytes,
            meta_magic="TSTFUL01",
        )
        stable_key = _create_object_key(43)
        stable_obj = _create_memory_obj(fill_value=43.0)

        adapter1 = RawBlockL2Adapter(config)
        try:
            assert _run_store(adapter1, [stable_key], [stable_obj]) is True
        finally:
            adapter1.close()

        adapter2 = RawBlockL2Adapter(config)
        adapter2_closed = False
        try:
            _, recovered = _run_lookup(adapter2, [stable_key])
            assert recovered is not None
            assert recovered.get_indices_list() == [0]
            adapter2.submit_unlock([stable_key])

            overflow_keys = [
                _create_object_key(i, model_name=f"overflow-{i}-" + "x" * 512)
                for i in range(5000, 5016)
            ]
            overflow_objs = [_create_memory_obj(fill_value=float(i)) for i in range(16)]
            overflow_key = overflow_keys[0]
            overflow_obj = overflow_objs[0]
            assert _run_store(adapter2, overflow_keys, overflow_objs) is True

            _, live_lookup = _run_lookup(adapter2, [overflow_key])
            assert live_lookup is not None
            assert live_lookup.get_indices_list() == [0]
            load_buffer = _create_memory_obj(fill_value=0.0)
            _, live_load = _run_load(adapter2, [overflow_key], [load_buffer])
            assert live_load is not None
            assert live_load.get_indices_list() == [0]
            assert torch.equal(load_buffer.tensor, overflow_obj.tensor)
            adapter2.submit_unlock([overflow_key])

            with patch(
                "lmcache.v1.storage_backend.raw_block.core.logger.warning"
            ) as warning_mock:
                adapter2.close()
                adapter2_closed = True
            assert any(
                "metadata payload too large" in str(call.args[0])
                for call in warning_mock.call_args_list
            )
        finally:
            if not adapter2_closed:
                adapter2.close()

        adapter3 = RawBlockL2Adapter(config)
        try:
            _, restarted = _run_lookup(adapter3, [stable_key, overflow_key])
            assert restarted is not None
            assert restarted.get_indices_list() == [0]
            adapter3.submit_unlock([stable_key, overflow_key])
        finally:
            adapter3.close()


def test_raw_block_l2_adapter_listener_errors_do_not_block_eventfds():
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dev.bin")
        with open(dev_path, "wb") as f:
            f.truncate(8 * 1024 * 1024)

        adapter = RawBlockL2Adapter(_make_config(dev_path))
        adapter.register_listener(_FailingListener())

        try:
            key = _create_object_key(29)
            obj = _create_memory_obj(fill_value=29.0)

            store_task_id = adapter.submit_store_task([key], [obj])
            assert _wait_event_fd(adapter.get_store_event_fd())
            assert adapter.pop_completed_store_tasks()[store_task_id] is True

            load_buffer = _create_memory_obj(fill_value=0.0)
            load_task_id = adapter.submit_load_task([key], [load_buffer])
            assert _wait_event_fd(adapter.get_load_event_fd())
            load_bitmap = adapter.query_load_result(load_task_id)
            assert load_bitmap is not None
            assert load_bitmap.get_indices_list() == [0]
        finally:
            adapter.close()


def test_raw_block_l2_adapter_recovery_from_checkpoint():
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dev.bin")
        with open(dev_path, "wb") as f:
            f.truncate(8 * 1024 * 1024)

        config = _make_config(dev_path)
        key1 = _create_object_key(31)
        key2 = _create_object_key(32)
        obj1 = _create_memory_obj(size=512, fill_value=31.0)
        obj2 = _create_memory_obj(size=1024, fill_value=32.0)

        adapter1 = RawBlockL2Adapter(config)
        try:
            assert _run_store(adapter1, [key1, key2], [obj1, obj2]) is True
        finally:
            adapter1.close()

        adapter2 = RawBlockL2Adapter(config)
        try:
            expected_usage = obj1.get_size() + obj2.get_size()
            assert adapter2.get_usage().total_bytes_used == expected_usage
            assert adapter2._core.indexed_key_count() == 2

            _, lookup_bitmap = _run_lookup(adapter2, [key1, key2])
            assert lookup_bitmap is not None
            assert lookup_bitmap.get_indices_list() == [0, 1]

            load_buffer1 = _create_memory_obj(size=512, fill_value=0.0)
            load_buffer2 = _create_memory_obj(size=1024, fill_value=0.0)
            _, load_bitmap = _run_load(
                adapter2,
                [key1, key2],
                [load_buffer1, load_buffer2],
            )
            assert load_bitmap is not None
            assert load_bitmap.get_indices_list() == [0, 1]
            assert torch.equal(load_buffer1.tensor, obj1.tensor)
            assert torch.equal(load_buffer2.tensor, obj2.tensor)
            assert adapter2.get_usage().total_bytes_used == expected_usage
            adapter2.submit_unlock([key1, key2])
        finally:
            adapter2.close()


def test_raw_block_l2_adapter_recovers_unknown_checkpoint_dtype():
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dev.bin")
        with open(dev_path, "wb") as f:
            f.truncate(8 * 1024 * 1024)

        config = _make_config(dev_path)
        key = _create_object_key(35)
        obj = _create_complex_memory_obj(fill_value=1 + 2j)

        adapter1 = RawBlockL2Adapter(config)
        try:
            assert _run_store(adapter1, [key], [obj]) is True
        finally:
            adapter1.close()

        adapter2 = RawBlockL2Adapter(config)
        try:
            load_buffer = _create_complex_memory_obj(fill_value=0j)
            _, load_bitmap = _run_load(adapter2, [key], [load_buffer])
            assert load_bitmap is not None
            assert load_bitmap.get_indices_list() == [0]
            assert load_buffer.metadata.dtype is torch.complex64
            assert torch.equal(load_buffer.tensor, obj.tensor)
        finally:
            adapter2.close()


def test_raw_block_l2_adapter_error_bitmaps_keep_submitted_size():
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dev.bin")
        with open(dev_path, "wb") as f:
            f.truncate(8 * 1024 * 1024)

        adapter = RawBlockL2Adapter(_make_config(dev_path))
        try:
            keys = [_create_object_key(41), _create_object_key(42)]
            objects = [_create_memory_obj(), _create_memory_obj()]

            with patch.object(
                adapter, "_run_lookup_task", side_effect=RuntimeError("lookup failed")
            ):
                lookup_task_id = adapter.submit_lookup_and_lock_task(keys)
                assert _wait_event_fd(adapter.get_lookup_and_lock_event_fd())
                lookup_bitmap = adapter.query_lookup_and_lock_result(lookup_task_id)
            assert lookup_bitmap is not None
            assert str(lookup_bitmap) == "00"

            with patch.object(
                adapter, "_run_load_task", side_effect=RuntimeError("load failed")
            ):
                load_task_id = adapter.submit_load_task(keys, objects)
                assert _wait_event_fd(adapter.get_load_event_fd())
                load_bitmap = adapter.query_load_result(load_task_id)
            assert load_bitmap is not None
            assert str(load_bitmap) == "00"
        finally:
            adapter.close()
