# SPDX-License-Identifier: Apache-2.0
"""fs_native L2: a file that fails to load must not keep serving lookup hits.

These tests drive a real fs_native adapter on a temporary directory through
the public L2 adapter interface and damage files on disk between calls.
"""

# Standard
from pathlib import Path
from typing import Any
import errno
import os
import sys
import time

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.internal_api import L2AdapterListener
from lmcache.v1.distributed.key_codec import encode_fs_filename
from lmcache.v1.distributed.l2_adapters.base import L2AdapterInterface
from lmcache.v1.distributed.l2_adapters.factory import create_l2_adapter_from_registry
from lmcache.v1.distributed.l2_adapters.fs_native_l2_adapter import (
    FSNativeL2AdapterConfig,
)
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObj,
    MemoryObjMetadata,
    TensorMemoryObj,
)

try:
    # First Party
    import lmcache.lmcache_fs  # noqa: F401
except ImportError:
    pytest.skip("native FS extension is not built", allow_module_level=True)

_NUM_FLOATS = 2048  # 8 KiB: half of it is still O_DIRECT block-aligned
_NUM_BYTES = _NUM_FLOATS * 4
_ALIGN = 4096  # O_DIRECT buffer alignment


class _DeleteRecorder(L2AdapterListener):
    def __init__(self) -> None:
        self.deleted: list[ObjectKey] = []

    def on_l2_keys_stored(self, keys: list[ObjectKey], sizes: list[int]) -> None:
        pass

    def on_l2_keys_accessed(self, keys: list[ObjectKey]) -> None:
        pass

    def on_l2_keys_deleted(self, keys: list[ObjectKey]) -> None:
        self.deleted.extend(keys)


def _key(chunk_id: int) -> ObjectKey:
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(chunk_id),
        model_name="test_model",
        kv_rank=0,
    )


def _obj(fill: float) -> MemoryObj:
    # Aligned so the same buffers also work with use_odirect.
    raw = torch.empty(_NUM_FLOATS + _ALIGN // 4, dtype=torch.float32)
    offset = (-raw.data_ptr() % _ALIGN) // 4
    data = raw[offset : offset + _NUM_FLOATS]
    data.fill_(fill)
    metadata = MemoryObjMetadata(
        shape=torch.Size([_NUM_FLOATS]),
        dtype=torch.float32,
        address=0,
        phy_size=_NUM_BYTES,
        fmt=MemoryFormat.KV_2LTD,
        ref_count=1,
    )
    return TensorMemoryObj(data, metadata, parent_allocator=None)


def _make_adapter(
    base: Path,
    shared: bool = False,
    num_read_workers: int = 4,
    fs_options: dict[str, Any] | None = None,
) -> L2AdapterInterface:
    cfg = FSNativeL2AdapterConfig(
        base_path=str(base), num_read_workers=num_read_workers, **(fs_options or {})
    )
    cfg.shared = shared
    return create_l2_adapter_from_registry(cfg)


def _wait(fn, timeout: float = 5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = fn()
        if result is not None:
            return result
        time.sleep(0.005)
    raise AssertionError("timed out waiting for adapter task")


def _store(adapter: L2AdapterInterface, keys: list[ObjectKey], fill: float) -> None:
    objs = [_obj(fill) for _ in keys]  # the native write reads them async
    task = adapter.submit_store_task(keys, objs)
    result = _wait(lambda: adapter.pop_completed_store_tasks().get(task))
    del objs
    assert result >= 0


def _lookup(adapter: L2AdapterInterface, keys: list[ObjectKey]) -> list[bool]:
    task = adapter.submit_lookup_and_lock_task(keys, {})
    bitmap = _wait(lambda: adapter.query_lookup_and_lock_result(task))
    hits = [bitmap.test(i) for i in range(len(keys))]
    adapter.submit_unlock([k for k, hit in zip(keys, hits, strict=True) if hit])
    return hits


def _settle(adapter: L2AdapterInterface) -> None:
    """Return once listener notifications for completions already handled
    have fired: the adapter fires them after publishing the load result, and a
    later completion is handled only after that."""
    _lookup(adapter, [_key(999)])


def _load(
    adapter: L2AdapterInterface, keys: list[ObjectKey]
) -> tuple[list[bool], list[MemoryObj]]:
    objs = [_obj(0.0) for _ in keys]
    task = adapter.submit_load_task(keys, objs)
    bitmap = _wait(lambda: adapter.query_load_result(task))
    _settle(adapter)
    return [bitmap.test(i) for i in range(len(keys))], objs


def _truncate(base: Path, key: ObjectKey, size: int = _NUM_BYTES // 2) -> Path:
    path = base / encode_fs_filename(key)
    os.truncate(path, size)
    return path


def _open_fifo_writer(path: Path, timeout: float = 5.0) -> int:
    """Open the write end of ``path`` once a reader holds it open.

    On timeout, release a reader blocked in open() and replace the FIFO, so a
    late reader cannot block ``adapter.close()``; the test then fails.
    """
    deadline = time.monotonic() + timeout
    while True:
        try:
            fd = os.open(path, os.O_WRONLY | os.O_NONBLOCK)
        except OSError as e:
            if e.errno != errno.ENXIO:
                raise
            if time.monotonic() > deadline:
                os.close(os.open(path, os.O_RDWR))
                os.unlink(path)
                raise AssertionError("the load never opened the FIFO") from e
            time.sleep(0.005)
            continue
        os.set_blocking(fd, True)
        return fd


def _odirect_supported(directory: Path) -> bool:
    probe = directory / "odirect_probe"
    try:
        fd = os.open(probe, os.O_CREAT | os.O_WRONLY | getattr(os, "O_DIRECT", 0))
    except OSError:
        return False
    os.close(fd)
    probe.unlink()
    return hasattr(os, "O_DIRECT")


@pytest.mark.parametrize("num_read_workers", [0, 4])
@pytest.mark.parametrize("restart", [False, True])
def test_truncated_file_is_invalidated_and_restored(
    tmp_path: Path, restart: bool, num_read_workers: int
) -> None:
    """A truncated file fails one load, is removed and reported deleted, and the
    next store writes a complete file, also across an adapter restart and with
    or without the read pool."""
    keys = [_key(1), _key(2), _key(3)]
    adapter = _make_adapter(tmp_path, num_read_workers=num_read_workers)
    try:
        _store(adapter, keys, 1.0)
        path = _truncate(tmp_path, keys[1])
        if restart:
            adapter.close()
            adapter = _make_adapter(tmp_path, num_read_workers=num_read_workers)
        recorder = _DeleteRecorder()
        adapter.register_listener(recorder)
        usage_before = adapter.get_usage().total_bytes_used

        assert _lookup(adapter, keys) == [True, True, True]
        loaded, _ = _load(adapter, keys)
        assert loaded == [True, False, True]

        assert not path.exists()
        assert recorder.deleted == [keys[1]]
        if not restart:
            assert adapter.get_usage().total_bytes_used == usage_before - _NUM_BYTES
        assert _lookup(adapter, keys) == [True, False, True]

        _store(adapter, [keys[1]], 2.0)
        assert path.stat().st_size == _NUM_BYTES
        loaded, objs = _load(adapter, [keys[1]])
        assert loaded == [True]
        assert torch.all(objs[0].tensor == 2.0)
    finally:
        adapter.close()


@pytest.mark.parametrize(
    ("fs_options", "truncate_to"),
    [
        pytest.param({"read_ahead_size": 1024}, 512, id="read_ahead_head_short"),
        pytest.param({"read_ahead_size": 1024}, 4096, id="read_ahead_tail_short"),
        pytest.param({"use_odirect": True}, 4096, id="odirect"),
        pytest.param({"use_odirect": True}, 0, id="odirect_empty"),
    ],
)
def test_short_read_is_invalidated_on_every_read_path(
    tmp_path: Path, fs_options: dict[str, object], truncate_to: int
) -> None:
    """The read-ahead head and tail reads and O_DIRECT reads detect a short file
    like a plain read does."""
    if fs_options.get("use_odirect") and not _odirect_supported(tmp_path):
        pytest.skip("O_DIRECT is not supported on this filesystem")
    key = _key(4)
    adapter = _make_adapter(tmp_path, fs_options=fs_options)
    try:
        recorder = _DeleteRecorder()
        adapter.register_listener(recorder)
        _store(adapter, [key], 1.0)
        loaded, objs = _load(adapter, [key])
        assert loaded == [True]
        assert torch.all(objs[0].tensor == 1.0)

        path = _truncate(tmp_path, key, truncate_to)
        assert _load(adapter, [key])[0] == [False]
        assert not path.exists()
        assert recorder.deleted == [key]
    finally:
        adapter.close()


def test_concurrent_failed_loads_report_one_delete(tmp_path: Path) -> None:
    """Two loads of one truncated file in flight together report it deleted
    once."""
    key = _key(5)
    adapter = _make_adapter(tmp_path)
    try:
        recorder = _DeleteRecorder()
        adapter.register_listener(recorder)
        _store(adapter, [key], 1.0)
        path = _truncate(tmp_path, key)

        objs = [_obj(0.0), _obj(0.0)]
        tasks = [adapter.submit_load_task([key], [obj]) for obj in objs]
        for task in tasks:
            assert not _wait(lambda t=task: adapter.query_load_result(t)).test(0)
        _settle(adapter)
        assert not path.exists()
        assert recorder.deleted == [key]
    finally:
        adapter.close()


@pytest.mark.skipif(sys.platform != "linux", reason="needs Linux /proc/self/mem")
def test_eio_keeps_file(tmp_path: Path) -> None:
    """A read that fails with EIO fails the load but leaves the entry: on
    network, FUSE and multipath storage EIO is often transient."""
    key = _key(7)
    adapter = _make_adapter(tmp_path)
    try:
        recorder = _DeleteRecorder()
        adapter.register_listener(recorder)
        path = tmp_path / encode_fs_filename(key)
        # Reading /proc/self/mem at offset 0 fails with EIO without root.
        os.symlink("/proc/self/mem", path)

        assert _load(adapter, [key])[0] == [False]
        assert os.path.lexists(path)
        assert recorder.deleted == []
    finally:
        adapter.close()


def test_failed_read_keeps_file_a_concurrent_store_published(tmp_path: Path) -> None:
    """A read that fails after a concurrent store renamed a good copy over its
    file leaves the new copy in place."""
    key = _key(10)
    adapter = _make_adapter(tmp_path)
    try:
        recorder = _DeleteRecorder()
        adapter.register_listener(recorder)
        path = tmp_path / encode_fs_filename(key)
        os.mkfifo(path)  # holds the read open until the test feeds it
        objs = [_obj(0.0)]
        load_task = adapter.submit_load_task([key], objs)
        wfd = _open_fifo_writer(path)
        try:
            # Publish a good copy the way a store does: write aside, rename.
            staged = tmp_path / "staged"
            staged.write_bytes(torch.full((_NUM_FLOATS,), 4.0).numpy().tobytes())
            os.rename(staged, path)
        finally:
            os.write(wfd, b"\0" * (_NUM_BYTES // 2))
            os.close(wfd)
        bitmap = _wait(lambda: adapter.query_load_result(load_task))
        assert not bitmap.test(0)
        _settle(adapter)

        assert path.stat().st_size == _NUM_BYTES
        assert recorder.deleted == []
        loaded, objs = _load(adapter, [key])
        assert loaded == [True]
        assert torch.all(objs[0].tensor == 4.0)
    finally:
        adapter.close()


@pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0,
    reason="root ignores directory permissions",
)
def test_failed_unlink_keeps_key_without_delete(tmp_path: Path) -> None:
    """When the short file cannot be removed, the load still fails and nothing
    is reported deleted, so accounting keeps matching the disk."""
    key = _key(11)
    adapter = _make_adapter(tmp_path)
    try:
        recorder = _DeleteRecorder()
        adapter.register_listener(recorder)
        _store(adapter, [key], 1.0)
        path = _truncate(tmp_path, key)
        usage_before = adapter.get_usage().total_bytes_used
        tmp_path.chmod(0o555)
        try:
            assert _load(adapter, [key])[0] == [False]
        finally:
            tmp_path.chmod(0o755)
        assert path.exists()
        assert recorder.deleted == []
        assert adapter.get_usage().total_bytes_used == usage_before
    finally:
        adapter.close()


def test_other_read_error_keeps_file(tmp_path: Path) -> None:
    """Any read error other than a short read (here EISDIR) fails the load but
    leaves the entry in place."""
    key = _key(8)
    adapter = _make_adapter(tmp_path)
    try:
        recorder = _DeleteRecorder()
        adapter.register_listener(recorder)
        target = tmp_path / "a_directory"
        target.mkdir()
        path = tmp_path / encode_fs_filename(key)
        os.symlink(target, path)

        assert _load(adapter, [key])[0] == [False]
        assert os.path.lexists(path)
        assert recorder.deleted == []
    finally:
        adapter.close()


def test_shared_mount_keeps_failed_file(tmp_path: Path) -> None:
    """On a shared mount a failed load deletes nothing: another instance may be
    the one writing the directory."""
    key = _key(9)
    adapter = _make_adapter(tmp_path, shared=True)
    try:
        recorder = _DeleteRecorder()
        adapter.register_listener(recorder)
        _store(adapter, [key], 1.0)
        path = _truncate(tmp_path, key)

        assert _load(adapter, [key])[0] == [False]
        assert path.stat().st_size == _NUM_BYTES // 2
        assert recorder.deleted == []
    finally:
        adapter.close()
