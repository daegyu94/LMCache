# SPDX-License-Identifier: Apache-2.0

# First Party
from lmcache.v1.storage_backend.raw_block.core import RawBlockCore


def test_raw_block_io_latency_callback_observes_completed_write():
    core = object.__new__(RawBlockCore)
    core._data_base_offset = 4096
    observed = []
    core._io_latency_callback = lambda *args: observed.append(args)
    completed = []

    def write_impl(*_args, **_kwargs):
        completed.append(True)

    core._write_buffers_impl = write_impl

    core._write_buffers(
        [8192, 12288],
        [bytearray(4096), bytearray(8192)],
        [4096, 8192],
        [4096, 8192],
    )

    assert completed == [True]
    assert len(observed) == 1
    operation, latency_s, io_class, physical_bytes, io_count, failed = observed[0]
    assert operation == "write"
    assert latency_s >= 0
    assert io_class == "data"
    assert physical_bytes == 12288
    assert io_count == 2
    assert failed is False


def test_raw_block_io_latency_callback_marks_failed_read():
    core = object.__new__(RawBlockCore)
    core._data_base_offset = 4096
    observed = []
    core._io_latency_callback = lambda *args: observed.append(args)

    def read_impl(*_args, **_kwargs):
        raise RuntimeError("read failed")

    core._read_buffers_impl = read_impl

    try:
        core._read_buffers([0], [bytearray(4096)], [4096], [4096])
    except RuntimeError:
        pass

    assert len(observed) == 1
    operation, _latency_s, io_class, physical_bytes, io_count, failed = observed[0]
    assert operation == "read"
    assert io_class == "metadata"
    assert physical_bytes == 4096
    assert io_count == 1
    assert failed is True
