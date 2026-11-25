# SPDX-License-Identifier: Apache-2.0
"""A crash model for reuse after a delete.

Every device write call is logged; device images for process crashes at
sampled points are rebuilt from the log (calls that finished land whole,
posix calls in flight as an in-order prefix, io_uring requests in flight
each as nothing, a block prefix or all), recovered, and every recovered key
must hold the bytes of one write of that key."""

# Standard
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
import importlib.util
import itertools
import random
import threading
import time

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.l2_adapters.raw_block import RawBlockCore, encode_object_key
from lmcache.v1.distributed.l2_adapters.raw_block.io_engine import (
    IoUringBlockIO,
    PosixBlockIO,
)
from tests.v1.distributed.raw_block.test_utils import (
    RAW_BLOCK_CI_BLOCK_ALIGN,
    RAW_BLOCK_CI_HEADER_BYTES,
    RAW_BLOCK_CI_META_TOTAL_BYTES,
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


def _spec(i: int):
    return encode_object_key(make_object_key(i))


def _config(tmp_path: Path, engine: str, data_bytes: int, odirect: bool = False):
    total = RAW_BLOCK_CI_META_TOTAL_BYTES + data_bytes
    path = make_raw_block_file(tmp_path, total)
    return replace(
        make_raw_block_core_config(path, total),
        io_engine=engine,
        use_odirect=odirect,
        meta_enable_periodic=False,
    )


class _WriteLog:
    """Records every engine write call with start/end tickets and its bytes."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.ticket = itertools.count()
        self.calls: list[dict] = []

    def wrap(self, real, kind: str):
        log = self

        def write_buffers(self, offsets, buffers, payload_lens, total_lens, *a, **k):
            sqes = []
            for off, buf, pl, tl in zip(
                offsets, buffers, payload_lens, total_lens, strict=True
            ):
                mv = memoryview(buf).cast("B")
                data = bytes(mv[:pl]) + bytes(tl - pl)
                sqes.append((int(off), data))
            with log.lock:
                call = {"start": next(log.ticket), "end": None, "kind": kind}
                call["sqes"] = sqes
                log.calls.append(call)
            real(self, offsets, buffers, payload_lens, total_lens, *a, **k)
            with log.lock:
                call["end"] = next(log.ticket)

        return write_buffers

    def image(self, base: bytes, t: int, rng: random.Random) -> bytearray:
        """Device bytes after a process crash at ticket ``t``.

        Completed calls land whole. A call in flight lands partly: posix as
        a prefix of its buffer sequence (pwrite in order, each a prefix);
        io_uring per request independently: nothing, a block prefix, or all.
        """
        img = bytearray(base)
        done = sorted(
            (c for c in self.calls if c["end"] is not None and c["end"] < t),
            key=lambda c: c["end"],
        )
        flying = [
            c
            for c in self.calls
            if c["start"] < t and (c["end"] is None or c["end"] > t)
        ]
        rng.shuffle(flying)
        for c in done:
            for off, data in c["sqes"]:
                img[off : off + len(data)] = data
        for c in flying:
            sqes = c["sqes"]
            if c["kind"] == "posix":
                cut = rng.randrange(len(sqes) + 1)
                for j, (off, data) in enumerate(sqes[: cut + 1]):
                    n = len(data) if j < cut else _prefix(rng, len(data))
                    img[off : off + n] = data[:n]
            else:
                for off, data in sqes:
                    n = rng.choice([0, len(data), _prefix(rng, len(data))])
                    img[off : off + n] = data[:n]
        return img


def _prefix(rng: random.Random, n: int) -> int:
    return min(n, ALIGN * rng.randrange(0, n // ALIGN + 1))


def _run_workload(core, versions, vcount, seed: int, ops: int = 120) -> _WriteLog:
    """Three threads of puts (1-3 keys, batched on io_uring), forced deletes
    and checkpoints on 24 keys; returns the log of every device write."""
    log = _WriteLog()
    vlock = threading.Lock()
    errors: list[BaseException] = []

    def worker(wid: int) -> None:
        rng = random.Random(seed * 100 + wid)
        try:
            for _ in range(ops):
                op = rng.random()
                if op < 0.6:
                    keys = rng.sample(range(24), rng.randint(1, 3))
                    objs, specs = [], []
                    for k in keys:
                        size = rng.choice([1, 2, 3, 5, 7]) * ALIGN - rng.choice(
                            [0, 100]
                        )
                        data = random.Random(next(vcount)).randbytes(size)
                        with vlock:
                            versions.setdefault(k, []).append(data)
                        objs.append(make_memory_obj(data))
                        specs.append(_spec(k))
                    core.put_many(specs, objs)
                elif op < 0.9:
                    keys = rng.sample(range(24), rng.randint(1, 3))
                    core.delete_many([_spec(k).encoded for k in keys], force=True)
                else:
                    core.checkpoint_now()
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    with (
        patch.object(
            PosixBlockIO, "write_buffers", log.wrap(PosixBlockIO.write_buffers, "posix")
        ),
        patch.object(
            IoUringBlockIO,
            "write_buffers",
            log.wrap(IoUringBlockIO.write_buffers, "uring"),
        ),
    ):
        threads = [threading.Thread(target=worker, args=(w,)) for w in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(120)
    assert not errors, errors
    return log


def _check_recovery(config, versions, where: str) -> tuple[int, list[str]]:
    """Recover the device file; return loaded keys and wrong-byte keys."""
    crashed = RawBlockCore(config)
    bad: list[str] = []
    loaded = 0
    try:
        keys = [k for k in range(24) if crashed.contains_key(_spec(k).encoded)]
        metas = crashed.get_metadata_many([_spec(k).encoded for k in keys])
        for k, meta in zip(keys, metas, strict=True):
            assert meta is not None
            obj = make_empty_memory_obj(meta.size)
            if crashed.load_many_into([_spec(k).encoded], [obj]) != [True]:
                continue
            loaded += 1
            if memory_obj_bytes(obj) not in versions.get(k, []):
                bad.append(f"{where} key={k}")
    finally:
        crashed.close()
    return loaded, bad


ENGINE_PARAMS = pytest.mark.parametrize(
    "engine,odirect",
    [("posix", False), ("io_uring", False), ("io_uring", True)],
    ids=["posix", "uring", "uring-odirect"],
)


@ENGINE_PARAMS
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_crash_at_any_point_never_returns_bytes_of_another_write(
    tmp_path, engine, odirect, seed
):
    """Concurrent puts (batched for io_uring), deletes and checkpoints on a
    small device so space is reused constantly. At sampled crash points the
    device image is built from the write log under the process-crash model,
    recovered, and every recovered key must load exactly the bytes of one
    write of that key."""
    config = _config(tmp_path, engine, 640 * 1024, odirect)
    versions: dict[int, list[bytes]] = {}
    vcount = itertools.count(seed * 1_000_000)
    core = RawBlockCore(config)
    base = Path(config.device_path).read_bytes()
    log = _run_workload(core, versions, vcount, seed)
    core.close()

    last = max(c["end"] for c in log.calls)
    rng = random.Random(seed)
    bad: list[str] = []
    total = 0
    for t in sorted(rng.sample(range(1, last + 1), min(60, last))):
        for _ in range(3):
            Path(config.device_path).write_bytes(log.image(base, t, rng))
            loaded, b = _check_recovery(config, versions, f"t={t}")
            total += loaded
            bad += b
    assert total > 50, total
    assert not bad, bad[:10]


@ENGINE_PARAMS
@pytest.mark.parametrize("seed", [1, 2])
# 256 KiB: stores often split, and loaded split entries are listed by both
# containers, so their deletes zero headers.
@pytest.mark.parametrize("data_kib", [640, 256])
def test_second_crash_after_recovering_from_a_first_one(
    tmp_path, engine, odirect, seed, data_kib
):
    """Crash, recover (possibly past a torn checkpoint copy, with loaded
    entries listed by both containers), run again and crash again."""
    config = _config(tmp_path, engine, data_kib * 1024, odirect)
    versions: dict[int, list[bytes]] = {}
    vcount = itertools.count(seed * 1_000_000)
    rng = random.Random(seed)
    core = RawBlockCore(config)
    base1 = Path(config.device_path).read_bytes()
    log1 = _run_workload(core, versions, vcount, seed)
    core.close()
    last1 = max(c["end"] for c in log1.calls)
    bad: list[str] = []
    total = 0
    for t1 in sorted(rng.sample(range(last1 // 2, last1 + 1), 6)):
        Path(config.device_path).write_bytes(log1.image(base1, t1, rng))
        core = RawBlockCore(config)
        base2 = Path(config.device_path).read_bytes()
        log2 = _run_workload(core, versions, vcount, seed + t1, ops=60)
        core.close()
        last2 = max(c["end"] for c in log2.calls)
        for t2 in sorted(rng.sample(range(1, last2 + 1), min(25, last2))):
            Path(config.device_path).write_bytes(log2.image(base2, t2, rng))
            loaded, b = _check_recovery(config, versions, f"t1={t1} t2={t2}")
            total += loaded
            bad += b
    assert total > 50, total
    assert not bad, bad[:10]


def test_a_deleted_listed_object_takes_a_whole_io_uring_batch(tmp_path):
    """A full device evicts one large object to make room for a batch of
    small ones. Whether or not a checkpoint listed it, the whole batch fits
    there: the stores behind the first one write after its header."""
    big = 16 * ALIGN - HEADER  # 16 blocks with its header
    # Marker block, four big objects, one spare block.
    config = _config(tmp_path, "io_uring", (1 + 4 * 16 + 1) * ALIGN)
    results = {}
    for listed in (False, True):
        core = RawBlockCore(config)
        try:
            i = 0
            while core.put_many([_spec(i)], [make_memory_obj(bytes(big))]).results == [
                True
            ]:
                i += 1
            assert i == 4
            if listed:
                core.checkpoint_now()
            before = core.space_exhausted_count()
            assert core.delete_many([_spec(0).encoded]) == [True]
            small = [make_memory_obj(bytes(ALIGN)) for _ in range(4)]
            r = core.put_many([_spec(1000 + j) for j in range(4)], small)
            results[listed] = (sum(r.results), core.space_exhausted_count() - before)
            core.delete_many([_spec(k).encoded for k in range(i)], force=True)
            core.delete_many([_spec(1000 + j).encoded for j in range(4)], force=True)
        finally:
            core.close()
    # Unlisted: all four 2-block objects fit in the 16-block hole.
    assert results[False] == (4, 0)
    assert results[True] == (4, 0), results


def test_dropping_a_loaded_index_also_forgets_its_retired_ranges(tmp_path):
    """Startup retires the space of an entry whose header cannot be read, then
    drops the whole index because the data-region marker is gone: the retired
    range must go with it."""
    config = _config(tmp_path, "posix", 1 << 20)
    core = RawBlockCore(config)
    assert core.put_many([_spec(1)], [make_memory_obj(bytes(2 * ALIGN))]).results == [
        True
    ]
    offset = core.entry_offset(_spec(1).encoded)
    marker = core.data_base_offset()
    core.close()
    with open(config.device_path, "r+b") as f:
        f.seek(marker)
        f.write(bytes(ALIGN))
    real_read = PosixBlockIO.read_buffers

    def unreadable(self, offsets, *args, **kwargs):
        if list(offsets) == [offset]:
            return [False]
        return real_read(self, offsets, *args, **kwargs)

    with patch.object(PosixBlockIO, "read_buffers", unreadable):
        core = RawBlockCore(config)
    try:
        status = core.report_status()
        assert core.indexed_key_count() == 0
        assert status["allocated_bytes"] == 0, status
        assert status["quarantined_bytes"] == 0, status
    finally:
        core.close()


def test_a_carved_store_writes_after_its_head_store_has_landed(tmp_path):
    """The head store's header covers one deleted object's start and its
    payload another's; a store carved from the fence over the second one's
    tail must not be submitted until that payload has landed."""
    config = _config(tmp_path, "io_uring", 1 << 20)
    core = RawBlockCore(config)
    calls: list[list[int]] = []
    real_write = IoUringBlockIO.write_buffers

    def record(self, offsets, *args, **kwargs):
        calls.append(list(offsets))
        return real_write(self, offsets, *args, **kwargs)

    try:
        objs = [bytes(ALIGN), bytes(8 * ALIGN - HEADER), bytes(ALIGN)]
        for k, data in enumerate(objs):
            assert core.put_many([_spec(k)], [make_memory_obj(data)]).results == [True]
        first = core.entry_offset(_spec(0).encoded)
        assert core.entry_offset(_spec(1).encoded) == first + 2 * ALIGN
        core.checkpoint_now()  # lists both
        assert core.delete_many([_spec(0).encoded, _spec(1).encoded]) == [True] * 2
        head, carved = bytes(4 * ALIGN - HEADER), bytes(ALIGN)
        with patch.object(IoUringBlockIO, "write_buffers", record):
            result = core.put_many(
                [_spec(10), _spec(11)], [make_memory_obj(head), make_memory_obj(carved)]
            )
        assert result.results == [True, True]
        assert core.entry_offset(_spec(10).encoded) == first
        carved_at = core.entry_offset(_spec(11).encoded)
        assert first + 4 * ALIGN <= carved_at < first + 10 * ALIGN
        head_payload = next(i for i, c in enumerate(calls) if first + HEADER in c)
        carved_header = next(i for i, c in enumerate(calls) if carved_at in c)
        assert head_payload < carved_header
    finally:
        core.close()


def test_a_store_needing_fenced_space_waits_for_the_write_holding_it(tmp_path):
    """Another thread's store needs the space a write in flight holds back
    as its fence: it waits for that write instead of failing (and asking for
    an eviction round)."""
    # Marker block, an 8-block object between two 1-block ones.
    config = _config(tmp_path, "posix", (1 + 2 + 8 + 2) * ALIGN)
    core = RawBlockCore(config)
    real_write = PosixBlockIO.write_buffers
    started, release = threading.Event(), threading.Event()

    def held(self, offsets, *args, **kwargs):
        if offsets[0] == target and not started.is_set():
            started.set()
            assert release.wait(10)
        return real_write(self, offsets, *args, **kwargs)

    try:
        for k, blocks in enumerate([2, 8, 2]):
            data = bytes(blocks * ALIGN - HEADER)
            assert core.put_many([_spec(k)], [make_memory_obj(data)]).results == [True]
        target = core.entry_offset(_spec(1).encoded)
        core.checkpoint_now()
        assert core.delete_many([_spec(1).encoded]) == [True]
        before = core.space_exhausted_count()
        results: list[bool] = []
        with patch.object(PosixBlockIO, "write_buffers", held):
            head = threading.Thread(
                target=lambda: results.append(
                    core.put_many([_spec(20)], [make_memory_obj(bytes(ALIGN))]).results
                    == [True]
                )
            )
            head.start()
            assert started.wait(10)
            threading.Timer(0.2, release.set).start()
            tail = core.put_many([_spec(21)], [make_memory_obj(bytes(5 * ALIGN))])
            head.join(10)
        assert results == [True]
        assert tail.results == [True]
        assert core.space_exhausted_count() == before
    finally:
        release.set()
        core.close()


def test_a_store_waiting_for_a_fence_yields_to_a_store_of_the_same_key(tmp_path):
    """While a store waits for fenced space, the lock is released: another
    store of the same key may land meanwhile, and the waiting one must then
    not allocate a second copy (its space would stay allocated with no
    owner)."""
    # Marker block, objects of 2, 8 and 2 blocks, then 2 spare blocks.
    config = _config(tmp_path, "posix", (1 + 2 + 8 + 2 + 2) * ALIGN)
    core = RawBlockCore(config)
    real_write = PosixBlockIO.write_buffers
    started, release = threading.Event(), threading.Event()

    def held(self, offsets, *args, **kwargs):
        if offsets[0] == target and not started.is_set():
            started.set()
            assert release.wait(10)
        return real_write(self, offsets, *args, **kwargs)

    try:
        for k, blocks in enumerate([2, 8, 2]):
            data = bytes(blocks * ALIGN - HEADER)
            assert core.put_many([_spec(k)], [make_memory_obj(data)]).results == [True]
        target = core.entry_offset(_spec(1).encoded)
        core.checkpoint_now()
        assert core.delete_many([_spec(1).encoded]) == [True]
        results: dict[str, bool] = {}

        def put(name: str, key: int, blocks: int) -> None:
            data = make_memory_obj(bytes(blocks * ALIGN - HEADER))
            results[name] = core.put_many([_spec(key)], [data]).results == [True]

        with patch.object(PosixBlockIO, "write_buffers", held):
            head = threading.Thread(target=put, args=("head", 20, 3))
            head.start()
            assert started.wait(10)  # holds 5 blocks of key 1's space
            waiter = threading.Thread(target=put, args=("waiter", 21, 4))
            waiter.start()
            time.sleep(0.1)  # the waiter is waiting for the fence
            put("other", 21, 2)  # same key, in the spare blocks
            release.set()
            head.join(10)
            waiter.join(10)
        assert results["head"] and results["other"]
        status = core.report_status()
        held_bytes = sum(n for _, n in core.snapshot_indexed_allocations())
        assert status["allocated_bytes"] == held_bytes
    finally:
        release.set()
        core.close()


@ENGINE_PARAMS
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_split_objects_crash_at_any_point_never_return_bytes_of_another_write(
    tmp_path, engine, odirect, seed
):
    """The same workload on a device a third the size: stores often find no
    single free range and split across several, and deletes of split objects
    a checkpoint lists zero their headers while the others stay hazards."""
    config = _config(tmp_path, engine, 256 * 1024, odirect)
    versions: dict[int, list[bytes]] = {}
    vcount = itertools.count(seed * 1_000_000)
    core = RawBlockCore(config)
    base = Path(config.device_path).read_bytes()
    log = _run_workload(core, versions, vcount, seed)
    status = core.report_status()
    core.close()
    assert status["multi_extent_alloc_count"] > 5, status
    assert status["header_invalidation_count"] > 0, status

    last = max(c["end"] for c in log.calls)
    rng = random.Random(seed)
    bad: list[str] = []
    total = 0
    for t in sorted(rng.sample(range(1, last + 1), min(60, last))):
        for _ in range(3):
            Path(config.device_path).write_bytes(log.image(base, t, rng))
            loaded, b = _check_recovery(config, versions, f"t={t}")
            total += loaded
            bad += b
    assert total > 20, total
    assert not bad, bad[:10]
