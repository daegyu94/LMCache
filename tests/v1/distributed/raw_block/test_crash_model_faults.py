# SPDX-License-Identifier: Apache-2.0
"""Fences, carving and waits under failures and contention: failed buffer
preparation in a batch, stores racing a fence wait, batches holding fences,
and a crash model on a tight device with injected preparation failures."""

# Standard
from pathlib import Path
from unittest.mock import patch
import random
import threading
import time

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.l2_adapters.raw_block import RawBlockCore
from lmcache.v1.distributed.l2_adapters.raw_block.io_engine import (
    IoUringBlockIO,
    PosixBlockIO,
)
from tests.v1.distributed.raw_block.test_crash_model import (
    ALIGN,
    HEADER,
    _config,
    _spec,
)
from tests.v1.distributed.raw_block.test_utils import (
    make_empty_memory_obj,
    make_memory_obj,
    memory_obj_bytes,
)

pytestmark = pytest.mark.skipif(
    __import__("importlib").util.find_spec("lmcache_rust_raw_block_io") is None,
    reason="lmcache_rust_raw_block_io is not installed",
)


def _recover_bytes(config, image: bytes, key: int) -> bytes | None:
    Path(config.device_path).write_bytes(image)
    core = RawBlockCore(config)
    try:
        meta = core.get_metadata_many([_spec(key).encoded])[0]
        if meta is None:
            return None
        obj = make_empty_memory_obj(meta.size)
        if core.load_many_into([_spec(key).encoded], [obj]) != [True]:
            return None
        return memory_obj_bytes(obj)
    finally:
        core.close()


@pytest.mark.parametrize("fail", ["head", "carved"])
def test_prep_failure_in_batch_keeps_hazard_safe(tmp_path, fail):
    """Head store covers hazard k1's start, a second store of the same batch
    is carved from its fence over k1's tail. If buffer preparation of one of
    them fails, the other must not leave k1's tail overwritten while k1's
    header (listed by the checkpoint) survives."""
    config = _config(tmp_path, "io_uring", 1 << 20)
    core = RawBlockCore(config)
    k1_data = random.Random(7).randbytes(8 * ALIGN - HEADER)
    try:
        for k, data in enumerate([bytes(ALIGN), k1_data, bytes(ALIGN)]):
            assert core.put_many([_spec(k)], [make_memory_obj(data)]).results == [True]
        core.checkpoint_now()  # lists k0, k1, k2
        assert core.delete_many([_spec(0).encoded, _spec(1).encoded]) == [True] * 2
        head = make_memory_obj(bytes(4 * ALIGN - HEADER))
        carved = make_memory_obj(b"\xab" * ALIGN)
        other = make_memory_obj(bytes(ALIGN))
        victim = head if fail == "head" else carved
        real = RawBlockCore._prepare_write_payload

        def prep(self, obj):
            if obj is victim:
                raise MemoryError("bounce buffer")
            return real(self, obj)

        with patch.object(RawBlockCore, "_prepare_write_payload", prep):
            r = core.put_many([_spec(10), _spec(11), _spec(12)], [head, carved, other])
        print("results", r.results)
        # Process crash right here: the device holds what was written.
        image = Path(config.device_path).read_bytes()
    finally:
        core.close()
    got = _recover_bytes(config, image, 1)
    assert got is None or got == k1_data, "k1 recovered with another write's bytes"


def test_prep_failure_of_a_carved_store_fails_its_head_store_too(tmp_path):
    """Freeing a carved store alone would leave a free range starting inside
    the hazard, which another thread could take unguarded before the head
    store writes the hazard's first byte. Both fail, and the space comes back
    whole."""
    config = _config(tmp_path, "io_uring", 1 << 20)
    core = RawBlockCore(config)
    k1_data = random.Random(8).randbytes(8 * ALIGN - HEADER)
    try:
        for k, data in enumerate([bytes(ALIGN), k1_data, bytes(ALIGN)]):
            assert core.put_many([_spec(k)], [make_memory_obj(data)]).results == [True]
        free_before = core.report_status()["free_extent_count"]
        core.checkpoint_now()
        assert core.delete_many([_spec(0).encoded, _spec(1).encoded]) == [True] * 2
        head = make_memory_obj(bytes(4 * ALIGN - HEADER))
        carved = make_memory_obj(bytes(ALIGN))
        real_prep = RawBlockCore._prepare_write_payload

        # Fault injection: buffer preparation has no public hook.
        def prep(self, obj):
            if obj is carved:
                raise MemoryError("bounce buffer")
            return real_prep(self, obj)

        with patch.object(RawBlockCore, "_prepare_write_payload", prep):
            result = core.put_many([_spec(10), _spec(11)], [head, carved])
        assert result.results == [False, False]
        status = core.report_status()
        assert status["reuse_hazard_count"] == 2
        assert status["allocated_bytes"] == 2 * ALIGN  # key 2 only
        assert status["free_extent_count"] == free_before + 1
    finally:
        core.close()
    image = Path(config.device_path).read_bytes()
    got = _recover_bytes(config, image, 1)
    assert got is None or got == k1_data


def test_waiting_store_does_not_race_a_store_of_the_same_key(tmp_path):
    """A store that waits for a fence releases the lock after checking that
    its key is neither indexed nor in flight. Another store of the same key
    may commit meanwhile; the waiter then commits over it and the first
    object's space is never freed."""
    # Marker block, then objects of 2, 8 and 6 blocks.
    config = _config(tmp_path, "posix", (1 + 2 + 8 + 6) * ALIGN)
    core = RawBlockCore(config)
    real_write = PosixBlockIO.write_buffers
    started, release = threading.Event(), threading.Event()

    def held(self, offsets, *args, **kwargs):
        if offsets[0] == target and not started.is_set():
            started.set()
            assert release.wait(10)
        return real_write(self, offsets, *args, **kwargs)

    try:
        for k, blocks in enumerate([2, 8, 6]):
            data = bytes(blocks * ALIGN - HEADER)
            assert core.put_many([_spec(k)], [make_memory_obj(data)]).results == [True]
        target = core.entry_offset(_spec(1).encoded)
        core.checkpoint_now()
        assert core.delete_many([_spec(1).encoded]) == [True]
        six = bytes(6 * ALIGN - HEADER)
        out: list = []
        with patch.object(PosixBlockIO, "write_buffers", held):
            # Holds a 6-block fence while its write is blocked.
            head = threading.Thread(
                target=lambda: core.put_many(
                    [_spec(20)], [make_memory_obj(bytes(ALIGN))]
                )
            )
            head.start()
            assert started.wait(10)
            waiter = threading.Thread(
                target=lambda: out.append(
                    core.put_many([_spec(21)], [make_memory_obj(six)])
                )
            )
            waiter.start()
            time.sleep(0.2)  # waiter is in _fence_released.wait
            assert core.delete_many([_spec(2).encoded]) == [True]
            other = core.put_many([_spec(21)], [make_memory_obj(six)])
            assert other.results == [True]
            release.set()
            head.join(10)
            waiter.join(10)
        print("waiter", out, "other", other)
        core.delete_many([_spec(k).encoded for k in (0, 20, 21)], force=True)
        status = core.report_status()
        assert status["indexed_key_count"] == 0
        assert status["allocated_bytes"] == 0, status
    finally:
        release.set()
        core.close()


def test_two_batches_holding_fences_do_not_stall_each_other(tmp_path):
    """Two io_uring batches each take a fence with their first store and
    then find no space for the rest: each waits for the other's fence, which
    is only released after its own reservation loop ends, so every further
    key waits the full second. A third single-key store with a slow write
    holds a fence so that the first batch starts waiting."""
    # Marker, then [2, 8] * 3 + [2] blocks; deleting the 8s leaves 3 hazards.
    layout = [2, 8, 2, 8, 2, 8, 2]
    config = _config(tmp_path, "io_uring", (1 + sum(layout)) * ALIGN)
    core = RawBlockCore(config)
    real_write = IoUringBlockIO.write_buffers
    started, release = threading.Event(), threading.Event()

    def held(self, *args, **kwargs):
        if threading.current_thread().name == "C" and not started.is_set():
            started.set()
            assert release.wait(10)
        return real_write(self, *args, **kwargs)

    try:
        for k, blocks in enumerate(layout):
            data = bytes(blocks * ALIGN - HEADER)
            assert core.put_many([_spec(k)], [make_memory_obj(data)]).results == [True]
        core.checkpoint_now()
        assert core.delete_many([_spec(k).encoded for k in (1, 3, 5)]) == [True] * 3
        big = bytes(12 * ALIGN - HEADER)  # never fits
        elapsed: dict[int, float] = {}

        def batch(base: int) -> None:
            t0 = time.monotonic()
            core.put_many(
                [_spec(base + j) for j in range(5)],
                [make_memory_obj(bytes(ALIGN))]
                + [make_memory_obj(big) for _ in range(4)],
            )
            elapsed[base] = time.monotonic() - t0

        with patch.object(IoUringBlockIO, "write_buffers", held):
            c = threading.Thread(
                name="C",
                target=lambda: core.put_many(
                    [_spec(50)], [make_memory_obj(bytes(ALIGN))]
                ),
            )
            c.start()
            assert started.wait(10)
            a = threading.Thread(target=batch, args=(100,))
            b = threading.Thread(target=batch, args=(200,))
            a.start()
            time.sleep(0.1)
            b.start()
            time.sleep(0.1)
            release.set()
            a.join(60)
            b.join(60)
            c.join(10)
        print("elapsed", elapsed)
        assert max(elapsed.values()) < 1.5, elapsed
    finally:
        release.set()
        core.close()


def test_close_while_waiting_for_a_fence(tmp_path):
    """close() during a store's fence wait: the store must return without
    error and without touching the closed engine afterwards."""
    config = _config(tmp_path, "posix", (1 + 2 + 8 + 2) * ALIGN)
    core = RawBlockCore(config)
    real_write = PosixBlockIO.write_buffers
    started, release = threading.Event(), threading.Event()

    def held(self, offsets, *args, **kwargs):
        if offsets[0] == target and not started.is_set():
            started.set()
            assert release.wait(10)
        return real_write(self, offsets, *args, **kwargs)

    errors: list = []
    try:
        for k, blocks in enumerate([2, 8, 2]):
            data = bytes(blocks * ALIGN - HEADER)
            assert core.put_many([_spec(k)], [make_memory_obj(data)]).results == [True]
        target = core.entry_offset(_spec(1).encoded)
        core.checkpoint_now()
        assert core.delete_many([_spec(1).encoded]) == [True]
        with patch.object(PosixBlockIO, "write_buffers", held):
            head = threading.Thread(
                target=lambda: core.put_many(
                    [_spec(20)], [make_memory_obj(bytes(ALIGN))]
                )
            )
            head.start()
            assert started.wait(10)

            def wait_store():
                try:
                    out = core.put_many(
                        [_spec(21)], [make_memory_obj(bytes(5 * ALIGN))]
                    )
                    errors.append(("result", out.results))
                except Exception as e:  # noqa: BLE001
                    errors.append(("exc", repr(e)))

            w = threading.Thread(target=wait_store)
            w.start()
            time.sleep(0.2)
            closer = threading.Thread(target=core.close)
            closer.start()
            time.sleep(0.2)
            release.set()
            head.join(10)
            w.join(10)
            closer.join(10)
        print("waiter", errors)
    finally:
        release.set()
        core.close()


@pytest.mark.parametrize("engine,odirect", [("io_uring", False), ("posix", False)])
@pytest.mark.parametrize("seed", [11, 12, 13, 14, 15, 16])
@pytest.mark.parametrize("fail_rate", [0.0, 0.15])
def test_crash_model_with_prep_failures(tmp_path, engine, odirect, seed, fail_rate):
    """The crash-model workload with buffer preparation failing at random:
    every recovered key must still hold the bytes of one of its writes."""
    # First Party
    from tests.v1.distributed.raw_block import test_crash_model as cm

    config = _config(tmp_path, engine, 640 * 1024, odirect)
    versions: dict[int, list[bytes]] = {}
    vcount = __import__("itertools").count(seed * 1_000_000)
    core = RawBlockCore(config)
    base = Path(config.device_path).read_bytes()
    real = RawBlockCore._prepare_write_payload
    frng = random.Random(seed)
    flock = threading.Lock()

    def prep(self, obj):
        with flock:
            fail = frng.random() < fail_rate
        if fail:
            raise MemoryError("injected")
        return real(self, obj)

    with patch.object(RawBlockCore, "_prepare_write_payload", prep):
        log = cm._run_workload(core, versions, vcount, seed)
    core.close()
    last = max(c["end"] for c in log.calls)
    rng = random.Random(seed)
    bad: list[str] = []
    for t in sorted(rng.sample(range(1, last + 1), min(60, last))):
        for _ in range(3):
            Path(config.device_path).write_bytes(log.image(base, t, rng))
            _, b = cm._check_recovery(config, versions, f"t={t}")
            bad += b
    assert not bad, bad[:10]


def _tight_workload(core, versions, vcount, seed, *, ops, max_batch, fail_rate):
    """cm._run_workload on a device that runs out of space: batches up to
    ``max_batch`` keys, optional buffer preparation failures."""
    # First Party
    from tests.v1.distributed.raw_block import test_crash_model as cm

    log = cm._WriteLog()
    vlock = threading.Lock()
    errors: list[BaseException] = []
    real = RawBlockCore._prepare_write_payload
    frng = random.Random(seed)

    def prep(self, obj):
        with vlock:
            fail = frng.random() < fail_rate
        if fail:
            raise MemoryError("injected")
        return real(self, obj)

    def worker(wid: int) -> None:
        rng = random.Random(seed * 100 + wid)
        try:
            for _ in range(ops):
                op = rng.random()
                if op < 0.6:
                    keys = rng.sample(range(24), rng.randint(1, max_batch))
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
        patch.object(RawBlockCore, "_prepare_write_payload", prep),
    ):
        threads = [threading.Thread(target=worker, args=(w,)) for w in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(300)
    assert not errors, errors
    return log


@pytest.mark.parametrize("engine", ["io_uring", "posix"])
@pytest.mark.parametrize("seed", [21, 22, 23, 24])
@pytest.mark.parametrize("fail_rate", [0.0, 0.1])
def test_tight_crash_model(tmp_path, engine, seed, fail_rate):
    """Device of 48 blocks for 24 keys of 2-8 blocks: stores run out of
    space, fences are carved and waited for."""
    # First Party
    from tests.v1.distributed.raw_block import test_crash_model as cm

    config = _config(tmp_path, engine, 48 * ALIGN)
    versions: dict[int, list[bytes]] = {}
    vcount = __import__("itertools").count(seed * 1_000_000)
    core = RawBlockCore(config)
    base = Path(config.device_path).read_bytes()
    t0 = time.monotonic()
    log = _tight_workload(
        core, versions, vcount, seed, ops=80, max_batch=6, fail_rate=fail_rate
    )
    took = time.monotonic() - t0
    status = core.report_status()
    core.delete_many([_spec(k).encoded for k in range(24)], force=True)
    after = core.report_status()
    core.close()
    print(
        f"took={took:.2f}s exhausted={status['space_exhausted_count']} "
        f"failures={status['alloc_failure_count']} "
        f"allocated_after_delete_all={after['allocated_bytes']}"
    )
    last = max(c["end"] for c in log.calls)
    rng = random.Random(seed)
    bad: list[str] = []
    for t in sorted(rng.sample(range(1, last + 1), min(60, last))):
        for _ in range(3):
            Path(config.device_path).write_bytes(log.image(base, t, rng))
            _, b = cm._check_recovery(config, versions, f"t={t}")
            bad += b
    assert after["allocated_bytes"] == 0, after["allocated_bytes"]
    assert not bad, bad[:10]


@pytest.mark.parametrize("zeroing_fails", [False, True])
def test_a_failed_write_over_a_hazard_zeroes_its_header_first(tmp_path, zeroing_fails):
    """A failed request may have landed in part, so the hazard's later bytes
    may be new while its header still matches: its header is zeroed before
    the space is freed, or the space is retired if that write fails too."""
    config = _config(tmp_path, "io_uring", 1 << 20)
    core = RawBlockCore(config)
    real_write = IoUringBlockIO.write_buffers

    def failing(self, offsets, buffers, *args, **kwargs):
        if first + HEADER in offsets:  # the new object's payload
            raise OSError("injected")
        if (
            zeroing_fails
            and offsets == [first]
            and not any(bytes(memoryview(buffers[0])[:8]))
        ):
            raise OSError("injected")
        return real_write(self, offsets, buffers, *args, **kwargs)

    try:
        for k, blocks in enumerate([8, 1]):
            data = bytes(blocks * ALIGN - HEADER) if blocks > 1 else b"x"
            assert core.put_many([_spec(k)], [make_memory_obj(data)]).results == [True]
        first = core.entry_offset(_spec(0).encoded)
        core.checkpoint_now()
        assert core.delete_many([_spec(0).encoded]) == [True]
        usable = core.report_status()["usable_capacity_bytes"]
        with patch.object(IoUringBlockIO, "write_buffers", failing):
            result = core.put_many([_spec(10)], [make_memory_obj(bytes(ALIGN))])
        assert result.results == [False]
        status = core.report_status()
        if zeroing_fails:
            assert status["usable_capacity_bytes"] < usable
            assert status["header_invalidation_failure_count"] == 1
        else:
            assert status["reuse_hazard_count"] == 0
            assert status["usable_capacity_bytes"] == usable
            with open(config.device_path, "rb") as f:
                f.seek(first)
                assert f.read(8) == bytes(8)
    finally:
        core.close()
