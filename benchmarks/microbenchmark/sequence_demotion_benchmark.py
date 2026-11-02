#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CPU simulation: does demoting superseded chunks raise the L1 hit rate?

Replays a synthetic agentic-RL workload against a real ``DramL1Manager`` with
the ``LRU`` eviction policy, once with superseded-chunk demotion off and once
on, and reports the L1 prefix hit rate. Every tagged request goes through the
real ``SequenceHistoryModule.on_lookup_admitted``; only the storage manager is
a stub that forwards ``demote_cached_keys`` to the one L1. Keys fan out over
2 object groups x 2 kv ranks per chunk, as for a hybrid model at TP=2.
Run with::

    python benchmarks/microbenchmark/sequence_demotion_benchmark.py

Workload (seeded, so every run is identical):

- ``--groups`` GRPO groups of ``--group-size`` trajectories. All requests
  start with a shared system prompt; each group shares a task prompt.
- Each trajectory runs ``--turns`` turns and appends 1-3 chunks per turn.
  Every ``--fold-every`` turns it folds: the chunks added since the last
  fold are replaced by one summary chunk, so the next request rewrites its
  history at the fold point.
- Untagged traffic: with ``--shared-prob`` a step is an untagged request that
  reuses a group's system + task prefix with a fresh suffix. With
  ``--replay-prob`` a step is an untagged replay of some trajectory's
  *pre-fold* history, i.e. traffic that reuses exactly the chunks demotion
  pushes out (the adversarial case for demotion).

Each request follows the server's order: prefix lookup in L1 (a chunk hits
when all of its keys are resident; hits stop at the first miss), the
module's observer call, store of the missing chunks, touch of all of the
request's keys (``end_session``), then one synchronous eviction step
with the eviction controller's default watermark (0.8) and ratio (0.2). The
background controller is emulated synchronously so the result is
deterministic; objects are 4 KiB and only L1 is simulated (no L2, no GPU,
no engine).
"""

# Standard
from dataclasses import dataclass, field
from typing import cast
import argparse
import hashlib
import random

# Third Party
import torch

# First Party
from lmcache.v1.distributed.api import (
    AttnWindowDesc,
    MemoryLayoutDesc,
    ObjectKey,
    ipc_key_to_chunk_major_object_keys,
)
from lmcache.v1.distributed.config import L1ManagerConfig, L1MemoryManagerConfig
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.eviction import L1EvictionPolicy
from lmcache.v1.distributed.eviction_policy import LRUEvictionPolicy
from lmcache.v1.distributed.l1_manager import DramL1Manager
from lmcache.v1.mp_observability.event_bus import EventBus, EventBusConfig
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.engine_context import (
    LayoutDescRegistry,
    MPCacheServerContext,
)
from lmcache.v1.multiprocess.modules.sequence_history import (
    SequenceHistoryModule,
    SupersededChunkAction,
)
from lmcache.v1.multiprocess.sequence_tracker import (
    SEQUENCE_ID_HINT,
    SequenceTracker,
)

LAYOUT = MemoryLayoutDesc([torch.Size([1024])], [torch.float32])  # 4 KiB
OBJECT_BYTES = 4096
MODEL = "sim"
WORLD_SIZE = 2
OBJECT_GROUPS = [0, 1]
KEYS_PER_CHUNK = WORLD_SIZE * len(OBJECT_GROUPS)
WATERMARK = 0.8
EVICTION_RATIO = 0.2


def _chain(contents: list[str]) -> list[bytes]:
    """Rolling prefix hashes, like the server's chunk chain."""
    hashes: list[bytes] = []
    prefix = b""
    for content in contents:
        prefix = hashlib.blake2b(prefix + content.encode(), digest_size=16).digest()
        hashes.append(prefix)
    return hashes


def _ipc_key(request_id: str, sequence_id: str | None) -> IPCCacheServerKey:
    return IPCCacheServerKey.from_token_ids(
        model_name=MODEL,
        world_size=WORLD_SIZE,
        worker_id=None,
        token_ids=[],
        request_id=request_id,
        request_configs=(
            {SEQUENCE_ID_HINT: sequence_id} if sequence_id is not None else None
        ),
        num_kv_readers=1,
    )


def _keys(hashes: list[bytes]) -> list[ObjectKey]:
    return ipc_key_to_chunk_major_object_keys(_ipc_key("", None), hashes, OBJECT_GROUPS)


@dataclass
class _Trajectory:
    seq_id: str
    group: int
    prefix: list[str]
    branch: list[str] = field(default_factory=list)
    turn: int = 0
    folds: int = 0
    # Full contents of each pre-fold request, for replay traffic.
    folded_histories: list[list[str]] = field(default_factory=list)


@dataclass
class _Stats:
    requested: int = 0
    hit: int = 0

    @property
    def hit_rate(self) -> float:
        return self.hit / self.requested if self.requested else 0.0

    @property
    def recomputed(self) -> int:
        return self.requested - self.hit


@dataclass
class _Result:
    tagged: _Stats = field(default_factory=_Stats)
    shared: _Stats = field(default_factory=_Stats)
    replay: _Stats = field(default_factory=_Stats)
    # Superseded chunk hashes passed to demote_keys.
    demoted: int = 0

    @property
    def total(self) -> _Stats:
        return _Stats(
            self.tagged.requested + self.shared.requested + self.replay.requested,
            self.tagged.hit + self.shared.hit + self.replay.hit,
        )


class _Cache:
    """A real L1 manager plus LRU policy, with a synchronous eviction step."""

    def __init__(self, capacity_chunks: int) -> None:
        self.l1 = DramL1Manager(
            L1ManagerConfig(
                L1MemoryManagerConfig(
                    size_in_bytes=capacity_chunks * KEYS_PER_CHUNK * OBJECT_BYTES,
                    use_lazy=False,
                    shm_name="",
                )
            )
        )
        self.policy = LRUEvictionPolicy()
        self.l1.register_listener(L1EvictionPolicy(self.policy))
        self.demotion_requests = 0

    def lookup(self, hashes: list[bytes]) -> int:
        """Return the number of leading chunks whose keys are all resident."""
        hit = 0
        for chunk_hash in hashes:
            keys = _keys([chunk_hash])
            reserved = self.l1.reserve_read(keys)
            ok = [k for k, (err, _) in reserved.items() if err is L1Error.SUCCESS]
            if ok:
                self.l1.finish_read(ok)
            if len(ok) != len(keys):
                break
            hit += 1
        return hit

    def store(self, hashes: list[bytes]) -> None:
        keys = _keys(hashes)
        if not keys:
            return
        reserved = self.l1.reserve_write(keys, [False] * len(keys), LAYOUT)
        written = [k for k, (err, _) in reserved.items() if err is L1Error.SUCCESS]
        if written:
            self.l1.finish_write(written)

    def touch(self, hashes: list[bytes]) -> None:
        self.l1.touch_keys(_keys(hashes))

    def evict_step(self) -> None:
        used, total = self.l1.get_memory_usage()
        if total == 0 or used / total < WATERMARK:
            return
        for action in self.policy.get_eviction_actions(
            EVICTION_RATIO, key_eligible_filter=self.l1.is_key_evictable
        ):
            self.l1.delete(action.keys)

    # StorageManager stub: the only call SequenceHistoryModule makes.
    def demote_cached_keys(self, keys: list[ObjectKey]) -> None:
        self.demotion_requests += len(keys) // KEYS_PER_CHUNK
        self.l1.demote_keys(keys)

    def close(self) -> None:
        self.l1.close()


class _BenchContext:
    """The parts of ``MPCacheServerContext`` the module reads."""

    def __init__(self, cache: _Cache) -> None:
        self.event_bus = EventBus(EventBusConfig(enabled=False))
        self.layout_desc_registry = LayoutDescRegistry()
        self.layout_desc_registry.register(
            MODEL,
            WORLD_SIZE,
            LAYOUT,
            AttnWindowDesc(num_chunks_in_sw=[-1, -1], world_size=WORLD_SIZE),
        )
        self.storage_manager = cache


def _serve(
    cache: _Cache, stats: _Stats, contents: list[str]
) -> tuple[list[bytes], int]:
    """Look the request up and return its chain and L1 prefix hit."""
    hashes = _chain(contents)
    hit = cache.lookup(hashes)
    stats.requested += len(hashes)
    stats.hit += hit
    return hashes, hit


def run(args: argparse.Namespace, capacity_chunks: int, demote: bool) -> _Result:
    """Replay the workload once and return its hit statistics."""
    rng = random.Random(args.seed)
    cache = _Cache(capacity_chunks)
    module = SequenceHistoryModule(
        cast(MPCacheServerContext, _BenchContext(cache)),
        SequenceTracker(args.tracker_chunks),
        SupersededChunkAction.DEMOTE_IN_L1 if demote else SupersededChunkAction.REPORT,
    )
    result = _Result()
    system = [f"sys:{i}" for i in range(args.system_chunks)]
    trajectories = [
        _Trajectory(
            seq_id=f"g{g}-t{t}",
            group=g,
            prefix=system + [f"task:{g}:{i}" for i in range(args.task_chunks)],
        )
        for g in range(args.groups)
        for t in range(args.group_size)
    ]
    active = list(trajectories)
    untagged = 0
    while active:
        roll = rng.random()
        if roll < args.shared_prob:
            untagged += 1
            group = rng.randrange(args.groups)
            contents = (
                system
                + [f"task:{group}:{i}" for i in range(args.task_chunks)]
                + [f"q:{untagged}:{i}" for i in range(rng.randint(1, 3))]
            )
            hashes, hit = _serve(cache, result.shared, contents)
        elif roll < args.shared_prob + args.replay_prob and any(
            t.folded_histories for t in trajectories
        ):
            source = rng.choice([t for t in trajectories if t.folded_histories])
            contents = rng.choice(source.folded_histories)
            hashes, hit = _serve(cache, result.replay, contents)
        else:
            traj = rng.choice(active)
            traj.branch += [
                f"turn:{traj.seq_id}:{traj.turn}:{j}" for j in range(rng.randint(1, 3))
            ]
            traj.turn += 1
            contents = traj.prefix + traj.branch
            hashes, hit = _serve(cache, result.tagged, contents)
            module.on_lookup_admitted(
                _ipc_key(f"{traj.seq_id}-{traj.turn}", traj.seq_id), hashes
            )
            if traj.turn % args.fold_every == 0:
                traj.folded_histories.append(contents)
                traj.prefix = traj.prefix + [f"sum:{traj.seq_id}:{traj.folds}"]
                traj.branch = []
                traj.folds += 1
            if traj.turn == args.turns:
                active.remove(traj)
        cache.store(hashes[hit:])
        cache.touch(hashes)
        cache.evict_step()
    result.demoted = cache.demotion_requests
    cache.close()
    return result


def _working_set(args: argparse.Namespace) -> int:
    """Distinct tagged chunks the workload creates (upper bound, approx.)."""
    per_traj = args.turns * 2 + args.turns // args.fold_every
    return (
        args.system_chunks
        + args.groups * args.task_chunks
        + args.groups * args.group_size * per_traj
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--groups", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--turns", type=int, default=24)
    parser.add_argument("--fold-every", type=int, default=4)
    parser.add_argument("--system-chunks", type=int, default=8)
    parser.add_argument("--task-chunks", type=int, default=8)
    parser.add_argument("--shared-prob", type=float, default=0.2)
    parser.add_argument("--replay-prob", type=float, default=0.0)
    parser.add_argument("--tracker-chunks", type=int, default=1 << 16)
    parser.add_argument(
        "--capacity-fractions",
        type=float,
        nargs="+",
        default=[0.1, 0.2, 0.3, 0.5],
        help="L1 capacity as a fraction of the approximate working set.",
    )
    args = parser.parse_args()

    working_set = _working_set(args)
    print(
        f"working set ~{working_set} chunks; shared_prob={args.shared_prob} "
        f"replay_prob={args.replay_prob} seed={args.seed}"
    )
    header = (
        f"{'L1 chunks':>9} {'demote':>6} {'hit rate':>8} {'recomputed':>10} "
        f"{'tagged':>7} {'shared':>7} {'replay':>7} {'demote req':>10}"
    )
    print(header)
    for fraction in args.capacity_fractions:
        capacity = max(1, int(working_set * fraction))
        for demote in (False, True):
            r = run(args, capacity, demote)
            print(
                f"{capacity:>9} {'on' if demote else 'off':>6} "
                f"{r.total.hit_rate:>8.3f} {r.total.recomputed:>10} "
                f"{r.tagged.hit_rate:>7.3f} {r.shared.hit_rate:>7.3f} "
                f"{r.replay.hit_rate:>7.3f} {r.demoted:>10}"
            )


if __name__ == "__main__":
    main()
