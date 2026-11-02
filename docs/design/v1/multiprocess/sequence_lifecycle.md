# Sequence History Tracking and Superseded-Chunk Demotion

Source: `lmcache/v1/multiprocess/sequence_tracker.py` (bookkeeping) and
`lmcache/v1/multiprocess/modules/sequence_history.py` (the
`SequenceHistoryModule`), attached to `LookupModule` through
`LookupModule.add_lookup_observer`. The demotion hook lives in
`lmcache/v1/distributed/` (`internal_api.py`, `eviction.py`,
`l1_manager/base.py`, `storage_manager.py`, `eviction_policy/lru.py`,
`eviction_policy/isolated_lru.py`). Benchmark:
`benchmarks/microbenchmark/sequence_demotion_benchmark.py`. User guide:
`docs/source/mp/sequence_history.rst`.

## Problem

Consecutive requests of one agent trajectory are often not appends. Context
folding, summarization, tool-output pruning and `<think>` stripping make
request `n+1` diverge from request `n` at chunk `d`. Chunks `n[d:]` will not be
hit again by this trajectory, yet L1 ranks them like any recently used chunk.
A divergence nobody intended (a template or
tokenization change between turns) is also a train/inference-mismatch signal
for RL that the cache server can see cheaply, because it already hashes every
prompt into a chunk chain.

## Contract

| Item | Value |
|---|---|
| Request hint | `lmcache.sequence_id`: `str`, 1–128 chars of `[A-Za-z0-9._:-]` |
| Server switch | `--sequence-tracker-max-chunks N`; 0 (default) does not build the module |
| Demotion switch | `--sequence-demotion` (experimental, off by default; requires the budget) |
| Without hint, or switch off | today's behavior; no tracker state, no event |
| Malformed hint | ignored; at most one warning per 60 s, raw value never logged |
| Effect | `MP_SEQUENCE_TRANSITION` event and metrics; with `--sequence-demotion`, on a rewrite, **experimental** L1 demotion of superseded chunks no tracked sequence references |
| Never | deletion from L1 or L2, lock changes, cancelled L2 writes, change to the lookup result |

**Hint name.** `lmcache.hint.*` is used for values an *engine integration*
derives and adds (`lmcache.hint.engine_computed_tokens`); clients never set
them. The sequence id is set by the *client* as a property of its request,
like `lmcache.skip_load`, so it lives at the same level. It is deliberately
workload-neutral ("this request continues sequence X") rather than an RL term.

## Flow

```text
LookupModule.lookup(key)
  chunk_hashes = hash(prompt)                 # computed by the lookup anyway
  ... layout checks, KV load policy ...       # declined lookups are not observed
  for observer in _lookup_observers:          # the only hook in lookup.py
      observer.on_lookup_admitted(key, chunk_hashes)
  ... touch covered prefix, submit prefetch (unchanged) ...

SequenceHistoryModule.on_lookup_admitted(key, chunk_hashes)
  sequence_id = parse_sequence_id(key.request_configs)   # None -> return
  t = tracker.observe(SequenceKey(model, world_size, cache_salt, id),
                      request_id, chunk_hashes)
  DEMOTE_IN_L1 and REWRITE with unreferenced superseded chunks ->
      storage_manager.demote_cached_keys(keys, chunk-major in chain order)
        -> every L1: demote_keys -> listener.on_l1_keys_demoted
           -> L1EvictionPolicy -> EvictionPolicy.on_keys_demoted
  APPEND / REWRITE -> publish MP_SEQUENCE_TRANSITION
```

`SequenceHistoryModule` is a `DiscoverableModule` with
`module_dependencies = (LookupModule,)`; its `create` returns `None` when the
budget is 0, so a default server builds nothing and registers no gauge.
`LookupObserver` is a documented `Protocol`: observers are registered while
modules are built and called synchronously after the load policy, in
registration (module build) order, from any thread of the BLOCKING handler
pool, so they must be thread-safe and non-blocking. An observer that raises
is logged with `logger.exception` and skipped; the lookup proceeds unchanged,
because observability must never fail a lookup.

## Tracker

`SequenceTracker.observe` keeps, per `SequenceKey`, the latest request id and
its chunk-hash chain (the 32-byte `bytes` hashes the lookup computed; no
tokens).

- `d` is the common-prefix length of the previous and new chain.
  `d == len(prev)` is `APPEND`; otherwise `REWRITE` (covers a rewrite at 0 and a
  shorter prefix). No previous chain is `FIRST`; the same request id looked up
  again (an engine re-lookup after an APC change or preemption) is `REPEAT`.
- Cache salt, model and world size are part of the key, as they are part of
  every object key: tenants using equal sequence ids stay independent.
- **Shared-chain guard.** A reference count over all tracked chains
  (`hash -> chains containing it`) is updated before the transition is
  computed, so `unreferenced_hashes` excludes any superseded hash still in
  another tracked sequence's latest chain or in this sequence's new chain.
- **Bound.** At most `max_tracked_chunks` hashes across all chains. Entries are
  ordered by last observation; storing a chain forgets the oldest sequences
  until the budget holds. A chain longer than the budget is not stored. There
  is no TTL: an idle sequence costs memory only until newer sequences push it
  out, and the budget is the one knob that bounds memory.
- One lock guards the entry map and reference counts. `observe` is
  O(previous + new chain), next to hashing that is already O(chain).

### Memory

Measured with `tracemalloc` on CPython 3.12 (1000 sequences, 32-byte hashes,
22-char sequence ids, 41-char request ids), including everything the tracker
retains:

| Item | Bytes |
|---|---|
| per tracked chunk: hash `bytes` object (65) + tuple slot (8) + reference-count `dict` entry (~41, amortized table) | ~114 |
| per tracked sequence: `SequenceKey` + sequence-id `str`, request-id `str`, `_SequenceEntry`, chain tuple header, `OrderedDict` node | ~450 |

So the budget `N` costs about `114·N + 450·S` bytes for `S` sequences, and
`S <= N` because empty chains are not stored. Example: `N = 262144`
(256 sequences × 1024 chunks) is about 29 MiB. A hash shared by several chains
has one reference-count entry, so shared prefixes cost slightly less. The
reference counts are kept whether or not `--sequence-demotion` is set, so the
figure holds for report-only tracking too. The
model-name and salt strings are shared with the request and not counted.

## Demotion (experimental)

### Why demotion, and why it is safe

Keys are content-addressed: "superseded for sequence A" does not mean "dead".
A GRPO sibling, an untagged request with the same prefix, or a forgotten
sequence may still read the key. The action is therefore the weakest one that
can help:

1. **No data is removed.** Demotion only reorders the policy's candidate list;
   the L1 eviction controller still evicts only above its watermark and still
   skips keys `is_key_evictable` rejects (read/write-locked), so a request that
   is retrieving a demoted key is unaffected.
2. **Use restores recency.** Any later `touch_keys` (lookup of a covered
   prefix, `end_session`) moves the key back to the MRU end.
3. **Known sharers are excluded** by the shared-chain guard.
4. **Worst case is a cache miss**, the same outcome as ordinary memory
   pressure: the engine recomputes. It is never a wrong result.
5. **Races are benign.** A demotion that reaches the policy after the key was
   evicted is a no-op (policies only move keys they track), so it cannot
   resurrect a key.

The residual risk is untracked traffic that reuses superseded chunks (the
"replay" rows below); it pays recomputation.

Deliberately not done:

- **L2 deletion** — no other reader can be ruled out (other servers may share
  the L2), and L2 copies are the cheap tier.
- **L2 demotion** — `L2AdapterInterface` has `submit_touch` but no demotion
  call; adding one to every adapter is out of scope.
- **Cancelling pending L2 writes** — `StoreListener` drains its queue on the
  next eventfd wake-up, so by the time the next turn of a trajectory arrives
  (seconds later) those writes are in flight or done; a superseded chunk may
  also be another request's prefix.
- **ARC** keeps the base no-op: artificial demotions would feed its
  ghost-list adaptation a signal that does not come from the workload.

### Interface additions

Both hooks are non-abstract no-ops, so no existing listener, policy or
out-of-tree implementation changes behavior:

- `L1ManagerListener.on_l1_keys_demoted(keys)` and
  `L1ManagerInterface.demote_keys(keys)`; `BaseL1Manager.demote_keys` forwards
  to its listeners and, like `touch_keys`, takes no lock (it publishes no
  event).
- `EvictionPolicy.on_keys_demoted(keys)`, implemented by `LRUEvictionPolicy`
  and `IsolatedLRUEvictionPolicy` (within the key's salt) by moving each
  tracked key to the front in the given order, so the last key passed is
  evicted first.
- `StorageManager.demote_cached_keys(keys)`: every L1, no L2.

The module passes keys **chunk-major in chain order**: all object groups and
kv ranks of chunk `d`, then of `d+1`, and so on. The chain's last chunk is
then evicted first in every group, and the head stays a usable prefix the
longest — the same rationale as the policies' reverse-order inserts. The
expansion is `ipc_key_to_chunk_major_object_keys` in
`lmcache/v1/distributed/api.py`, which `LookupModule._all_object_keys` (the
`end_session` touch) now uses too, so a touch ranks whole chunks the same way
in every object group instead of group by group. These
calls are also the surface an explicit agent directive API ("drop this
branch") could use later.

### Evidence (CPU simulation)

`benchmarks/microbenchmark/sequence_demotion_benchmark.py` drives the real
`SequenceHistoryModule.on_lookup_admitted` (with `REPORT` vs `DEMOTE_IN_L1`)
against a real `DramL1Manager` + `LRUEvictionPolicy`; only the storage
manager is a stub forwarding `demote_cached_keys` to that L1. Keys fan out
over 2 object groups × 2 kv ranks per chunk (a chunk hits only when all four
keys are resident). The eviction controller's default watermark/ratio
(0.8/0.2) is applied synchronously after each request. Workload: 8 GRPO groups
× 8 trajectories, 24 turns, 1–3 new chunks per turn, a fold (branch replaced by
one summary chunk) every 4 turns, plus 20% untagged requests sharing a group's
system + task prefix. It is a **CPU simulation** of L1 only: no L2, no GPU, no
engine, 4 KiB objects. Seed 0; seeds 1 and 2 agree within ±1 point of hit
rate.

Default run (`--replay-prob 0`):

| L1 capacity (chunks, ≈ fraction of working set) | demotion | L1 hit rate | recomputed chunks | untagged shared-prefix hit rate |
|---|---|---|---|---|
| 352 (10%) | off / on | 0.783 / 0.786 | 9323 / 9208 (−1.2%) | 0.880 / 0.880 |
| 705 (20%) | off / on | 0.857 / 0.864 | 6172 / 5838 (−5.4%) | 0.886 / 0.886 |
| 1058 (30%) | off / on | 0.886 / 0.898 | 4906 / 4388 (−10.6%) | 0.886 / 0.886 |
| 1764 (50%) | off / on | 0.900 / 0.902 | 4283 / 4236 (−1.1%) | 0.886 / 0.886 |

Adversarial run (`--replay-prob 0.1`: 10% untagged replays of pre-fold
histories, which reuse exactly the chunks demotion pushes out):

| L1 capacity | demotion | L1 hit rate | recomputed chunks | replay hit rate |
|---|---|---|---|---|
| 352 (10%) | off / on | 0.771 / 0.773 | 10846 / 10733 (−1.0%) | 0.687 / 0.686 |
| 705 (20%) | off / on | 0.842 / 0.848 | 7478 / 7193 (−3.8%) | 0.755 / 0.741 |
| 1058 (30%) | off / on | 0.874 / 0.882 | 5984 / 5568 (−7.0%) | 0.805 / 0.799 |
| 1764 (50%) | off / on | 0.897 / 0.899 | 4886 / 4773 (−2.3%) | 0.872 / 0.875 |

Reading: demotion helps modestly and only in a band of memory pressure where
superseded branches compete with live prefixes; with very little or ample
memory it is neutral. Shared-prefix traffic is unaffected thanks to the guard,
but traffic that re-reads superseded chunks loses up to 1.4 points of hit rate
even while the total improves. Whether this carries over to real serving (L2
refills, real chunk sizes, real fold rates) is not measured. Hence demotion is
experimental and needs its own flag, `--sequence-demotion` (off by default),
on top of `--sequence-tracker-max-chunks`; without it the module only reports.

## Observability

`MP_SEQUENCE_TRANSITION` (see `../mp_observability/EVENTS.md`) is published for
`APPEND` and `REWRITE` and feeds `SequenceMetricsSubscriber`
(`../mp_observability/METRICS.md`): a transition counter by `kind`, a
superseded-chunk counter, a demotion-requested counter (hashes passed to
`demote_cached_keys`; counted even when the policy ignores demotion, e.g.
ARC, or L1 no longer holds the chunk) and a divergence-chunk histogram with
fixed buckets
(0, then powers of two up to 4096). Labels are `model_name` and `kind` only —
never the sequence id. A DEBUG line names the request id and divergence chunk
of a rewrite, at most once per 60 s. `lmcache_mp.sequence_tracker_chunks`
gauges tracker memory.

## Limitations

- **Prompt chunks only.** The chain is the lookup's prompt, not decode tokens
  stored with `save_decode_cache`. Consequently `<think>` stripping shows up as
  a rewrite only where the stripped reasoning was part of an earlier *prompt*
  chain, e.g. a multi-step tool loop whose intermediate assistant turns (with
  reasoning) are re-sent until the next user message. In a plain multi-turn
  chat the reasoning exists only in the previous response (decode), so the next
  prompt is still an extension of the previous prompt and reports `APPEND`.
- Demotion takes effect only with the `LRU` and `IsolatedLRU` L1 eviction
  policies; L2 eviction order is unchanged.
- Tracking is per MP server process; a trajectory routed to several servers is
  tracked independently on each.
- Only the standard `LookupModule` notifies observers; CacheBlend lookups are
  not tracked.
- Engines forward the hint only if their integration forwards `lmcache.*`
  request configs on lookup (the vLLM MP connector does).
- Reusing one id for unrelated or concurrent trajectories makes them look like
  rewrites of each other.

## Tests

- `tests/v1/multiprocess/test_sequence_tracker.py`: divergence cases,
  shared-chain guard, budget, concurrent observers, hint parsing.
- `tests/v1/multiprocess/test_sequence_history_module.py`: module build and
  observer wiring, no-hint and switch-off paths, declined lookups, malformed
  hints, salt scoping, demotion order and fan-out over groups and kv ranks.
- `tests/v1/mp_observability/subscribers/metrics/test_sequence.py`: metrics
  and label set.
- `tests/v1/distributed/test_l1_demotion.py`: eviction order through a real
  `DramL1Manager` for LRU, IsolatedLRU and ARC; demoted keys stay readable; a
  read-locked demoted key is skipped by the evictability filter; a demotion
  racing an eviction (forced interleaving) does not resurrect the key.
