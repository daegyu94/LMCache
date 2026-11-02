# Sequence History Tracking

Source: `lmcache/v1/multiprocess/sequence_tracker.py` (bookkeeping) and
`lmcache/v1/multiprocess/modules/sequence_history.py` (the
`SequenceHistoryModule`), attached to `LookupModule` through
`LookupModule.add_lookup_observer`. User guide:
`docs/source/mp/sequence_history.rst`.

## Problem

Consecutive requests of one agent trajectory are often not appends. Context
folding, summarization, tool-output pruning and `<think>` stripping make
request `n+1` diverge from request `n` at chunk `d`. Chunks `n[d:]` will not be
hit again by this trajectory. A divergence nobody intended (a template or
tokenization change between turns) is also a train/inference-mismatch signal
for RL that the cache server can see cheaply, because it already hashes every
prompt into a chunk chain.

## Contract

| Item | Value |
|---|---|
| Request hint | `lmcache.sequence_id`: `str`, 1–128 chars of `[A-Za-z0-9._:-]` |
| Server switch | `--sequence-tracker-max-chunks N`; 0 (default) does not build the module |
| Without hint, or switch off | today's behavior; no tracker state, no event |
| Malformed hint | ignored; at most one warning per 60 s, raw value never logged |
| Effect | `MP_SEQUENCE_TRANSITION` event and metrics; no cache side effect |

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
- **Bound.** At most `max_tracked_chunks` hashes across all chains. Entries are
  ordered by last observation; storing a chain forgets the oldest sequences
  until the budget holds. A chain longer than the budget is not stored. There
  is no TTL: an idle sequence costs memory only until newer sequences push it
  out, and the budget is the one knob that bounds memory.
- One lock guards the entry map; the divergence scan runs outside it on
  immutable tuples. `observe` is O(previous + new chain), next to hashing that
  is already O(chain).

### Memory

Measured with `tracemalloc` on CPython 3.12 (1000 sequences, 32-byte hashes,
22-char sequence ids, 41-char request ids), including everything the tracker
retains:

| Item | Bytes |
|---|---|
| per tracked chunk: hash `bytes` object (65) + tuple slot (8) | ~73 |
| per tracked sequence: `SequenceKey` + sequence-id `str`, request-id `str`, `_SequenceEntry`, chain tuple header, `OrderedDict` node | ~450 |

So the budget `N` costs about `73·N + 450·S` bytes for `S` sequences, and
`S <= N` because empty chains are not stored. Example: `N = 262144`
(256 sequences × 1024 chunks) is about 19 MiB. The model-name and salt strings
are shared with the request and not counted.

## Observability

`MP_SEQUENCE_TRANSITION` (see `../mp_observability/EVENTS.md`) is published for
`APPEND` and `REWRITE` and feeds `SequenceMetricsSubscriber`
(`../mp_observability/METRICS.md`): a transition counter by `kind`, a
superseded-chunk counter and a divergence-chunk histogram with fixed buckets
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
- Tracking is per MP server process; a trajectory routed to several servers is
  tracked independently on each.
- Only the standard `LookupModule` notifies observers; CacheBlend lookups are
  not tracked.
- Engines forward the hint only if their integration forwards `lmcache.*`
  request configs on lookup (the vLLM MP connector does).
- Reusing one id for unrelated or concurrent trajectories makes them look like
  rewrites of each other.

## Tests

- `tests/v1/multiprocess/test_sequence_tracker.py`: divergence cases, budget,
  concurrent observers, hint parsing.
- `tests/v1/multiprocess/test_sequence_history_module.py`: module build and
  observer wiring, no-hint and switch-off paths, declined lookups, malformed
  hints, salt scoping.
- `tests/v1/mp_observability/subscribers/metrics/test_sequence.py`: metrics
  and label set.
