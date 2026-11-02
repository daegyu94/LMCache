# Generation-Scoped Cache Reset (Multiprocess Mode)

Covers `lmcache/v1/multiprocess/cache_generation.py` and the hash call sites in
the MP server that consume it. User-facing flag: `--clear-mode generation`
(`docs/source/mp/configuration.rst`, "Generation-scoped cache reset").

## 1. Problem

In RL post-training (GRPO, PPO) the rollout engine (vLLM with
`LMCacheMPConnector`) loads new policy weights every training step, then calls
`reset_prefix_cache(reset_connector=True)`, which sends a blocking `CLEAR` to
the MP server. Every KV object computed under the old weights is stale.

`ManagementModule.clear` -> `StorageManager.clear` frees only unlocked L1
objects:

- L2 is untouched, so the next lookup promotes old-weight KV back from L2
  (a stale hit — wrong results, not just wasted work).
- Locked L1 objects and in-flight stores survive and can be served or written
  to L2 after the clear.

Deleting every L2 key inside `CLEAR` closes the L2 hole but makes the clear
O(#L2 keys) storage deletes. The engine blocks on it (up to `mq_timeout`), and
that latency is GPU idle time at every RL step boundary.

## 2. Key Idea

The MP server computes every chunk hash itself (`TokenHasher`, rolling prefix
hashes seeded with `none_hash`). If the seed depends on a server-side
**cache generation**, bumping the generation changes the hash of every token
sequence, so every object stored before the bump — in L1, in L2, or still being
written — is unreachable in O(1). No per-key delete, no change to `ObjectKey`
or any L2 key codec. The unreachable L2 objects are reclaimed later, off the
critical path.

```
seed(g) = none_hash                                  if g == 0
        = H(none_hash, [g] + [0] * chunk_size)       otherwise
h_0     = H(seed(g), chunk_0)
h_i     = H(h_{i-1}, chunk_i)
```

- **Generation 0 is bit-identical** to the pre-feature hashes, so the default
  mode, existing caches, coordinators and tests are unaffected.
- **No aliasing.** The seed hashes a `chunk_size + 1`-token pseudo-chunk.
  Real chunks always have exactly `chunk_size` tokens, so a seed never equals
  the hash of a real prefix; otherwise generation-`g` hashes of `T` would equal
  generation-0 hashes of a longer sequence `P + T`.
- `TokenHasher.generation_prefix_hash(g)` computes the seed (memoizing the last
  non-zero generation); call sites pass it as `prefix_hash` to the existing
  `compute_*_chunk_hashes` methods.

## 3. Components

| Component | Where | Role |
|-----------|-------|------|
| `CacheGeneration` | `cache_generation.py`, owned by `MPCacheServerContext.cache_generation` | Locked monotonic counter; survives chunk-size rebinds (passed to every new `SessionManager`). Stays 0 unless the clear mode is `generation`. |
| `Session.cache_generation` | `session.py` | Generation the session's cached hashes belong to. Stamped at creation; re-stamped by `begin_lookup`. |
| `StaleL2KeyCollector` | `cache_generation.py`, owned by `ManagementModule` in generation mode only | `L2AdapterListener` that tracks stored L2 keys and deletes retired ones on a background `PeriodicThread`. |
| `MPServerConfig.clear_mode` | `config.py`, `--clear-mode {l1,generation}` | Opt-in; `l1` keeps today's behavior exactly. |

## 4. Clear Flow (generation mode)

```
CLEAR (ManagementModule.clear, under _clear_lock)
  1. collector.retire_tracked_keys()   # swap tracked set -> pending queue, wake GC
  2. ctx.cache_generation.bump()       # g -> g+1 (O(1))
  3. storage_manager.clear(force)      # existing L1 clear, frees memory
  return                                # never waits for L2

GC thread (every 1 s, or woken)
  pop <= 1024 keys -> adapter.delete(batch) for every L2 adapter
  <= 64 batches per cycle, re-wakes itself while a backlog remains
```

**Why retire before bumping.** Any key stored before the bump hashes under a
generation `<= g`, all of which become unreachable. A key hashed under `g+1`
cannot exist before step 2 returns. Retiring first therefore never schedules a
live key; an old-generation store that completes between steps 1 and 2 is
tracked in the new set and collected by the next clear.

**Collector contract.** The collector does not know generations: "every key
whose store completed before `retire_tracked_keys` returned is deleted." The
listener API has no adapter argument, so each batch is offered to every L2
adapter; adapters skip keys they do not hold. Keys deleted by L2 eviction
(`on_l2_keys_deleted`) are dropped from the tracked set.

**Memory bound.** At most `MAX_TRACKED_L2_KEYS` (1M) keys are held, tracked
plus pending. Overflow is counted as `untracked_keys` and left to L2 eviction.

## 5. Exact Semantics for Requests Spanning a Bump

The generation a request's keys are computed under is fixed per lookup:

| Path | Generation used |
|------|-----------------|
| `LookupModule.lookup` | `cache_generation.current`, read **once**; the same value seeds the lookup hashes and is recorded with `Session.begin_lookup`. |
| STORE / RETRIEVE / engine-driven PREPARE/COMMIT / qstore (`ctx.resolve_obj_keys`) | The session's `cache_generation`. |
| Lock release (`free_lookup_locks`, failed-RETRIEVE release via `resolve_prefetched_obj_keys`) | The session's `cache_generation` — the generation that took the locks, not the current one. |
| `end_session` touch, `MP_TOKENS` parent hash | The session's `cache_generation`. |
| Token-addressed prefetch (`PrefetchService`, `/cache/prefetch`) | `cache_generation.current`. |

Consequences:

1. **Lookup before the bump, store after it** (the request kept decoding
   across the weight update, or its store was already in flight): the store
   resolves through the session, so it lands under the *old* generation —
   unreachable garbage, collected at the next clear. Old-weight KV can never
   enter the new generation.
2. **Lookup after the bump** (e.g. the engine preempted the request on
   `reset_prefix_cache` and re-schedules it): `begin_lookup` re-stamps the
   session and drops its cached hashes. The lookup only hits new-generation
   objects; the engine recomputes the rest with the new weights, and the
   following stores land under the new generation. This assumes the engine's
   own prefix cache was reset together with the connector (true for
   `reset_prefix_cache(reset_connector=True)`).
3. **Lookup racing the bump**: a lookup that read `g` before the bump
   completes as if it ran before the clear (it may hit generation-`g` L1
   objects it locked). vLLM issues `reset_prefix_cache` between scheduler
   steps, so a single engine never overlaps the two.
4. **Sessions created after the bump** (`SessionManager.get_or_create`) start
   in the current generation.

## 6. Paths Not Made Generation-Aware

- **`--engine-type blend`** is rejected with `--clear-mode generation`
  (`MPServerConfig.__post_init__`). The CacheBlend fingerprint index maps
  content to stored chunk hashes; after a bump it would keep resolving
  old-generation objects. Making blend safe needs the index cleared or
  generation-tagged on bump.
- **MP coordinator** (`resolve_object_keys`: pin, eviction, directory APIs)
  runs in another process with its own `TokenHasher` and always resolves
  generation-0 keys; after the first bump they address nothing.
- **Worker-built KV store events** (`_kv_event_hasher` in the vLLM adapter,
  used when the server event channel is disabled) hash with generation 0;
  server-sourced events use the server's real keys and stay consistent.

## 7. Limitations

- The counter is per process and restarts at 0. A persistent L2 reused across
  server restarts can see a generation number again with different weights;
  use a per-run L2 location. Servers sharing one L2 share hits only while their
  generations agree.
- GC deletes race in-flight L2 loads of old-generation keys like L2 eviction
  does (the load becomes a miss); no new-generation reader can be affected.
- Old-generation L1 objects that were locked during the clear survive it and
  are reclaimed by L1 eviction.
- Adapters whose `delete` is the base-class no-op keep old objects until their
  own eviction removes them; the collector still reports them as deleted
  (offered to every adapter's `delete`).

## 8. Observability

- `GET /status` -> `cache_generation`: `generation` plus
  `stale_l2_gc` = `tracked_keys`, `pending_delete_keys`, `retired_keys`,
  `deleted_keys`, `untracked_keys`, `failed_delete_batches`. Absent in `l1`
  mode, so the default status is unchanged.
- `INFO` log per bump with the new generation and the number of retired keys;
  `DEBUG` log per GC cycle; `logger.exception` per failed delete batch.

## 9. Tests

- `tests/v1/multiprocess/test_cache_generation.py`: seed and hash properties
  (generation 0 unchanged, generations distinct and deterministic, no
  aliasing), counter concurrency, session stamping/re-stamping, collector
  non-blocking retirement, batching, re-tracking and memory bound.
- `tests/v1/multiprocess/test_generation_clear.py`: real context, mock L2
  adapter, `LookupModule` + `ManagementModule`: default mode still serves L2
  after clear; generation mode misses L1 and L2 and GCs old L2 keys; a pre-bump
  store committed after the bump is invisible and collected next clear;
  new-generation round trip; re-looked-up pre-bump session stores under the new
  generation; blend rejected.
