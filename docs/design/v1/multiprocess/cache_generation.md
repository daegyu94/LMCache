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

## 2. Scope: synchronous RL

The design assumes **synchronous** rollouts: the trainer drains in-flight
requests, loads the new weights, then clears. Exactly one weight version is
live per server between two clears, so "every object stored before the clear
is stale" holds.

It is **not** for asynchronous / off-policy RL (in-flight weight updates,
several live policy versions). A global bump there throws away valid KV of
lagging versions. Scope those deployments by weight version instead — a
per-version `cache_salt` in `kv_transfer_params` — and keep `--clear-mode l1`;
do not combine the two.

## 3. Key Idea

The MP server computes every chunk hash itself (`TokenHasher`, rolling prefix
hashes seeded with `none_hash`). If the seed depends on a server-side **cache
generation**, bumping the generation changes the hash of every token sequence,
so every object stored before the bump — in L1, in L2, or still being written
— is unreachable in O(1). No per-key delete, no change to `ObjectKey` or any L2
key codec. The unreachable L2 objects are reclaimed later, off the critical
path.

```
l1 mode:          seed = none_hash                             (bit-identical)
generation mode:  seed(g)   = D([GEN, inc_hi, inc_lo, g_hi, g_lo])
                  untrusted = D([STRN, inc_hi, inc_lo])
D(words) = H(none_hash, words zero-padded to >= chunk_size + 1 tokens)
h_0 = H(seed, chunk_0);  h_i = H(h_{i-1}, chunk_i)
```

- **`l1` mode is bit-identical** to the pre-feature hashes; generation-mode
  generation 0 is not (bit-identity only matters for the default).
- **Incarnation.** `inc` is a random 64-bit value per server process. A
  restarted server never reproduces a previous process's keys, so a persistent
  or shared L2 cannot serve objects that a previous incarnation stored under
  the same generation number.
- **No aliasing.** `TokenHasher.domain_prefix_hash` hashes a pseudo-chunk of
  at least `chunk_size + 1` tokens. Real chunks always have exactly
  `chunk_size` tokens, so a seed never equals the hash of a real prefix;
  otherwise scoped hashes of `T` would equal unscoped hashes of `P + T`. The
  `GEN` / `STRN` tag words keep the two domains disjoint.

## 4. Components

| Component | Where | Role |
|-----------|-------|------|
| `CacheGeneration` | `cache_generation.py`, owned by `MPCacheServerContext.cache_generation` | Clear mode, incarnation and locked counter. `current_scope(hasher)` returns a `HashScope(generation, prefix_hash)`; `bump()` refuses in `l1` mode. Survives chunk-size rebinds (passed to every new `SessionManager`). |
| `Session.lookup_scope` / `get_hashes` | `session.py` | Scope of the latest lookup; RETRIEVE, lock release and touches resolve in it. |
| `Session.store_prefix_hash` / `get_store_hashes` | `session.py` | Store seed, fixed at session creation; every store resolves in it. |
| `MPCacheServerContext.resolve_store_obj_keys` | `engine_context.py` | Store-scope keys, used by LMCache-driven STORE, engine-driven PREPARE/COMMIT_STORE, qstore and blend store. `resolve_obj_keys` stays the lookup-scope resolver. |
| `StaleL2KeyCollector` | `cache_generation.py`, owned by `ManagementModule` in generation mode only | `L2AdapterListener` that tracks stored L2 keys and deletes retired ones on a background `PeriodicThread`. |
| `MPServerConfig.clear_mode`, `validate_clear_mode` | `config.py`, `--clear-mode {l1,generation}` | Opt-in; rejects blend, P2P and a coordinator (§7). |
| `LMCacheDrivenTransferModule.register_kv_cache` | `modules/lmcache_driven_transfer.py` | Rejects non-vLLM workers in generation mode (§7). |

## 5. Clear Flow (generation mode)

```
CLEAR (ManagementModule.clear, under _clear_lock)
  1. collector.retire_tracked_keys()   # swap tracked set -> pending queue, wake GC
  2. ctx.cache_generation.bump()       # g -> g+1 (O(1))
  3. storage_manager.clear(force)      # existing L1 clear, frees memory
  return                                # never waits for L2

GC thread (every 1 s, or woken)
  while pending and not stop_requested:
      pop <= 1024 keys -> adapter.delete(batch) for every L2 adapter
```

**Why retire before bumping.** Any key stored before the bump hashes under a
generation `<= g` (or the untrusted seed), all unreachable after it. A key
hashed under `g+1` cannot exist before step 2 returns. Retiring first therefore
never schedules a live key; an old-generation store completing between steps
1 and 2 is tracked in the new set and collected by the next clear.

**Collector contract.** The collector does not know generations: "every key
whose store completed before `retire_tracked_keys` returned is deleted, unless
a bound or a failure drops it to L2 eviction." The listener API has no adapter
argument, so each batch is offered to every L2 adapter; adapters skip keys they
do not hold. Keys deleted by L2 eviction (`on_l2_keys_deleted`) are dropped
from the tracked set. Adapters that announce resident keys on listener
registration (raw block) hand the collector their pre-existing keys too; they
belong to earlier incarnations and are collected at the first clear.

**Bounds.** Tracked keys (stored since the last clear) and pending keys
(retired, not yet deleted) have separate 1M-key bounds, so a slow adapter whose
pending queue is full never stops tracking new stores. Excess is counted as
`untracked_keys` / `dropped_keys` and left to L2 eviction.

**Shutdown.** `ManagementModule.close` (modules close before the context, so
before `StorageManager.close` closes the adapters) unregisters the listener
and stops the thread. The thread checks `stop_requested` (under the
collector lock) before each batch, and `close` then waits without a timeout
for a batch already inside an adapter `delete`, so no delete overlaps adapter
shutdown. That wait is bounded only by the adapters' own delete timeouts.

## 6. Exact Semantics for Requests Spanning a Bump

A request's keys are resolved in one of two scopes:

| Path | Scope |
|------|-------|
| `LookupModule.lookup` | `current_scope`, read **once**; it seeds the lookup hashes and is recorded with `Session.begin_lookup` (and is the store seed of a session the lookup creates). |
| RETRIEVE, engine-driven PREPARE/COMMIT_RETRIEVE, `end_session` touch | The session's `lookup_scope` (latest lookup). |
| `free_lookup_locks` current range, failed-RETRIEVE release | The scope recorded with that lookup's lock state (`lookup_scope`, or the snapshot from `prepare_failed_retrieve_release`). |
| `free_lookup_locks` `extra_ranges` (abandoned lookups) | `Session.abandoned_lookup_scope(hit)`: the scopes of recorded lookups with that hit boundary (bounded history of 16), plus the scopes of every entry that aged out of the history (any of them may have had that hit); if none is recorded, the session's only lookup scope. If the candidates span several generations the range is **not** released — L1 locks are anonymous refcounts and the wrong generation would decrement other readers' locks. Skipped ranges rely on the L1 read-lock TTL (`read_ttl_seconds`), which must not be disabled. |
| STORE (all transfer modes), qstore, `MP_TOKENS` parent hash | The session's `store_prefix_hash`, fixed at creation. |
| Token-addressed prefetch (`PrefetchService`) | `current_scope`. |

Store seed at session creation:

- Created by a **lookup** (`get_or_create_for_lookup`): that lookup's
  generation seed. This includes lookups that return early with no complete chunk
  (`empty_chunk_hashes`, e.g. a prompt shorter than one chunk whose decode
  later crosses a chunk boundary) or with no registered GPU context.
- Recreated after **TTL reaping** (by any path): in generation mode the
  reaped session's store seed is remembered (bounded, 64K requests, dropped by
  `END_SESSION`) and inherited, so a request outliving
  `--session-ttl-seconds` keeps storing in the scope its lookup proved — the
  old one if a bump happened meanwhile.
- Created by **any other path** (`get_or_create` with no live or remembered
  session): in generation mode the **untrusted** seed, which no lookup ever
  computes. Nothing proves which weights computed that KV, so it is stored as
  garbage (collected at the next clear). In `l1` mode it is `none_hash`, as
  before.

Consequences:

1. **A store issued before the bump** — resolved before it, or arriving after
   it (in-flight store, lazy-offload drain, delayed engine-driven COMMIT) —
   lands in the session's creation generation: unreachable after the bump.
2. **A session alive at the bump never stores into the new generation**, even
   after a re-lookup under the new weights. Its new-weight KV is wasted (a
   small hit cost, typically one request per preempted sequence) in exchange
   for never admitting old-weight KV. Because the store seed never changes,
   engine-driven PREPARE and COMMIT always resolve identical keys.
3. **Lookups after the bump** only hit new-generation objects (sessions
   created before the bump included), and their RETRIEVE and release resolve
   in that lookup's scope.
4. **A lookup racing the bump** that read `g` before it completes as if it ran
   before the clear: it may hit generation-`g` objects (e.g. L1 objects that
   were locked during the clear). Under the synchronous-RL assumption (§2)
   vLLM issues `reset_prefix_cache` between scheduler steps, so a single
   engine never overlaps the two. Its locks are released in `g`.
5. **Stores of sessions created after the bump by a lookup** land in the new
   generation and are served.

## 7. Paths Not Made Generation-Aware

- **`--engine-type blend`** is rejected (`MPServerConfig.__post_init__`). The
  CacheBlend fingerprint index maps content to stored chunk hashes; after a
  bump it would keep resolving old-generation objects. Making blend safe needs
  the index cleared or generation-tagged on bump.
- **P2P** is rejected (`MPServerConfig.__post_init__`): peers would address
  each other's objects with their own generations.
- **Non-vLLM workers** are rejected at `REGISTER_KV_CACHE`
  (`EngineType` other than `VLLM`). Generation mode trusts a request's stores
  only after its LOOKUP stamped the session, and only the vLLM connector looks
  up every request; the SGLang connector, for example, sends no LOOKUP for a
  sub-chunk prompt, so all its stores would be untrusted. Engine-driven
  (PREPARE/COMMIT) workers carry no engine type and are assumed to be vLLM.
- **A coordinator** is rejected at startup (`validate_clear_mode`, called by
  `run_cache_server`): its pin, eviction and directory APIs resolve token ids
  to generation-free keys in their own process and would silently address
  nothing after the first bump while reporting success.
- **KV events.** A bump emits no removal events for the dropped objects
  (they are removed only when GC or eviction deletes them), so a KV-event
  consumer such as a router may keep advertising old-generation prefixes until
  then. Worker-built store events (`_kv_event_hasher` in the vLLM adapter,
  used when the server event channel is disabled) hash without a generation
  and stop matching server keys after the first bump.

## 8. Limitations

- Servers sharing one L2 in generation mode never share hits (different
  incarnations), and keys stored by an earlier incarnation are not tracked by
  the new one's collector — they are reclaimed by L2 eviction (use a per-run L2
  location when the L2 has no eviction).
- GC deletes race in-flight L2 loads of old keys like L2 eviction does (the
  load becomes a miss); no new-generation reader can be affected. On a shared
  L2, the collector deletes only keys this process stored or was announced.
- Old-generation L1 objects that were locked during the clear survive it and
  are reclaimed by L1 eviction.
- Adapters whose `delete` is the base-class no-op keep old objects until their
  own eviction removes them; the collector counts their batches as deleted.
- A store is untrusted (never served) when its request has no live or
  remembered session: no LOOKUP ever reached this server for it, END_SESSION
  already ran, or the reaped-seed record (64K requests) evicted it.
- Session TTL (`--session-ttl-seconds`, measured from creation) still drops
  the lookup lock state, as in `l1` mode: a `free_lookup_locks` after reaping
  releases nothing and the locks wait for the read-lock TTL.
- Abandoned-lookup releases whose generation is ambiguous are left to the
  read-lock TTL (§6).

## 9. Observability

- `GET /status` -> `cache_generation`: `generation` plus `stale_l2_gc` =
  `tracked_keys`, `pending_delete_keys`, `retired_keys`, `deleted_keys` (in
  batches every adapter accepted), `failed_delete_calls` (adapter `delete`
  calls that raised; those batches are left to eviction), `untracked_keys`,
  `dropped_keys`. Absent in `l1` mode, so the default status is unchanged.
- `INFO` log per bump with the new generation and the number of retired keys;
  `DEBUG` log per GC cycle; `logger.exception` per failed delete call;
  `WARNING` per abandoned range left to the TTL.

## 10. Tests

- `tests/v1/multiprocess/test_cache_generation.py`: domain-seed properties
  (deterministic, disjoint, no aliasing, uint32 check); `l1` bit-identity and
  no bump; distinct scopes per generation and per incarnation; counter
  concurrency; store scope fixed across bump and re-lookup; untrusted stores;
  abandoned-lookup scope resolution and ambiguity; failed-RETRIEVE snapshot;
  collector non-blocking retirement, batching, re-tracking, eviction sync,
  failure path, independent bounds, close.
- `tests/v1/multiprocess/test_generation_clear.py`: real context, mock L2,
  `LookupModule` + `ManagementModule`: default mode serves L2 after clear;
  generation clear misses L1 and L2 and GCs old keys; new-generation round
  trip; pre-bump store committed after the bump; store arriving after a
  re-lookup; store without a lookup; sub-chunk lookup then decode store;
  store after TTL reaping; release after the bump; abandoned range across a
  bump (only its share released); lookup racing the clear (threads + events);
  close; non-vLLM registration; CLI parsing and rejected combinations.
- Unit tests additionally cover aged-out history ambiguity, reaped-session
  seed inheritance (with and without a bump, and forgetting on END_SESSION)
  and `close` waiting for an in-flight delete.
