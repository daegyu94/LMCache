# `kv_version.py`: Bounded-Staleness, Version-Qualified KV

The MP server can keep KV of several producer versions apart, serve a request
KV that is at most `max_kv_staleness` versions old, and report which version
each served span came from. The driving workload is asynchronous RL
(PipelineRL, AReaL, prime-rl): the rollout fleet updates weights in flight,
several weight versions are live, and a request sampled under version `v` may
continue after the engine moved to `v+1`. Without this feature a deployment
has two choices: one key space for all versions (stale reuse, unbounded and
invisible), or a per-version `cache_salt` (hard isolation, no cross-version
reuse). The feature is opt-in per request; a request without the hint
resolves exactly today's keys.

Nothing in the server interprets "version" as weights: it is an opaque
integer the client attaches to the KV it produces, in the spirit of
`lmcache.skip_load`. This is a proof of concept; where a choice was needed
the simplest safe semantics won.

## Request hints

| Key (`request_configs`) | Type | Meaning |
|---|---|---|
| `lmcache.kv_version` | int, `0 <= v < 2^63` | version of the KV this request computes |
| `lmcache.max_kv_staleness` | int `>= 0`, default 0, capped at 16 | accept cached KV with version in `[v - s, v]` |

Hints the server cannot honour never widen what the client asked for:

| Hint | Treatment |
|---|---|
| `kv_version` not an int (bool, str, float), negative, `>= 2^63` | fail closed: no load (lookup reports a miss without prefetching) and no store |
| `max_kv_staleness` not an int or negative | window narrowed to the exact version |
| `max_kv_staleness > 16` | capped to 16 (bounds the lookup rounds, below) |
| `max_kv_staleness` without `kv_version` | ignored |
| `max_kv_staleness > 0` on a layout with a sliding-window group | narrowed to 0 (see Non-scope) |
| any `kv_version` on a server not in LMCache-driven transfer mode | fail closed: no load (see Capability) |

Problems go to one process-wide warner that reports each problem at most once
per 60 s with an occurrence count, and never logs the client's raw value.

## Where the version lives: key namespace, not object metadata

A versioned request addresses chunk `i` under

```
versioned_chunk_hash(h_i, version) =
    blake2b(h_i || version.to_bytes(8, "big"),
            person=b"lmcache.kv_ver", digest_size=len(h_i))
```

where `h_i` is the ordinary prefix-chained token hash. The version is part of
`ObjectKey.chunk_hash`, so it is part of the object's identity in L1, in every
L2 adapter's on-media key and in P2P transfers, with no change to `ObjectKey`,
serde or any adapter. The token chain is not re-hashed per version: `h_i`
depends on tokens only, so chunks of different versions can be mixed along
one served prefix.

The alternative, the version stored as object metadata with an unversioned
key, was rejected:

| | Key namespace (chosen) | Metadata on the object |
|---|---|---|
| L1 + every L2 adapter | works unchanged (opaque hash) | version plumbed through `MemoryObj` metadata, serde and each adapter's stored format |
| two versions of the same tokens | coexist as different objects | collide: L1 rejects the second write (`KEY_NOT_WRITABLE`), the first version sticks until eviction and newer requests miss |
| lookup cost | up to `s + 1` prefetch rounds | 1 round plus a version check |
| capacity | one copy per live version | one copy |

## What is stored: only KV of exactly one version

KV is stored only under the request's own `kv_version`, and only if every run
the request's lookup served has that version
(`KVVersionAssignment.allows_store`). A request served older KV computes the
rest of its prompt with new weights attending to the old KV, so that KV
belongs to no single version:

- stored under the request's version, a later exact-version request would get
  KV derived from older weights as an exact hit;
- stored under the older version, a later exact older-version request would
  get KV computed with newer weights, and that older version's real KV could
  never be written (`KEY_NOT_WRITABLE`).

Either breaks the window, so it is not stored. The same holds for KV the
engine already holds in its own prefix cache, which LMCache cannot version.
Example: request A (`v10`, `s=2`) loads `v8` KV into vLLM blocks that vLLM
registers by token hash. Request B (`v10`, `s=0`) then gets a vLLM prefix-cache
hit on those blocks, so its lookup starts past them, finds nothing to serve,
and would store `v8`-derived KV as exact `v10`.

The gate, `MPCacheServerContext.kv_version_allows_store`, therefore allows a
versioned store only when the session records the request's current lookup,
and:

- that lookup finished and recorded the versions it served (no store without
  a lookup, while a re-lookup such as a preemption's is unconsumed, after a
  declined lookup, or on a session recreated after its TTL);
- it started at token 0 and the engine reported exactly 0 prefix-cache tokens
  (`lmcache.hint.engine_computed_tokens` present and 0; vLLM's eager lookups
  omit the hint and so fail closed);
- every served run has the request's version.

A malformed `kv_version` is never stored. The LMCache-driven STORE checks the
gate after key resolution: a refused store commits nothing and answers
success with an empty event handle (no device work). The engine-driven STORE
paths refuse every versioned store (see Capability).

The cost, stated plainly: a request that reused older KV, or vLLM-held KV,
adds nothing to LMCache. In a multi-turn rollout whose turns keep reusing an
in-window older prefix, each turn's new suffix stays uncached, so recompute
grows with the conversation (roughly quadratic in the number of turns) until
a turn misses the old prefix and stores fresh KV. A key that records both the
weights version and the oldest version it depends on (`(version, floor)`),
with lookups accepting `floor >= v - s`, would let such KV be stored and
reused within the window; that two-dimensional key is the natural follow-up.

## Lookup

```
lookup(key, v, s), chunk hashes h_0..h_{n-1}, p = key.start (engine-covered prefix)
  s == 0: one ordinary prefetch of [p, n) under version v      run (p, hit, v)
  s >  0: a probe worker, newest first, one round per version:
    round 0: prefetch [p, n) under v      -> hits [p, p+r0)   run (p, p+r0, v)
    round 1: prefetch [p+r0, n) under v-1 -> hits r1          run (.., v-1)
    ...      until p reaches n or the version drops below max(0, v-s)
  the session records the runs before the hit length
```

Each round is an ordinary `PrefetchTaskSpec` (prefix policy, read locks) over
the remaining suffix, exactly like the engine-covered suffix lookup that
already exists. Properties:

- A chunk whose cached versions all lie outside the window is a miss and ends
  the prefix.
- Every round lowers the version, so at most `s + 1` rounds run and served
  versions never increase along the prompt. The probe prefers the newest
  version at the start of each run rather than per chunk; per-chunk newest
  would cost up to `s + 1` rounds per run.
- `s = 0` is one prefetch: the same cost as an unversioned lookup.

### The probe dispatcher

Rounds of an `s > 0` lookup are driven on the server by one dispatcher thread
per lookup module, started on the first probe. It checks the in-flight round
of every live probe with the non-blocking `query_prefetch_status` and submits
the next round as soon as one finishes, so a round costs a prefetch (plus at
most a 2 ms poll), not a scheduler step plus an RPC, and a slow round never
delays another probe. With no live probe it sleeps on an event.
`QUERY_PREFETCH_STATUS` reports `None` until the dispatcher has recorded the
result; `WAIT_PREFETCH_STATUS` waits for it, its timeout bounding all rounds.

Ownership is explicit: only the dispatcher submits rounds. It records the
result on the session (runs, then hit length and L1 owners) under the probe
lock, and only if the probe was not cancelled. Cancellation happens when the
lookup is abandoned before its result is recorded:

| Event | Effect |
|---|---|
| `FREE_LOOKUP_LOCKS` mid-probe | the probe is cancelled and its job dropped; only `extra_ranges` are released by the call |
| a new `LOOKUP` of the same request | the old probe is cancelled before the new lookup touches the session |
| `END_SESSION` mid-probe | the probe is cancelled before the session is removed |
| an unexpected error while advancing | the probe is cancelled; the lookup reports a miss |
| the module closes | no further round is submitted; rounds in flight keep their locks until the read-lock TTL |

For a cancelled probe the dispatcher releases every read lock its rounds
took, including the round in flight once it ends. Submitting is the last
step of advancing a probe, so an error never leaves a submitted round
unaccounted. No other path releases them, so there is
neither a leak nor a double release, and a cancelled probe can never write
into a newer lookup's session or job (the worker never registers jobs, and a
consumed job is only removed if it is still the registered one). A probe whose
result is already recorded is a plain lookup result and is released by
`FREE_LOOKUP_LOCKS` through the session as before.

Locks are taken per round on the served keys only; their union is the served
prefix, so `free_lookup_locks` releases exactly them. A request's earlier,
abandoned lookups (vLLM's APC-shrink re-lookup) keep their runs on the
session, so `extra_ranges` release under the versions those lookups locked; a
range no lookup served is left to the read-lock TTL rather than guessed.

## Chunk-to-version map (`KVVersionAssignment`)

Retrieve, lock release and the `END_SESSION` touch resolve keys through one
map built from the session's served runs: a chunk inside a served run keeps
that run's version; every other chunk uses the request's `kv_version`.
`MPCacheServerContext.resolve_obj_keys` applies it for STORE and RETRIEVE;
`LookupModule` applies it for lock release (`resolve_prefetched_obj_keys`
takes it as a required argument, so no caller can silently release
unversioned keys). The touch of the engine-covered prefix at lookup is
skipped for versioned requests: LMCache did not serve those chunks, so which
version's copy to refresh is unknown.

## Capability

An older server ignores the hints and would serve and store the request in
the unversioned key space. A server therefore advertises
`KV_VERSION_CAPABILITY` (`"kv_version"`) through `GET_EXPERIMENTAL`, and only
in the LMCache-driven transfer mode, whose STORE implements the gate above.
The engine-driven (`PREPARE`/`COMMIT`) store paths do not; a server in
`engine_driven` or `auto` mode does not advertise the capability, its lookup
reports a miss for any versioned request, and its engine-driven STORE stores
nothing for a request carrying `lmcache.kv_version`, valid or malformed
(`PREPARE_STORE` reserves no chunk, `COMMIT_STORE` answers `False`). In
`auto` mode the LMCache-driven STORE refuses such requests too, since no
versioned lookup result is ever recorded.

The vLLM scheduler adapter checks the capability on every server at startup.
Without it, or with a malformed `kv_version`, the connector bypasses LMCache
for the request: no lookup, and `max_offload_tokens = 0` so nothing is
stored. One warning is logged per connector.

## Several servers

Each server holds a shard of the KV (a TP rank set) and would resolve
versions independently, so with a window two shards could serve one token
from different versions and store on top of them. The scheduler adapter
narrows `max_kv_staleness` to 0 when it has more than one server (one
warning): every shard then serves and stores only `kv_version`, so the shards
agree by construction. Bounded staleness with several servers would need one
version decision shared across servers; that is future work.

## Provenance: `QUERY_KV_VERSION_SPANS`

```
connector                               adapter                       server LookupModule
 get_num_new_matched_tokens  ── check_lookup_result resolves ──►
                                        QUERY_KV_VERSION_SPANS(req) ─► session runs as
                                        (futures kept, not awaited)    [(start, end, version)]
 request_finished ───────────── take_kv_version_spans (never waits)
   merge over servers (oldest version wins), clip to [vLLM hit, LMCache hit)
   return_params["kv_version_spans"] = [[0, vllm_hit, None], [s, e, v], ...]
 END_SESSION
```

The RPC is new (ZMQ by name, gRPC `LookupService.QueryKvVersionSpans`) rather
than a wider `QUERY_PREFETCH_STATUS` reply, so the existing contract stays
untouched. It answers `None` while the lookup's result is unconsumed and when
the server has no session, `[]` when the finished lookup served nothing
versioned. The adapter sends it when the lookup resolves (every server has
handed its result over by then) and `request_finished` only collects replies
that have arrived, so the scheduler thread never blocks; a missing, failed or
`None` reply omits `kv_version_spans`.

The connector reports the tokens LMCache loaded with their versions, led by
`[0, n, None]` for the `n` tokens vLLM's own prefix cache supplied: their
version is unknown to LMCache. The output contract of
`kv_transfer_params["kv_version_spans"]` is a JSON list of
`[start_token, end_token, version]` with `version` an int or `null`,
non-overlapping and in token order; the field is absent when unknown.

The spans are diagnostics: a framework can log them per trajectory, mask
stale spans out of the loss, or decide to recompute. They are not needed for
importance-sampling correctness when the sampler records behavior logprobs:
decoupled PPO/IS already corrects for what was actually sampled, stale KV
included.

## Assumptions

- **vLLM's prefix cache.** LMCache cannot version KV the engine holds itself
  and reports it as `[0, n, None]`. Two ways keep it out of the window
  accounting: reset vLLM's prefix cache when loading new weights (keeps
  cross-version reuse through LMCache), or give requests a per-version vLLM
  `cache_salt` (for example `f"v{weight_version}"`). The salt is part of the
  LMCache key as well, so it also isolates LMCache by version: the window then
  only ever matches the exact version, which is hard isolation again.
- **Declared version.** A request that straddles an in-flight weight update
  keeps its declared version: its stored KV is labelled `v` although part of
  it may have been computed with `v+1` weights. Only the lower bound is
  conservative (no KV is older than its label); the label is not exact.
- Spans are chunk-aligned.

## Non-scope

- **Sliding-window / hybrid layouts**: a later round's window check would need
  the trailing chunks of an earlier round, which its rows do not contain, and
  the session lock model would no longer match. `max_kv_staleness` is
  narrowed to 0 (exact version, one round).
- **Engine-driven transfer**: versioning is disabled and fails closed for
  both load and store.
- **CacheBlend** (`CB_UNIFIED_LOOKUP`, blend store) ignores the hints and uses
  unversioned keys. Do not combine `lmcache.kv_version` with blend.
- **Token-addressed cache control** (HTTP pin / prefetch / delete through
  `key_resolver`, coordinator directory queries) addresses unversioned keys
  only.
- **Connectors**: only the vLLM MP connector checks the capability, bypasses
  unservable requests and returns `kv_version_spans`.
- **vLLM prefix caching** limits what versioned requests store: any request
  with a vLLM prefix-cache hit stores nothing (fail closed, above). Tagging
  the vLLM-side block hashes with the version would lift this and is not
  done here.
- **Capacity**: each live version holds its own copy; eviction treats them as
  unrelated objects. Old versions age out under the normal eviction policy.
- **KV events**: server-side stored/removed events (`MP_TOKENS` bindings) of
  versioned objects carry the version-qualified hash. The vLLM worker
  adapter's fallback store events, which hash tokens locally, are not emitted
  for versioned stores, since a token-only hash would name an object that
  does not exist. A router that recomputes token-only hashes does not match
  versioned KV.
- **Capability changes** after the adapter started (a server replaced by an
  older one) are not detected.
