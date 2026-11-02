# `kv_version.py`: Bounded-Staleness, Version-Qualified KV

The MP server can keep KV of several producer versions apart and serve a
request KV that is at most `max_kv_staleness` versions old, reporting which
version each served span came from. The driving workload is asynchronous RL
(PipelineRL, AReaL, prime-rl): the rollout fleet updates weights in flight,
several weight versions are live, and a request sampled under version `v` may
continue after the engine moved to `v+1`. Without this feature a deployment
has two choices: one key space for all versions (stale reuse, unbounded and
invisible), or a per-version `cache_salt` (hard isolation, no cross-version
reuse at all). The feature is opt-in per request; a request without the hint
resolves exactly today's keys.

Nothing in the server interprets "version" as weights: it is an opaque
integer the client attaches to the KV it produces (weights, adapter, or any
other producer revision), in the spirit of `lmcache.skip_load`.

## Request hints

| Key (`request_configs`) | Type | Meaning |
|---|---|---|
| `lmcache.kv_version` | int, `0 <= v < 2^63` | version of the KV this request computes |
| `lmcache.max_kv_staleness` | int, `0..16`, default 0 | accept cached KV with version in `[v - s, v]` |

Malformed hints never widen what the client asked for:

| Hint | Treatment |
|---|---|
| `kv_version` not an int (bool, str, float), negative, `>= 2^63` | versioning off for the request: today's unversioned keys |
| `max_kv_staleness` not an int or negative | window narrowed to the exact version |
| `max_kv_staleness > 16` | capped to 16 (bounds lookup rounds, below) |
| `max_kv_staleness` without `kv_version` | ignored |
| `max_kv_staleness > 0` on a layout with a sliding-window group | narrowed to 0 (see Non-scope) |

Each problem is logged by a rate-limited warner (once per 60 s per problem,
with an occurrence count) that never logs the client's raw value. The lookup
and the store-key resolution are the two places a request's hints first
reach the server, so only they warn.

## Where the version lives: key namespace, not object metadata

A versioned request addresses chunk `i` under

```
versioned_chunk_hash(h_i, version) =
    blake2b(h_i || version.to_bytes(8, "big"),
            person=b"lmcache.kv_ver", digest_size=len(h_i))
```

where `h_i` is the ordinary prefix-chained token hash. The version is part of
`ObjectKey.chunk_hash`, so it is part of the object's identity in L1, in every
L2 adapter's on-media key, in P2P transfers and in KV events, with no change
to `ObjectKey`, serde or any adapter. The token chain is not re-hashed per
version: `h_i` depends on tokens only, so chunks of different versions can be
mixed along one prefix.

The alternative, the version stored as object metadata with an unversioned
key, was rejected:

| | Key namespace (chosen) | Metadata on the object |
|---|---|---|
| L1 + every L2 adapter | works unchanged (opaque hash) | version must be plumbed through `MemoryObj` metadata, serde and each adapter's stored format |
| two versions of the same tokens | coexist as different objects | collide: L1 rejects the second write (`KEY_NOT_WRITABLE`), so the first version sticks until eviction and newer requests miss forever |
| lookup cost | up to `s + 1` prefetch rounds | 1 round, plus a post-lookup version check |
| capacity | one copy per live version | one copy |

Correctness for L1 *and* L2 with zero adapter changes decided it; the probe
cost is bounded (`s <= 16`, typically 1-4) and is paid only by requests that
ask for a window.

## Lookup: newest-first probe, one round per version

```
lookup(key, v, s), chunk hashes h_0..h_{n-1}, p = key.start (engine-covered prefix)
  round 0: prefetch [p, n) under version v       -> hits [p, p+r0)   run (p, p+r0, v)
  round 1: prefetch [p+r0, n) under v-1          -> hits r1          run (.., v-1)
  ...       until p reaches n or the version drops below max(0, v-s)
  hit = final p; session records the runs
```

Each round is an ordinary `PrefetchTaskSpec` (prefix policy, read locks) over
the remaining suffix, exactly like the engine-covered suffix lookup that
already exists. A finished round is consumed by `query_prefetch_status` /
`wait_prefetch_status`; if chunks and versions remain, the module submits the
next round and reports "still running" (`None`), so the connector's polling
contract is unchanged. `wait_prefetch_status`'s timeout bounds all rounds.

Properties:

- A chunk whose only cached versions lie outside the window is a miss and
  ends the prefix: no round can serve it.
- Every round lowers the version, so at most `s + 1` rounds run, and served
  versions never increase along the prompt. The probe prefers the newest
  version at the start of each run rather than per chunk; per-chunk newest
  would cost up to `(s + 1)` rounds per run.
- `s = 0` is one round: the same cost as an unversioned lookup.
- Locks are taken per round on the served keys only; their union is the
  served prefix, so `free_lookup_locks` releases exactly them.

## Chunk-to-version map (`KVVersionAssignment`)

Store, retrieve, lock release and `END_SESSION` touch resolve keys through one
map built from the session's served runs:

- a chunk inside a served run keeps that run's version (retrieve reads and
  releases exactly what the lookup locked; a re-store of a served chunk hits
  the existing object instead of re-stamping it);
- every other chunk is stamped `write_version = min(kv_version, oldest served
  version)`.

The `min` is the staleness guarantee. A chunk computed under `v` attends to
the reused version-`u` prefix, so its KV depends on `u`; stamping it `u`
makes "every served chunk was influenced only by versions `>= v - s`" hold
transitively. Without it, a multi-turn rollout that reuses the previous turn
each time would chain staleness without bound. The cost: KV computed on a
stale prefix is not reusable by a stricter request, which recomputes from
scratch and then stores fresh KV.

`MPCacheServerContext.resolve_obj_keys` applies the map for every
engine-driven and LMCache-driven STORE/RETRIEVE; `LookupModule` applies it
for lock release (`resolve_prefetched_obj_keys` takes it as a required
argument, so no caller can silently release unversioned keys) and for the
`END_SESSION` touch. A request's earlier, abandoned lookups (vLLM's APC-shrink
re-lookup) keep their runs on the session, so `extra_ranges` release under
the versions those lookups locked; a range no lookup served is left to the
read-lock TTL rather than guessed.

## Provenance: `QUERY_KV_VERSION_SPANS`

```
connector.request_finished(req)              server LookupModule
  if lmcache.kv_version in request configs:
    QUERY_KV_VERSION_SPANS(req_id) ───────►  session runs as
                                             [(start_tok, end_tok, version), ...]
                                             None if no session
    merge over servers, clip to [vLLM hit, LMCache hit)
    return_params["kv_version_spans"] = [[s, e, v], ...]
  END_SESSION(req_id)
```

A new RPC (ZMQ by name, gRPC `LookupService.QueryKvVersionSpans`) rather than
a wider `QUERY_PREFETCH_STATUS` reply, so the existing contract and every
engine connector stay untouched. With several servers (TP shards) each server
resolves versions independently; the adapter keeps tokens every server
served and reports the oldest version among them. The connector clips to the
tokens LMCache actually loaded: KV below the vLLM prefix-cache hit came from
the engine. `None` (server unhealthy, timeout, no session, e.g. reaped after
`--session-ttl-seconds`) omits the field instead of reporting an empty list.

The spans are diagnostics: a framework can log them per trajectory, mask
stale spans out of the loss, or decide to recompute. They are not needed for
importance-sampling correctness when the sampler records behavior logprobs:
decoupled PPO/IS already corrects for what was actually sampled, stale KV
included.

## Assumptions

- The engine's own prefix cache holds no KV older than the request's
  `kv_version` (reset the engine's prefix cache after loading new weights,
  or accept that engine-held KV is outside this accounting). LMCache stamps
  what the engine stores with the request's write version.
- The stamp records the *declared* version. A request that straddles an
  in-flight weight update keeps its declared version, which is the oldest
  weights that contributed: conservative, never fresher than reality.
- Chunk granularity: spans are chunk-aligned.

## Non-scope

- **Sliding-window / hybrid layouts**: a later round's window check would need
  the trailing chunks of an earlier round, which its rows do not contain, and
  the session lock model would no longer match. `max_kv_staleness` is
  narrowed to 0 (exact version, one round) with a warning.
- **CacheBlend** (`CB_UNIFIED_LOOKUP`, blend store): ignores the hints and
  uses unversioned keys. Do not combine `lmcache.kv_version` with blend.
- **Token-addressed cache control** (HTTP pin / prefetch / delete through
  `key_resolver`, coordinator directory queries) addresses unversioned keys
  only.
- **Connectors**: only the vLLM MP connector returns `kv_version_spans`. Other
  MP connectors get versioned keys if they forward the request's `lmcache.*`
  configs, but no spans.
- **Capacity**: each live version holds its own copy; eviction treats them as
  unrelated objects. Old versions age out under the normal eviction policy.
- **KV events / routers**: stored/removed events of versioned objects carry
  the version-qualified hash; a router that recomputes token-only hashes does
  not match them.
