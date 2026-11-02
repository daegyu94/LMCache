# Per-request L2 store opt-out (`lmcache.skip_l2_store`)

Design of the `lmcache.skip_l2_store` request hint honored by
`lmcache/v1/distributed/storage_controllers/store_controller.py`
(`split_by_skip_l2_store_hint`, `StoreController._process_new_keys`).

## Motivation

Every key written to L1 is offered to the store policy, which (for every
built-in policy) writes it through to L2. Some clients know that a request's
KV will never be read again after L1 stops holding it, and for them that
write is pure cost: L2 bandwidth, SSD write amplification, and contention
with stores that are later read.

The motivating case is RL rollouts (GRPO/PPO, agentic multi-turn RL), where
KV is only valid for one policy-weight version:

- A GRPO group prompt is read by the other `G - 1` samples in the same step
  and by nothing after the weight update.
- The last turn of a trajectory is never resent.

The rollout framework, not LMCache, decides which requests fall into that
case and sets one generic hint on them. LMCache needs no RL vocabulary.

## Hint

| Request config | Type | Meaning |
|---|---|---|
| `lmcache.skip_l2_store` | `bool` | `true`: keys this request newly writes to L1 are not stored to L2. |

It follows the existing generic per-request bools (`lmcache.skip_load` in
`kv_load_policy.py`). The request is still served normally; only the
L1-to-L2 write-through of its keys is skipped. L1 retention of a skipped key
is still the store policy's decision, through a separate hook,
`StorePolicy.select_l1_deletions_for_skipped`. It is separate from
`select_l1_deletions` because that hook's contract is "keys already safe in
L2"; a policy that drops such keys from L1 must not silently drop the only
copy of a skipped key. The base class keeps all skipped keys. Under `default` they
stay resident, serve L1 hits, and leave through normal L1 eviction; under
`skip_l1` they are deleted right away (deferred while a reader holds them), so
buffer-only mode with `--eviction-policy noop` still drains L1.

## Data path

No new plumbing: the MP server already copies a request's `lmcache.*`
request configs into `ObjectKey.extra_attributes` of every key it builds for
the request, and the store controller receives those same key objects.

```
kv_transfer_params {"lmcache.skip_l2_store": true}
  -> LMCacheMPConnector: request_configs (lmcache.* only)
  -> IPCCacheServerKey.request_configs (STORE)
  -> EngineContext.resolve_obj_keys -> ObjectKey.extra_attributes
  -> StorageManager.reserve_write / finish_write (same key objects)
  -> L1Manager.finish_write -> StoreListener.on_l1_keys_write_finished
  -> StoreController._process_new_keys
       split_by_skip_l2_store_hint
         -> other keys:   StorePolicy.select_store_targets -> L2 store tasks
         -> skipped keys: StorePolicy.select_l1_deletions_for_skipped
                          -> L1 delete / defer
                          then one L2_STORE_SKIPPED event
```

The filter runs in the controller before policy routing, so it composes with
every `--l2-store-policy` (`default`, `skip_l1`, plugins) and policies stay
pure. The filter applies only while at least one non-draining adapter is
attached; without one nothing is stored anyway, so the batch goes to the
policy unchanged and no skip is reported. Only the vLLM `LMCacheMPConnector` forwards client request configs
today; the SGLang and TensorRT-LLM MP connectors only attach their own
engine hints, so their requests are never hinted.

## Combining writers

`ObjectKey` equality ignores `extra_attributes`, so requests that share a
prefix produce equal keys with possibly different hints.

- **Within one batch** (the keys of one `StoreListener.pop_pending_keys`
  call, i.e. one `_process_new_keys` call, before shape grouping): a key is
  skipped only if every occurrence carries `true`. Any occurrence without
  the hint, or with `false`, keeps the key.
- **Across batches:** the controller only sees writes that make a key
  resident. `L1Manager.reserve_write` rejects a key that is already
  resident and `finish_write` discards a staging copy whose key became
  resident first; neither notifies the store listener. So the request whose
  write makes a key resident decides for that residency. A later request that
  writes the same resident key cannot change the decision; after the key is
  evicted, the next write decides again.

Consequence: a shared prefix first written by an opting-out request stays
out of L2 until it leaves L1, even if other requests would have wanted it in L2.
The hint is safe when the client owns all traffic under the keys'
`cache_salt` (e.g. a trainer with its own salt), because then no other client
depends on that prefix being in L2.

## Malformed hints

A value that is not a JSON boolean (e.g. the string `"true"` or `1`) is
treated as absent, so the key is stored: wrongly storing costs one L2 write,
wrongly skipping costs a recompute after L1 eviction. The controller logs a
`WARNING` naming the value's type (not the client-controlled value) and a
sample key, at most once every
`INVALID_HINT_WARNING_INTERVAL_S` (10 s).

## Which chunks a hint covers

The hint is request-scoped: every chunk the request newly writes gets the
same decision. With the vLLM connector's default
(`lmcache.mp.save_decode_cache = false`) a request writes only complete
prompt chunks. A per-chunk prompt/generated signal would need new
engine-to-server plumbing and is not part of this design.

## Observability

Each batch that skips at least one key publishes one `L2_STORE_SKIPPED`
event after the batch's store submissions and L1 deletions (`reason = "request_hint"`,
`key_count` of distinct skipped keys, `key_count_per_salt`).
`L2MetricsSubscriber` turns it into `lmcache_mp.l2_store_skipped_objects`
with the `reason` attribute and the `cache_salt` attribute convention of the
other L2 object counters (no label for the empty salt). Stored keys are
counted by the existing `lmcache_mp.l2_store_submitted_objects`. Skipped
bytes are not reported; they equal skipped chunks times the chunk object
size. Keys a store policy leaves out of its plan are not reported.

## Compatibility

Without the hint the filter returns the batch unchanged and publishes
nothing; the policy receives the same list as before.
