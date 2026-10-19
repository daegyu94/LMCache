# Lookup hold for in-flight stores

Opt-in (`--lookup-pending-write-wait-ms`, default `0` = off, behavior
unchanged). Code: `lmcache/v1/multiprocess/modules/lookup.py`
(`LookupModule`), `L1Manager.probe_keys`, `StorageManager.has_pending_write`.

## Problem

A store RPC is acknowledged once its device work is enqueued; the L1 commit
(`finish_write`) runs later as a stream-ordered completion callback. Until
then the keys are *staged* (write-reserved) and `L1Manager.reserve_read`
reports them `KEY_NOT_EXIST`. In multi-turn agent rollouts the next turn's
lookup can land in that window and miss a prefix the server already holds.

## What counts as pending

"Staged" means *any* live L1 write reservation, not only a store: another
request's L2->L1 prefetch load (`prefetch_controller.py`) and serde temporary
buffers also reserve before they commit. A lookup whose first missing chunk is
being loaded from L2 by someone else is therefore held too (and then hits L1).

## Behavior

- `L1Manager.probe_keys(keys) -> list[L1KeyState]` (`RESIDENT`, `STAGED`,
  `ABSENT`) is read-only: no read or write lock, no recency touch, no events.
  A key whose write reservations all expired is `ABSENT`.
- `StorageManager.has_pending_write(chunk_keys)` merges all L1 managers and
  is true iff the first chunk not fully resident (across every object group
  and kv rank) has a staged key. A gap with no writer is not waited for,
  and a staged chunk beyond a gap is irrelevant (the prefix ends at the gap).
- `LookupModule.lookup`, just before `submit_prefetch_task`, records a
  deferred job (spec, job template, deadline) instead of submitting when the
  hold is enabled and `has_pending_write` is true.
- `query_prefetch_status` re-probes: while still staged and before the
  deadline it returns `None` (the vLLM adapter maps this to "not ready" and
  re-polls next scheduler step; no wire or connector change). Otherwise it
  submits the prefetch and continues the normal path. `wait_prefetch_status`
  polls the same check (2 ms) within its own timeout.
- A deadline expiry logs at DEBUG and looks up anyway (a miss if the store
  is still not committed).
- A deferred job holds no L1 locks. `end_session`, `free_lookup_locks` and a
  re-lookup of the same request id drop it. At most 256 jobs are held; beyond
  that a lookup is not held. Entries unpolled 30 s past their deadline are
  pruned when the cap is hit.
- Threading: handlers run on a thread pool; the deferred dict is guarded by
  `_deferred_lock`. `_resolve_deferred` holds it across pop + submit +
  register, so concurrent pollers submit exactly once (lock order:
  `_deferred_lock` then `_prefetch_job_lock`).
- With the hold off (`0`) no hook runs: lookup, status, free and end-session
  take the original code path and never touch the deferred state.

## Why waiting cannot deadlock

- Commits do not depend on lookup polling. The MP store paths commit from
  transfer completion: `lmcache_driven_transfer.py` submits
  `finish_write_by_owner` as a stream callback after the D2H copies;
  `engine_driven_transfer.py` / `server_transfer.py` call
  `finish_write(_by_owner)` after the copy. None waits on a lookup.
- A writer that dies or aborts leaves a reservation that expires after
  `write_ttl` (default 600 s); the hold is bounded by the (much shorter)
  deadline, so the worst case is one deadline of added latency.
- A held request is still waiting in the engine's scheduler and holds no KV
  blocks, so holding does not starve the store.

## Trade-offs

A held lookup delays only that request's scheduling by at most the
configured wait. Choose a value above the typical store-to-commit latency
(tens of ms) and well below the cost of recomputing the prefix.

## Scope limits

- CacheBlend lookups are not covered; that path has its own hold (see
  upstream PR #5775).
- There is no hold before the store reaches `reserve_write`: a lookup that
  arrives earlier still misses.
- Only the first non-resident chunk is examined. Sliding-window groups whose
  early chunks are absent make that chunk `ABSENT`, so no hold happens
  (a false negative, never an over-wait).
