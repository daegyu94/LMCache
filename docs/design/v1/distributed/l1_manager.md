# L1Manager: staging objects and write tags

Design of the write path in `lmcache/v1/distributed/l1_manager.py`.

## Problem

Before this design, `reserve_write` created the object directly in the
resident table, write-locked. Every other component saw it immediately:
`reserve_read` returned `KEY_NOT_READABLE`, and a second `reserve_write`
returned `KEY_NOT_WRITABLE`. Two prefetch requests loading the same L2 object
therefore collided: the second reservation failed, the prefetch controller
abandoned its whole L2 load (all-or-nothing) and reported a non-hit for a
chunk that was seconds away from being resident.

## Contract

Two kinds of objects live in the manager:

| | resident object | staging object |
|---|---|---|
| owner | the key | `(key, tag)` -- one per writer tag |
| created by | admission of a staging object | `reserve_write` of a key that is not resident |
| readers | `reserve_read` / `unsafe_read` / `finish_read` | invisible (`KEY_NOT_EXIST`) |
| writers | `reserve_write` refuses it (`KEY_NOT_WRITABLE`) | another tag may stage the same key |
| ends | `delete`, eviction, temporary object read out | admission, `finish_write_and_delete`, reclaim after write-lock expiry |

**Tag.** A plain `str` naming the writer (a transaction): the prefetch
controller uses `prefetch:<request_id>`, `StorageManager` uses
`storage_manager` for engine stores, the serde wrapper `serde_wrapper`. The
default is `""`. Writers sharing a tag exclude each other on a key
(`KEY_NOT_WRITABLE`), which keeps the de-duplication engine stores rely on;
writers with different tags never block each other.

**Admission.** `finish_write`, `finish_write_and_reserve_read` and
`finish_write_and_delete` take the tag used at reservation and act on the
staging object `(key, tag)`:

- key not resident: the staging object *becomes* the resident object (tag
  dropped). `finish_write_and_reserve_read` read-locks it in the same
  critical section.
- key resident (another writer admitted first): the staging object is
  discarded (memory freed, debug log, `SUCCESS` returned -- the data is in
  L1 either way). `finish_write_and_reserve_read` takes its read locks on the
  resident object and returns *that* `MemoryObj`, so the caller always ends
  up holding what readers see.
- `finish_write_and_delete` never admits; it frees the staging object.

**Expiry and eviction.** A staging object keeps its write TTL lock. Once the
lock expires the reservation is abandoned:

- `finish_write*` on it returns `KEY_IN_WRONG_STATE` and leaves it in place;
- `is_key_evictable(key)` is true and `delete([key])` reclaims every expired
  staging object of the key (all of them with `force=True`). `delete` then
  returns `SUCCESS` when nothing is left for the key, and `KEY_IS_LOCKED`
  while a live staging object remains -- a live reservation pins the key
  even if its resident object is unlocked. `clear()` reclaims expired
  staging objects too.

**Clear fence.** `clear()` bumps a monotonically increasing clear generation
kept by the manager (it owns `clear`, the staging table and admission, all
under one lock). Every reservation is stamped with the generation at
`reserve_write` time (this covers GPU stores, L2 prefetch landings and serde
temps alike). With `clear(force=False)`, which keeps live work:

- `finish_write` and `finish_write_and_reserve_read` on a reservation from an
  older generation discard it (memory freed, never admitted, no write-finished
  notification) and return `RESERVATION_CLEARED`; listeners get the
  "staging discarded, key now absent" treatment below. Callers treat it like
  any other failed write: the store is dropped. `finish_write_and_delete`
  is unaffected.
- A read-locked resident object is kept for its holders but marked stale:
  `reserve_read` reports it as `KEY_NOT_EXIST`, `unsafe_read` as
  `KEY_CLEARED` (distinct, because the caller still holds a read lock that it
  must release; `StorageManager.read_prefetched_results` and the SHM
  `prepare_retrieve` release exactly those keys), and `finish_read` frees it
  when the last read lock is released (the same path as temporary objects, so
  it emits `on_l1_keys_deleted_by_manager` and `L1_KEYS_EVICTED`). If the
  holders never release, the read TTL expires and the next `reserve_read`,
  `reserve_write` or `finish_read` on the key reclaims it. A write to a stale
  key is rejected with `KEY_NOT_WRITABLE` until then.
- Callers of `finish_write_and_reserve_read` must treat a non-`SUCCESS` result
  as "no read lock taken": the prefetch controller counts only admitted keys as
  hits and locked keys, and the serde wrapper drops the whole store.

**Fail-closed trade-off.** A request whose lookup preceded the clear and whose
retrieve follows it now gets a load failure instead of old-weight KV. With
vLLM's default `kv_load_failure_policy="fail"` such a request finishes with
`FINISHED_ERROR`; set `kv_load_failure_policy="recompute"` to recompute
transparently. Serving stale KV after a weight update was judged worse.

An in-flight L1->L2 store of a pre-clear object still completes in L2, and this
change does not clear L2; only L1 visibility is fenced.

The eviction policy learns about the key from `on_l1_keys_reserved_write`
(`L1EvictionPolicy` maps it to `on_keys_created`), so the regular eviction
loop -- `get_eviction_actions(key_eligible_filter=is_key_evictable)` followed
by `delete` -- covers abandoned reservations with no special casing. Policies
and controllers only ever see original keys; tags never leave the manager.

**Listener and event semantics.**

| step | listeners | event bus |
|---|---|---|
| `reserve_write` | `on_l1_keys_reserved_write(keys)` | `L1_WRITE_RESERVED` (`keys`, `tag`) |
| admission (non-temporary) | `on_l1_keys_write_finished` / `on_l1_keys_finish_write_and_reserve_read` | `L1_WRITE_FINISHED` / `L1_WRITE_FINISHED_AND_READ_RESERVED` |
| admission lost to a resident object | `on_l1_keys_reserved_read` (only for `_and_reserve_read`) | `L1_READ_RESERVED` |
| staging discarded / reclaimed, key still resident or still staged | none | none (debug log) |
| staging discarded / reclaimed, key now absent | `on_l1_keys_deleted_by_manager` | none |
| staging discarded by the clear fence | `on_l1_keys_deleted_by_manager` if the key is absent | none |
| resident object deleted | `on_l1_keys_deleted_by_manager` | `L1_KEYS_EVICTED` |

Discards are deliberately not published as `L1_KEYS_EVICTED`: the coordinator
cache-event reporter and the L1 byte metrics treat that event as the deletion
of an admitted object.

## Staging area accounting

`get_staging_memory_usage()` returns the bytes held by staging objects (a
subset of `get_memory_usage()`'s used bytes); `report_status()` adds
`staging_object_count` / `staging_bytes`,
and the gauge `lmcache_mp.l1_staging_bytes` is registered next to the L1
usage gauges. It is meant for observability and for future in-flight
prefetch planning (how much L1 is committed to loads that have not landed).

## Residual race

A prefetch request still aborts its L2 load when a key becomes resident
between its L1 lock pass and its `reserve_write`
(`KEY_NOT_WRITABLE`, reported as `L2_PREFETCH_FAILED{reason="l1_contended"}`).
That window is the L2 lookup latency, not the L2 load time as before; see
`storage_controllers/prefetch_controller.md`.

## Contract-anchoring tests

`tests/v1/distributed/test_l1_manager.py` (`TestStaging*` classes) and
`tests/v1/distributed/test_prefetch_controller.py::TestConcurrentPrefetchSameKeys`.
