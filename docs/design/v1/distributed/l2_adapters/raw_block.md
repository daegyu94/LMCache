# Raw Block L2 Adapter Design

This document describes the built-in `raw_block` L2 adapter for LMCache MP
mode. It covers the adapter shape, the shared raw-block core, and the recovery
model.

## Overview

`raw_block` is a persistent MP L2 adapter backed by a raw block device or a
dedicated file. It is designed to keep the MP request flow unchanged while
reusing the existing raw-block on-device metadata format and the low-level Rust
raw-device I/O path.

```text
StoreController / PrefetchController
                |
                v
        RawBlockL2Adapter
                |
                v
          RawBlockCore
      (index, locks, allocation, checkpoints)
                |
                v
          RawBlockIO engine
         (posix / io_uring)
                |
                v
          lmcache_rust_raw_block_io
       (pwrite_from_buffer / pread_into)
                |
                v
         raw block device / file
```

## Goals

- Support LMCache MP mode using raw block storage as an L2 cache.
- Reuse the same durable metadata and checkpoint model as the existing
  non-MP raw-block backend.
- Reuse the existing Rust raw-device I/O layer.
- Preserve restart recovery semantics.
- Keep the MP controller flow unchanged: store, lookup-and-lock, load, unlock.

## FDP Placement Base

The MP `raw_block` adapter supports NVMe Flexible Data Placement (FDP) status
discovery when `io_engine="io_uring"` and `use_uring_cmd=true`. At startup, it
queries FDP reclaim unit handle status from the device and reports the
discovered mapping. Startup fails if the query fails or the device reports no
placement identifiers.

FDP plumbing is split by layer: `RawBlockL2Adapter` discovers and registers
non-zero placement identifiers, while `RawBlockCore` enforces that explicit
identifier 0 is never used. `fdp_placement_ids` is the KV data placement pool:
explicit data identifiers are rejected if they overlap with
`meta_checkpoint_placement_id` or are not reported by the device. If
`fdp_placement_ids` is omitted, the adapter registers all device-reported
non-zero identifiers except the metadata checkpoint identifier.

The adapter maps KV data writes onto FDP placement identifiers with a
cache-salt prefix policy. It derives a case-insensitive bucket from the part of
`ObjectKey.cache_salt` before `:` only when the separator is present, assigns
buckets to placement identifiers in first-seen order, and reuses the same
identifier for later writes in that bucket. Values without `:` or with an empty
prefix omit the directive; `rag:` is a valid opt-in to the `rag` bucket. The
mapping is availability-first: if the number of buckets exceeds the number of
registered data placement identifiers, extra buckets fall back to no directive
and the adapter emits one warning. Status reporting tracks a fallback count and
a bounded bucket sample rather than retaining all fallback bucket names. Empty
`cache_salt` values also omit the directive. The mapping is process-local and
may change after restart; read correctness is unaffected because `cache_salt` is
part of the object key rather than the read path's FDP directive. Metadata
checkpoint writes can use an explicit configured placement identifier while
defaulting to no directive when unset. User-facing FDP configuration rules live in
`docs/source/mp/l2_storage/raw_block.rst`; low-level NVMe command encoding
details live in `rust/raw_block/README.md`.

When `pid_affinity` reuse is enabled, freed space keeps the placement
identifier it was last written with, in memory. Among free extents of exactly
the requested length, the allocator prefers one with the requested identifier;
otherwise it takes the normal best fit. Coalescing keeps an identifier only when
both neighbours share it. Space above the highest offset ever allocated has
never been written, so taking it counts neither as an affinity hit nor as a
fallback.

Affinity is not checkpointed because FDP placement assignments are
process-local. After recovery, free space has no recorded affinity until it is
reused.

## Key Design Choice

The implementation is split into:

- `RawBlockCore` in `lmcache/v1/distributed/l2_adapters/raw_block/`
- `RawBlockL2Adapter` in `lmcache/v1/distributed/l2_adapters/`
- `RustRawBlockBackend` as the legacy non-MP wrapper

`RawBlockCore` owns the durable state and orchestrates I/O through a pluggable
`RawBlockIO` engine:

- raw device open/close
- in-memory key index
- free-space (extent) allocation
- lock refcounts used by MP lookup/load/unlock
- metadata checkpointing and recovery

The low-level "direct reads and writes through the Rust binding" responsibility
lives in the `RawBlockIO` engine so storage and transfer behavior stays behind a
single interface. `PosixBlockIO` implements synchronous `pread`/`pwrite` for
`io_engine="posix"`; `IoUringBlockIO` implements batched io_uring reads/writes for
`io_engine="io_uring"`, including optional NVMe raw-command (`use_uring_cmd`)
passthrough. `RawBlockCore` holds one `RawBlockIO` instance and leases buffers
through its `lease_buffer` / `lease_buffers` context managers, so callers never
touch the engine's buffer primitives directly. This keeps the durable-state logic
in one place while letting the I/O strategy be swapped without touching the core.

This avoids maintaining separate raw-block implementations for MP and non-MP
mode.

## Adapter Contract

`RawBlockL2Adapter` implements `L2AdapterInterface` directly. It exposes:

- three distinct eventfds: store, lookup, load
- non-blocking task submission APIs
- worker-thread execution for blocking raw-device operations
- result maps keyed by adapter-local task id
- listener notifications for stored, accessed, and deleted keys

The adapter uses caller-provided `MemoryObj` buffers for load operations. It
does not allocate destination buffers on the load path.

## Worker Failure and Task Completion

Recoverable io_uring submission errors (`EAGAIN`, `EINTR`, and `EBUSY`) retry
after completion progress or exponential backoff from 1 to 100 ms. Permanent
errors, or 30 seconds without submission or completion progress, stop new
native I/O admission. `RawBlockCore.report_status()` then reports
`is_healthy=False` and the terminal reason in `worker_error`. An ordinary
per-request I/O error does not mark the worker as failed. On a confirmed NVMe
unrecovered payload read, the core quarantines the object's space until
close, invalidates its owner and keeps successful sibling keys in the load bitmap.
The prefetch controller combines available tiers and applies the request’s
fetching policy. A full-attention prefix ends at the first missing chunk.

An open adapter reports a known worker failure through its task results, not
through a synchronous exception from `submit_*`. It allocates a task ID and
passes an already-failed future to the existing completion callback without
queuing work in the thread pool:

- lookup and load complete with all-zero bitmaps of the submitted key count;
- store completes with `L2StoreResult(False, 0)` without writing data;
- each completion decrements its in-flight counter, publishes the result, and
  signals the corresponding eventfd. Results are consumed once.

This lets the controllers release failed-load buffers and lookup/store locks
through their normal completion paths. Existing L1 and healthy-adapter hits
remain usable; missing data can be recomputed by the serving engine. Unlocks
remain available after worker failure. Invalid arguments and submissions after
adapter close still raise. The legacy non-MP backend skips new stores after a
known worker failure without retaining objects or raising to its caller.

I/O already accepted by the kernel keeps its buffers until terminal completion
or cancellation completion; a worker error alone must not complete those
tasks. Close and destructor cleanup release the GIL while draining and joining
the worker. Cleanup can still wait indefinitely for uncancellable I/O: the
submission retry budget is not a shutdown timeout.

## Locking Model

LMCache MP already uses L1 locks for CPU-memory object lifetime. `raw_block`
adds a separate L2-side lock refcount so a looked-up key cannot be deleted
between `lookup_and_lock` and `load`.

Rules:

- `exists_many(..., lock=True)` increments the refcount for hits
- `unlock_many(keys)` decrements and floors at zero
- `delete(keys)` skips locked entries

## Space Allocation

### Why not fixed slots

Fixed slots needed the object size at server start. In MP mode the byte size
of an object is only known after engines register, and one server holds many
sizes at once:

- per model: layers, KV heads, head size and dtype;
- per chunk size (MP multi-chunk-size, #6765 onward);
- per object group (`--separate-object-groups`, the default for hybrid models
  since #6041): full-attention, sliding-window, MLA and state groups differ;
- unfull chunks of sliding-window groups narrower than a chunk carry one
  extra lead block (`get_unfull_chunk_layout_desc`), so they are larger than
  full chunks;
- serde changes the length (AES-GCM adds a tag; quantizing codecs shrink it);
- TP changes the per-rank size.

A slot sized for the largest object wastes the rest of the slot on every
smaller one; a smaller slot rejects the larger objects outright.

### Structure

`ExtentAllocator` (`raw_block/extent_allocator.py`) keeps maximal free extents
indexed by start and by end (O(1) coalescing on free) and by length (a sorted
list of distinct lengths for best fit); Ceph BlueStore's AVL allocator keeps
the same two orders in trees. The unit is `block_align`. An object needs
`header_bytes + round_up(payload_len, block_align)` bytes, at one offset:

- Best fit: the head of the smallest free extent that fits; the remainder
  stays free. With same-size churn this is an exact fit, the same reuse
  pattern as slots. Among equal candidates the most recently freed is taken.
- A store fails when no single free extent fits, even if the free bytes add
  up; eviction then frees more (see Rejected alternatives for how often).
- Free: immediate coalescing with free neighbours. Freeing a range that starts
  or ends where free space does raises, which catches most double frees.
- Cost under `_lock`: dictionary operations plus, when a length appears or
  disappears, one insert into or delete from the sorted length list (memmove,
  O(k) for k distinct free lengths).

The object layout is the slot layout: header, then payload, as two write
requests. Zero-copy direct views, in-place tail zeroing, bounce buffers,
io_uring fixed buffers and io_uring_cmd splitting are unchanged.

The first `block_align` bytes of the data region hold a format marker (see
Upgrades and downgrades); objects are allocated after it.

### Checkpoint format

Format version 2 drops `slot_bytes` and `next_slot`; an entry keeps `offset`
and `size`, and its length follows from `size`. Recovery rebuilds free space
as everything not held by a valid entry; entries outside the data region,
off the block grid, or overlapping another entry are dropped (overlap means
corruption, so both go).

### Upgrades and downgrades

Version 1 (fixed-slot) checkpoints are imported: each object keeps only the
space it needs, and the slot tail is free. The entry in slot 0 is dropped,
since that block becomes the format marker. The checkpoint is then rewritten
as version 2.

A fixed-slot binary rejects version 2, starts empty and writes its first
object at slot 0, which overwrites the marker. A version 2 checkpoint found
without the marker therefore means such a run may have reused its space, and
it is not loaded. This covers a downgraded run that stopped before writing a
checkpoint of its own, unless it stopped after later slots reached the device
but before slot 0 did (one io_uring batch has no order, and posix runs several
store workers); the user docs ask to
clear the metadata region before a downgrade for that reason. An unreadable
marker next to a version 2 checkpoint stops startup rather than guess. The
marker is written before an imported checkpoint is rewritten, so a crash in
between imports it again.

### Rejected alternatives

A trace simulation (`alloc_sim.py`, kept outside the tree; 16 GiB, LRU
eviction controller, working set 3x capacity, 200k operations) compares the
allocators. Every policy treats the device as full after a failed store, as
`space_exhausted_count` does here. Hit rate / failed stores:

| Workload | Fixed slot | Exact-size slabs | 1.25x classes, segments returned | 1.25x classes, no return | This allocator |
|---|---|---|---|---|---|
| Watermark 0.8 (default), eviction ratio 0.2 | | | | | |
| One size | 0.374 / 0 % | 0.374 / 0 % | 0.374 / 0 % | 0.374 / 0 % | 0.374 / 0 % |
| Hybrid (gpt-oss-like groups) | 0.295 / 0 % | 0.362 / 0 % | 0.346 / 0 % | 0.346 / 0 % | 0.362 / 0 % |
| Several models and chunk sizes | 0.227 / 0 % | 0.374 / 2.2 % | 0.366 / 1.1 % | 0.365 / 0 % | 0.373 / 0 % |
| Model mix changes halfway | 0.158 / 0 % | 0.290 / 1.4 % | 0.280 / 1.2 % | 0.221 / 24 % | 0.289 / 0 % |
| 99 % small objects | 0.010 / 0 % | 0.346 / 0 % | 0.340 / 0 % | 0.340 / 0 % | 0.346 / 0 % |
| 1 byte and 64 MiB | 0.226 / 0 % | 0.384 / 8.6 % | 0.384 / 8.6 % | 0.390 / 0 % | 0.390 / 0 % |
| Log-uniform 4 KiB to 64 MiB | 0.069 / 0 % | 0.044 / 48 % | 0.259 / 0.6 % | 0.350 / 0 % | 0.379 / 0 % |
| Two sizes 12 % apart (serde) | 0.356 / 0 % | 0.372 / 0 % | 0.356 / 0 % | 0.356 / 0 % | 0.372 / 0 % |
| Watermark 0.95, eviction ratio 0.05 | | | | | |
| One size | 0.455 / 8.1 % | 0.419 / 27 % | 0.419 / 27 % | 0.455 / 8.1 % | 0.455 / 8.1 % |
| Several models and chunk sizes | 0.288 / 61 % | 0.425 / 19 % | 0.428 / 19 % | 0.437 / 4.9 % | 0.461 / 11 % |
| Model mix changes halfway | 0.206 / 65 % | 0.355 / 30 % | 0.350 / 31 % | 0.284 / 37 % | 0.362 / 18 % |
| Log-uniform 4 KiB to 64 MiB | 0.090 / 74 % | 0.097 / 64 % | 0.311 / 2.1 % | 0.413 / 0.8 % | 0.449 / 0.9 % |
| Two sizes 12 % apart (serde) | 0.435 / 11 % | 0.432 / 23 % | 0.400 / 28 % | 0.435 / 11 % | 0.443 / 11 % |

Slabs carve 256 MiB segments per exact size and return a segment once it is
empty. The 1.25x classes start at 64 KiB and top out at the largest object
(the full-slot class); "segments returned" carves them like the slabs,
"no return" from one shared region with fallback to a larger class.
Power-of-two classes do worse than 1.25x classes everywhere (several models
at the default watermark: 0.286). Bytes evicted
per store are within a few percent across policies, lowest for this
allocator. Near a full device, free space splits into ranges too short for
the next object; without `space_exhausted_count`, usage stays below the watermark
and such stores keep failing (two sizes 12 % apart: 92 % instead of 11 %).
The larger objects, for example one model's chunks, are then almost never
cached. Hit rate counted per lookup can even rise (0.457 instead of 0.443),
since smaller objects fill the device; the rule trades that for not
starving one object size. At the default watermark it changes nothing.

Random instead of LRU eviction order changes these by a few points.

- Size classes (the proposal in issue #3392): no splitting, so each object
  wastes up to the gap to its class; power-of-two classes waste about a
  quarter of KV sizes, which are not powers of two. Finer classes come close
  at a fixed mix, but space carved for a class only serves that class: without
  returning segments it calcifies when the mix changes (memcached's slab
  rebalancing problem), and with it, a segment comes back only once every
  object in it is evicted, so spread sizes still lose (log-uniform 0.259 vs
  0.379).
- Exact-size slabs that return empty segments: an LRU rarely empties a whole
  segment, and with many distinct sizes each one holds segments of its own.
- Buddy allocation: the same power-of-two rounding.
- Multi-extent objects (SPDK blobs, BlueStore `PExtentVector`): splitting an
  object across up to 8 free ranges when no single one fits cut the failed
  stores at watermark 0.95 (several models 11 % to 0.9 %) but moved hit rate
  by about one point either way (several models −1.2, two sizes +0.8), and
  split no object at the default watermark. It costs more requests per object
  and a header that must also pin the layout, so it is a separate follow-up.
  It does matter when space cannot be evicted (see Known limitations).
- Region or log-structured allocation (CacheLib BlockCache): reclaims whole
  regions, which conflicts with the per-key LRU eviction controller and needs
  reinsertion of live objects.

## Known limitations

- **Unevictable objects split free space for good.** Delete skips locked keys,
  and raw-block locks have no TTL, so leaked locks pin their objects until a
  restart. With fixed slots each one pins a slot; here, locked objects spread
  closer together than a large object leave no range long enough for it,
  whatever is evicted. `space_exhausted_count()` then makes the controller evict
  below the watermark on every such failed store, which shrinks the cache
  without the store ever fitting. In a simulation with the real allocator
  (1 GiB, 1 MiB and 16 MiB objects, one 1 MiB object locked every 8 MiB after a
  small-object warm-up, 10 % of the device), 33 % of 16 MiB stores failed and
  mean usage fell from 0.74 to 0.55; up to 128 locked objects at random places
  had no effect. Splitting an object across up to 8 free ranges removes the
  failures (0 %, usage 0.73); it is the planned follow-up.
  `alloc_failure_count` and `largest_free_extent_bytes` show the condition.

## Persistence and Recovery

`RawBlockCore` keeps the existing metadata checkpoint model:

- metadata region reserved on the same device
- periodic checkpointing
- optional checkpoint load on startup
- optional verification on load
- recovery by loading the latest durable checkpoint and rebuilding the in-memory
  index
- POSIX recovery validates per-object headers with an internal pool of 8 reader
  threads. Both regular `io_uring` and `io_uring_cmd` batch header reads up to
  `iouring_queue_depth`.

The data layout (object header, then payload) is unchanged; the checkpoint
format changed with extent allocation (see Space Allocation).

Recovered keys are exposed to the shared L2 eviction policy on adapter startup,
so reclaimed space comes from global L2 eviction or explicit `delete()` calls.

### Reuse after a delete

Recovery trusts the newest valid checkpoint and verification rereads only
each object's header. With variable sizes, space a deleted object held can be
reused by an object that starts inside it, so the old header can survive
while its payload is overwritten; recovery would then map the old key to new
bytes. The invariant: no byte of a deleted object that a checkpoint copy may
list is overwritten unless its first byte is overwritten no later.

A delete writes nothing. Each entry carries a bit per metadata container that
may list it (set when a snapshot for that container is taken; both bits for
entries loaded at startup). A deleted entry with any bit set becomes a
hazard: its range is free, and the hazard records its end and those bits.

```text
delete(K, listed) ──► range free + hazard(K.start, K.end, containers)
store Y at the head of a free range covering hazard H's start
   ├─ fence: H's free bytes past Y are taken until Y's write ends
   ├─ H.start in Y's header request ──► io_uring: header, wait, payload
   └─ Y written ──► H dropped; fence freed
checkpoint into container c (snapshot after H) ──► clear bit c; none left ──► H dropped
```

- Allocation always takes the head of a free extent. A hazard's bytes are
  freed together, so the extent holding them begins at or before its first
  byte; the only way an extent can begin inside it is a store covering that
  byte, and that store's fence holds the rest of it. So no store starts
  inside a hazard, and the store covering its first byte holds all its other
  bytes it does not overwrite itself.
- Within that store, the hazard's other bytes follow its first byte in the
  same request (a process crash leaves a prefix of a posix write or of one
  io_uring request: posix loops over short writes in order, and the plain
  io_uring path sends each buffer as one SQE and resubmits only the rest of
  a short write), unless the first byte is in the header request: then
  io_uring submits the payload after the header completed. posix writes in
  order anyway.
- With LRU eviction of same-size objects nearly every store reuses a listed
  object at its start, so on io_uring this is the common case, not the
  exception. On a ramdisk with O_DIRECT (batches of 8, every victim listed)
  a put took 3.1 µs more at 4 KiB than with fixed slots (+22 %), +17 % at
  64 KiB and +2.6 % at 1 MiB; posix +16 %, +12 % and +1.9 %. A reviewer
  measured more on an ext4 file on NVMe (+32 to 54 % at 64 KiB in batches of
  8). Linking the payload to the header (`IOSQE_IO_LINK`) in the Rust engine
  would remove the wait; it is not part of this change.
- A batch reserves all its stores first. A later store of the same batch
  that finds no other space takes the head of an earlier store's fence, and
  is submitted after every other write of the batch, so after the earlier
  store's request covering a hazard start, header or payload. A store of
  another thread that finds no space waits (up to 1 s) for a fence to be
  released before it fails.
- io_uring_cmd splits a request into NVMe commands that complete in any
  order, so with `use_uring_cmd` a delete zeroes the header (one
  `header_bytes` write with the object's placement id, batched, outside
  `_lock`) and frees the space after it returns; a range whose write fails
  again on its own is retired like quarantined space.
- A failed or cancelled store frees its range and fence and keeps the hazards;
  only a successful write drops them.
- Retiring free bytes that meet a hazard retires the whole hazard object, so
  no free extent starts inside it.
- Checkpoints are serialized; a snapshot marks every entry for its container
  before the write, so an entry deleted while that checkpoint is in flight
  stays a hazard for it.
- Without I/O the store bookkeeping is 4.1 instead of 3.1 µs per put, about
  0.6 µs of it in the allocator's dict updates; on a ramdisk with O_DIRECT a
  put took 1 to 3 µs more per object than with fixed slots; loads are
  unchanged.
- Each write puts a random 64-bit generation in the header (bytes 24-32) and
  the checkpoint entry (`gen`); recovery compares it with the key and size.
  A key re-stored where it was listed (best fit puts it back there) leaves a
  header of another generation, so a crash between its header and payload
  writes drops the entry instead of serving the old range's new bytes. For
  the same reason the space of a failed or cancelled store is free at once.
  An entry imported from fixed slots takes the generation bytes its header
  has, which earlier versions left unset.
- An entry dropped by header validation at startup is freed at once when its
  header did not match; when the header could not be read it is retired,
  since the header on the device may still be valid.
- Hazards are tracked with `meta_verify_on_load` off too: that run's recovery
  reads no headers, but a later run with verification may load the same
  checkpoint.
- Sequence numbers continue after the highest valid header on the device, so
  a checkpoint written later supersedes the older copies. A run that loads no
  checkpoint (skipped, rejected, or written with another `meta_magic` or
  `meta_version`, which it cannot see) zeroes the header block of every
  metadata container before any store, since its stores can reuse space an
  old checkpoint lists without overwriting those headers; with a rejected
  checkpoint it also zeroes the containers of that checkpoint's own
  `meta_total_bytes`. A metadata read that fails stops startup instead.
- With `meta_verify_on_load=false` there is no header check, so reused space
  can return wrong bytes after a crash, as with fixed slots.
- Nothing is flushed, so the ordering holds for process crashes, not for power
  loss on drives with volatile write caches (issue #6568).

## Configuration

The MP adapter is configured through `--l2-adapter` JSON:

```json
{
  "type": "raw_block",
  "device_path": "/dev/nvme0n1",
  "capacity_bytes": 0,
  "use_odirect": true,
  "block_align": 4096,
  "header_bytes": 4096,
  "meta_total_bytes": 268435456,
  "meta_magic": "LMCIDX01",
  "meta_version": 1,
  "meta_checkpoint_interval_sec": 60,
  "meta_enable_periodic": true,
  "load_checkpoint_on_init": true,
  "meta_verify_on_load": true,
  "num_store_workers": 2,
  "num_lookup_workers": 1,
  "num_load_workers": 4
}
```

For FDP configuration examples, see `docs/source/mp/l2_storage/raw_block.rst`.

Important validation rules:

- `block_align` must be a power of two
- `header_bytes` and `meta_total_bytes` must be aligned to `block_align`;
  `slot_bytes` is accepted for old configs but ignored
- with `use_uring_cmd=true`, `block_align` must be a multiple of the NVMe
  namespace LBA size
- `per_tp_device_paths` is rejected in MP mode
- `load_checkpoint_on_init=false` starts with an empty in-memory index instead
  of loading the latest on-device metadata checkpoint
- with `use_odirect=true` or `use_uring_cmd=true`, MP L1 alignment must satisfy
  `l1_align_bytes >= block_align`
- with `use_odirect=true`, raw-block I/O rejects offsets and total I/O lengths
  that are not aligned to `block_align`; misaligned write buffers use an
  aligned bounce buffer
- with `io_engine="io_uring"` and `enable_zero_copy=true`, a payload that is
  not a multiple of `block_align` (for example after AES-GCM serde) is
  transferred straight from its L1 allocation when that allocation covers the
  aligned length, which the L1 alignment rule above guarantees for L1 objects.
  Writes zero the padding after the payload in place; objects without such
  padding keep the bounce buffer

## Relationship to Non-MP Mode

The legacy `RustRawBlockBackend` now acts as a thin facade over `RawBlockCore`.
It preserves non-MP behavior such as prefix-oriented contains/get semantics,
while the MP adapter uses the core's full-bitmap lookup/load API.

## References

- Implementation: `lmcache/v1/distributed/l2_adapters/raw_block_l2_adapter.py`
- Shared core: `lmcache/v1/distributed/l2_adapters/raw_block/core.py`
- I/O engine: `lmcache/v1/distributed/l2_adapters/raw_block/io_engine.py`
- User docs: `docs/source/mp/l2_storage/raw_block.rst`
- Rust device layer: `rust/raw_block/README.md`
