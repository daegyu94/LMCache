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
      (index, locks, slots, checkpoints)
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

The MP `raw_block` adapter supports NVMe Flexible Data Placement (FDP)
discovery only when `io_engine="io_uring"` and `use_uring_cmd=true`. During
startup, the adapter queries the device for FDP reclaim unit handles and, when
`fdp_placement_handles` is configured, intersects the discovered handles with
that explicit list. Startup fails if the query fails, if the resulting handle
list is empty, or if any explicitly configured handle is not reported by the
device.

FDP directives are sent for raw-block data slot writes, including both the
per-slot header and payload. Metadata checkpoint reads and writes omit placement
handles, so they use the device's default placement. This keeps checkpoint
metadata lifetime separate from cached KV slot lifetime.

`None` and integer `0` have different meanings. `None` means no directive and
passes `dtype=0, dspec=0` to NVMe. Integer `0` is a valid explicit FDP placement
handle and sends the FDP directive with `dspec=0`.

`rank_isolation`, `domain_isolation`, and `model_isolation` are sibling FDP
placement policies. All three select placement handles for raw-block data slot
writes, but they use different isolation keys. The adapter supports one
`fdp_policy` value per configuration; composite policies such as rank plus model
isolation are not supported yet.

`rank_isolation` maps the local rank encoded in `ObjectKey.kv_rank` to FDP
placement handles. FDP SSDs are node-local NVMe devices shared by GPU workers
on the same LMCache server, so this policy isolates local writers. Local ranks
are registered lazily when store keys first arrive from the engine. Local rank N
maps to selected handle N, preserving explicit handle order. If a store sees a
local rank without a selected handle, that store fails instead of sharing another
rank's handle. The selected handles are also claimed exclusively per device
while the adapter is running, so another local adapter using the same device must
select a disjoint handle subset.

`domain_isolation` maps `cache_salt` values to FDP placement handles when they
are first observed in a store request. The first observed store for a new
`cache_salt` consumes the next selected handle; later stores for the same
`cache_salt` reuse that handle. If the selected handle pool is exhausted, new
`cache_salt` values are recorded with `None`, which means the write uses default
device placement. The exhaustion warning is emitted once so store workers do not
log on every I/O after the pool is exhausted.

`model_isolation` maps `ObjectKey.model_name` values to FDP placement handles
when they are first observed in a store request. It uses the same handle
assignment and exhaustion behavior as `domain_isolation`, but isolates by model
because different models can have different workload characteristics and cache
lifetimes.

## Key Design Choice

The implementation is split into:

- `RawBlockCore` in `lmcache/v1/storage_backend/raw_block/`
- `RawBlockL2Adapter` in `lmcache/v1/distributed/l2_adapters/`
- `RustRawBlockBackend` as the legacy non-MP wrapper

`RawBlockCore` owns the durable state and blocking I/O:

- raw device open/close
- in-memory key index
- free-slot tracking
- lock refcounts used by MP lookup/load/unlock
- metadata checkpointing and recovery
- direct reads and writes through the Rust binding

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

## Locking Model

LMCache MP already uses L1 locks for CPU-memory object lifetime. `raw_block`
adds a separate L2-side lock refcount so a looked-up key cannot be deleted
between `lookup_and_lock` and `load`.

Rules:

- `exists_many(..., lock=True)` increments the refcount for hits
- `unlock_many(keys)` decrements and floors at zero
- `delete(keys)` skips locked entries

## Persistence and Recovery

`RawBlockCore` keeps the existing metadata checkpoint model:

- metadata region reserved on the same device
- periodic checkpointing
- optional checkpoint load on startup
- optional verification on load
- recovery by loading the latest durable checkpoint and rebuilding the in-memory
  index

The on-device format is intentionally unchanged by the MP adapter work.

Recovered keys are exposed to the shared L2 eviction policy on adapter startup,
so reclaimed slots come from global L2 eviction or explicit `delete()` calls.

## Configuration

The MP adapter is configured through `--l2-adapter` JSON:

```json
{
  "type": "raw_block",
  "device_path": "/dev/nvme0n1",
  "slot_bytes": 1048576,
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

For FDP, start from the base `raw_block` configuration above, switch to the NVMe
character namespace device, and enable `io_uring_cmd`. Add or override these
fields to restrict the selected placement handles to an explicit subset reported
by the device:

```json
{
  "device_path": "/dev/ng0n1",
  "use_odirect": false,
  "io_engine": "io_uring",
  "use_uring_cmd": true,
  "fdp_enabled": true,
  "fdp_policy": "rank_isolation",
  "fdp_placement_handles": [0, 1]
}
```

For `domain_isolation`, use the same FDP base configuration and set the policy
to `domain_isolation`:

```json
{
  "type": "raw_block",
  "device_path": "/dev/ng0n1",
  "slot_bytes": 1048576,
  "io_engine": "io_uring",
  "use_uring_cmd": true,
  "fdp_enabled": true,
  "fdp_policy": "domain_isolation",
  "fdp_placement_handles": [0, 1, 2, 3]
}
```

For `model_isolation`, set `fdp_policy` to `model_isolation`. Each distinct
`ObjectKey.model_name` receives the next selected placement handle when first
observed in a store request:

```json
{
  "type": "raw_block",
  "device_path": "/dev/ng0n1",
  "slot_bytes": 1048576,
  "io_engine": "io_uring",
  "use_uring_cmd": true,
  "fdp_enabled": true,
  "fdp_policy": "model_isolation",
  "fdp_placement_handles": [0, 1, 2, 3]
}
```

Important validation rules:

- `slot_bytes`, `header_bytes`, and `meta_total_bytes` must be aligned to
  `block_align`
- `slot_bytes >= header_bytes + 1`
- `per_tp_device_paths` is rejected in MP mode
- `load_checkpoint_on_init=false` starts with an empty in-memory index instead
  of loading the latest on-device metadata checkpoint
- with `use_odirect=true`, MP L1 alignment must satisfy
  `l1_align_bytes >= block_align`

## Relationship to Non-MP Mode

The legacy `RustRawBlockBackend` now acts as a thin facade over `RawBlockCore`.
It preserves non-MP behavior such as prefix-oriented contains/get semantics,
while the MP adapter uses the core's full-bitmap lookup/load API.

## References

- Implementation: `lmcache/v1/distributed/l2_adapters/raw_block_l2_adapter.py`
- Shared core: `lmcache/v1/storage_backend/raw_block/core.py`
- User docs: `docs/source/mp/l2_storage/raw_block.rst`
- Rust device layer: `rust/raw_block/README.md`
