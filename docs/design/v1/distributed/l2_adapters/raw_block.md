# Raw Block L2 Adapter Design

This document describes the built-in `raw_block` L2 adapter for LMCache MP
mode. It covers the adapter shape, the shared raw-block core, and the recovery
model.

## Overview

`raw_block` is a persistent MP L2 adapter backed by a raw block device or a
dedicated file. It keeps the MP request flow unchanged while reusing the
low-level Rust raw-device I/O path. Allocation is selectable:

- `allocator="fixed"` preserves the original fixed-size slot layout and
  checkpoint payload. Each stored object consumes one whole `slot_bytes` slot.
- `allocator="buddy"` keeps the same top-level fixed slots, but treats each
  slot as a buddy arena. A stored object consumes the smallest power-of-two
  subslot that can hold the header and payload, and the allocator extends the
  existing checkpoint payload with buddy-specific fields.

```text
fixed allocator, slot_bytes = 1MiB

slot 0: [ object A uses the whole 1MiB slot ]
slot 1: [ object B uses the whole 1MiB slot ]

slot buddy allocator, slot_bytes = 1MiB, min_subslot_bytes = 8KiB

slot 0: [ A:8KiB ][ B:16KiB ][ free:8KiB ][ C:64KiB ][ ... free ... ]
slot 1: [ free 1MiB buddy root block ]
```

The slot-buddy mode therefore changes only the allocator inside each fixed
slot. It does not split one object across multiple slots, and it does not make
`slot_bytes` a separate allocation knob from the buddy root size.

```text
StoreController / PrefetchController
                |
                v
        RawBlockL2Adapter
                |
                v
           RawBlockCore
      (index, locks, allocator, checkpoints)
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

## TODO

- FDP / placement-hint support.
- A raw NVMe command path.

## Key Design Choice

The implementation is split into:

- `RawBlockCore` in `lmcache/v1/storage_backend/raw_block/`
- `RawBlockL2Adapter` in `lmcache/v1/distributed/l2_adapters/`
- `RustRawBlockBackend` as the legacy non-MP wrapper

`RawBlockCore` owns the durable state and blocking I/O:

- raw device open/close
- in-memory key index
- fixed-slot or slot-buddy allocation
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

`RawBlockCore` keeps the existing metadata checkpoint container model:

- metadata region reserved on the same device
- periodic checkpointing
- optional checkpoint load on startup
- optional verification on load
- recovery by loading the latest durable checkpoint and rebuilding the in-memory
  index

The fixed allocator keeps the original checkpoint payload shape. The buddy
allocator extends that payload with allocator-specific fields such as buddy free
lists and per-entry `allocated_bytes`. A buddy core does not auto-migrate
fixed-slot checkpoints; start it on an empty metadata region or set
`load_checkpoint_on_init=false` when switching allocator policies.

Recovered keys are exposed to the shared L2 eviction policy on adapter startup,
so reclaimed bytes come from global L2 eviction or explicit `delete()` calls.
For slot buddy allocations, store/delete/recovery accounting uses allocated
block bytes instead of logical payload bytes.

## Configuration

The MP adapter is configured through `--l2-adapter` JSON:

```json
{
  "type": "raw_block",
  "device_path": "/dev/nvme0n1",
  "slot_bytes": 1048576,
  "allocator": "fixed",
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

Slot buddy allocation example:

```json
{
  "type": "raw_block",
  "device_path": "/dev/nvme0n1",
  "slot_bytes": 1048576,
  "allocator": "buddy",
  "min_subslot_bytes": 8192,
  "block_align": 4096,
  "header_bytes": 4096,
  "meta_total_bytes": 268435456
}
```

Important validation rules:

- `slot_bytes`, `header_bytes`, and `meta_total_bytes` must be aligned to
  `block_align`
- `slot_bytes >= header_bytes + 1`
- `allocator` is either `fixed` or `buddy`; the default is `fixed`
- for buddy allocation, `slot_bytes` is the buddy root size and must be a
  power-of-two value
- buddy `min_subslot_bytes` must be a power-of-two value aligned to
  `block_align`, with `header_bytes + 1 <= min_subslot_bytes <= slot_bytes`
- `per_tp_device_paths` is rejected in MP mode
- `load_checkpoint_on_init=false` starts with an empty in-memory index instead
  of loading the latest on-device metadata checkpoint
- with `use_odirect=true`, MP L1 alignment must satisfy
  `l1_align_bytes >= block_align`

Slot buddy allocation does not compact or span one object across multiple
blocks. Watermark eviction reduces total L2 capacity pressure, but it does not
guarantee that a specific block size is available inside a buddy arena. If
fragmentation leaves no single free block large enough for a payload, the store
fails for that key even when total free capacity is below the eviction
watermark. A later retry can succeed after normal eviction frees a suitable
block.

## Relationship to Non-MP Mode

The legacy `RustRawBlockBackend` now acts as a thin facade over `RawBlockCore`.
It preserves non-MP behavior such as prefix-oriented contains/get semantics,
while the MP adapter uses the core's full-bitmap lookup/load API.

## References

- Implementation: `lmcache/v1/distributed/l2_adapters/raw_block_l2_adapter.py`
- Shared core: `lmcache/v1/storage_backend/raw_block/core.py`
- User docs: `docs/source/mp/l2_storage/raw_block.rst`
- Rust device layer: `rust/raw_block/README.md`
