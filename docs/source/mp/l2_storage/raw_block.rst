Raw Block (Rust)
================

A built-in L2 adapter that stores KV objects in fixed-size slots on a raw block
device or pre-sized file using the Rust raw-device I/O bindings. It reuses the
existing raw-block metadata checkpoint model and writes directly into the
caller-provided load buffers during prefetch.

**Required fields:**

- ``device_path``: Raw device path or pre-sized file path.
- ``slot_bytes``: Fixed slot size in bytes. Must be aligned to ``block_align``.

**Optional fields:**

- ``capacity_bytes``: Optional cap on the usable device bytes. Default ``0``
  means use the full device/file size.
- ``use_odirect``: ``true`` or ``false`` (default ``true``).
- ``block_align``: Device alignment in bytes (default ``4096``).
- ``header_bytes``: Per-slot header reservation (default ``4096``).
- ``meta_total_bytes``: Reserved metadata checkpoint region (default ``256MiB``).
- ``meta_magic`` / ``meta_version``: Metadata checkpoint identity/version knobs.
- ``meta_checkpoint_interval_sec`` / ``meta_idle_quiet_ms`` /
  ``meta_enable_periodic`` / ``meta_verify_on_load``: Checkpoint and recovery
  controls carried over from the legacy raw-block backend.
- ``load_checkpoint_on_init``: Load an existing on-device metadata checkpoint
  during startup (default ``true``). Set to ``false`` to start with an empty
  in-memory index instead.
- ``enable_zero_copy``: Try aligned direct-buffer I/O when possible.
- ``io_engine``: Rust raw-block I/O engine. Valid values are ``"posix"``
  (default synchronous ``pread``/``pwrite`` path), ``"io_uring"`` (direct Rust
  io_uring syscall path).
- ``use_uring_cmd``: Enable NVMe passthrough via io_uring command interface
  for direct device access. Requires ``io_engine="io_uring"`` and NVMe
  character device node (e.g., ``/dev/ng0n1``).
- ``iouring_queue_depth``: Queue depth for ``io_engine="io_uring"``.
- ``max_data_transfer_size``: Maximum data transfer size for
  ``use_uring_cmd=true``. Large transfers are split into smaller chunks
  that fit within device limits.
- ``fdp_enabled``: Enables NVMe Flexible Data Placement (FDP) discovery
  and non-zero handle registration. Requires ``io_engine="io_uring"`` and
  ``use_uring_cmd=true``.
- ``fdp_placement_handles``: Optional exact non-zero handle list.
- ``fdp_policy``: FDP placement policy. Use ``"none"`` (default) or
  ``"class_isolation"``.
- ``fdp_class_granularity``: Placement class granularity, ``"coarse"``
  (default) or ``"fine"``.
- ``fdp_class_placement_map``: Optional explicit mapping from class or hint name
  to FDP placement handle. The same handle may be assigned to multiple classes.
- ``num_store_workers`` / ``num_lookup_workers`` / ``num_load_workers``:
  Worker-thread counts for each operation type.

**Notes:**

- ``raw_block`` is a server-owned MP adapter. It does **not** support
  per-TP device-path mappings in MP mode.
- ``raw_block`` remains ``"type": "raw_block"`` for all supported engines.
- ``raw_block`` owns on-device slot allocation, checkpointing, and recovery
  through ``RawBlockCore``. Slot reclamation is driven by the shared/global
  L2 eviction controller or explicit ``delete()`` calls.
- If ``use_odirect`` is enabled, the server's ``--l1-align-bytes`` should be
  at least ``block_align``.
- ``persist_enabled`` must remain ``true`` for this adapter.
- For ``use_uring_cmd=true``, ``device_path`` must use the NVMe character
  device node (e.g., ``/dev/ng0n1``) instead of the block device node
  (``/dev/nvme0n1``). The character device provides direct NVMe
  command passthrough.
- ``use_uring_cmd`` requires ``io_engine="io_uring"`` to be set.
- When ``use_uring_cmd=true``, ``use_odirect`` is ignored for NVMe namespace
  character devices. FDP examples set ``use_odirect=false`` because
  ``io_uring_cmd`` uses NVMe passthrough rather than the POSIX write path.
- FDP registers only non-zero handles. If ``fdp_placement_handles`` is
  omitted, all discovered non-zero handles are used; if provided, the list must
  exactly match the device's non-zero handle set and must not contain 0.
  Checkpoint metadata writes use default NVMe placement with no directive.
- ``fdp_policy="class_isolation"`` binds each engine instance to a single
  FDP placement handle at REGISTER time. The engine's registration hint is
  normalized to a placement class (see the class-selection table below) and
  looked up in the class→handle map. Every store from that instance uses the
  bound handle. If REGISTER is issued without a placement hint under this
  policy, the adapter rejects the registration.
- ``fdp_policy`` is a per-adapter concern: adapters that use another policy
  (or a non-raw_block backend) ignore placement hints entirely. The
  storage manager rejects registering a second FDP-enabled raw_block
  adapter against the same device namespace.
- If ``fdp_class_placement_map`` is omitted, ``class_isolation`` requires enough
  non-zero FDP handles for every class in the selected granularity and assigns
  handles in deterministic class order. If the handles are insufficient, server
  startup fails.
- vLLM passes the per-instance placement hint through
  ``kv_connector_extra_config`` with key ``lmcache.fdp.placement_hint``.
  SGLang and TensorRT-LLM can read the same
  key from LMCache ``extra_config``; for example,
  ``LMCACHE_EXTRA_CONFIG='{"lmcache.fdp.placement_hint":"cold_reuse"}'``.
  TensorRT-LLM MP mode also accepts ``placement_hint``, ``fdp_placement_hint``,
  or ``lmcache_fdp_placement_hint`` on ``kv_connector_config``.

**FDP class selection:**

The serving engine provides one ``placement_hint`` when it registers a KV-cache
instance. ``raw_block`` normalizes the hint according to
``fdp_class_granularity`` and looks up the class in the class→handle map.

.. list-table::
   :header-rows: 1
   :widths: 32 24 24

   * - ``placement_hint`` value
     - Class/result with ``coarse``
     - Class/result with ``fine``
   * - ``hot_churn``
     - ``hot_churn``
     - ``hot_churn``
   * - ``hot_short``, ``hot_session``
     - ``hot_churn``
     - Same as the hint
   * - ``cold_reuse``
     - ``cold_reuse``
     - ``cold_reuse``
   * - ``cold_prefix``, ``cold_rag``, ``cold_checkpoint``
     - ``cold_reuse``
     - Same as the hint
   * - ``bulk_stream``
     - ``bulk_stream``
     - ``bulk_stream``
   * - ``bulk_prefill``, ``bulk_import``
     - ``bulk_stream``
     - Same as the hint
   * - ``ephemeral``
     - ``ephemeral``
     - ``ephemeral``
   * - ``temp_decode``, ``temp_speculative``
     - ``ephemeral``
     - Same as the hint
   * - Missing hint
     - REGISTER is rejected
     - REGISTER is rejected
   * - Unknown hint
     - REGISTER is rejected
     - REGISTER is rejected

**Configuration examples:**

.. code-block:: bash

    # Basic raw_block with posix I/O
    --l2-adapter '{"type": "raw_block", "device_path": "/dev/nvme0n1", "slot_bytes": 1048576, "block_align": 4096, "header_bytes": 4096, "meta_total_bytes": 268435456, "use_odirect": true, "num_store_workers": 2, "num_lookup_workers": 1, "num_load_workers": 4}'

    # With io_uring
    --l2-adapter '{"type": "raw_block", "device_path": "/dev/nvme0n1", "slot_bytes": 1048576, "io_engine": "io_uring", "iouring_queue_depth": 256, "use_odirect": true}'

    # With io_uring_cmd (NVMe passthrough)
    --l2-adapter '{"type": "raw_block", "device_path": "/dev/ng0n1", "slot_bytes": 1048576, "io_engine": "io_uring", "use_uring_cmd": true, "iouring_queue_depth": 256, "max_data_transfer_size": 131072, "use_odirect": false}'

    # With FDP discovery enabled, registering all non-zero device handles
    --l2-adapter '{"type": "raw_block", "device_path": "/dev/ng0n1", "slot_bytes": 1048576, "io_engine": "io_uring", "use_uring_cmd": true, "fdp_enabled": true, "use_odirect": false}'

    # With FDP class isolation and explicit class-to-handle aliases
    --l2-adapter '{"type": "raw_block", "device_path": "/dev/ng0n1", "slot_bytes": 1048576, "io_engine": "io_uring", "use_uring_cmd": true, "fdp_enabled": true, "fdp_policy": "class_isolation", "fdp_class_placement_map": {"hot_churn": 1, "cold_reuse": 1, "bulk_stream": 2, "ephemeral": 2}, "use_odirect": false}'

    # With eviction
    --l2-adapter '{"type": "raw_block", "device_path": "/dev/nvme0n1", "slot_bytes": 1048576, "load_checkpoint_on_init": false, "eviction": {"eviction_policy": "LRU", "trigger_watermark": 0.9, "eviction_ratio": 0.1}}'
