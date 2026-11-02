Secondary KV Storage
====================

LMCache multiprocess mode supports a two-tier storage architecture:

- **L1 (fast tier)** -- CPU memory by default, or an NVMe slab via GPUDirect
  Storage (cuFile) when ``--gds-l1-path`` is set, managed by the L1 Manager.
  All KV cache chunks live here during active use. (Byte-array L2 adapters are
  unsupported under the GDS L1 tier, which exposes no L1 memory buffer.)
- **L2 (persistent)** -- Durable storage backends (NIXL-based or plain
  file-system/raw-block).  The StoreController asynchronously pushes data from L1
  to L2, and the PrefetchController loads data from L2 back into L1 on
  cache misses.

.. contents::
   :local:
   :depth: 2

Data Flow
---------

**Write path (L1 -> L2):**

1. vLLM stores KV cache chunks into L1 via the ``STORE`` RPC.
2. The ``StoreController`` detects new objects (via eventfd) and
   asynchronously submits store tasks to each configured L2 adapter.
3. The L2 adapter writes the data to its backend (e.g., local SSD via GDS).

**Read path (L2 -> L1):**

1. A ``LOOKUP`` RPC checks L1 for prefix hits.
2. For keys not found in L1, the ``PrefetchController`` submits lookup
   requests to L2 adapters.
3. If found in L2, the data is loaded back into L1 and read-locked for the
   pending ``RETRIEVE`` RPC.

Adapter Types
-------------

LMCache ships several L2 storage backends, grouped by medium under
:doc:`Supported Backends <supported_storages>`. Select one or more with the
``--l2-adapter`` flag.

Every adapter accepts, alongside its type-specific keys, the common
``"shared": true`` option. Set it when the adapter mounts a storage
domain that several LMCache instances share (one S3 bucket, one NFS
export): with :doc:`coordinator event reporting <../coordinator>`
enabled, the coordinator's key directory then identifies the pool by
the adapter's type name — keep **one pool per adapter type** across the
fleet — deduplicates its placements across instances, and keeps them
across instance restarts (the bytes outlive any single reporter).
Storage private to the instance (the default, ``false``) needs nothing.

.. code-block:: bash

    --l2-adapter '{"type": "fs", "base_path": "/mnt/nfs/lmcache", "shared": true}'

.. toctree::
   :maxdepth: 2

   supported_storages

Multiple Adapters (Cascade)
---------------------------

You can configure multiple L2 adapters by repeating the ``--l2-adapter``
argument.  Adapters are used in the order they are specified.  The
``StoreController`` pushes data to all configured adapters, and the
``PrefetchController`` queries adapters in order during lookups.

.. code-block:: bash

    # SSD (fast, smaller) + NVMe GDS (larger capacity)
    --l2-adapter '{"type": "nixl_store", "backend": "POSIX", "backend_params": {"file_path": "/data/ssd/l2", "use_direct_io": "false"}, "pool_size": 64}' \
    --l2-adapter '{"type": "nixl_store", "backend": "GDS", "backend_params": {"file_path": "/data/nvme/l2", "use_direct_io": "true"}, "pool_size": 128}'

Store and Prefetch Policies
----------------------------

The **store policy** controls how keys flow from L1 to L2: which adapters
receive each key and whether keys are deleted from L1 after a successful
L2 store.  The **prefetch policy** controls how keys flow from L2 back to
L1: when multiple adapters have the same key, the policy decides which
adapter loads it.

Select policies via CLI:

.. code-block:: bash

    --l2-store-policy default \
    --l2-prefetch-policy default

**Built-in policies:**

.. list-table::
   :header-rows: 1
   :widths: 15 15 70

   * - Flag
     - Name
     - Behaviour
   * - ``--l2-store-policy``
     - ``default``
     - Store all keys to all adapters.  Never delete from L1.
   * - ``--l2-store-policy``
     - ``skip_l1``
     - Buffer-only mode.  Store all keys to all adapters, then
       **delete them from L1** immediately.  Pair with
       ``--eviction-policy noop`` to avoid useless LRU overhead.
   * - ``--l2-prefetch-policy``
     - ``default``
     - For each key, pick the first (lowest-indexed) adapter that has it.
       Prefetched keys are **temporary** (deleted after the reader finishes).
   * - ``--l2-prefetch-policy``
     - ``retain``
     - Same load plan as ``default``, but prefetched keys are **retained**
       permanently in L1.  Useful when prefetched data is likely reused
       by subsequent requests (e.g. shared system-prompt chunks).

Prefetch Concurrency
~~~~~~~~~~~~~~~~~~~~~

The number of concurrent L2-to-L1 prefetches is bounded by L1 memory, not
by a fixed count.  A queued prefetch request starts only when both hold:

* the L1 memory being written by in-flight prefetches and stores is below
  ``--l2-prefetch-inflight-watermark`` of the L1 capacity, and
* L1 has enough free space for the request's KV cache.

Many small prefetches therefore run at once, while a few large ones cannot
fill L1.  When no prefetch is in flight, the next request always starts, so
a request larger than the free space is served partially rather than
waiting.  A higher watermark increases L2-to-L1 throughput but leaves less
L1 room for stores and resident data.

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Flag
     - Default
     - Description
   * - ``--l2-prefetch-inflight-watermark``
     - ``0.5``
     - Fraction of L1 capacity, in (0, 1], that in-flight prefetch loads may
       stage before new prefetch requests wait.
   * - ``--l2-prefetch-max-in-flight``
     - (none)
     - Deprecated and ignored; logs a warning when set.

Buffer-Only Mode
~~~~~~~~~~~~~~~~~

When L1 is used purely as a write buffer (all data lives in L2), use
``--l2-store-policy skip_l1`` together with ``--eviction-policy noop``.
This combination deletes keys from L1 as soon as they are stored to L2
and disables the LRU eviction tracker entirely, reducing memory and CPU
overhead. A key that a request is still reading when its L2 store finishes
is deleted shortly after that read ends.

.. code-block:: bash

    --eviction-policy noop \
    --l2-store-policy skip_l1 \
    --l2-prefetch-policy default

Per-Request L2 Store Opt-Out
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A request that sets ``"lmcache.skip_l2_store": true`` in its
``kv_transfer_params`` opts out of L2 write-through: the chunks it newly
writes to L1 are not stored to any L2 adapter.  The request is served
normally.  The hint works with every ``--l2-store-policy``, and the policy
still decides L1 retention: with ``default`` the opted-out chunks stay in L1,
serve L1 hits, and leave through normal L1 eviction; with ``skip_l1`` they are
deleted from L1 as soon as their readers release them, so buffer-only mode
with ``--eviction-policy noop`` does not fill up.  Under ``skip_l1`` the hint
therefore means "do not cache": the chunks are lost to later requests in the
same step.  For rollouts that want in-step L1 hits (for example the other
samples of a GRPO group), use ``--l2-store-policy default`` with
``--eviction-policy LRU``.

.. note::

   Only the vLLM ``LMCacheMPConnector`` forwards client request configs to
   the MP server.  Requests from the SGLang and TensorRT-LLM MP connectors
   cannot carry this hint.

Use it when the client knows the KV will not be read after L1 drops it.  The
motivating case is RL rollouts (GRPO/PPO, agentic multi-turn RL), where KV is
only valid until the next policy-weight update and most write-through is never
read back from L2.  The rollout framework decides per request; set the hint
when no other request will read this request's KV (expected reuse 0) **and**
the trajectory does not send another turn that resends this context.

.. list-table::
   :header-rows: 1
   :widths: 40 20 40

   * - Request
     - ``lmcache.skip_l2_store``
     - Why
   * - GRPO sample, group size ``G > 1``
     - ``false`` (or omit)
     - The other ``G - 1`` samples read the group prompt.
   * - Single-shot sample (``G = 1``), no further turn
     - ``true``
     - Nothing reads the KV again.
   * - Multi-turn trajectory, a turn that will be followed by another
     - ``false`` (or omit)
     - The next turn resends this context.
   * - Multi-turn trajectory, last turn, no other reader
     - ``true``
     - Its new chunks are never read again.

For example, the last turn of a single-sample agentic trajectory:

.. code-block:: json

    {
        "model": "Qwen/Qwen3-8B",
        "prompt": "<conversation so far>",
        "max_tokens": 1024,
        "kv_transfer_params": {
            "lmcache.skip_l2_store": true
        }
    }

The hint is an opt-out by the writing request, with these semantics:

- Requests that share a prefix produce the same chunk keys.  The request whose
  write makes a chunk resident in L1 decides.  A shared prefix first written
  by an opting-out request is not stored to L2 until it leaves L1, even if later
  requests omit the hint; a later request writing a chunk that is already
  resident does not change the decision.  The hint is safe when the client
  owns all traffic under its ``cache_salt``.
- The hint covers every chunk the request newly writes.  With the vLLM
  connector's default ``lmcache.mp.save_decode_cache: false`` that is the
  prompt only.
- A value that is not a JSON boolean (for example ``"true"``) is ignored, so
  the chunks are stored, and the server logs a rate-limited warning.

Skipped chunks are counted by ``lmcache_mp.l2_store_skipped_objects``
(``reason="request_hint"``); stored chunks appear in the usual
``lmcache_mp.l2_store_submitted_objects``.

Policies are extensible -- new policies can be added by creating a file
in ``storage_controllers/`` and calling ``register_store_policy()`` or
``register_prefetch_policy()`` at import time.  See the design doc
``l2_adapters/design_docs/overall.md`` for details.

Serde (compression / quantization)
----------------------------------

Each adapter can optionally run a **serde** (serializer / deserializer)
that transforms data on the way in and out of L2 — e.g. fp8 quantization
for disk backends, or encryption for remote adapters. See
:doc:`KV Cache Compression </mp/serde>` for details and configuration.

.. toctree::
   :maxdepth: 1

   /mp/serde

Eviction
--------

LMCache supports eviction at both storage tiers so that each tier
can operate within a fixed capacity budget.

L1 Eviction
~~~~~~~~~~~

L1 eviction runs a single background thread that monitors overall L1
memory usage. When usage exceeds ``trigger_watermark``, the eviction
policy evicts a fraction of the least-recently-used keys.

**CLI flags:**

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Flag
     - Default
     - Description
   * - ``--eviction-policy``
     - *(required)*
     - Policy name: ``LRU``, ``ARC``, ``IsolatedLRU``, or ``noop``. ``ARC``
       balances recent and frequent keys using key-only ghost history.
       ``IsolatedLRU``
       maintains one LRU list per ``cache_salt`` and relies on per-salt
       quotas registered via the ``/quota`` HTTP endpoints; see
       :doc:`/mp/configuration` for the full description.
   * - ``--eviction-trigger-watermark``
     - ``0.8``
     - L1 usage fraction [0, 1] above which eviction is triggered.
   * - ``--eviction-ratio``
     - ``0.2``
     - Fraction of currently allocated L1 memory to evict per cycle.

**Example:**

.. code-block:: bash

    --eviction-policy LRU \
    --eviction-trigger-watermark 0.8 \
    --eviction-ratio 0.2

L2 Eviction
~~~~~~~~~~~

L2 eviction is **per-adapter** and **opt-in**. Each adapter can
independently declare an eviction policy by adding an ``"eviction"``
sub-object to its ``--l2-adapter`` JSON spec. Adapters without an
``"eviction"`` key have no eviction controller.

When L2 eviction is enabled for an adapter, a dedicated background
thread monitors that adapter's ``get_usage()`` value. Once usage
exceeds ``trigger_watermark``, the policy evicts keys until usage
drops by ``eviction_ratio``.

**``"eviction"`` sub-object fields:**

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Field
     - Default
     - Description
   * - ``eviction_policy``
     - *(required)*
     - Policy name: ``"LRU"``, ``"ARC"``, ``"IsolatedLRU"``, or ``"noop"``.
   * - ``trigger_watermark``
     - ``0.8``
     - Adapter usage fraction [0, 1] above which eviction is triggered.
   * - ``eviction_ratio``
     - ``0.2``
     - Fraction of used capacity to evict per cycle.

**Example — nixl_store with LRU eviction:**

.. code-block:: bash

    --l2-adapter '{
      "type": "nixl_store",
      "backend": "POSIX",
      "backend_params": {"file_path": "/data/lmcache/l2", "use_direct_io": "false"},
      "pool_size": 128,
      "eviction": {
        "eviction_policy": "LRU",
        "trigger_watermark": 0.8,
        "eviction_ratio": 0.2
      }
    }'

To use ARC for an adapter, set ``"eviction_policy": "ARC"`` in that
adapter's ``"eviction"`` object. ARC operates independently for each adapter
because every ``L2AdapterEvictionState`` owns its own policy instance.

**Adapter support:**

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Adapter
     - L2 Eviction Support
   * - ``nixl_store``
     - Full support. ``delete`` frees pool slots; pinned keys (in-flight
       loads) are skipped and retried on the next cycle.
   * - ``nixl_store_dynamic``
     - Full support. ``delete`` removes data files from disk; pinned
       keys are skipped. ``get_usage`` is byte-based
       (``_total_bytes / max_capacity_bytes``).
   * - ``mock``
     - Full support. Useful for testing eviction behaviour without
       real storage hardware.
   * - ``raw_block``
     - Full shared/global eviction support. ``delete`` recycles raw-block
       slots; locked entries are skipped and retried on the next cycle.
   * - ``s3``
     - ``delete`` removes objects from the bucket and frees aggregate
       byte accounting. ``get_usage`` reports ``usage_fraction == -1.0``
       when ``max_capacity_gb`` is ``0`` (disabled); set a non-zero
       ``max_capacity_gb`` to enable the watermark-triggered eviction
       controller.
   * - ``hfbucket``
     - ``delete`` removes objects from the bucket and frees aggregate
       byte accounting. ``get_usage`` reports ``usage_fraction == -1.0``
       when ``max_capacity_gb`` is ``0`` (disabled); set a non-zero
       ``max_capacity_gb`` to enable the watermark-triggered eviction
       controller. Locked keys (in-flight loads) are skipped.
   * - ``dax``
     - Full support. ``delete`` removes unlocked keys from the in-memory
       index immediately and recycles fixed slots once active read borrows
       drain. Usage is slot-based.
   * - ``mooncake_store``
     - No eviction support (native connector adapter).
   * - ``fs``
     - No eviction support (``delete`` and ``get_usage`` are no-ops).
   * - native connectors
     - No eviction support.

.. note::

   Each L2 adapter instance gets its own independent eviction
   controller and policy.  Two adapters of the same type can have
   different watermarks or policies.

Combined L1 + L2 Eviction Example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    --l1-size-gb 100 \
    --eviction-policy LRU \
    --eviction-trigger-watermark 0.8 \
    --eviction-ratio 0.2 \
    --l2-adapter '{
      "type": "nixl_store",
      "backend": "GDS",
      "backend_params": {"file_path": "/data/nvme/l2", "use_direct_io": "true"},
      "pool_size": 256,
      "eviction": {
        "eviction_policy": "LRU",
        "trigger_watermark": 0.9,
        "eviction_ratio": 0.1
      }
    }'

In this setup:

- L1 evicts from memory when it is 80 % full, reclaiming 20 % of
  allocated memory per cycle.
- L2 (NIXL/GDS) evicts from the storage pool when 90 % of pool slots
  are occupied, reclaiming 10 % per cycle.
- Both tiers use independent LRU policies, so each evicts its own
  least-recently-used keys.

Verifying L2 Storage
--------------------

Set ``LMCACHE_LOG_LEVEL=DEBUG`` to see L2 activity in the server logs:

.. code-block:: bash

    LMCACHE_LOG_LEVEL=DEBUG lmcache server \
        --l1-size-gb 100 --eviction-policy LRU \
        --l2-adapter '{"type": "nixl_store", "backend": "POSIX", "backend_params": {"file_path": "/data/lmcache/l2", "use_direct_io": "false"}, "pool_size": 64}'

Expected log messages when L2 is active:

.. code-block:: text

    LMCache DEBUG: Submitted store task ...
    LMCache DEBUG: L2 store task N completed ...
    LMCache DEBUG: Prefetch request submitted: X total keys, Y L1 prefix hits, Z remaining for L2
