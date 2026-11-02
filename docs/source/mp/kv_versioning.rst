Bounded-Staleness KV Reuse
==========================

When the KV a request computes depends on a changing producer, such as the
policy weights of an asynchronous RL rollout fleet, a request can tag its KV
with a version and accept cached KV that is at most a few versions old.
LMCache keeps every version apart, serves only versions inside the request's
window, and reports which version each served token span came from.

This feature is opt-in per request. A request without the hints behaves
exactly as before. The LMCache server needs no flags, but it must run in the
default LMCache-driven transfer mode (``--supported-transfer-mode
lmcache_driven``); servers in another mode, and older servers, do not
advertise the feature and versioned requests then bypass LMCache.

.. contents::
   :local:
   :depth: 2

Why
---

In asynchronous RL (for example PipelineRL, AReaL or prime-rl) the trainer
pushes new weights to the rollout engines while they keep generating, so
several weight versions are live at once. Without versioning there are two
options, both all-or-nothing:

- **One shared key space**: KV computed under old weights is reused by new
  requests with no bound on how old it is, and nothing tells you it happened.
- **A per-version** ``cache_salt``: complete isolation, so KV is never reused
  across versions, not even one version back.

With ``lmcache.kv_version`` and ``lmcache.max_kv_staleness`` a request accepts
KV whose version lies in ``[kv_version - max_kv_staleness, kv_version]``.
Anything older (or newer) is a miss and is recomputed.

Request hints
-------------

Pass the hints in the request's ``kv_transfer_params``; the vLLM
``LMCacheMPConnector`` forwards every ``lmcache.*`` entry to the server.

.. list-table::
   :header-rows: 1
   :widths: 28 22 50

   * - Key
     - Value
     - Meaning
   * - ``lmcache.kv_version``
     - int, ``>= 0``
     - Version of the KV this request computes, e.g. the weight version the
       engine serves it with. KV the request stores is stored under it.
   * - ``lmcache.max_kv_staleness``
     - int, ``>= 0``, default ``0``
     - How many versions older than ``kv_version`` to accept. Each extra
       version can cost one more lookup round, so values above 16 are capped
       to 16.

Malformed values are logged (rate limited, without the value) and never widen
the window: a malformed ``kv_version`` makes the request bypass LMCache (no
load, no store), and a malformed ``max_kv_staleness`` means "exact version
only".

Example
-------

A rollout worker currently serving weight version 42 accepts KV up to two
versions old:

.. code-block:: bash

   curl http://localhost:8000/v1/completions \
     -H "Content-Type: application/json" \
     -d '{
       "model": "my-policy",
       "prompt": "...multi-turn trajectory so far...",
       "max_tokens": 256,
       "kv_transfer_params": {
         "lmcache.kv_version": 42,
         "lmcache.max_kv_staleness": 2
       }
     }'

The response's ``kv_transfer_params`` reports what LMCache served:

.. code-block:: json

   {
     "kv_version_spans": [[0, 256, null], [256, 1024, 42], [1024, 1536, 40]]
   }

Each entry is ``[start_token, end_token, kv_version]``: prompt tokens
``[256, 1024)`` were loaded from version-42 KV and ``[1024, 1536)`` from
version 40. ``[0, 256)`` came from vLLM's own prefix cache, whose version
LMCache cannot know (``null``). Tokens not listed were computed by the
engine. The field is omitted when the spans are not known (for example the
lookup never finished); it is never fetched with a blocking call.

How it behaves
--------------

- **Newest first.** The lookup tries ``kv_version`` first and continues with
  one version older wherever the previous version's KV ends, so a prompt can
  be served from several versions (a *mixed* prefix, reported as several
  spans). Versions never increase along the prompt.
- **Out of window is a miss.** The first chunk with no version inside the
  window ends the hit; the engine recomputes the rest.
- **Only single-version KV is stored.** A request stores its KV under
  ``kv_version``, and only if everything LMCache served it had that exact
  version. A request that reused older KV computes the rest of its prompt
  attending to that older KV, so its new KV belongs to no single version and
  is not stored. Staleness therefore cannot accumulate across turns that each
  reuse the previous turn; the next request that misses recomputes and stores
  fresh KV.
- **Cost.** ``max_kv_staleness: 0`` costs exactly one lookup, like an
  unversioned request. A window of ``s`` can cost up to ``s + 1`` prefetch
  rounds, which the server runs back to back. Every version keeps its own
  copy in L1 and L2.

Relation to other features
--------------------------

- **cache_salt.** A per-version salt is hard isolation. Versioning is reuse
  within a bounded window. A per-tenant salt combines with versioning: the
  salt separates tenants, versions are kept apart inside each tenant.
- **In-flight weight updates.** Set ``lmcache.kv_version`` to the weight
  version the request is *submitted* under. If the engine switches weights
  while the request is running, its KV keeps the submitted (older) version,
  which errs on the stale side.
- **vLLM's own prefix cache.** LMCache cannot version KV that vLLM keeps
  itself. Reset vLLM's prefix cache when loading new weights so it never
  serves older KV outside this accounting. A per-version vLLM ``cache_salt``
  would also enforce that, but the salt is part of the LMCache key too, so it
  reduces the window to the exact version.
- **Importance sampling.** If the sampler records behavior log-probabilities,
  decoupled PPO / importance sampling already corrects for what was actually
  sampled, stale KV included. ``kv_version_spans`` are for diagnostics,
  masking stale spans, or a recompute-if-too-stale policy, not a requirement
  for correctness.

Limitations
-----------

- ``max_kv_staleness`` needs every attention group to be full attention.
  Models with sliding-window or hybrid layers get exact-version matching
  (a warning is logged).
- With more than one LMCache server per engine, versioned requests get
  exact-version matching (a warning is logged): each server holds a shard
  of the KV and could otherwise pick different versions for one token.
- CacheBlend ignores the hints; do not combine them.
- Only the vLLM ``LMCacheMPConnector`` returns ``kv_version_spans``.
- Token-addressed cache operations (HTTP pin, prefetch, delete) address only
  unversioned KV, and KV-event consumers that recompute token-only hashes do
  not recognize versioned KV.

Design details: ``docs/design/v1/multiprocess/kv_version.md``.
