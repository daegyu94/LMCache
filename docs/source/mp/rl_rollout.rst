Using MP Mode for RL Rollouts
=============================

RL post-training and agent-RL loops use vLLM as the rollout engine: the policy
weights change every training step, many samples share a prompt (for example
GRPO group sampling), and multi-turn agents resend a growing conversation.
This page describes how ``LMCacheMPConnector`` behaves in those loops and how
to avoid serving KV computed with stale weights.

.. contents::
   :local:
   :depth: 2

Cache Reset After a Weight Update
---------------------------------

KV cache entries are keyed by tokens, not by model weights. After the policy
weights are updated, any entry stored under the old weights is stale. Plain
``llm.reset_prefix_cache()`` does **not** clear LMCache: vLLM only resets the
connector cache when ``reset_connector=True``.

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - vLLM call
     - Clears LMCache (L1)?
   * - ``llm.reset_prefix_cache()``
     - No (``reset_connector`` defaults to ``False``).
   * - ``llm.reset_prefix_cache(reset_connector=True)``
     - Yes.
   * - ``POST /reset_prefix_cache`` (HTTP server)
     - No, unless ``?reset_external=true`` is passed.
   * - ``llm.sleep(level=1)`` or higher
     - Yes (pauses the scheduler and clears caches, including the connector).
       The default ``mode="abort"`` aborts in-flight requests.
   * - ``AsyncLLM.pause_generation(clear_cache=True)`` (the default)
     - Yes (same cascade). vLLM documents this call as the way to pause
       generation for model weight updates. The default ``mode="abort"``
       aborts in-flight requests; use ``mode="wait"`` to let them finish.

The HTTP routes ``/reset_prefix_cache``, ``/sleep`` and ``/pause`` exist only
when the vLLM server runs with ``VLLM_SERVER_DEV_MODE=1``. The HTTP ``/pause``
route's ``clear_cache`` parameter is deprecated and is ignored when
``mode="keep"``.

``reset_connector=True`` calls ``LMCacheMPConnector.reset_cache()``, which sends
a non-forced ``CLEAR`` to every LMCache server and waits for the reply, so it
is synchronous and can block the engine for up to ``lmcache.mp.mq_timeout``
(default 300 seconds). It returns ``False`` on timeout or RPC failure.
``reset_prefix_cache`` also returns ``False`` if the GPU prefix-cache reset
fails because requests are still running. Check the returned boolean and
retry or alert on ``False``.

What is and is not cleared
~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Cleared**: unlocked objects in L1 (CPU memory).
- **Kept**: objects held by in-flight reads or writes (a store that is still
  being written is not removed, and may commit after the reset).
- **Not cleared**: L2 storage. Entries already pushed to L2 survive the reset,
  and a later lookup can load them back into L1.

The same L1-only clear is available from the MP HTTP server as
``POST /cache/clear`` (see :doc:`http_api`); ``force=true`` also drops locked
objects but may corrupt in-flight store and prefetch operations.

Because the reset is best-effort and L2 is untouched, do not rely on it alone
for correctness when L2 is enabled or when requests may still be in flight.

Strict isolation with a per-version ``cache_salt``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``cache_salt`` is part of the cache key: the same tokens under different salts
map to different keys in L1 and L2. vLLM also mixes the salt into its own
prefix-cache block hashes. Using a new salt for each weight version means a
request is never served from LMCache with entries stored under another
version, regardless of what remains in L1 or L2.

Pass the salt per request. With the vLLM ``LLM`` API it is a field of the
prompt dict (the OpenAI-compatible completions and chat endpoints accept a
``cache_salt`` request field):

.. code-block:: python

   from vllm import LLM, SamplingParams
   from vllm.config import KVTransferConfig

   llm = LLM(
       model="Qwen/Qwen3-14B",
       kv_transfer_config=KVTransferConfig(
           kv_connector="LMCacheMPConnector",
           kv_role="kv_both",
           kv_connector_extra_config={
               "lmcache.mp.host": "localhost",
               "lmcache.mp.port": 5555,
           },
       ),
   )

   def rollout(prompts: list[str], run_id: str, version: int):
       salt = f"{run_id}-v{version}"
       return llm.generate(
           [{"prompt": p, "cache_salt": salt} for p in prompts],
           SamplingParams(max_tokens=512),
       )

   # Pseudo-code for the training loop; run_id is unique per training run,
   # num_steps, train_one_step, update_engine_weights and next_prompts are
   # placeholders.
   for version in range(num_steps):
       rollout(next_prompts(), run_id, version)
       train_one_step()
       update_engine_weights()  # drain or pause in-flight requests first
       if not llm.reset_prefix_cache(reset_connector=True):
           raise RuntimeError("cache reset failed")  # or log and retry

Notes:

- A salt must be at most 128 characters and must not contain ``@``, ``/``,
  ``\`` or NUL. A bad salt is rejected inside LMCache when the cache key is
  built, not by vLLM's request API.
- Salts must be unique per weight version for the lifetime of the L2 store.
  A bare ``policy-v{step}`` collides across training runs when L2 persists, so
  include a run identifier.
- Old-salt entries are not deleted by switching salts. L1 entries age out
  through the configured L1 eviction policy (see :doc:`configuration`). Do not
  set L1 ``--eviction-policy IsolatedLRU`` for this recipe: its eviction
  raises ``ValueError`` when no ``cache_salt`` is given.
- L2 entries are not removed by switching salts either. They are removed by
  LMCache's L2 eviction, by the explicit deletes below, or by the backend
  itself. L2 eviction is enabled for an adapter by an ``eviction`` block in its config. ``LRU`` and ``ARC``
  apply only when the adapter has ``max_capacity_bytes > 0``; ``IsolatedLRU``
  enforces per-salt quotas. If an adapter uses ``IsolatedLRU``, a salt with no
  registered quota has an effective limit of ``0`` bytes on that adapter, so
  register a quota for each active salt (see :ref:`mp-http-quota-api`) or use
  ``LRU`` / ``ARC`` there. ``DELETE /quota/<old-salt>`` drops a retired
  version's data on such an adapter. Individual keys can also be deleted with
  ``DELETE /cache/objects`` (see :doc:`http_api`).
- Keep the same salt for all requests of one weight version, including the
  turns of one multi-turn rollout; otherwise they cannot reuse each other's KV.
- Drain or pause in-flight requests before the weight update; a request
  that spans the update mixes KV from two weight versions.

Multi-Turn Agent Rollouts
-------------------------

By default MP mode stores only complete chunks of the **prompt**. The
response tokens of a turn are not stored, so when the next turn resends the
conversation (previous prompt, previous response, new observation), the
cached prefix ends at the end of the previous prompt's last complete chunk
and the previous response is not served from LMCache.

To also store chunks that contain generated tokens, set
``lmcache.mp.save_decode_cache`` in ``kv_connector_extra_config``:

.. code-block:: python

   kv_connector_extra_config={
       "lmcache.mp.host": "localhost",
       "lmcache.mp.port": 5555,
       "lmcache.mp.save_decode_cache": True,
   }

Caveats:

- The value must be a boolean; any other type raises ``ValueError`` at
  connector start-up.
- Only complete chunks are stored; a trailing partial chunk is not.
- Reuse requires the next turn's prompt token IDs to start with exactly the
  tokens that were generated. If your agent framework re-tokenizes the
  response text, the token IDs may differ and the generated chunks will not
  match.
- Sampled outputs become cache entries, which uses more cache capacity than
  prompt-only storage.
- This setting is separate from the in-process connector's
  ``save_decode_cache``. See :doc:`configuration`.

GRPO Group Sampling
-------------------

When several samples of one prompt (a GRPO group) are
submitted together, they are identical prompts, so their stores target the same
keys. LMCache keeps a single copy: a second write to a key that is already
stored, or being written, is rejected as not writable
(``KEY_NOT_WRITABLE``).

LMCache does **not** make a lookup wait for a store that is still in flight.
An object that is only being written is not readable, so a sample whose lookup
arrives before the first sample's store commits sees a miss and computes the
prefix itself. Samples that arrive after the commit hit L1. In a fully
concurrent group, expect little LMCache reuse inside the group itself; the
benefit comes from reuse across steps and across prompts that share a prefix
(for example a long system prompt or tool schema).

Related knobs, both off by default and independent of the above:

- ``lmcache.mp.eager_prefetch`` submits the lookup when a request enters
  vLLM's waiting queue, overlapping L2-to-L1 loading with queue wait (see
  :doc:`configuration`).
- ``lmcache.mp.lazy_offload`` defers GPU-to-LMCache stores (see
  :doc:`lazy_offload`). Its default policy is ``EVICTION_AWARE`` and it
  requires vLLM prefix caching (``enable_prefix_caching=True``). Deferring
  stores delays when entries become available to other requests.

Checklist
---------

- Pass a per-weight-version ``cache_salt`` on every rollout request.
- After each weight update, call ``reset_prefix_cache(reset_connector=True)``
  (or ``AsyncLLM.pause_generation`` / ``sleep`` with cache clearing); for the HTTP
  server use ``/reset_prefix_cache?reset_external=true``.
- Do not assume L2 was cleared; rely on the salt for correctness.
- The HTTP reset routes require ``VLLM_SERVER_DEV_MODE=1``.
- Drain in-flight requests before updating weights.
- Enable ``lmcache.mp.save_decode_cache`` for multi-turn rollouts that resend
  prior responses as token-identical prefixes.
- Use a unique salt per run and weight version, and check the boolean returned
  by ``reset_prefix_cache``.
- Plan capacity for old-salt entries (L1 eviction, L2 eviction config,
  per-salt quotas).
