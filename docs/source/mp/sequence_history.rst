Sequence History Rewrites
=========================

Multi-turn agents do not always *append* to their history. Context folding
collapses a finished sub-task into a summary, summarizing agents replace old
turns, tool-response pruning drops old tool output, and some chat templates
strip earlier turns' ``<think>`` blocks. Each of these makes the next request
diverge from the previous one at some chunk ``d``.

When a request names the sequence it continues, the MP server compares its
chunk chain with that sequence's previous request and counts appends and
rewrites, including *where* a rewrite happened. As a separate, experimental
opt-in it can also demote the superseded chunks to the front of the L1
eviction order, so they are evicted before chunks that live trajectories
still use. An unexpected rewrite — a
template or tokenization change between turns, a common source of
train/inference mismatch in agentic RL — then becomes visible on a dashboard.

The feature is off by default and changes nothing for requests without the
hint.

.. contents::
   :local:
   :depth: 1

Enable It
---------

Give the server a tracking budget, the total number of chunk hashes kept
across all sequences (about 114 bytes per chunk plus 450 bytes per sequence;
the least recently seen sequences are forgotten first):

.. code-block:: bash

   lmcache server --sequence-tracker-max-chunks 262144

Then tag each request of a conversation or trajectory with the same
``lmcache.sequence_id`` in ``kv_transfer_params``. The vLLM
``LMCacheMPConnector`` forwards every ``lmcache.*`` entry to the server:

.. code-block:: bash

   curl http://localhost:8000/v1/chat/completions \
     -H 'Content-Type: application/json' \
     -d '{
       "model": "Qwen/Qwen3-8B",
       "messages": [...],
       "kv_transfer_params": {"lmcache.sequence_id": "rollout-42:traj-7"}
     }'

The id must be 1 to 128 characters from ``[A-Za-z0-9._:-]``. A malformed id
is ignored (the request is served normally) and logged as a rate-limited
warning that does not include the value.

Example: Agentic RL Rollouts
----------------------------

In a GRPO-style rollout every trajectory gets its own id; siblings that
share a prompt must **not** share an id, because the server compares each
request with the previous request *of the same id*.

.. code-block:: python

   import openai

   client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="-")

   def step(traj_id: str, messages: list[dict]) -> str:
       out = client.chat.completions.create(
           model="Qwen/Qwen3-8B",
           messages=messages,
           extra_body={"kv_transfer_params": {"lmcache.sequence_id": traj_id}},
       )
       return out.choices[0].message.content

   # Turn k: the full branch of a sub-task is in the context.
   # Turn k+1: the branch is folded into a summary. The server counts a
   # rewrite at the chunk where the branch started.

On each tagged lookup the server computes ``d``, the length of the common
chunk prefix with the sequence's previous request. ``d`` equal to the
previous length is an **append**; anything shorter is a **rewrite**, and the
previous chunks ``[d, previous_length)`` are *superseded*.

Superseded-Chunk Demotion (Experimental)
----------------------------------------

Enable it explicitly on top of tracking:

.. code-block:: bash

   lmcache server --sequence-tracker-max-chunks 262144 --sequence-demotion

Without ``--sequence-demotion`` the server only reports rewrites and the
cache behaves exactly as without tracking. With it, on a rewrite, superseded
chunks that no other tracked sequence's latest request contains are moved to
the front of the L1 eviction order. They stay
resident and readable: another request that uses them hits as usual and
restores their recency, and a read-locked chunk is never evicted. Nothing is
deleted from L1 or L2.

In a CPU simulation of a folding workload (8 GRPO groups of 8 trajectories,
a fold every 4 turns, 20% untagged shared-prefix traffic; see
``benchmarks/microbenchmark/sequence_demotion_benchmark.py``), demotion
recomputed 5–11% fewer chunks when L1 held 20–30% of the working set and
was neutral with much less or much more memory. Untagged traffic that
re-reads pre-fold histories lost up to 1.4 points of hit rate. Leave it off
if such traffic matters to you. These are
simulation numbers, not serving measurements.

Observability
-------------

With metrics enabled the server exports, labeled only by ``model_name``
(and ``kind``):

- ``lmcache_mp_sequence_transitions_requests_total{kind="append"|"rewrite"}``
- ``lmcache_mp_sequence_superseded_chunks_chunks_total``
- ``lmcache_mp_sequence_demotion_requested_chunks_chunks_total`` (chunks for
  which demotion was requested; 0 without ``--sequence-demotion``, and also
  counted when the policy ignores it)
- ``lmcache_mp_sequence_divergence_chunk_chunks`` (histogram of ``d`` for
  rewrites)
- ``lmcache_mp_sequence_tracker_chunks`` (gauge, bounded by the budget)

A rewrite rate you did not expect, concentrated in the lowest divergence
buckets, usually means the prompt prefix itself changed between turns.
At ``DEBUG`` level the server also logs a rate-limited line with the request
id and divergence chunk of a rewrite.

Limitations
-----------

- **Only prompt chunks are compared**, not decode tokens stored with
  ``lmcache.mp.save_decode_cache``. ``<think>`` stripping therefore appears
  as a rewrite only where earlier reasoning was part of a previous *prompt*,
  for example a multi-step tool loop that re-sends the intermediate
  assistant turns (with reasoning) until the next user message strips them.
  In a plain multi-turn chat the reasoning only ever appeared in the previous
  *response*, so the next prompt still extends the previous prompt and is
  counted as an append.
- **Demotion is a hint.** Chunk keys are content-addressed and may be shared
  with requests the tracker does not know about (untagged requests,
  forgotten sequences); those may pay a recomputation. Demotion takes effect
  with the ``LRU`` and ``IsolatedLRU`` L1 eviction policies only; ``ARC``
  ignores it and L2 eviction order is unchanged.
- **Bounded memory.** When the budget is exhausted the least recently seen
  sequences are forgotten, and a single prompt longer than the whole budget
  is not tracked. A forgotten sequence's next request starts a new chain.
- **One id per trajectory.** Reusing an id for unrelated or concurrent
  trajectories makes them look like rewrites of each other.
- **Engine support.** The vLLM ``LMCacheMPConnector`` forwards the hint.
  Other engine integrations forward request configs differently and may not
  pass it; CacheBlend lookups (``--engine-type blend``) do not track
  sequences.
