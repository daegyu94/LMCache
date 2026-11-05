# Prefill leases and sibling-load coalescing (GRPO group rollouts)

Code: `lmcache/v1/multiprocess/prefill_lease.py`, `modules/lookup.py`
(leases, server side) and `lmcache/integration/vllm/lmcache_mp_connector.py`
(`get_num_new_matched_tokens`, coalescing, engine side).

## Problem

GRPO (and RLOO, DAPO, best-of-N) samples N completions of every prompt. Async
RL frameworks submit the N samples together and their load balancer spreads
them over data-parallel engine replicas: verl's agent loop (least in-flight),
AReaL (round-robin), SkyRL (hash of a per-sample session id) and TRL's
`vllm serve --data-parallel-size` all do. Each engine's prefix cache shares
the prompt among its own siblings, but the engines miss in LMCache, because
no sibling has stored the prompt yet, so **every engine prefills it**.

Measured, one LMCache server, 2 vLLM engines, 16 prompts × 8 samples of
8–15k tokens: 2.00× the unique prompt tokens computed, 0 LMCache hits.

## Mechanism

```
engine A: a1 lookup ──lease chunks──> miss ─> prefill ─> store ─┐
engine B: b1 lookup ──leased by A──> deferred (None) ...........├─> resident
                                                                 v
          b1 re-polled ─> released ─> lookup hits ─> load ─> APC
          b2..b4 (same hit as b1, loading) ─> wait (None) ─> APC hit
```

1. **Lease** (server, `PrefillLeaseTable`). A `"prefix"` lookup leases the
   keys of its full-attention rows to its request. A lookup from another
   engine that needs a leased chunk is deferred: `QUERY_PREFETCH_STATUS`
   (and `WAIT_PREFETCH_STATUS`, without blocking) return `None`, so the
   engine keeps the request waiting and does no work for it. The engine is
   named by the `lmcache.hint.engine_id` the vLLM scheduler adapter adds to
   every key; requests of one engine never wait on each other (its prefix
   cache shares the prompt).
2. **Release.** A deferred lookup runs (and hits) once every awaited chunk is
   resident in L1 (`L1ManagerInterface.is_resident`, no side effects) or no
   longer leased. A lease is dropped when its owner's lookup hits the chunk
   (only misses stay leased), when the owner's session ends, or by a sweep
   once the session is reaped. A lease lives as long as its owner: there is
   no wall-clock lifetime to tune.
3. **Give up.** A waiter whose owner stored none of the awaited chunks for
   `--prefill-lease-timeout-seconds` (the owner is queued behind other work,
   or will not store) runs its lookup and computes as without leases. The
   timer restarts on every chunk that lands, so a long prefill that makes
   progress is waited for.
4. **Coalescing** (engine). Released siblings on one engine would each load
   the same KV into their own GPU blocks: vLLM does not share blocks between
   concurrent external loads. A request whose LMCache hit (cache salt +
   tokens of the hit) another request of the engine is loading gets
   `(None, True)` until that load finishes (`finished_recving`, or the
   request ends); its re-poll then sees the prefix-cache hit and loads
   nothing.

Leasing alone made rollouts slower (3B +11%, 8B +4%): the siblings' duplicate
GPU blocks cost more than the prefill saved. Coalescing makes it a win.

## Configuration

```bash
lmcache server ... --prefill-lease-timeout-seconds 10   # 0 (default) = off
```

Coalescing needs no flag. It also applies without leases to any burst of
same-prefix requests that hit LMCache on one engine.

## Results

One NVIDIA GB10 (both engines share its GPU), vLLM MP connector, L1 only,
GRPO 16 prompts × 8 samples, prompts 8–15k tokens, 256 output tokens,
32 in flight, samples round-robin over 2 engines, 2 interleaved runs each:

| Model | Metric | dev | leases + coalescing |
|---|---|---|---|
| Qwen2.5-3B | rollout wall time | 142.2 / 141.4 s | 132.6 / 132.8 s (−6.4%) |
| Qwen3-8B | rollout wall time | 423.6 / 424.5 s | 380.2 / 346.5 s (−14.4%) |
| both | prompt tokens computed | 2.00× unique | 1.01–1.09× unique |

Waits: 233 of 237 served, 4 gave up. Waiting siblings start later, but the
engines spend no time on duplicate prefills; mean TTFT was −21% (3B) and −5%
(8B).

## Limits

- Only engines that share one LMCache server (one node) coordinate; siblings
  on other nodes still prefill. Leases cover `"prefix"` lookups only.
- A waiter waits for the owner's whole prefill; with its own idle GPU it
  might have finished sooner, so TTFT of a waiter can rise even when the
  rollout finishes earlier.
- The residency probe checks L1; a chunk that lands only in L2 releases
  the waiter through the idle timeout.
- Only clients that name their engine (`lmcache.hint.engine_id`, sent by the
  vLLM MP connector) take part; others look up as without leases.
- An owner that does not store on prompt completion (a `kv_consumer` engine,
  or `lazy_offload`, which stores later) holds its waiters until the idle
  timeout. Enable leases where every engine stores its prefills.
- A waiter shares the owner's chunks as stored; a coalesced sibling shares
  the loader's GPU blocks. Both are the same KV, so outputs do not change.
- Measured on a shared single GPU; engines with their own GPUs (DP across
  devices) were not measured.

## Contract-anchoring tests

`tests/v1/multiprocess/test_prefill_lease.py` (waits, progress, give-up,
same-engine, owner release/end/death) and
`tests/v1/test_mp_connector_sibling_loads.py` (a sibling waits for the
loading request, then loads nothing).
