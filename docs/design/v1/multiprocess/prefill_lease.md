# Prefill leases and sibling-load coalescing (GRPO group rollouts)

Code: `lmcache/v1/multiprocess/prefill_lease.py`, `modules/lookup.py`
(leases, server side) and `lmcache/integration/vllm/lmcache_mp_connector.py`
(`get_num_new_matched_tokens`, coalescing, engine side).

## Problem

Group-sampling methods such as GRPO (also RLOO, DAPO, best-of-N) sample N
completions of every prompt. Async
RL frameworks submit the N samples together and their load balancer spreads
them over data-parallel engine replicas. As of their GitHub `main` in November
2026: verl's agent loop gives each request a fresh id and picks the least-loaded
server; SkyRL routes on a per-sample session id; AReaL deals requests round-robin,
where leases apply only on vLLM, since only the vLLM connector names its engine;
TRL's async GRPO sends samples separately, and its `vllm_serve` documents keeping
data parallelism at 1 for dense models. Each engine's prefix cache shares
the prompt among its own siblings, but the engines miss in LMCache, because
no sibling has stored the prompt yet, so **every engine prefills it**.

Measured, one LMCache server, 2 vLLM instances, 32 prompts × 8 samples of
8–15k tokens: 1.89–1.94× the unique prompt tokens computed.

This speeds up the prefill of a prompt that siblings share: a whole single-turn
rollout, or the first turn of a multi-turn one, after which samples diverge.

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
   engine keeps the request waiting and polls, computing nothing for it. It
   still leases its free keys: prompts sharing a prefix would otherwise all wait on the
   prefix, then all compute their own suffixes. The engine is
   named by the `lmcache.hint.engine_id` the vLLM scheduler adapter adds to
   every key; requests of one engine never wait on each other (its prefix
   cache shares the prompt).
2. **Release.** A deferred lookup runs (and hits) once every awaited chunk is
   resident in L1 (`L1ManagerInterface.is_resident`, no side effects) or no
   longer leased by the request it waited on. A lease is dropped when a lookup sees its chunk resident,
   when its owner's lookup hits the chunk (only misses stay leased), when the
   owner's session ends, or by a sweep once the session is reaped. There is no
   wall-clock lifetime to tune.
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

Leases release many siblings at once, so without coalescing each of them
would hold its own GPU copy of the same KV. Enable coalescing with leases for
that reason.

## Configuration

```bash
lmcache server ... --prefill-lease-timeout-seconds 10   # 0 (default) = off
```

Coalescing is a separate engine option, because the engine cannot see the
server's flags:

```bash
--kv-transfer-config '{..., "kv_connector_extra_config": {"lmcache.mp.coalesce_sibling_loads": true}}'
```

Both default to off, so a deployment that sets neither behaves as before. A
waiting request starts later, which is why neither is on by default.

## Results

2 vLLM instances, one GPU each (RTX PRO 4000 Blackwell, 24 GB), one LMCache
server, Qwen3-4B, L1 only, 10 s timeout, both options on. A rollout is 32 prompts
× 8 samples. "Traces" uses the first turn of real agent sessions
(`sammshen/lmcache-agentic-traces`, up to 15,360 tokens, most sharing a long
prefix) with output lengths drawn from those turns; samples go to the instance
with the fewest in-flight requests. "Synthetic" uses random 8–15k-token prompts,
256 output tokens, round-robin. Outputs run to a set length, so latency is
comparable between runs. Run counts are as shown, and rows not marked otherwise
are one run; most lease arms were measured after their baselines, not interleaved.
On traces at 64 in flight, baseline runs span about 2% and lease runs about 6%;
the slowest lease run still beats the baseline mean by about 7%.

| Workload | Metric | baseline | leases + coalescing | change |
|---|---|---|---|---|
| Traces, 64 in flight (4 vs 3 runs) | rollout wall time | 56.2 s | 49.9 s | −11.2% |
| | prompt tokens computed | 146–151k | 81–89k | −44% |
| | mean TTFT | 1.03 s | 1.21 s | **+18%** |
| | median TTFT | 0.52 s | 0.77 s | **+47%** |
| Same (2 vs 3 runs) | mean request latency | 11.09 s | 9.45 s | −14.8% |
| Synthetic (2 vs 2 runs) | rollout wall time | 151.5 s | 119.0 s | −21.5% |
| | mean TTFT | 12.1 s | 7.3 s | −40% |
| | prompt tokens computed | 1.89–1.94× unique | 1.00× unique | |

A waiting sample starts later, so where the queue is short TTFT rises even though
requests finish sooner. Leases trade TTFT for rollout throughput, which is why
they suit batch rollouts and not interactive traffic.

How the traces rollout responds, rollout wall time, baseline → this change:

| Question | Case | baseline | this change | notes |
|---|---|---|---|---|
| Does off change anything? | both options off | 56.2 s | 56.2 s | 2 runs |
| Is coalescing needed? | leases only | 56.2 s | 55.0 s | 3.7× the KV loads of both on, mean TTFT 1.70 s |
| Prompts in flight | 8 (one at a time) | 99.2 s | 97.3 s | |
| | 256 (all at once), 2 runs | 55.6 s | 57.0 s | mean latency 33.3 → 29.6 s; 24 timeouts |
| Group size | 2 | 23.7 s | 23.0 s | |
| | 4 | 41.6 s | 36.5 s | |
| | 16 (16 prompts) | 51.1 s | 46.3 s | |
| Prompt length, no shared prefix | 1k | 23.5 s | 24.2 s | prefill is cheap: no gain |
| | 4k | 48.6 s | 45.1 s | |
| | 12k | 143.9 s | 113.1 s | |
| Group on one instance | | 49.9 s | 48.4 s | no harm |
| No siblings | group of 1 | 13.8 s | 13.8 s | no harm |
| L1 below the 11 GB working set | 8 GB | 56.3 s | 50.7 s | mean TTFT 1.02 → 1.30 s |
| | 2 GB | 56.1 s | 52.8 s | mean TTFT 1.02 → 2.63 s; 8 timeouts |
| More instances (Qwen2.5-1.5B) | 2 | 17.9 s | 15.7 s | |
| | 4, sharing 2 GPUs | 26.2 s | 22.7 s | computed 0.98× → 0.31× unique |

With all 256 requests in at once, a single long output sets the wall time, so mean
latency is the better measure there. A 30 s timeout removed the timeouts and
computed 9% less, but raised p90 latency (38.3 s → about 42 s) and its wall time
varied (47.8 and 61.3 s).

Sending a whole group to the least-loaded instance took about as long as spreading
it with leases on (both 49.9 s). On synthetic fixed-length outputs, an earlier
revision measured affinity 9% ahead. Leases help frameworks that already spread
samples without changing them; they do not replace affinity routing where that is
an option.

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
  the loader's GPU blocks. Both are the same KV by construction; outputs were
  not compared in the results above.
- A chunk evicted before any lookup saw it resident stays leased while its
  owner runs: the next lookup from another engine waits for the idle timeout,
  then recomputes and stores it.
- A coalesced sibling has no deadline of its own. It waits for the loader's
  `finished_recving` or the end of a request, so a retrieve that hangs without
  failing also holds its siblings until they are aborted.
- Coalescing has no effect with prefix caching off, since a waiter could not
  reuse the loaded blocks.
- `END_SESSION` racing a deferred lookup's submit can leave a prefetch job with
  no consumer, as can an abort racing any in-flight lookup. Its read locks last
  until the read-lock TTL.
- A coalesced sibling already holds read locks on its hit while it waits. If it is
  aborted then, no FREE is sent and they last until the read-lock TTL, which blocks
  evicting those chunks.
- A wait cycle is still possible when a request re-looks up (APC shrink, or
  resuming after preemption, the latter from reading the code) while another engine
  waits on it. Each side then idles for one timeout
  and computes as without leases.
- A deferred request keeps polling the server, which costs CPU under
  `_prefetch_job_lock` per poll; this was not measured separately.
- Eager prefetch leases at request arrival, possibly long before the owner gets a
  slot, so its waiters are more likely to reach the idle timeout. This combination
  was not measured.
- Measured with one model on one kind of GPU, mostly 2 instances. Multi-turn
  rollouts, training, weight updates, real sampled output lengths and methods
  other than a GRPO-shaped rollout were not measured. End to end, the gain shrinks
  with the first turn's share of a rollout.
- The gain depends on prefill cost against KV load cost, which depends on the
  GPU. Load speed was not measured separately.
- Coalescing keys on the cache salt and tokens, not the LoRA adapter. A request
  with the same tokens on another adapter waits for the load, then loads its own
  KV; it never shares the other adapter's blocks.

## Contract-anchoring tests

`tests/v1/multiprocess/test_prefill_lease.py` (waits, progress, give-up,
same-engine, owner release/end/death),
`tests/v1/multiprocess/test_lookup_prefill_lease.py` (the lookup and status
handlers around a deferred lookup) and
`tests/v1/test_mp_connector_sibling_loads.py` (a sibling waits for the
loading request, then loads nothing).
