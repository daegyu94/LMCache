**What this PR does / why we need it**:

vLLM's waiting queue can hold far more requests than `max_num_seqs`, for example after a burst of arrivals or when clients keep more requests in flight than one instance can run. While a connector lookup is pending, `get_num_new_matched_tokens` returns `None`, and vLLM's scheduler skips that request without spending token budget and keeps walking the queue (vLLM 0.24 `scheduler.py`, waiting loop). So whenever the running batch has room, `LMCacheMPConnector` submits a LOOKUP for every waiting request in a single scheduler step: the number of LOOKUPs in flight follows the queue depth, not what vLLM can run.

On the server, LOOKUP and PING are both `HandlerType.BLOCKING` handlers in the NORMAL pool, sized by `--max-cpu-workers` (falls back to `--max-workers`, default 1). The heartbeat PING queues behind the backlog, misses its 10 s timeout, and the scheduler adapter enters degraded mode; while unhealthy, `check_lookup_result` returns 0, so lookups checked in that window count as misses and are recomputed.

This PR bounds lookups that are submitted but not yet consumed (`cleanup_lookup_result`) by vLLM's `max_num_seqs`:

- `LMCacheMPSchedulerAdapter.can_submit_lookup(request_id)`: True if the request already has a lookup (so its result can still be polled) or `len(lookups) < max_pending_lookups`. New keyword `max_pending_lookups`, default `None` (no bound), so other adapter callers are unaffected.
- `LMCacheMPConnector.get_num_new_matched_tokens` checks it before submitting and returns `(None, True)` when full, so vLLM polls the request again next step.
- The bound is disabled when `lmcache.mp.eager_prefetch` is on: that mode submits on enqueue by design, and bounding only the scheduler-time path would starve requests that still need it (e.g. preempted ones).

No new config option. The design doc `docs/design/integration/vllm/mp_lookup_state.md` gains a "Bounded submission" rule and the new adapter call.

**Measurement** (1x RTX PRO 4000 Blackwell, Qwen3-0.6B, `--max-num-seqs 256`, prefix caching on, `lmcache server --l1-size 40 --chunk-size 256`; N = prompts x 16 samples, 1024-token random prompts, 16 output tokens, all requests sent at once over HTTP streaming; two runs each, seconds from burst start)

| N | metric | vanilla vLLM | LMCache (dev) | LMCache (this PR) |
|---|---|---|---|---|
| 2048 | 256th-fastest TTFT | 1.87 / 1.88 | 3.25 / 3.07 | 2.22 / 2.33 |
| 2048 | makespan | 11.1 / 11.1 | 12.8 / 12.2 | 11.1 / 11.1 |
| 8192 | median TTFT | 28.4 / 28.9 | 39.0 / 43.7 | 28.3 / 28.0 |
| 8192 | makespan | 50.6 / 51.6 | 62.6 / 67.6 | 48.9 / 49.7 |
| 8192 | degraded-mode entries | - | 1 | 0 |

At N=512 all three are within noise. Vanilla first-token time itself grows with N (~9 s at N=8192, API-server ingestion), so compare columns, not rows.

**Special notes for your reviewers**:

- AI assistance: the investigation, code, tests and benchmark were produced with an AI coding agent (Claude Code) and reviewed in three independent AI review passes (two reviewers). <!-- AUTHOR: confirm you reviewed every changed line before marking ready -->
- Known limitation (PRIORITY scheduling policy): while the bound is full, a high-priority request deferred early in a step can lose a freed slot to a lower-priority request polled later in the same step. It never deadlocks and FCFS (default) is unaffected; it only matters when lookups, not compute or memory, are the bottleneck and the backlog exceeds `max_num_seqs`. A possible follow-up: once one request is deferred in a step, defer all new lookups for the rest of that step.
- Under `lmcache.mp.eager_prefetch` the burst protection does not apply (unchanged behavior).
- The bound reduces, but does not remove, PING starvation: a cold burst still sends up to `max_num_seqs` LOOKUPs in one step.
- Not measured: TP > 1. LOOKUP count is per request (not per rank), but server-side work per lookup grows with KV ranks and each worker also sends its own PING through the same pool, so the effect is expected to be larger at high TP.
- Lookup latency stats (`record_lookup`) start at submission, so time spent deferred is not counted.

Local checks:

- `pytest -p no:cacheprovider` over every test file referencing the MP connector/adapter (16 files): 402 passed, 3 failed; the same 3 fail on unmodified `dev` (`test_free_locks_request_transport[grpc]`, `test_multi_connector_child_role_and_completion[*-kv_both]`).
- The 5 new tests fail on unmodified source and pass with this change.
- `ruff check`, `ruff format --check` (0.11.7), `isort --check-only` (6.0.1), `codespell` (2.4.1) on changed files: clean. `mypy` (1.17.1) on the two source files: 5 errors, all on lines this PR does not touch.

**AI-generated PR explainer** (required when AI-generated code is included):

<!-- AUTHOR: attach pr-explainer.html (covers commit 63b78a16) and replace this line with the attachment link -->

**If applicable**:

- [ ] this PR contains user facing changes - docs added
- [x] this PR contains unit tests

<details>
<summary>Before marking ready for review</summary>

- [ ] I reviewed every changed line since the latest update, including generated or
      agent-assisted changes.
- [x] I ran the relevant local checks and recorded the commands and results above.
- [x] I confirmed the description matches the final diff and documents any missing coverage
      or compatibility concerns.
- [x] I confirmed every commit has a DCO `Signed-off-by` trailer.
- [x] I stated whether AI assistance was used in **Special notes for your reviewers**.
- [ ] If AI-generated code is included, this PR is a draft and its title, description, and
      bilingual HTML explainer all match the current head.
- [ ] The HTML explainer is detailed enough for a reviewer without prior context, and its
      diagrams, optional motion, code walkthrough, and test rationale match the implementation.

</details>
