<!-- Draft PR text kept on the candidate branch at the user's request; this file and
     its commit are not part of the upstream change and must be dropped before
     opening the PR. -->

# [Perf][MP] fs L2 adapter: check a lookup batch in one executor call

**What this PR does / why we need it**:

`FSL2Adapter._execute_lookup` awaited `aiofiles.os.path.exists(path)` once per key.
`aiofiles.os.path.exists` is `wrap(os.path.exists)`, i.e. `loop.run_in_executor(None,
os.path.exists, path)` (aiofiles/base.py), so a lookup of N keys paid N thread-pool
round trips and N GIL re-acquisitions on the adapter's event loop. This PR runs the
same `os.path.exists` for the whole batch in one `run_in_executor(None, ...)` call
(new module helper `_paths_exist`) and removes the now-unused `_key_exists_on_disk`.

- Same function, same default executor, same failure mode (`os.path.exists` maps
  `OSError`/`ValueError` to `False`), same sequential order of checks.
- No caching: every lookup still reads the real file-system state, so shared-FS
  setups (another node writes or deletes a file) behave as before.
- Same pattern the adapter already uses for O_DIRECT reads and writes
  (`self._loop.run_in_executor(None, ...)`).

Why lookup is on the hot path: the prefetch controller sends every suffix key that
L1 misses to L2 (see the TODO in `prefetch_controller.py` near the L2 submit), so in
the agentic rollout below most lookups reach the fs adapter, and most keys are
absent. The `fs_native` adapter already avoids per-key Python awaits; this change is
for the documented Python `fs` adapter.

**Evidence**

Adapter microbenchmark (`benchmarks/microbenchmark/fs_l2_lookup_benchmark.py`, 32
absent keys, 60 lookups per row, one GB10). "Before" is the same script run against
base `386473da`. Busy threads are CPU-bound Python threads holding the GIL; they are
a stress model of other server threads, not a measured server load.

| busy threads | before p50 / p90 | after p50 / p90 |
| --- | --- | --- |
| 0 | 8.51 / 11.38 ms | 0.74 / 0.92 ms |
| 1 | 800.75 / 914.86 ms | 141.91 / 197.10 ms |
| 2 | 1026.91 / 1624.63 ms | 122.70 / 215.17 ms |

```bash
python benchmarks/microbenchmark/fs_l2_lookup_benchmark.py --base-path /mnt/l2-bench
```

Agentic rollout on a real server: one GB10, vLLM 0.33.0, Qwen2.5-0.5B-Instruct,
LMCache MP with LRU L1 4 GB + `fs` L2 (O_DIRECT, local NVMe), token-in/token-out
replay of 64 SWE-agent trajectories (about 620 requests; driver:
`benchmarks/agentic_rl/tito_rollout.py` from the companion per-request
`save_decode_cache` branch, without `--save-decode`). 3 runs per side, fresh servers
and empty L2, mean ± 95% CI with t(df = 2), two-sided Welch t-test:

| metric | before | after | |
| --- | --- | --- | --- |
| server-side prefetch completion time (mp_server.log), median per run | 19.6 ± 3.4 ms | 8.3 ± 1.8 ms | -58%, significant |
| same, mean per run | 27.4 ± 5.1 ms [29.7, 26.5, 25.8] | 19.5 ± 8.4 ms [15.6, 21.6, 21.3] | -29%, significant |
| vLLM "LMCache MP lookup avg latency" | 32.8 ± 1.3 ms | 27.9 ± 11 ms | no significant change |
| mean TTFT | 0.124 ± 0.005 s | 0.118 ± 0.019 s | no significant change |
| request latency p99 | 5.79 ± 0.24 s | 5.68 ± 0.62 s | no significant change |
| generated tok/s | 956 ± 31 | 973 ± 27 | no significant change |
| prompt tokens vLLM computed | 744,800 | 746,000 | +0.2% (same cache behavior) |
| LMCache hit tokens | 1,115,000 | 1,116,000 | +0.1% (same cache behavior) |

The server finishes the L2 existence check sooner, but in this setup vLLM's scheduler
only sees a lookup result at its next check, so the end-to-end lookup latency and TTFT
do not move beyond noise. n = 3 is a small sample and the per-run mean after the
change varies (15.6 / 21.6 / 21.3 ms); the median is the stable signal. This PR is
therefore a removal of per-key overhead (less server CPU and GIL churn per lookup),
not a user-visible latency win in this setup. Deployments with more contention or
longer suffixes may gain more; that was not measured.

Tests:

- New `tests/v1/distributed/test_fs_l2_adapter_lookup.py` (public interface only): the
  bitmap reports each key in order for a mixed present/absent batch; files removed
  behind the adapter become misses on the next lookup (no caching); an empty lookup
  completes. These also pass on the old code, as expected for a change that keeps
  behavior.
- Existing fs adapter tests: 108 passed, 15 skipped.
- `SKIP=rust-fmt,rust-clippy pre-commit run --files <changed files>`: passed.

**Special notes for your reviewers**:

- AI assistance: the change, tests, benchmark and this description were written with
  an AI coding agent. The submitting human still has to review every changed line and
  rerun the validation before this leaves draft.
- One batch occupies one default-executor worker for its whole sequential pass,
  instead of queuing one task per key; before and after are equally sequential.
- Interactions: #4366 (parallel `_execute_load`) touches the same file but a different
  function; #4853 touches lines around `_key_exists_on_disk` and needs a trivial rebase.
- `docs/source/mp/l2_storage/fs.rst` ("async I/O (aiofiles)") stays accurate: store,
  load and delete still use aiofiles.

**AI-generated PR explainer** (required when AI-generated code is included):

TODO before opening the PR: generate the bilingual HTML explainer with
`.claude/skills/create-pr-explainer/SKILL.md` and link it here.

**If applicable**:

- [ ] this PR contains user facing changes - docs added
- [x] this PR contains unit tests

<details>
<summary>Before marking ready for review</summary>

- [ ] I reviewed every changed line since the latest update, including generated or
      agent-assisted changes.
- [ ] I ran the relevant local checks and recorded the commands and results above.
- [ ] I confirmed the description matches the final diff and documents any missing coverage
      or compatibility concerns.
- [ ] I confirmed every commit has a DCO `Signed-off-by` trailer.
- [ ] I stated whether AI assistance was used in **Special notes for your reviewers**.
- [ ] If AI-generated code is included, this PR is a draft and its title, description, and
      bilingual HTML explainer all match the current head.
- [ ] The HTML explainer is detailed enough for a reviewer without prior context, and its
      diagrams, optional motion, code walkthrough, and test rationale match the implementation.

</details>
