# GRPO rollout benchmark: prefill leases + sibling-load coalescing

Measures the change on branch `feat/mp-grpo-prefill-lease` against `dev`.
The workload is a GRPO rollout: `--prompts` prompts, each with `--group`
samples. The whole group is submitted together, and its samples are dealt
round-robin over the engines, like verl's agent loop, AReaL, SkyRL or TRL's
data-parallel server. Token content is random because it does not change
prefill or KV cost.

## Setup (once per node)

```bash
git clone https://github.com/LMCache/LMCache.git lmcache-dev
cd lmcache-dev && git checkout dev && pip install -e . --no-build-isolation
cd ..
git clone https://github.com/daegyu94/LMCache.git lmcache-lease
cd lmcache-lease && git checkout feat/mp-grpo-prefill-lease && pip install -e . --no-build-isolation
cd ..
# Use separate venvs, or build both trees in place with
# `python setup.py build_ext --inplace`: the scripts put each tree on
# PYTHONPATH, so one venv with vLLM is enough.
```

## Run (multi-GPU node: one engine per GPU)

```bash
B=benchmarks/grpo_rollout   # in a checkout of branch exp/grpo-lease-bench
WL="--prompts 32 --group 8 --inflight 64 --len-dist uniform:8192:15360 --max-tokens 256"
LEASE="--prefill-lease-timeout-seconds 10"
export MODEL=Qwen/Qwen3-8B PER_GPU=1 L1=40 OUT=results
for n in 2 4 8; do
  for r in 1 2; do
    NENG=$n $B/run_ab.sh e${n}_lease_r$r /path/lmcache-lease "$LEASE" $WL
    NENG=$n $B/run_ab.sh e${n}_base_r$r  /path/lmcache-dev   ""       $WL
  done
  NENG=$n $B/run_ab.sh e${n}_group_r1 /path/lmcache-dev "" $WL --route group
done
python $B/summarize.py results
```

Check:

- `tokens.local_compute` against `unique_prompt_tokens`: dev computes about
  `min(NENG, group)` times the unique prompt tokens; the change should be
  close to 1×.
- `wall_s` and `gen_tok_s` give the rollout time and throughput.
- `ttft_*` gives TTFT. Waiting siblings start later, so a higher mean TTFT can
  go with a shorter rollout.
- `*.status.json` holds `prefill_lease_{waits,served,timeouts,wait_seconds}`.
  Many timeouts mean owners queue too long; raise the timeout.

## Results so far (one GB10, 2 engines sharing its GPU, 16 prompts × 8)

| Model | dev wall | change wall | prefill computed |
|---|---|---|---|
| Qwen2.5-3B | 142.2 / 141.4 s | 132.6 / 132.8 / 135.6 / 124.4 s (−7.3%) | 2.00× → 1.01× |
| Qwen3-8B | 423.6 / 424.5 s | 380.2 / 346.5 / 413.0 / 339.1 s (−12.8%) | 2.00× → 1.01–1.17× |

No siblings (`--group 1`) and a whole group on one engine (`--route group`)
are unchanged, with no lease waits.

Run-to-run spread on the change follows `prefill_lease_timeouts`: a run with
8 timeouts gained 2.6%, runs with none gained 17–20%. Timeouts mean an owner
sat in its engine's queue longer than the timeout. On a busy node, also try
`--prefill-lease-timeout-seconds 30`.

On this single-GPU host, sending a whole group to one engine (`--route
group`, 112 s for 3B) beat spreading it, with or without leases. Measure that
arm too: the frameworks spread groups to balance load, which matters more
with real, variable-length outputs and one GPU per engine.
