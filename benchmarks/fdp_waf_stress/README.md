# LMCache FDP WAF Stress Harness

This benchmark runs several LMCache storage trace replays concurrently against
one raw-block NVMe namespace. It is designed to create cache-like churn and make
SSD garbage-collection pressure easier to observe.

FDP RUH IDs are placement hints only. Byte isolation comes from unique
`base_offset_bytes` and `capacity_bytes` windows.

## Modes

- `no_fdp`: same replay concurrency and byte windows without FDP.
- `mixed`: FDP enabled, with hot, cold, metadata-heavy, small-object, and
  large-object workloads intentionally sharing the same data RUH pool.
- `separated`: FDP enabled, with hot/churn data, cold/RAG data, and metadata
  assigned to separate RUH pools.

## Run

First inspect the generated commands and windows:

```bash
uv run --no-sync python benchmarks/fdp_waf_stress/run_fdp_waf_stress.py \
  --config benchmarks/fdp_waf_stress/config.example.yaml \
  --mode mixed \
  --iterations 8 \
  --warmup-iterations 2 \
  --output-dir /mnt/hc-ssd/waf-fdp-mixed \
  --dry-run
```

## Generate CPU-Only Synthetic Traces

The stress harness can use real `.lct` files, but the repo also provides a
CPU-only synthetic generator for servers without GPUs:

```bash
uv run --no-sync python benchmarks/fdp_waf_stress/generate_synthetic_traces.py \
  --output-dir /mnt/hc-ssd/lmcache-fdp-waf-stress/traces \
  --config-out /mnt/hc-ssd/lmcache-fdp-waf-stress/config.128ruh.yaml \
  --manifest-out /mnt/hc-ssd/lmcache-fdp-waf-stress/trace_manifest.generated.yaml \
  --summary-out /mnt/hc-ssd/lmcache-fdp-waf-stress/trace_generation_summary.json \
  --ruh-count 128 \
  --scale stress
```

The generator writes storage-level LMCache trace records directly. It does not
run a model, allocate CUDA tensors, or touch NVMe. Replay later allocates CPU L1
objects from the recorded `MemoryLayoutDesc` and writes synthetic payloads to
the configured raw-block L2 adapter.

`--scale smoke` retains the compact validation recipe. `--scale stress` and
`--scale 1` select the stress baseline; a positive integer scales its store/prefetch
batch counts and worker capacities together. The generated config uses packed,
non-overlapping worker windows, so the LBA span equals the sum of worker
capacities. For example, `--scale 108` creates about 1,026 GiB of worker
capacity.

For an 8-RUH device, pass `--ruh-count 8`; this uses data RUHs 0-6 and
metadata RUH 7, with the separated-class mapping from the 8-RUH evaluation
configuration. Other RUH counts must be at least 128.

Use `--root` to relocate the generated traces, config, manifest, and summary
together. `--device-path` and `--block-device-path` override the corresponding
replay device paths. The summary prints the expected packed worker LBA ranges,
capacity sum, and span.

When `--config-out` is omitted, the generated config name includes the RUH
count and scale, for example `config.8ruh.50scale.yaml`. It uses compact RUH
ranges and expands them to explicit RUH arrays when building replay commands.
In 128-RUH `separated` mode, data RUHs are split by workload family and
metadata RUHs use IDs above 100.

Dry-run the generated config:

```bash
uv run --no-sync python benchmarks/fdp_waf_stress/run_fdp_waf_stress.py \
  --config /mnt/hc-ssd/lmcache-fdp-waf-stress/config.128ruh.yaml \
  --mode separated \
  --iterations 1 \
  --warmup-iterations 0 \
  --output-dir /mnt/hc-ssd/waf-128ruh-separated-dry-run \
  --dry-run
```

Run the three comparisons:

```bash
# 1. No FDP baseline
uv run --no-sync python benchmarks/fdp_waf_stress/run_fdp_waf_stress.py \
  --config benchmarks/fdp_waf_stress/config.example.yaml \
  --mode no_fdp \
  --iterations 8 \
  --warmup-iterations 2 \
  --output-dir /mnt/hc-ssd/waf-no-fdp

# 2. FDP mixed / adversarial placement
uv run --no-sync python benchmarks/fdp_waf_stress/run_fdp_waf_stress.py \
  --config benchmarks/fdp_waf_stress/config.example.yaml \
  --mode mixed \
  --iterations 8 \
  --warmup-iterations 2 \
  --output-dir /mnt/hc-ssd/waf-fdp-mixed

# 3. FDP separated placement
uv run --no-sync python benchmarks/fdp_waf_stress/run_fdp_waf_stress.py \
  --config benchmarks/fdp_waf_stress/config.example.yaml \
  --mode separated \
  --iterations 8 \
  --warmup-iterations 2 \
  --output-dir /mnt/hc-ssd/waf-fdp-separated
```

Expected comparison:

- `no_fdp`: non-FDP baseline.
- `mixed`: intentionally bad FDP placement, expected to create higher WAF
  pressure.
- `separated`: tests whether FDP lowers WAF by separating lifetimes.

Do not expect a fixed WAF number. Results depend on SSD firmware, namespace
provisioning, FDP implementation, workload shape, and media-write counter
availability.

## Time-Based Runs

Use `--duration-seconds` instead of `--iterations` to run measurement replays
until a wall-clock deadline:

```bash
uv run --no-sync python benchmarks/fdp_waf_stress/run_fdp_waf_stress.py \
  --config benchmarks/fdp_waf_stress/config.example.yaml \
  --mode mixed \
  --duration-seconds 1800 \
  --warmup-iterations 2 \
  --sample-interval-seconds 300 \
  --output-dir /mnt/hc-ssd/waf-fdp-mixed-30m
```

Each worker starts again from the first trace record on every replay pass, with
a new `--replay-cache-salt-suffix`. A pass that has already started is allowed
to finish cleanly after the deadline.

## Output

The example config defaults generated data to:

```text
/mnt/hc-ssd/lmcache-fdp-waf-stress/
```

Each run writes:

- `run_config.resolved.yaml`
- `commands.txt`
- `workers.json`
- `measurement_before.json`
- `measurement_after_warmup.json`
- `measurement_after_benchmark.json`
- `measurement_after.json`
- `summary.json`
- `summary.md`
- `l2_latency_summary.json`
- `worker_logs/`
  - each replay directory contains `l2_latency.jsonl`
- `waf_samples.tsv`, when `--sample-interval-seconds` is greater than 0

`waf_samples.tsv` records one sample per interval during measurement with
`timestamp`, interval-delta `fdp_host_write_mb`, interval-delta
`fdp_media_write_mb`, interval `fdp_waf`, cumulative
`device_write_multiplier`, and `sample_status`. Some NVMe controllers do not
refresh FDP stats on every sample interval; when both observed MB deltas are zero,
the delta and WAF fields are left blank and `sample_status` is `stale`. The
default interval is 300 seconds; pass `--sample-interval-seconds 0` to disable
periodic sampling.

`summary.json`, `summary.md`, and `l2_latency_summary.json` report
`count`, `error_count`, `avg_ms`, `p90_ms`, and `p99_ms` for L2 reads
and writes. `l2_e2e_read/write` measures raw-block adapter task submission
through asynchronous completion, including its queue and core overhead.
`raw_block_read/write` measures the underlying raw-device call through
completion. The aggregate uses measurement-phase data I/O only; the worker
JSONL retains metadata I/O samples with `"io_class":"metadata"`.
If a vendor media/NAND write counter is configured, `summary.json` includes
`media_write_bytes_delta` and `waf`. Without that counter, WAF is marked
unavailable rather than estimated.

The runner waits 600 seconds after warmup, when warmup is enabled, before
capturing the WAF baseline. It also saves an immediate post-benchmark counter
snapshot, waits 600 seconds for delayed controller accounting, then captures the
final counter used for WAF.

## Trace Guidance

Use user-provided `.lct` files and collect at least these trace families:

1. Small model, small chunk size, chat-like prompts.
2. Large model, large chunk size, long-context prompts.
3. RAG/shared-prefix trace with repeated prefixes.
4. Random prompt trace with low reuse.
5. Metadata-heavy/small-object trace.

For storage-level replay, actual KV tensor bytes are not stored in the trace.
Replay uses synthetic payloads from recorded layout descriptors. Different
models matter only when they change storage-level behavior: object sizes,
number of chunks, timing, key reuse, and store/retrieve mix.

The harness preflights each trace and warns when a worker window is likely too
large to churn or too small to fit the largest object plus metadata reservation.
Capacity should usually be smaller than total unique bytes written during the
measurement phase but large enough for the active working set.

## Local Replay Command

This checkout has been run with `uv run --no-sync` and a small OpenTelemetry
logger-provider shim. The example config uses that as `global.replay_binary`.
If the `lmcache` console script is installed on `PATH`, replace it with:

```yaml
global:
  replay_binary: lmcache
```
