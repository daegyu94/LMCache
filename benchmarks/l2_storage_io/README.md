# CPU-only L2 storage I/O benchmark

`benchmark.py` sends deterministic synthetic KV chunks through LMCache's real
L2 adapter interface without starting a model server, allocating CUDA memory,
or requiring a GPU.
It measures storage I/O only: request E2E is admission/scheduling through
adapter completion, not TTFT.

Run it as a module from an LMCache checkout so its relative imports resolve.

```bash
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=""
python -m benchmarks.l2_storage_io.benchmark \
  --lmcache-checkout "$PWD" \
  --storage-root /mnt/benchmark/l2 \
  --output /mnt/benchmark/l2/results \
  --profile xfs-fs-native \
  --operation mixed \
  --concurrency 4
```

The four profile names are labels, not filesystem detection:

| Profile | Adapter | Expected user-supplied mount/backend |
| --- | --- | --- |
| `xfs-fs-native` | `fs_native` | XFS mount |
| `pnfs-fs-native` | `fs_native` | pNFS mount |
| `pnfs-nixl-posix` | `nixl_store_dynamic` / `POSIX` | pNFS mount |
| `3fs-nixl-hf3fs` | `nixl_store_dynamic` / `HF3FS` | 3FS mount and HF3FS plugin |

Run only a profile whose root is actually mounted on the intended storage.
The benchmark records the supplied root and marks mount verification as false;
it never claims that directory labels prove filesystem identity.
HF3FS is not installed in the local development environment, so its profile
cannot be validated there.

Before measured requests, the benchmark stores each working-set key once in
bounded batches and then starts a fresh internal collector.
Measured reads select a deterministic hot/prefix reuse set, poison destination
buffers before loads, and verify payload bytes.
Measured writes use unique keys and require the adapter to report the expected
byte count, avoiding duplicate-store throughput.
Mixed requests require at least two chunks so they contain both an established
read and a fresh write.

The result JSON excludes prepopulation, failures, timeouts, and integrity
failures from byte throughput.
It includes hit/miss counts, request p50/p95/p99 timings, and opt-in adapter
spans.
Native connector `queue_io_completion` is an opaque queue + I/O + completion
interval.
NIXL `transfer_wait` includes the adapter's 10 ms polling cadence; per-file
open/register/prepare/deregister/publish spans can overlap and must not be
summed as an E2E decomposition.

The dynamic NIXL adapter uses page indices relative to its registered L1 arena.
This corrects the prior absolute-address indexing, which fails local POSIX
transfers when a normal CPU arena is registered.

## CPU setup and limits

Use the LMCache checkout's existing project virtual environment, never a system
runtime.
If it does not exist, create it with Python 3.12 and install a CPU build before
running this changed checkout:

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
NO_GPU_EXT=1 uv pip install -e . --no-build-isolation
```

The editable install matters: a wheel installed from the unchanged Tracebench
pin will not contain this benchmark or adapter instrumentation.

Synthetic payloads repeat a deterministic 16-byte seed.
They exercise access reuse and integrity, not real KV entropy, compression, or
deduplication behavior.
Choose `chunk_bytes` to match an illustrative KV geometry such as
`2 * layers * local_kv_heads * head_dim * tokens * dtype_bytes`.
The harness polls completions at 1 ms; dynamic NIXL additionally polls transfers
at 10 ms, so those polling intervals affect reported latency.
