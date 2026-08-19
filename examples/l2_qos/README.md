# Weighted L2 request scheduling Docker example

This harness demonstrates cache-salt-based weighted L2 request scheduling.
Clients send only a salt, and the server resolves its effective scheduling
weight. Unregistered salts use the default weight while explicit registrations
override it.

## Build

From the repository root:

```bash
docker build -f examples/l2_qos/Dockerfile -t lmcache-l2-qos .
```

## Run

Start a server with default scheduling weight 100 and an explicit scheduling
weight 500 for `high`:

```bash
docker run --rm --network host --name lmcache-l2-qos-server \
  lmcache-l2-qos server --clients 2 --default-sched-weight 100 \
  --weight high=500
```

Run the clients concurrently in two other terminals:

```bash
docker run --rm --network host lmcache-l2-qos \
  client 127.0.0.1 --cache-salt low --tasks 100
```

```bash
docker run --rm --network host lmcache-l2-qos \
  client 127.0.0.1 --cache-salt high --tasks 100
```

The `low` salt is intentionally unregistered and therefore uses scheduling
weight 100. The server prints total and first-32 admission counts. Both clients
eventually complete 100 tasks, while `high` should receive most early
admissions. This is a relative admitted-request share, not a physical disk or
network bandwidth guarantee.

## Steady-state L2 adapter benchmark

The storage benchmark starts exactly three Docker workload clients and runs
actual store or load requests through the selected L2 adapter. The `--adapter`
option currently supports only `fs`; other L2 adapters are TBD (future work). Use
`--io-mode direct` for O_DIRECT or `--io-mode buffered` for page-cache-backed
I/O. The benchmark measures weighted L2 request scheduling at the adapter
submission boundary; backend writeback, device queues, and remote queues are
outside its control.
By default, tenants with weights `100`, `200`, and `400` use 4 MiB, 2 MiB, and
1 MiB objects respectively. This makes request share and byte share distinct;
the benchmark validates completed byte share at the adapter wrapper against the
configured weights while all tenants remain backlogged. Set
`--max-inflight-tasks` to compare dispatcher concurrency. The selected value is
recorded in `result.json`. Passing this check demonstrates weighted admission;
it does not demonstrate physical device bandwidth isolation. Pass `--object-mib`
to use one object size for every tenant.

```bash
docker build --pull=false \
  -f examples/l2_qos/Dockerfile.benchmark \
  -t lmcache-l2-qos-benchmark:local .

python examples/l2_qos/benchmark.py run \
  --base-path /mnt/nvme/lmcache-l2-qos \
  --tenants-per-client 1 --operation store
```

To evaluate a different dispatcher concurrency, repeat the run with the desired
value, for example `8`, and compare `aggregate_mib_per_second` and per-tenant
`byte_share`:

```bash
python examples/l2_qos/benchmark.py run \
  --base-path /mnt/nvme/lmcache-l2-qos \
  --tenants-per-client 1 --operation store \
  --max-inflight-tasks 8
```

Use `--operation load` for the read path. Run `--tenants-per-client` with `2`
and `3` to cover 6 and 9 tenants as well. Store runs retain their generated
files, and every run writes `result.json` in a unique directory. The result
records both `request_share` and `byte_share`; `relative_error` is calculated
from `byte_share`. Direct I/O is the closer end-to-end signal because buffered
I/O completion can precede device writeback.
