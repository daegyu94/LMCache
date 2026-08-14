# L2 QoS Docker example

This directory contains a manual integration harness for the shared-L2 QoS
dispatcher. It uses two containers with different cgroup I/O weights and
checks that the higher-weight client receives a larger share of early admissions.

The harness is not a pytest test: it requires a Linux host with Docker
configured to expose cgroup v2 I/O weights.

## Build

Run this command from the repository root:

```bash
docker build -f examples/l2_qos/Dockerfile -t lmcache-l2-qos .
```

## Run

Start the server in one terminal:

```bash
docker run --rm --network host --name lmcache-l2-qos-server \
  lmcache-l2-qos server --clients 2
```

Run both clients in separate terminals:

```bash
docker run --rm --network host --blkio-weight 100 \
  -e LMCACHE_QOS_DOMAIN=low lmcache-l2-qos client 127.0.0.1 --tasks 100
```

```bash
docker run --rm --network host --blkio-weight 500 \
  -e LMCACHE_QOS_DOMAIN=high lmcache-l2-qos client 127.0.0.1 --tasks 100
```

Each client should report `"source":"cgroup"`. The server prints total
and first-32 admission counts. With equal finite task counts, total counts
converge; the `high` client should dominate the first-32
admissions. Docker's `--blkio-weight` mapping can vary on cgroup v2, so
compare the weights reported by the clients rather than assuming the raw
Docker values are preserved.
