# Cache-salt L2 request scheduling

## Scope

This design covers weighted admission of asynchronous L2 adapter requests in
MP mode. It spans the registry and scheduler in
`lmcache/v1/multiprocess/qos.py` and the adapter-boundary integration in
`lmcache/v1/distributed/l2_qos_adapter.py`.

The scheduler controls when LMCache calls an adapter's `submit_*` method. It
does not control filesystem writeback, device queues, network routing, or a
remote backend scheduler after submission.

## Scheduling domain and weights

The exact `ObjectKey.cache_salt` is the scheduling domain. The empty string is
a valid domain for requests without a salt.

`CacheSaltQosManager` stores only explicitly configured salt-to-weight entries.
An unregistered salt uses `default_sched_weight` immediately and is not added
to a discovered-salt collection. This keeps request cardinality from growing
registry state.

Weights are relative service shares under sustained contention, not bandwidth
or latency guarantees. Weight `200` receives twice the deficit quantum of
weight `100`. Updates affect queued work; already submitted backend work cannot
be reordered.

Changing the default weight publishes one default-policy update. Each active
dispatcher applies it to domains whose profile still has `source="default"`.
An explicit salt update targets only that salt, and deleting it changes the
active domain back to the current default.

## Resource groups

Each L2 adapter gets an independent dispatcher by default because adapters
normally represent independent resources. Its internal group key is
`adapter:<adapter_id>`.

Adapters configured with the same non-empty `qos_resource_group` share one
dispatcher. Use this only when they contend for the same device, mount,
network path, or backend admission window. Named groups use a separate
internal namespace from default adapter-local groups, so a configured name
cannot collide with `adapter:<adapter_id>`.

The weight registry is common to the MP server, while DRR queues and in-flight
limits are independent per resource group.

## Adapter boundary

`QosL2Adapter` preserves the asynchronous `L2AdapterInterface` contract:

1. It returns a wrapper task ID immediately.
2. It queues concrete adapter `submit_*` calls in the group dispatcher.
3. It translates concrete task IDs and completion events back to the wrapper
   task ID.
4. It releases dispatcher in-flight accounting when the concrete result is
   consumed.

`submit_unlock`, management calls, and eviction traffic bypass weighted
admission. Unlock is control traffic required to release backend state and
must not wait behind data-plane work.

### Mixed-salt batches

The public adapter interface permits one lookup or load request to contain
keys from different salts. The wrapper partitions every store, lookup, and
load task into salt-homogeneous subtasks while preserving the order of keys
within each salt.

Each subtask enters the correct salt's DRR queue. The caller still observes one
wrapper task:

- Lookup and load bitmaps are mapped from subtask-local positions back to the
  original key positions.
- Store completion succeeds only when every subtask succeeds; transferred
  bytes are summed.
- A failed submission contributes an all-zero bitmap or a failed store result.

This fan-out belongs in the wrapper rather than individual controllers so all
callers receive the same scheduling contract.

## DRR admission

Each dispatcher keeps a FIFO queue and deficit counter per active salt. One
logical round grants:

```text
quantum = max(1, quantum_bytes * sched_weight // 100)
```

Store and load cost is the sum of object sizes. Lookup cost is at least one and
otherwise the number of keys. A head task is admitted when its deficit covers
its cost and the resource-group in-flight limits permit it. Its cost is then
subtracted from the deficit.

Logical rounds advance immediately; they are not tied to wall-clock time. A
domain blocked only by deficit therefore accumulates service without an
artificial delay. Admission backpressure does not add deficit, preventing a
blocked domain from building an unbounded burst.

`max_inflight_tasks` and `max_inflight_bytes` apply to the entire resource
group. There are no per-salt operation or byte quotas in `QosProfile`; storage
capacity quotas are a separate feature.

When a domain has no queued or in-flight tasks, the dispatcher removes its
queue, profile, counters, and deficit. A later request recreates it from the
current manager profile.

## Management API

The MP HTTP server exposes:

| Method | Path | Purpose |
|---|---|---|
| GET, PUT | `/qos/config` | Read or change the default weight |
| GET | `/qos/cache-salt` | List explicit entries |
| GET, PUT, DELETE | `/qos/cache-salt/{cache_salt}` | Manage one salt |

Because an empty string cannot occupy a path segment, `_default` represents
the empty cache salt. This creates an intentional path-level collision: a
literal salt named `_default` cannot be managed separately through these
endpoints. Deployments that need to manage that literal value must choose a
different salt name.

## Concurrency and lifecycle

Registry state, dispatcher state, and wrapper task mappings use separate
locks. Registry listeners run without the registry lock. A failing listener is
logged and does not prevent later resource-group listeners from receiving an
update.

The wrapper serializes calls into the concrete adapter with an adapter-call
lock. Dispatcher-thread ``submit_*`` calls therefore cannot overlap controller-
thread ``pop_*`` or ``query_*`` calls, including for adapters whose internal
bookkeeping was designed around the two controller threads. The lock covers
control and management calls as well as data-plane calls.

`L2QoSDispatcher.close()` fails queued work and joins the dispatch thread. The
thread executes only the asynchronous adapter `submit_*` call, not the full
backend I/O. Joining is an intentional shutdown barrier: an adapter must not be
closed while one of its submission methods is still running. A concrete
adapter that blocks indefinitely in `submit_*` can therefore delay shutdown
and violates the asynchronous adapter expectation.

The dispatcher pool reference-counts adapters in each resource group. The last
release unregisters the group's listener before closing its dispatcher.

## Validation

`tests/v1/multiprocess/test_qos.py` covers weighted service, runtime updates,
resource-group isolation and sharing, mixed-salt fan-out/result merging,
bounded domain lifecycle, listener isolation, adapter-call serialization,
and wrapper event translation.
`tests/v1/multiprocess/test_http_qos_endpoints.py` covers the HTTP contract.
