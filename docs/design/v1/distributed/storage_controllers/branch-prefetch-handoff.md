# Partial branch prefetch and temporary-object handoff

`prefetch_flights.py` optionally coordinates partially overlapping requests on the MP PrefetchController loop.
Concurrent Agent-RL branches can request a common history plus distinct suffixes.
Exact-request singleflight cannot merge these different grids, while retaining every loaded object permanently would change the cache policy.

## Contract and sequence

1. An admitted request indexes only cells absent from its fresh L1 read-lock result.
   Each cell is qualified by ObjectKey (including salt, rank, group and model), layout shapes/dtypes, window and fetching policy.
2. A pending request that overlaps an indexed cell waits for its owner; unrelated requests still enter free slots.
   Waiters are registered even when all slots are occupied, so registration precedes the leader's result publication.
3. In `handoff` mode, immediately before settling the leader's locks, each dependent waiter reserves one extra L1 read lock per distinct resident overlapping key.
   Failure to find a cell produces no lock and no success assertion.
4. Leader retirement removes the dependency; the waiter takes its own ordinary read locks, then returns the extra handoff locks.
   Existing L2 lookup, planning, load, grouped-prefix folding and result construction are unchanged.
5. Completed, failed and stopped requests return rendezvous-owned read locks.
   A full waiter table bypasses coordination and runs the ordinary path.

The dependency graph is acyclic because only admitted requests own indexed cells, while deferred requests own none.
Resident shared cells are not indexed again, allowing branch suffixes to load independently after the common flight completes.
No payload pointer, success bitmap, ownership count or cache identity is borrowed from another request.
A follower with three downstream readers still obtains three independent locks after receiving one temporary handoff lock.

## Ablations and costs

`off` keeps the ordinary admission loop and disables cell indexing.
`overlap` waits for matching flights without preserving temporary data; it can reload data if the leader's reader finishes before the follower takes its lock.
`handoff` adds explicit temporary-data lifetime transfer.

The PoC delays the entire follower, including its unique suffix, until the earlier overlapping flight retires.
Low-overlap or slow-leader workloads can therefore lose latency even while transferring fewer bytes.
This is not fully parallel per-cell MSHR execution, and it does not reduce the existing full-grid L2 metadata lookup.
Default capacity is 64 waiting requests, including ready requests that still hold handoff pins; cell metadata and pins also scale with the sizes of those requests.
There is no independent byte credit or timeout policy.
`report_status()['branch_flights']` exposes waits, cap bypasses, pin acquisition/release counts, peak/current pins, and indexed cells.
Cap bypasses count scheduling attempts, not unique overflowed requests.

## Correctness boundaries

Flights do not replace policy-version namespaces; a caller must still use a distinct valid namespace after weight changes.
The existing L1 read-lock TTL remains authoritative, so a stalled waiter beyond that TTL is outside the validated handoff lifetime.
Baseline issue [#5263](https://github.com/LMCache/LMCache/issues/5263) describes active adapter writes and buffer reclamation during shutdown or TTL expiry.
This patch neither fixes that issue nor treats an open proposal as implemented upstream.
Its cleanup releases only extra rendezvous read pins; it does not claim adapter quiescence or reclaim active I/O safely.
Shutdown tests use completed payloads and no outstanding adapter writes.

A helper test deterministically consumes the leader before follower admission, checks the exact payload and independent reader counts, and shows the difference between overlap-only and handoff.
Controller tests cover partially overlapping branches, single-slot admission, failure fallback, bounded waiters and namespace/layout/window separation.

## Optional adapter submission failure

A synchronous lookup submission rejection (`RuntimeError` or `ValueError`) skips only that adapter.
Previously accepted lookups remain tracked and finish through the ordinary load/unlock path; aborting the entire request here could orphan a late lookup result and its L2 locks.
If no adapter accepts a lookup, the controller finishes from its fresh L1 read-lock set, including any missing suffix.
This rule assumes a rejected submission has not accepted asynchronous work, as the caller received no task identifier.
Two-adapter and all-rejected regression tests check independent L1/L2 ownership and completed flight cleanup in both disabled and handoff modes.
