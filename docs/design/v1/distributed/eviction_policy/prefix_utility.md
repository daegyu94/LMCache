# Bounded prefix interest

A terminal trajectory withdraws only its own interest in a shared prefix.
The policy keeps namespace-qualified ordered manifests and defines value as the sum of active manifests' longest resident prefixes.
A candidate's marginal loss is the number of usable chunks lost if it is removed; ties use the existing reverse-batch LRU ordering.
Each planned victim changes the next score, but membership changes only after confirmed manager deletion.
Staging reservations are tracked but do not contribute to readable prefix value.

The policy is advice, not validity or ownership.
The engine hook resolves from chunk zero even when STORE receives only a suffix and applies only to one full-attention object group.
Invalid advice is ignored, descriptor storage is bounded, and an empty manifest table preserves LRU ordering.
A terminal manifest remains for topology comparators until bounded replacement; stale interest can waste capacity, never authorize a stale KV hit.

`PrefixLRU` restricts candidates to known resident leaves, falling back to eligible ancestors if no leaf is eligible.
`SessionRefcount` orders these candidates by active-manifest reference count.
These are controlled comparison algorithms, not reproductions of SGLang or RR-Evict.
The exact `PrefixUtility` algorithm costs O(victims * references), so CPU and metadata must accompany hit-rate results.

Advice has no sequence, deadline, or automatic terminal timeout.
Missing/malformed/oversized advice leaves existing interests unchanged; a late `soon` can reactivate a completed trajectory.
This does not authorize stale KV but can waste capacity until bounded descriptor replacement.
The engine hook observes STORE only, so a terminal request without STORE cannot withdraw its interest.
Direct replay advice on every event is a stronger delivery contract and must not be reported as verified engine lifecycle coverage.
