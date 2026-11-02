# SPDX-License-Identifier: Apache-2.0

"""Sequence metrics subscriber — history appends vs rewrites.

Driven by ``MP_SEQUENCE_TRANSITION``, which the lookup module publishes for
requests tagged with ``lmcache.sequence_id`` that continue a tracked
sequence. The rewrite share is

    rate(lmcache_mp_sequence_transitions_total{kind="rewrite"}[5m])
    / rate(lmcache_mp_sequence_transitions_total[5m])

and the divergence-chunk histogram shows *where* histories are rewritten:
mass in the lowest buckets usually means the client's prompt template or
tokenization changed between turns rather than an intended fold.

Labels are bounded: ``model_name`` everywhere plus ``kind`` (``append`` or
``rewrite``) on the transition counter. Sequence ids are never labels.
"""

# Future
from __future__ import annotations

# Third Party
from opentelemetry import metrics

# First Party
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import EventCallback, EventSubscriber

# Divergence positions in chunks: 0, then powers of two up to 4096 (1M
# tokens at 256-token chunks), so the bucket count is fixed.
_DIVERGENCE_CHUNK_BUCKETS = (0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)


class SequenceMetricsSubscriber(EventSubscriber):
    """Maintains OTel instruments for sequence history transitions.

    Metrics (all labeled by ``model_name``):
    - ``lmcache_mp.sequence_transitions`` — tracked requests that continued
      their sequence, with extra ``kind`` label (``append`` / ``rewrite``).
    - ``lmcache_mp.sequence_superseded_chunks`` — previous-request chunks
      at or after the divergence chunk.
    - ``lmcache_mp.sequence_divergence_chunk`` — histogram of the
      divergence chunk index of rewrites.
    """

    def __init__(self) -> None:
        meter = metrics.get_meter("lmcache.sequence")
        self._transitions = meter.create_counter(
            "lmcache_mp.sequence_transitions",
            description=(
                "Requests tagged with lmcache.sequence_id that continued a "
                "tracked sequence, labeled by kind (append, rewrite)."
            ),
            unit="requests",
        )
        self._superseded = meter.create_counter(
            "lmcache_mp.sequence_superseded_chunks",
            description=(
                "Chunks of a sequence's previous request at or after the "
                "divergence chunk (never hit again by that sequence)."
            ),
            unit="chunks",
        )
        self._divergence = meter.create_histogram(
            "lmcache_mp.sequence_divergence_chunk",
            description="Chunk index at which a rewrite diverged.",
            unit="chunks",
            explicit_bucket_boundaries_advisory=_DIVERGENCE_CHUNK_BUCKETS,
        )

    def get_subscriptions(self) -> dict[EventType, EventCallback]:
        return {
            EventType.MP_SEQUENCE_TRANSITION: self._on_sequence_transition,
        }

    def _on_sequence_transition(self, event: Event) -> None:
        attrs = {"model_name": str(event.metadata["model_name"])}
        kind = event.metadata["kind"]
        self._transitions.add(1, attributes={**attrs, "kind": kind})
        self._superseded.add(event.metadata["superseded_chunks"], attributes=attrs)
        if kind == "rewrite":
            self._divergence.record(
                event.metadata["divergence_chunk"], attributes=attrs
            )
