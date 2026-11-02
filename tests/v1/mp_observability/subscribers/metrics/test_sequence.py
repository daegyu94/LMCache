# SPDX-License-Identifier: Apache-2.0

"""Tests for SequenceMetricsSubscriber.

Events are delivered through the subscriber's own callbacks (the contract
the event bus uses), so no drain thread or sleep is involved. Counters are
read back from the shared ``InMemoryMetricReader`` as deltas.
"""

# Standard
from typing import Any

# First Party
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.subscribers.metrics.sequence import (
    SequenceMetricsSubscriber,
)
from tests.v1.mp_observability.subscribers.metrics.otel_setup import (
    counter_delta,
    histogram_count,
    read_counters,
    reader,
)


def _deliver(sub: SequenceMetricsSubscriber, **metadata: Any) -> None:
    event = Event(event_type=EventType.MP_SEQUENCE_TRANSITION, metadata=metadata)
    sub.get_subscriptions()[EventType.MP_SEQUENCE_TRANSITION](event)


def _transition(kind: str, divergence: int, previous: int) -> dict:
    return {
        "kind": kind,
        "divergence_chunk": divergence,
        "previous_chunks": previous,
        "current_chunks": divergence + 1,
        "superseded_chunks": previous - divergence,
        "model_name": "m",
    }


def _transition_attrs() -> set[tuple]:
    data = reader.get_metrics_data()
    attrs: set[tuple] = set()
    for rm in data.resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                if metric.name == "lmcache_mp.sequence_transitions":
                    for dp in metric.data.data_points:
                        attrs.add(tuple(sorted(dict(dp.attributes).items())))
    return attrs


def test_counts_appends_rewrites_and_chunks() -> None:
    sub = SequenceMetricsSubscriber()
    before = read_counters()
    hist_before = histogram_count("lmcache_mp.sequence_divergence_chunk")

    _deliver(sub, **_transition("append", divergence=3, previous=3))
    _deliver(sub, **_transition("rewrite", divergence=1, previous=4))

    delta = counter_delta(before, read_counters())
    assert delta["lmcache_mp.sequence_transitions"] == 2
    assert delta["lmcache_mp.sequence_superseded_chunks"] == 3
    # Only rewrites have a meaningful divergence position.
    assert histogram_count("lmcache_mp.sequence_divergence_chunk") - hist_before == 1


def test_labels_are_bounded_to_model_and_kind() -> None:
    sub = SequenceMetricsSubscriber()
    _deliver(sub, **_transition("rewrite", divergence=0, previous=2))
    _deliver(sub, **_transition("append", divergence=2, previous=2))

    assert _transition_attrs() <= {
        (("kind", "append"), ("model_name", "m")),
        (("kind", "rewrite"), ("model_name", "m")),
    }
