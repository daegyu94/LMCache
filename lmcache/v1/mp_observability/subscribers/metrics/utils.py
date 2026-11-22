# SPDX-License-Identifier: Apache-2.0

"""Shared cache_salt and L2 adapter label helpers for metrics subscribers."""

# Future
from __future__ import annotations

# Standard
from collections import Counter
from typing import TYPE_CHECKING

# Third Party
from opentelemetry import metrics

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.mp_observability.event import Event

# Event metadata keys that identify the L2 adapter an event belongs to.
_L2_ADAPTER_ATTR_KEYS = ("l2_name", "peer")


def group_by_salt(keys: list) -> dict[str, int]:
    """Group *keys* by ``cache_salt``, returning salt → count."""
    return Counter(getattr(k, "cache_salt", "") for k in keys)


def emit_salt_counts(
    counter: metrics.Counter,
    salt_counts: dict[str, int],
    attributes: dict[str, str] | None = None,
) -> None:
    """Add to *counter* once per entry in *salt_counts*.

    *salt_counts* maps ``cache_salt`` values to key counts.
    Empty salt adds no salt label. Optional attributes label every increment.
    """
    for salt, count in salt_counts.items():
        attrs = dict(attributes or {})
        if salt:
            attrs["cache_salt"] = salt
        counter.add(count, attributes=attrs)


def l2_adapter_attrs(event: Event) -> dict[str, str]:
    """Copy the L2 adapter labels (``l2_name``, ``peer``) in *event*'s
    metadata into a metric attributes dict.

    Keys the event lacks are left out, so a non-P2P adapter keeps the
    ``l2_name``-only series and an event with neither key stays unlabeled.
    """
    return {
        key: str(event.metadata[key])
        for key in _L2_ADAPTER_ATTR_KEYS
        if key in event.metadata
    }
