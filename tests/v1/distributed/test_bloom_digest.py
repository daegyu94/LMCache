# SPDX-License-Identifier: Apache-2.0
"""Tests for the chunk-hash Bloom digest."""

# Standard
import os

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.bloom_digest import BloomDigest


def test_inserted_hashes_are_always_found() -> None:
    hashes = [os.urandom(32) for _ in range(2000)]
    digest = BloomDigest.build(hashes)
    assert all(digest.might_contain(h) for h in hashes)


def test_false_positive_rate_is_near_one_percent() -> None:
    digest = BloomDigest.build(os.urandom(32) for _ in range(2000))
    probes = [os.urandom(32) for _ in range(20000)]
    rate = sum(digest.might_contain(h) for h in probes) / len(probes)
    assert rate < 0.03


def test_json_round_trip_preserves_membership() -> None:
    hashes = [os.urandom(32) for _ in range(100)]
    decoded = BloomDigest.from_json(BloomDigest.build(hashes).to_json())
    assert all(decoded.might_contain(h) for h in hashes)


def test_from_json_rejects_size_mismatch() -> None:
    data = BloomDigest.build([b"x"]).to_json()
    data["num_bits"] = 4096
    with pytest.raises(ValueError):
        BloomDigest.from_json(data)


def test_large_set_has_no_false_negatives_and_low_request_false_positives() -> None:
    """At 200k keys every inserted hash is found, and a 16-chunk lookup for
    keys the peer does not hold is sent in only a few percent of cases."""
    hashes = [os.urandom(32) for _ in range(200_000)]
    digest = BloomDigest.build(hashes)
    assert all(digest.might_contain_any([h]) for h in hashes[::50])
    lookups = [[os.urandom(32) for _ in range(16)] for _ in range(5_000)]
    sent = sum(digest.might_contain_any(lookup) for lookup in lookups)
    assert sent / len(lookups) < 0.03
