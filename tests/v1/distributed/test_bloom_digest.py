# SPDX-License-Identifier: Apache-2.0
"""Tests for the chunk-hash Bloom digest and the P2P lookup skip rule."""

# Standard
import os

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.bloom_digest import BloomDigest
from lmcache.v1.distributed.l2_adapters.p2p_l2_adapter import PeerLookupDigest


def _key(chunk_hash: bytes) -> ObjectKey:
    return ObjectKey(chunk_hash=chunk_hash, kv_rank=0, model_name="m")


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


def test_lookup_digest_excludes_only_when_no_key_may_be_present() -> None:
    held = os.urandom(32)
    lookup_digest = PeerLookupDigest()
    keys = [_key(os.urandom(32)), _key(held)]
    # No digest yet: never skip.
    assert not lookup_digest.excludes_all(keys)
    lookup_digest.digest = BloomDigest.build([held])
    assert not lookup_digest.excludes_all(keys)
    lookup_digest.digest = BloomDigest.build([])
    assert lookup_digest.excludes_all(keys)
