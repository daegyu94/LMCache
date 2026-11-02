# SPDX-License-Identifier: Apache-2.0
"""ipc_key_to_chunk_major_object_keys: same keys as ipc_key_to_object_keys,
ordered chunk first."""

# First Party
from lmcache.v1.distributed.api import (
    ipc_key_to_chunk_major_object_keys,
    ipc_key_to_object_keys,
)
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey


def _key(world_size: int) -> IPCCacheServerKey:
    return IPCCacheServerKey.from_token_ids(
        model_name="m",
        world_size=world_size,
        worker_id=None,
        token_ids=[],
        request_id="r",
        cache_salt="salt",
        request_configs={"lmcache.sequence_id": "s"},
        num_kv_readers=1,
    )


def test_same_keys_as_group_major_expansion() -> None:
    key, hashes, groups = _key(2), [b"a", b"b", b"c"], [0, 1]
    chunk_major = ipc_key_to_chunk_major_object_keys(key, hashes, groups)
    group_major = [k for g in ipc_key_to_object_keys(key, hashes, groups) for k in g]
    assert sorted(chunk_major, key=repr) == sorted(group_major, key=repr)
    assert all(k.cache_salt == "salt" for k in chunk_major)
    assert all(k.extra_attributes == key.request_configs for k in chunk_major)


def test_keys_are_ordered_chunk_then_group_then_rank() -> None:
    key, hashes, groups = _key(2), [b"a", b"b"], [1, 0]
    keys = ipc_key_to_chunk_major_object_keys(key, hashes, groups)
    ranks = sorted({k.kv_rank for k in keys})
    assert [(k.chunk_hash, k.object_group_id, k.kv_rank) for k in keys] == [
        (h, g, r) for h in hashes for g in groups for r in ranks
    ]
