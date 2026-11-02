# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for the ``lmcache.skip_l2_store`` request hint.

Covers ``split_by_skip_l2_store_hint`` (the store controller's filter) and the
propagation of request configs into the keys the store controller receives.
The end-to-end controller flow is in ``test_store_controller.py``.
"""

# Standard
from typing import Any

# Third Party
import msgspec
import pytest

# First Party
from lmcache.v1.distributed.api import ObjectKey, ipc_key_to_object_keys
from lmcache.v1.distributed.storage_controllers.store_controller import (
    SKIP_L2_STORE_CONFIG,
    split_by_skip_l2_store_hint,
)
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey

# =============================================================================
# Helpers
# =============================================================================

SKIP = {SKIP_L2_STORE_CONFIG: True}
KEEP = {SKIP_L2_STORE_CONFIG: False}


def make_object_key(chunk_id: int, configs: dict[str, Any] | None = None) -> ObjectKey:
    """Create a test ObjectKey carrying ``configs`` as request configs."""
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(chunk_id),
        model_name="test_model",
        kv_rank=0,
        extra_attributes=configs,
    )


def make_ipc_key(request_configs: dict[str, Any] | None) -> IPCCacheServerKey:
    """Create a worker-scoped IPC store key with ``request_configs``."""
    return IPCCacheServerKey.from_token_ids(
        model_name="test_model",
        world_size=1,
        worker_id=0,
        token_ids=list(range(16)),
        end=16,
        request_id="req-0",
        request_configs=request_configs,
    )


# =============================================================================
# split_by_skip_l2_store_hint
# =============================================================================


class TestSplitBySkipL2StoreHint:
    """Test the per-batch skip decision."""

    def test_unhinted_batch_is_returned_unchanged(self):
        keys = [make_object_key(i) for i in range(3)]

        split = split_by_skip_l2_store_hint(keys)

        assert split.to_store == keys
        assert split.skipped == []
        assert split.invalid == []

    def test_hinted_keys_are_skipped(self):
        keys = [
            make_object_key(0, SKIP),
            make_object_key(1, KEEP),
            make_object_key(2),
            make_object_key(3, SKIP),
        ]

        split = split_by_skip_l2_store_hint(keys)

        assert split.to_store == [keys[1], keys[2]]
        assert split.skipped == [keys[0], keys[3]]
        assert split.invalid == []

    def test_unrelated_request_configs_do_not_skip(self):
        keys = [make_object_key(0, {"lmcache.skip_load": True})]

        assert split_by_skip_l2_store_hint(keys).to_store == keys

    def test_duplicate_key_skipped_only_if_every_writer_skips(self):
        keys = [make_object_key(5, SKIP), make_object_key(5, KEEP)]
        keys += [make_object_key(6, SKIP), make_object_key(6)]
        keys += [make_object_key(7, SKIP), make_object_key(7, SKIP)]

        split = split_by_skip_l2_store_hint(keys)

        assert split.to_store == keys[:4]
        assert split.skipped == [make_object_key(7)]

    @pytest.mark.parametrize("value", ["true", 1, None, [True]])
    def test_non_bool_hint_is_ignored_and_reported(self, value):
        keys = [make_object_key(0, {SKIP_L2_STORE_CONFIG: value}), make_object_key(1)]

        split = split_by_skip_l2_store_hint(keys)

        assert split.to_store == keys
        assert split.skipped == []
        assert split.invalid == [keys[0]]


# =============================================================================
# Hint propagation: request_configs -> ObjectKey.extra_attributes
# =============================================================================


class TestHintPropagation:
    """The hint sent as a request config reaches the store controller's keys."""

    def test_request_configs_reach_store_keys(self):
        hashes = [ObjectKey.IntHash2Bytes(i) for i in range(2)]
        configs = {**SKIP, "lmcache.tag.tenant": "t"}

        keys = ipc_key_to_object_keys(make_ipc_key(configs), hashes, [0])[0]

        assert [k.extra_attributes for k in keys] == [configs, configs]
        assert split_by_skip_l2_store_hint(keys).skipped == keys

    def test_request_without_configs_is_unhinted(self):
        hashes = [ObjectKey.IntHash2Bytes(0)]

        keys = ipc_key_to_object_keys(make_ipc_key(None), hashes, [0])[0]

        assert keys[0].extra_attributes == {}
        assert split_by_skip_l2_store_hint(keys).to_store == keys

    def test_hint_survives_finish_write_payload_encoding(self):
        # The MP store path hands keys to ``finish_write`` through a
        # msgpack-encoded ``dict[ObjectKey, int]`` device host-func payload.
        hashes = [ObjectKey.IntHash2Bytes(i) for i in range(2)]
        keys = ipc_key_to_object_keys(make_ipc_key(SKIP), hashes, [0])[0]

        decoded = msgspec.msgpack.decode(
            msgspec.msgpack.encode({key: 0 for key in keys}),
            type=dict[ObjectKey, int],
        )

        assert list(decoded) == keys
        assert [k.extra_attributes for k in decoded] == [SKIP, SKIP]
        assert split_by_skip_l2_store_hint(list(decoded)).skipped == keys
