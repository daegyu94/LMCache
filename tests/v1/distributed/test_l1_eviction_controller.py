# SPDX-License-Identifier: Apache-2.0
"""Tests for L1EvictionController.execute_eviction_action slicing."""

# Standard
from collections.abc import Iterator
from unittest.mock import patch

# Third Party
import pytest
import torch

# First Party
from lmcache import torch_dev, torch_device_type
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.config import (
    EvictionConfig,
    L1ManagerConfig,
    L1MemoryManagerConfig,
)
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.internal_api import EvictionAction, EvictionDestination
from lmcache.v1.distributed.l1_manager import L1Manager
from lmcache.v1.distributed.storage_controllers.eviction_controller import (
    L1_DELETE_BATCH_SIZE,
    L1EvictionController,
)
from tests.v1.distributed.utils import should_use_lazy_alloc

if not torch_dev.is_available():
    pytest.skip(
        f"Requires available {torch_device_type} runtime",
        allow_module_level=True,
    )

_LAYOUT = MemoryLayoutDesc(shapes=[torch.Size([16])], dtypes=[torch.bfloat16])


def _make_key(i: int) -> ObjectKey:
    return ObjectKey(chunk_hash=ObjectKey.IntHash2Bytes(i), model_name="m", kv_rank=0)


@pytest.fixture
def l1_manager() -> Iterator[L1Manager]:
    config = L1ManagerConfig(
        memory_config=L1MemoryManagerConfig(
            size_in_bytes=64 * 1024 * 1024,
            use_lazy=should_use_lazy_alloc(),
            init_size_in_bytes=64 * 1024 * 1024,
            align_bytes=0x1000,
        ),
        write_ttl_seconds=600,
        read_ttl_seconds=300,
    )
    mgr = L1Manager(config)
    yield mgr
    mgr.close()


@pytest.fixture
def controller(l1_manager: L1Manager) -> L1EvictionController:
    return L1EvictionController(l1_manager, EvictionConfig(eviction_policy="LRU"))


def _populate(l1_manager: L1Manager, count: int) -> list[ObjectKey]:
    keys = [_make_key(i) for i in range(count)]
    l1_manager.reserve_write(keys, [False] * count, _LAYOUT)
    l1_manager.finish_write(keys)
    return keys


def _run_action(
    controller: L1EvictionController, l1_manager: L1Manager, keys: list[ObjectKey]
) -> list[list[ObjectKey]]:
    """Execute a DISCARD action and return the key batches given to delete."""
    batches: list[list[ObjectKey]] = []
    real_delete = l1_manager.delete

    def spy(
        batch: list[ObjectKey], *args: object, **kwargs: object
    ) -> dict[ObjectKey, L1Error]:
        batches.append(list(batch))
        return real_delete(batch, *args, **kwargs)

    with patch.object(l1_manager, "delete", side_effect=spy):
        controller.execute_eviction_action(
            EvictionAction(destination=EvictionDestination.DISCARD, keys=keys)
        )
    return batches


def test_large_action_is_deleted_in_bounded_slices(
    controller: L1EvictionController, l1_manager: L1Manager
) -> None:
    count = 2 * L1_DELETE_BATCH_SIZE + 7
    keys = _populate(l1_manager, count)

    batches = _run_action(controller, l1_manager, keys)

    assert len(batches) == 3
    assert all(len(b) <= L1_DELETE_BATCH_SIZE for b in batches)
    # Same keys in the same order as the action, none dropped or duplicated.
    assert [k for b in batches for k in b] == keys
    for key in keys:
        assert l1_manager.reserve_read([key])[key][0] == L1Error.KEY_NOT_EXIST


def test_small_action_is_a_single_delete(
    controller: L1EvictionController, l1_manager: L1Manager
) -> None:
    keys = _populate(l1_manager, 5)

    batches = _run_action(controller, l1_manager, keys)

    assert batches == [keys]
    for key in keys:
        assert l1_manager.reserve_read([key])[key][0] == L1Error.KEY_NOT_EXIST


def test_empty_action_does_not_call_delete(
    controller: L1EvictionController, l1_manager: L1Manager
) -> None:
    assert _run_action(controller, l1_manager, []) == []


def test_locked_key_in_one_slice_does_not_block_other_slices(
    controller: L1EvictionController, l1_manager: L1Manager
) -> None:
    keys = _populate(l1_manager, L1_DELETE_BATCH_SIZE + 10)
    locked = keys[0]
    l1_manager.reserve_read([locked])

    _run_action(controller, l1_manager, keys)

    assert l1_manager.is_key_evictable(locked) is False
    assert l1_manager.get_object_state(locked) is not None
    for key in keys[1:]:
        assert l1_manager.reserve_read([key])[key][0] == L1Error.KEY_NOT_EXIST


def test_unsupported_destination_is_treated_as_discard(
    controller: L1EvictionController, l1_manager: L1Manager
) -> None:
    count = L1_DELETE_BATCH_SIZE + 1
    keys = _populate(l1_manager, count)
    destination = next(
        d for d in EvictionDestination if d != EvictionDestination.DISCARD
    )

    with patch.object(l1_manager, "delete", wraps=l1_manager.delete) as delete:
        controller.execute_eviction_action(
            EvictionAction(destination=destination, keys=keys)
        )

    assert delete.call_count == 2
    for key in keys:
        assert l1_manager.reserve_read([key])[key][0] == L1Error.KEY_NOT_EXIST
