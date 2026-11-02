# SPDX-License-Identifier: Apache-2.0
"""The LMCache-driven STORE refuses KV the server's version gate rejects.

Only the native completion dispatcher and device module are replaced; the
store handler is called through its public request-handler method.
"""

# Standard
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.multiprocess import native_completion
from lmcache.v1.multiprocess.modules import lmcache_driven_transfer as transfer


@pytest.mark.parametrize("allowed", [False, True])
def test_store_commits_nothing_when_the_version_gate_refuses(
    monkeypatch: pytest.MonkeyPatch, allowed: bool
) -> None:
    monkeypatch.setattr(
        native_completion.DeviceHostFuncDispatcher, "start", lambda self: None
    )
    monkeypatch.setattr(transfer, "torch_dev", MagicMock())
    keys = [ObjectKey(bytes([i]) * 8, "model", 0) for i in range(2)]
    ctx = MagicMock()
    ctx.chunk_size = 1
    ctx.resolve_obj_keys.return_value = [keys]
    ctx.kv_version_allows_store.return_value = allowed
    module = transfer.LMCacheDrivenTransferModule(cast(Any, ctx))
    event_backend = MagicMock()
    cache_context = MagicMock()
    cache_context.kv_layer_groups_manager.num_object_groups = 1
    # Short block ids end an allowed store early, before any device copy.
    cache_context.calculate_num_blocks.return_value = 4
    entry = SimpleNamespace(
        cache_context=cache_context, model_name="model", event_backend=event_backend
    )
    monkeypatch.setattr(module, "get_and_touch_context_entry", lambda _: entry)
    key = SimpleNamespace(request_id="req", worker_id=0)
    try:
        handle, ok = module.store(cast(Any, key), 1, [[1]], b"producer")
    finally:
        module.close()

    ctx.kv_version_allows_store.assert_called_once_with(key)
    if allowed:
        # The store went on to its normal checks (here: block id underflow).
        assert ok is False
        event_backend.create_event.assert_called_once()
    else:
        assert (handle, ok) == (b"", True)
        event_backend.create_event.assert_not_called()
        event_backend.import_event.assert_not_called()
        ctx.storage_manager.reserve_write.assert_not_called()
