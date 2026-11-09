# SPDX-License-Identifier: Apache-2.0
"""Connector initialization rejects unsupported auto-start deployments early."""

# Standard
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, PropertyMock

# Third Party
import pytest

# First Party
from tests.v1.mp_scheduler_adapter_test_utils import (
    make_request_client,
    make_scheduler_adapter,
)


@pytest.fixture
def connector_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Isolate engine geometry and network adapters, not connector URL parsing."""
    pytest.importorskip("vllm")
    # First Party
    from lmcache.integration.vllm import lmcache_mp_connector as module

    monkeypatch.setattr(
        module.KVConnectorBase_V1, "__init__", MagicMock(return_value=None)
    )
    for name in (
        "validate_mamba_step_alignment",
        "validate_kv_cache_groups",
        "validate_dcp_support",
    ):
        monkeypatch.setattr(module, name, MagicMock())
    monkeypatch.setattr(module, "get_group_tokens_per_block", lambda *args: [16])
    monkeypatch.setattr(module, "get_vllm_scheduler_block_size", lambda *args: 16)
    monkeypatch.setattr(
        module, "get_dcp_decorated_model_name", lambda *args: "test-model"
    )
    monkeypatch.setattr(
        module,
        "build_parallel_strategy_from_vllm_config",
        lambda *args: SimpleNamespace(
            dcp_size=1, vllm_world_size=2, vllm_worker_id=0, kv_tp_size=2
        ),
    )
    monkeypatch.setattr(
        module,
        "LMCacheMPSchedulerAdapter",
        MagicMock(
            return_value=SimpleNamespace(
                lmcache_tokens_per_chunk=256, experimental=set()
            )
        ),
    )
    monkeypatch.setattr(module, "LMCacheMPWorkerAdapter", MagicMock())
    monkeypatch.setattr(
        module.LMCacheMPConnector,
        "transfer_intermediate_tensors",
        PropertyMock(return_value=False),
    )
    return module


@pytest.mark.parametrize("role_name", ["SCHEDULER", "WORKER"])
@pytest.mark.parametrize("enabled", [True, "true", False, "false"])
@pytest.mark.parametrize(
    "urls", [["localhost:5555", "localhost:5556"], "localhost:5555, localhost:5556"]
)
def test_connector_multiserver_autostart_fails_before_network(
    connector_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    role_name: str,
    enabled: bool | str,
    urls: list[str] | str,
) -> None:
    """Both roles reject auto-start, but retain connect-only multi-server mode."""
    module = connector_module
    role = getattr(module.KVConnectorRole, role_name)
    monkeypatch.setattr(
        module.LMCacheMPConnector, "role", PropertyMock(return_value=role)
    )
    extra = {"lmcache.mp.server_urls": urls, "lmcache.mp.autostart": enabled}
    config = SimpleNamespace(
        cache_config=SimpleNamespace(mamba_cache_mode="none"),
        kv_transfer_config=SimpleNamespace(
            kv_connector_extra_config=extra,
            get_from_extra_config=extra.get,
            is_kv_producer=True,
        ),
        parallel_config=SimpleNamespace(
            world_size=2, data_parallel_size=1, pipeline_parallel_size=1
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=256),
    )
    if enabled in (True, "true"):
        with pytest.raises(ValueError, match="only supports a single server"):
            module.LMCacheMPConnector(config, role)
        module.LMCacheMPSchedulerAdapter.assert_not_called()
        module.LMCacheMPWorkerAdapter.assert_not_called()
        module.validate_dcp_support.assert_not_called()
    else:
        module.LMCacheMPConnector(config, role)
        adapter = (
            module.LMCacheMPSchedulerAdapter
            if role_name == "SCHEDULER"
            else module.LMCacheMPWorkerAdapter
        )
        adapter.assert_called_once()


@pytest.mark.parametrize("role_name", ["SCHEDULER", "WORKER"])
@pytest.mark.parametrize("explicit_url", [True, False])
def test_connector_single_server_autostart_keeps_resolved_endpoint(
    connector_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    role_name: str,
    explicit_url: bool,
) -> None:
    """Single-server auto-start reaches the adapter using connector URL precedence."""
    module = connector_module
    role = getattr(module.KVConnectorRole, role_name)
    monkeypatch.setattr(
        module.LMCacheMPConnector, "role", PropertyMock(return_value=role)
    )
    extra: dict[str, object] = {
        "lmcache.mp.autostart": True,
        "lmcache.mp.host": "127.0.0.1",
        "lmcache.mp.port": 5555,
    }
    if explicit_url:
        extra["lmcache.mp.server_urls"] = ["localhost:6000"]
    config = SimpleNamespace(
        cache_config=SimpleNamespace(mamba_cache_mode="none"),
        kv_transfer_config=SimpleNamespace(
            kv_connector_extra_config=extra,
            get_from_extra_config=extra.get,
            is_kv_producer=True,
        ),
        parallel_config=SimpleNamespace(
            world_size=2, data_parallel_size=1, pipeline_parallel_size=1
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=256),
    )
    module.LMCacheMPConnector(config, role)
    expected = "tcp://localhost:6000" if explicit_url else "tcp://127.0.0.1:5555"
    if role_name == "SCHEDULER":
        assert module.LMCacheMPSchedulerAdapter.call_args.kwargs["server_urls"] == [
            expected
        ]
    else:
        assert module.LMCacheMPWorkerAdapter.call_args.kwargs["server_url"] == expected


def _scheduler_config(eager: bool, max_num_seqs: int) -> SimpleNamespace:
    """Single-server, non-autostart scheduler config for lookup-bound tests."""
    extra: dict[str, object] = {
        "lmcache.mp.host": "127.0.0.1",
        "lmcache.mp.port": 5555,
        "lmcache.mp.eager_prefetch": eager,
    }
    return SimpleNamespace(
        cache_config=SimpleNamespace(mamba_cache_mode="none"),
        kv_transfer_config=SimpleNamespace(
            kv_connector_extra_config=extra,
            get_from_extra_config=extra.get,
            is_kv_producer=True,
        ),
        parallel_config=SimpleNamespace(
            world_size=2, data_parallel_size=1, pipeline_parallel_size=1
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs),
    )


@pytest.mark.parametrize(("eager", "expected"), [(False, 128), (True, None)])
def test_scheduler_lookup_bound_follows_max_num_seqs_unless_eager(
    connector_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    eager: bool,
    expected: int | None,
) -> None:
    """Lookups are bounded by max_num_seqs, except under eager prefetch."""
    module = connector_module
    monkeypatch.setattr(
        module.LMCacheMPConnector,
        "role",
        PropertyMock(return_value=module.KVConnectorRole.SCHEDULER),
    )
    module.LMCacheMPConnector(
        _scheduler_config(eager, max_num_seqs=128), module.KVConnectorRole.SCHEDULER
    )

    kwargs = module.LMCacheMPSchedulerAdapter.call_args.kwargs
    assert kwargs["max_pending_lookups"] == expected


def test_connector_defers_lookup_until_capacity_frees(
    connector_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """At the bound the connector defers, then submits once capacity frees."""
    # Third Party
    from vllm.v1.request import RequestStatus

    module = connector_module
    monkeypatch.setattr(
        module.LMCacheMPConnector,
        "role",
        PropertyMock(return_value=module.KVConnectorRole.SCHEDULER),
    )
    adapter = MagicMock(name="scheduler_adapter")
    adapter.lmcache_tokens_per_chunk = 16
    adapter.experimental = set()
    adapter.can_submit_lookup.return_value = False
    adapter.check_lookup_result.return_value = None
    module.LMCacheMPSchedulerAdapter.return_value = adapter
    connector = module.LMCacheMPConnector(
        _scheduler_config(eager=False, max_num_seqs=128),
        module.KVConnectorRole.SCHEDULER,
    )
    tokens = list(range(64))
    request = SimpleNamespace(
        request_id="req-1",
        status=RequestStatus.WAITING,
        cache_salt="",
        lora_request=None,
        prompt_token_ids=tokens,
        num_prompt_tokens=len(tokens),
        all_token_ids=tokens,
        num_computed_tokens=0,
        mm_features=[],
        sampling_params=None,
    )

    assert connector.get_num_new_matched_tokens(request, 0) == (None, True)
    adapter.maybe_submit_lookup_request.assert_not_called()

    adapter.can_submit_lookup.return_value = True
    assert connector.get_num_new_matched_tokens(request, 0) == (None, True)
    adapter.maybe_submit_lookup_request.assert_called_once()


def test_connector_sends_at_most_bound_lookups_through_real_adapter(
    connector_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Polling a burst sends LOOKUPs only up to the bound, then one per free slot."""
    # Third Party
    from vllm.v1.request import RequestStatus

    module = connector_module
    monkeypatch.setattr(
        module.LMCacheMPConnector,
        "role",
        PropertyMock(return_value=module.KVConnectorRole.SCHEDULER),
    )
    client = make_request_client(chunk_size=256)
    adapter = make_scheduler_adapter(
        monkeypatch, {"server-a": client}, 16, max_pending_lookups=2
    )
    module.LMCacheMPSchedulerAdapter.return_value = adapter
    connector = module.LMCacheMPConnector(
        _scheduler_config(eager=False, max_num_seqs=128),
        module.KVConnectorRole.SCHEDULER,
    )
    tokens = list(range(512))
    requests = [
        SimpleNamespace(
            request_id=f"req-{i}",
            status=RequestStatus.WAITING,
            cache_salt="",
            lora_request=None,
            prompt_token_ids=tokens,
            num_prompt_tokens=len(tokens),
            all_token_ids=tokens,
            num_computed_tokens=0,
            mm_features=[],
            sampling_params=None,
        )
        for i in range(5)
    ]

    # Two scheduler steps over the whole queue: lookups are still in flight,
    # so every poll defers and only the first two requests ever submit.
    for _ in range(2):
        for request in requests:
            assert connector.get_num_new_matched_tokens(request, 0) == (None, True)
    assert client.lookup.call_count == 2

    # Consuming one lookup frees one slot for the next deferred request.
    adapter.cleanup_lookup_result("req-0")
    for request in requests[1:]:
        connector.get_num_new_matched_tokens(request, 0)
    assert client.lookup.call_count == 3
