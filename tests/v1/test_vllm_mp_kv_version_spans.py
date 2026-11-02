# SPDX-License-Identifier: Apache-2.0
"""The scheduler adapter's handling of ``lmcache.kv_version`` requests.

Servers are mocked request clients; the adapter is driven through its public
API only. No vLLM install or GPU is needed.
"""

# Standard
from typing import Any
from unittest.mock import MagicMock

# Third Party
import pytest

# First Party
from lmcache.integration.vllm.vllm_multi_process_adapter import (
    LMCacheMPSchedulerAdapter,
    merge_kv_version_spans,
)
from lmcache.v1.multiprocess.futures import MessagingFuture
from lmcache.v1.multiprocess.kv_version import (
    KV_VERSION_CAPABILITY,
    KV_VERSION_CONFIG,
    MAX_KV_STALENESS_CONFIG,
)
from lmcache.v1.multiprocess.token_codec import pack_token_ids
from tests.v1.mp_scheduler_adapter_test_utils import (
    make_request_client,
    make_scheduler_adapter,
    ready_future,
)

CHUNK = 16
TOKENS = pack_token_ids(list(range(4 * CHUNK)))
VERSIONED = {KV_VERSION_CONFIG: 9, MAX_KV_STALENESS_CONFIG: 2}


def test_merge_keeps_common_tokens_and_the_oldest_version() -> None:
    server_a = [(0, 32, 7), (32, 64, 6)]
    server_b = [(0, 16, 7), (16, 48, 5)]
    assert merge_kv_version_spans([server_a, server_b]) == [
        (0, 16, 7),
        (16, 48, 5),
    ]


def test_merge_coalesces_equal_neighbours_and_handles_misses() -> None:
    assert merge_kv_version_spans([[(0, 16, 3), (16, 32, 3)]]) == [(0, 32, 3)]
    assert merge_kv_version_spans([[(0, 16, 3)], []]) == []
    assert merge_kv_version_spans([]) == []


def _adapter(
    monkeypatch: pytest.MonkeyPatch,
    urls: list[str],
    capabilities: list[str] | None = None,
) -> tuple[LMCacheMPSchedulerAdapter, dict[str, MagicMock]]:
    clients = {url: make_request_client(CHUNK) for url in urls}
    for client in clients.values():
        client.get_experimental.return_value = ready_future(
            [KV_VERSION_CAPABILITY] if capabilities is None else capabilities
        )
    return make_scheduler_adapter(monkeypatch, clients, CHUNK), clients


def _resolve_lookup(
    adapter: LMCacheMPSchedulerAdapter,
    clients: dict[str, MagicMock],
    configs: dict[str, Any],
    hit_chunks: int = 3,
) -> int | None:
    assert adapter.maybe_submit_lookup_request("req", TOKENS, request_configs=configs)
    for client in clients.values():
        client.query_prefetch_status.return_value = ready_future(hit_chunks)
        client.lookup.return_value.set_result(None)
    return adapter.check_lookup_result("req")


def test_old_servers_do_not_support_versions(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter, _ = _adapter(monkeypatch, ["a", "b"], capabilities=[])
    assert not adapter.supports_kv_versions
    adapter, _ = _adapter(monkeypatch, ["a"])
    assert adapter.supports_kv_versions


def test_spans_are_requested_when_the_lookup_resolves_and_taken_without_waiting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, clients = _adapter(monkeypatch, ["a"])
    clients["a"].query_kv_version_spans.return_value = ready_future(
        [(0, 32, 9), (32, 48, 8)]
    )

    assert _resolve_lookup(adapter, clients, VERSIONED) == 3 * CHUNK
    clients["a"].query_kv_version_spans.assert_called_once_with("req")
    # vLLM's own prefix cache covered [0, 20).
    assert adapter.take_kv_version_spans("req", 20, 48) == [
        (20, 32, 9),
        (32, 48, 8),
    ]
    # Taken once.
    assert adapter.take_kv_version_spans("req", 20, 48) is None


def test_unversioned_requests_query_no_spans(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter, clients = _adapter(monkeypatch, ["a"])
    assert _resolve_lookup(adapter, clients, {}) == 3 * CHUNK
    clients["a"].query_kv_version_spans.assert_not_called()
    assert adapter.take_kv_version_spans("req", 0, 48) is None


def test_missing_or_unknown_replies_make_the_spans_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, clients = _adapter(monkeypatch, ["a"])
    clients["a"].query_kv_version_spans.return_value = MessagingFuture()
    _resolve_lookup(adapter, clients, VERSIONED)
    # The reply has not arrived: unknown, and nothing blocks.
    assert adapter.take_kv_version_spans("req", 0, 48) is None

    adapter, clients = _adapter(monkeypatch, ["a"])
    clients["a"].query_kv_version_spans.return_value = ready_future(None)
    _resolve_lookup(adapter, clients, VERSIONED)
    assert adapter.take_kv_version_spans("req", 0, 48) is None

    failed: MessagingFuture = MessagingFuture()
    failed.set_exception(RuntimeError("server error"))
    adapter, clients = _adapter(monkeypatch, ["a"])
    clients["a"].query_kv_version_spans.return_value = failed
    _resolve_lookup(adapter, clients, VERSIONED)
    assert adapter.take_kv_version_spans("req", 0, 48) is None


def test_several_servers_accept_only_the_exact_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Shards resolve versions independently, so a window could serve one
    token from different versions; with several servers it is narrowed."""
    adapter, clients = _adapter(monkeypatch, ["a", "b"])
    for client in clients.values():
        client.query_kv_version_spans.return_value = ready_future([(0, 48, 9)])

    assert _resolve_lookup(adapter, clients, VERSIONED) == 3 * CHUNK
    for client in clients.values():
        sent = client.lookup.call_args.args[0].request_configs
        assert sent[MAX_KV_STALENESS_CONFIG] == 0
        assert sent[KV_VERSION_CONFIG] == 9
    assert adapter.take_kv_version_spans("req", 0, 48) == [(0, 48, 9)]


def test_divergent_server_answers_report_the_oldest_common_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, clients = _adapter(monkeypatch, ["a", "b"])
    clients["a"].query_kv_version_spans.return_value = ready_future([(0, 48, 9)])
    clients["b"].query_kv_version_spans.return_value = ready_future(
        [(0, 16, 9), (16, 32, 8)]
    )
    _resolve_lookup(adapter, clients, VERSIONED)
    assert adapter.take_kv_version_spans("req", 0, 48) == [
        (0, 16, 9),
        (16, 32, 8),
    ]


def test_single_server_keeps_the_window(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter, clients = _adapter(monkeypatch, ["a"])
    _resolve_lookup(adapter, clients, VERSIONED)
    sent = clients["a"].lookup.call_args.args[0].request_configs
    assert sent[MAX_KV_STALENESS_CONFIG] == 2


def test_end_session_drops_pending_spans(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter, clients = _adapter(monkeypatch, ["a"])
    clients["a"].query_kv_version_spans.return_value = ready_future([(0, 48, 9)])
    _resolve_lookup(adapter, clients, VERSIONED)
    adapter.end_session("req")
    assert adapter.take_kv_version_spans("req", 0, 48) is None
