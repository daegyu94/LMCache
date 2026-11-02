# SPDX-License-Identifier: Apache-2.0
"""The scheduler adapter reports which KV versions LMCache served a request.

Servers are mocked request clients; the adapter is driven through its public
API only. No vLLM install or GPU is needed.
"""

# Standard
from unittest.mock import MagicMock

# Third Party
import pytest

# First Party
from lmcache.integration.vllm.vllm_multi_process_adapter import (
    LMCacheMPSchedulerAdapter,
    merge_kv_version_spans,
)
from lmcache.v1.multiprocess.futures import MessagingFuture
from tests.v1.mp_scheduler_adapter_test_utils import (
    make_request_client,
    make_scheduler_adapter,
    ready_future,
)

CHUNK = 16


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
    monkeypatch: pytest.MonkeyPatch, answers: dict[str, MessagingFuture]
) -> tuple[LMCacheMPSchedulerAdapter, dict[str, MagicMock]]:
    clients = {url: make_request_client(CHUNK) for url in answers}
    for url, future in answers.items():
        clients[url].query_kv_version_spans.return_value = future
    return make_scheduler_adapter(monkeypatch, clients, CHUNK), clients


def test_spans_are_merged_across_servers_and_clipped_to_the_loaded_range(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, clients = _adapter(
        monkeypatch,
        {
            "a": ready_future([(0, 48, 9)]),
            "b": ready_future([(0, 32, 9), (32, 64, 8)]),
        },
    )
    # vLLM's own prefix cache covered [0, 20); LMCache's hit ends at 48.
    assert adapter.query_kv_version_spans("req", 20, 48) == [
        (20, 32, 9),
        (32, 48, 8),
    ]
    for client in clients.values():
        client.query_kv_version_spans.assert_called_once_with("req")


def test_nothing_loaded_reports_no_spans(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter, _ = _adapter(monkeypatch, {"a": ready_future([(0, 32, 4)])})
    assert adapter.query_kv_version_spans("req", 32, 32) == []


def test_unknown_session_makes_the_answer_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, _ = _adapter(
        monkeypatch, {"a": ready_future([(0, 32, 4)]), "b": ready_future(None)}
    )
    assert adapter.query_kv_version_spans("req", 0, 32) is None


def test_failed_server_makes_the_answer_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failed: MessagingFuture = MessagingFuture()
    failed.set_exception(RuntimeError("server error"))
    adapter, _ = _adapter(monkeypatch, {"a": failed})
    assert adapter.query_kv_version_spans("req", 0, 32) is None
