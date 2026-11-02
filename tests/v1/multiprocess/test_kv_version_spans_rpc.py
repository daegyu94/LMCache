# SPDX-License-Identifier: Apache-2.0
"""QUERY_KV_VERSION_SPANS round-trips over every request transport.

A stub module answers the RPC so the test covers the wire contract only:
spans, an empty answer, and the unknown (None) answer must stay distinct.
"""

# Standard
from typing import cast
import socket

# Third Party
import pytest

# First Party
from lmcache.v1.multiprocess.engine_context import MPCacheServerContext
from lmcache.v1.multiprocess.engine_module import EngineModule
from lmcache.v1.multiprocess.request_handler import HandlerType, request_handler
from lmcache.v1.multiprocess.transport.factory import RequestClientFactory
from lmcache.v1.multiprocess.transport.server_factory import create_request_server

# Test helpers
from tests.v1.multiprocess.transport_test_utils import (
    REQUEST_TRANSPORTS,
    RequestTransport,
    request_server_config,
    request_server_url,
)

_ANSWERS: dict[str, list[tuple[int, int, int]] | None] = {
    "mixed": [(0, 256, 9), (256, 512, 8)],
    "missed": [],
    "unfinished": None,
}


class _SpansModule(EngineModule):
    @property
    def context(self) -> MPCacheServerContext:
        """The wire-only test does not use an engine context."""
        raise NotImplementedError

    def report_status(self) -> dict:
        """Return the empty status of this test module."""
        return {}

    def close(self) -> None:
        """Release no resources for this test module."""

    @request_handler(HandlerType.BLOCKING)
    def query_kv_version_spans(
        self, request_id: str
    ) -> list[tuple[int, int, int]] | None:
        return _ANSWERS[request_id]


def _unused_tcp_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return cast(int, sock.getsockname()[1])


@pytest.mark.parametrize("request_transport", REQUEST_TRANSPORTS)
def test_spans_round_trip(request_transport: RequestTransport) -> None:
    server_url = request_server_url(request_transport, _unused_tcp_port())
    server = create_request_server(
        [_SpansModule()], request_server_config(request_transport, server_url)
    )
    server.start()
    client = RequestClientFactory.create(server_url)
    try:
        for request_id, expected in _ANSWERS.items():
            got = client.query_kv_version_spans(request_id).result(timeout=10)
            assert got == expected
            assert got is None if expected is None else got is not None
    finally:
        client.close()
        server.close()
