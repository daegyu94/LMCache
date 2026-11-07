# SPDX-License-Identifier: Apache-2.0
"""Tests for vLLM requests that bypass prefix-cache reads."""

# Standard
from types import SimpleNamespace
from unittest.mock import MagicMock

# Third Party
import pytest

pytest.importorskip("vllm", reason="MP connector imports vLLM at module top")

# Third Party
from vllm.distributed.kv_transfer.kv_connector.v1.base import (  # noqa: E402
    KVConnectorRole,
)
from vllm.sampling_params import SamplingParams  # noqa: E402
from vllm.v1.request import Request  # noqa: E402

# First Party
from lmcache.integration.vllm.lmcache_mp_connector import (  # noqa: E402
    LMCacheMPConnector,
)
from lmcache.integration.vllm.lmcache_mp_metadata import (  # noqa: E402
    LMCacheMPRequestState,
)

pytestmark = pytest.mark.no_shared_allocator


def _request(request_id: str) -> Request:
    """Return a vLLM request whose prompt-logprob mode skips cache reads."""
    return Request(
        request_id,
        list(range(64)),
        SamplingParams(max_tokens=1, prompt_logprobs=1),
        None,
    )


def _connector(*, eager_prefetch: bool) -> LMCacheMPConnector:
    """Build a scheduler connector around a mocked MP scheduler adapter."""
    connector = LMCacheMPConnector.__new__(LMCacheMPConnector)
    connector._role = KVConnectorRole.SCHEDULER
    connector._eager_prefetch = eager_prefetch
    connector._reserve_last_token_for_lookup = False
    connector._can_store = True
    connector.lazy_offload = False
    connector._mamba_relocation_window = 0
    connector._group_tokens_per_block = [16]
    connector.scheduler_adapter = MagicMock()
    connector.request_trackers = {}
    return connector


def test_prompt_logprobs_request_skips_eager_prefetch() -> None:
    connector = _connector(eager_prefetch=True)
    request = _request("prompt-logprobs-eager")

    assert request.skip_reading_prefix_cache
    connector.on_new_request(request)

    connector.scheduler_adapter.maybe_submit_lookup_request.assert_not_called()
    assert connector.request_trackers == {}


def test_prompt_logprobs_request_skips_lookup_and_keeps_store_tracking() -> None:
    connector = _connector(eager_prefetch=False)
    request = _request("prompt-logprobs-lazy")

    assert request.skip_reading_prefix_cache
    assert connector.get_num_new_matched_tokens(request, 0) == (0, False)
    connector.scheduler_adapter.maybe_submit_lookup_request.assert_not_called()

    connector.update_state_after_alloc(
        request,
        SimpleNamespace(get_block_ids=lambda: ([1, 2, 3, 4],)),
        0,
    )

    assert connector.request_trackers[request.request_id].state == (
        LMCacheMPRequestState.READY
    )
    assert connector.request_trackers[request.request_id].allocated_block_ids == {
        0: [1, 2, 3, 4]
    }
    assert connector.request_finished(request, [1, 2, 3, 4])[0]
