# SPDX-License-Identifier: Apache-2.0
"""A request whose LMCache hit another request of the same engine is loading
waits for that load and then takes the prefix-cache hit instead of loading the
same KV again."""

# Standard
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock
import importlib.util

# Third Party
import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("vllm") is None,
    reason="drives the vLLM connector, which imports vLLM at module top",
)

CHUNK = 16
PROMPT = list(range(8 * CHUNK))
HIT = 6 * CHUNK


class _FakeRequest:
    """Duck-typed vLLM Request carrying only what the tracker reads."""

    def __init__(self, request_id: str, prompt: list[int], cache_salt: str = ""):
        # Third Party
        from vllm.v1.request import RequestStatus

        self.request_id = request_id
        self.status = RequestStatus.WAITING
        self.cache_salt = cache_salt
        self.prompt_token_ids = list(prompt)
        self.num_prompt_tokens = len(prompt)
        self.all_token_ids = list(prompt)
        self.num_computed_tokens = 0
        self.mm_features: list[Any] = []
        self.sampling_params = None


def _connector(hit: int = HIT):
    # First Party
    from lmcache.integration.vllm.lmcache_mp_connector import LMCacheMPConnector

    adapter = MagicMock()
    adapter.lmcache_tokens_per_chunk = CHUNK
    adapter.maybe_submit_lookup_request.return_value = True
    adapter.check_lookup_result.return_value = hit
    connector = LMCacheMPConnector.__new__(LMCacheMPConnector)
    connector.request_trackers = {}
    connector._loading_prefixes = {}
    connector._prefix_key_of = {}
    connector.scheduler_adapter = adapter
    connector._reserve_last_token_for_lookup = False
    connector._hit_alignment_tokens = CHUNK
    connector._prefix_replay_tokens = 0
    connector._connector_stats = MagicMock()
    connector.lazy_offload = False
    connector._mamba_relocation_window = 0
    return connector


def _start_load(connector, request, num_external_tokens: int = HIT) -> None:
    """vLLM allocated blocks for the request and committed to its load."""
    blocks = SimpleNamespace(get_block_ids=lambda: ([0] * (HIT // CHUNK),))
    connector.update_state_after_alloc(request, blocks, num_external_tokens)


def _loads_done(connector, *request_ids: str) -> None:
    connector.update_connector_output(
        SimpleNamespace(
            kv_cache_events=None,
            kv_connector_worker_meta=None,
            finished_recving=set(request_ids),
        )
    )


def test_sibling_waits_for_the_load_then_uses_the_prefix_cache():
    connector = _connector()
    first, second = _FakeRequest("a", PROMPT), _FakeRequest("b", PROMPT)

    assert connector.get_num_new_matched_tokens(first, 0) == (HIT, True)
    _start_load(connector, first)
    # Same hit while "a" loads it: ask again later.
    assert connector.get_num_new_matched_tokens(second, 0) == (None, True)
    assert connector.get_num_new_matched_tokens(second, 0) == (None, True)

    _loads_done(connector, "a")
    # The loaded blocks are now prefix-cache hits: nothing left to load.
    assert connector.get_num_new_matched_tokens(second, HIT) == (0, False)


def test_without_a_prefix_cache_hit_the_sibling_loads_after_the_wait():
    connector = _connector()
    first, second = _FakeRequest("a", PROMPT), _FakeRequest("b", PROMPT)
    connector.get_num_new_matched_tokens(first, 0)
    _start_load(connector, first)
    _loads_done(connector, "a")
    assert connector.get_num_new_matched_tokens(second, 0) == (HIT, True)


def test_a_load_vllm_did_not_commit_to_holds_nobody():
    """vLLM may decline or cap a hit after the connector reported it (no
    blocks, prompt logprobs, a longer local hit); only a committed load
    makes siblings wait."""
    connector = _connector()
    first, second = _FakeRequest("a", PROMPT), _FakeRequest("b", PROMPT)
    assert connector.get_num_new_matched_tokens(first, 0) == (HIT, True)
    assert connector.get_num_new_matched_tokens(second, 0) == (HIT, True)
    _start_load(connector, first, num_external_tokens=0)
    third = _FakeRequest("c", PROMPT)
    assert connector.get_num_new_matched_tokens(third, 0) == (HIT, True)


def test_different_prompt_or_salt_does_not_wait():
    connector = _connector()
    first = _FakeRequest("a", PROMPT)
    connector.get_num_new_matched_tokens(first, 0)
    _start_load(connector, first)
    other = [t + 1 for t in PROMPT]
    assert connector.get_num_new_matched_tokens(_FakeRequest("b", other), 0) == (
        HIT,
        True,
    )
    salted = _FakeRequest("c", PROMPT, cache_salt="tenant")
    assert connector.get_num_new_matched_tokens(salted, 0) == (HIT, True)


def test_the_loader_itself_is_never_held():
    connector = _connector()
    first = _FakeRequest("a", PROMPT)
    assert connector.get_num_new_matched_tokens(first, 0) == (HIT, True)
    assert connector.get_num_new_matched_tokens(first, 0) == (HIT, True)
