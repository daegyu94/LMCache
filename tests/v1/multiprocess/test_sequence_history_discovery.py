# SPDX-License-Identifier: Apache-2.0
"""The server discovers SequenceHistoryModule, builds it only when tracking
is configured, and builds it after the lookup module it observes."""

# Standard
from unittest.mock import MagicMock

# Third Party
import pytest

# First Party
from lmcache.v1.multiprocess import server as server_mod
from lmcache.v1.multiprocess.config import CoordinatorConfig, MPServerConfig
from lmcache.v1.multiprocess.engine_module import discover_modules
from lmcache.v1.multiprocess.modules.lookup import LookupModule
from lmcache.v1.multiprocess.modules.sequence_history import SequenceHistoryModule


def _build(max_chunks: int) -> list:
    return server_mod._build_modules(
        MagicMock(name="ctx"),
        MPServerConfig(
            worker_reap_timeout_seconds=0, sequence_tracker_max_chunks=max_chunks
        ),
        CoordinatorConfig(),
    )


def test_module_is_discovered() -> None:
    assert SequenceHistoryModule in discover_modules()


def test_module_is_not_built_without_a_budget(stub_module_resources) -> None:
    modules = _build(0)
    assert not any(isinstance(m, SequenceHistoryModule) for m in modules)
    stub_module_resources[SequenceHistoryModule].assert_not_called()


def test_module_is_built_after_lookup_and_registered_as_its_observer(
    monkeypatch: pytest.MonkeyPatch, stub_module_resources
) -> None:
    # The fixture stubs module constructors, so record the registration.
    add_observer = MagicMock()
    monkeypatch.setattr(LookupModule, "add_lookup_observer", add_observer)

    modules = _build(1024)

    [history] = [m for m in modules if isinstance(m, SequenceHistoryModule)]
    [lookup] = [m for m in modules if isinstance(m, LookupModule)]
    assert modules.index(lookup) < modules.index(history)
    add_observer.assert_called_once_with(history)
