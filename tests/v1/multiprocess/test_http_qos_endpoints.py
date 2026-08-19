# SPDX-License-Identifier: Apache-2.0
"""Tests for cache-salt QoS HTTP endpoints."""

# Standard
from collections.abc import Generator
from unittest.mock import MagicMock

# Third Party
from fastapi.testclient import TestClient
import pytest

# First Party
from lmcache.v1.multiprocess.http_server import app
from lmcache.v1.multiprocess.qos import CacheSaltQosManager


@pytest.fixture
def client() -> Generator[tuple[TestClient, MagicMock], None, None]:
    """Return a client with a fresh in-memory QoS registry."""
    engine = MagicMock()
    engine.storage_manager.cache_salt_qos_manager = CacheSaltQosManager()
    engine.storage_manager.l2_qos_enabled = True
    app.state.engine = engine
    test_client = TestClient(app)
    yield test_client, engine
    if hasattr(app.state, "engine"):
        delattr(app.state, "engine")


def test_unknown_salt_uses_default_then_accepts_runtime_registration(
    client: tuple[TestClient, MagicMock],
) -> None:
    """An observed salt does not require provisioning before first use."""
    test_client, _ = client

    initial = test_client.get("/qos/cache-salt/tenant-a")
    updated = test_client.put(
        "/qos/cache-salt/tenant-a",
        json={"sched_weight": 700},
    )
    effective = test_client.get("/qos/cache-salt/tenant-a")

    assert initial.json() == {
        "cache_salt": "tenant-a",
        "sched_weight": 100,
        "source": "default",
        "exists": False,
    }
    assert updated.status_code == 200
    assert effective.json()["sched_weight"] == 700
    assert effective.json()["source"] == "explicit"
    assert effective.json()["exists"] is True


@pytest.mark.parametrize("weight", [0, 10001, 1.5, True, "700"])
def test_put_rejects_invalid_sched_weight(
    client: tuple[TestClient, MagicMock],
    weight: object,
) -> None:
    """The API rejects values outside the integer weight contract."""
    test_client, _ = client

    response = test_client.put(
        "/qos/cache-salt/tenant-a",
        json={"sched_weight": weight},
    )

    assert response.status_code == 400
    assert "sched_weight" in response.json()["error"]


def test_default_weight_and_empty_salt_sentinel(
    client: tuple[TestClient, MagicMock],
) -> None:
    """Config updates affect unknown salts and _default maps to empty salt."""
    test_client, engine = client

    config_response = test_client.put("/qos/config", json={"sched_weight": 250})
    salt_response = test_client.put(
        "/qos/cache-salt/_default",
        json={"sched_weight": 600},
    )

    assert config_response.json()["default_sched_weight"] == 250
    assert salt_response.json()["cache_salt"] == "_default"
    manager = engine.storage_manager.cache_salt_qos_manager
    assert manager.get_profile("").weight == 600


def test_list_and_delete_explicit_weights(
    client: tuple[TestClient, MagicMock],
) -> None:
    """List exposes explicit entries and delete restores the default."""
    test_client, _ = client
    test_client.put("/qos/cache-salt/bob", json={"sched_weight": 300})
    test_client.put("/qos/cache-salt/alice", json={"sched_weight": 500})

    listed = test_client.get("/qos/cache-salt")
    deleted = test_client.delete("/qos/cache-salt/alice")
    effective = test_client.get("/qos/cache-salt/alice")

    assert listed.json()["weights"] == {"alice": 500, "bob": 300}
    assert deleted.json()["status"] == "removed"
    assert effective.json()["sched_weight"] == 100
    assert effective.json()["source"] == "default"


def test_qos_endpoint_returns_503_before_engine_startup() -> None:
    """Management requests fail clearly when the MP engine is unavailable."""
    if hasattr(app.state, "engine"):
        delattr(app.state, "engine")

    response = TestClient(app).get("/qos/config")

    assert response.status_code == 503
