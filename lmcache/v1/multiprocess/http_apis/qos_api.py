# SPDX-License-Identifier: Apache-2.0
"""Management endpoints for weighted L2 request scheduling."""

# Standard
from typing import Protocol, cast

# Third Party
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

# First Party
from lmcache.v1.multiprocess.qos import CacheSaltQosManager

router = APIRouter()

# The empty cache salt cannot occupy a path segment, so the API maps it to this
# sentinel. A literal ``cache_salt="_default"`` therefore cannot be managed
# separately through these path-based endpoints.
_DEFAULT_SALT_SENTINEL = "_default"


class _StorageManagerLike(Protocol):
    @property
    def cache_salt_qos_manager(self) -> CacheSaltQosManager: ...

    @property
    def l2_qos_enabled(self) -> bool: ...


class _EngineLike(Protocol):
    storage_manager: _StorageManagerLike


def _unescape_salt(path_salt: str) -> str:
    """Translate the URL sentinel to the empty cache salt."""
    return "" if path_salt == _DEFAULT_SALT_SENTINEL else path_salt


def _escape_salt(cache_salt: str) -> str:
    """Translate the empty cache salt to its URL sentinel."""
    return _DEFAULT_SALT_SENTINEL if cache_salt == "" else cache_salt


def _get_storage_manager(request: Request) -> _StorageManagerLike:
    """Return the live storage manager or raise a startup-race response."""
    engine = getattr(request.app.state, "engine", None)
    if engine is None:
        raise HTTPException(
            status_code=503,
            detail="engine not initialized",
        )
    return cast(_EngineLike, engine).storage_manager


async def _read_sched_weight(request: Request) -> int | JSONResponse:
    """Parse the public ``sched_weight`` scheduling value from the request body."""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "invalid JSON body"})
    if not isinstance(body, dict) or "sched_weight" not in body:
        return JSONResponse(
            status_code=400,
            content={"error": "body must be {'sched_weight': <integer>}"},
        )
    weight = body["sched_weight"]
    if isinstance(weight, bool) or not isinstance(weight, int):
        return JSONResponse(
            status_code=400,
            content={"error": "sched_weight must be an integer"},
        )
    return weight


@router.get("/qos/config", response_model=None)
async def get_qos_config(request: Request) -> dict[str, object] | JSONResponse:
    """Return the default scheduling weight and scheduler state."""
    storage_manager = _get_storage_manager(request)
    return {
        "enabled": storage_manager.l2_qos_enabled,
        "default_sched_weight": (
            storage_manager.cache_salt_qos_manager.default_sched_weight
        ),
    }


@router.put("/qos/config", response_model=None)
async def set_qos_config(request: Request) -> dict[str, object] | JSONResponse:
    """Update the default scheduling weight for unregistered cache salts."""
    storage_manager = _get_storage_manager(request)
    weight = await _read_sched_weight(request)
    if isinstance(weight, JSONResponse):
        return weight
    try:
        storage_manager.cache_salt_qos_manager.set_default_sched_weight(weight)
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})
    return {
        "enabled": storage_manager.l2_qos_enabled,
        "default_sched_weight": weight,
        "status": "ok",
    }


@router.get("/qos/cache-salt", response_model=None)
async def list_cache_salt_sched_weights(
    request: Request,
) -> dict[str, object] | JSONResponse:
    """List every explicitly registered cache-salt scheduling weight."""
    storage_manager = _get_storage_manager(request)
    manager = storage_manager.cache_salt_qos_manager
    return {
        "enabled": storage_manager.l2_qos_enabled,
        "default_sched_weight": manager.default_sched_weight,
        "weights": {
            _escape_salt(cache_salt): weight
            for cache_salt, weight in manager.list_sched_weights().items()
        },
    }


@router.put("/qos/cache-salt/{cache_salt}", response_model=None)
async def set_cache_salt_sched_weight(
    cache_salt: str,
    request: Request,
) -> dict[str, object] | JSONResponse:
    """Create or update the scheduling weight for one cache salt."""
    storage_manager = _get_storage_manager(request)
    weight = await _read_sched_weight(request)
    if isinstance(weight, JSONResponse):
        return weight
    salt = _unescape_salt(cache_salt)
    try:
        storage_manager.cache_salt_qos_manager.set_sched_weight(salt, weight)
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})
    return {
        "cache_salt": _escape_salt(salt),
        "sched_weight": weight,
        "status": "ok",
    }


@router.get("/qos/cache-salt/{cache_salt}", response_model=None)
async def get_cache_salt_sched_weight(
    cache_salt: str,
    request: Request,
) -> dict[str, object] | JSONResponse:
    """Return the effective scheduling weight for one cache salt."""
    storage_manager = _get_storage_manager(request)
    salt = _unescape_salt(cache_salt)
    manager = storage_manager.cache_salt_qos_manager
    profile = manager.get_profile(salt)
    return {
        "cache_salt": _escape_salt(salt),
        "sched_weight": profile.weight,
        "source": profile.source,
        "exists": manager.has_sched_weight(salt),
    }


@router.delete("/qos/cache-salt/{cache_salt}", response_model=None)
async def delete_cache_salt_sched_weight(
    cache_salt: str,
    request: Request,
) -> dict[str, object] | JSONResponse:
    """Remove one explicit weight and restore the effective default."""
    storage_manager = _get_storage_manager(request)
    salt = _unescape_salt(cache_salt)
    manager = storage_manager.cache_salt_qos_manager
    removed = manager.delete_sched_weight(salt)
    return {
        "cache_salt": _escape_salt(salt),
        "sched_weight": manager.get_profile(salt).weight,
        "status": "removed" if removed else "not_found",
    }
