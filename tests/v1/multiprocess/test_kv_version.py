# SPDX-License-Identifier: Apache-2.0
"""Hint parsing, version windows and version-qualified hashes."""

# Standard
from collections.abc import Iterator
import logging

# Third Party
import pytest

# First Party
from lmcache.v1.multiprocess.kv_version import (
    KV_VERSION_CONFIG,
    MAX_KV_STALENESS_CONFIG,
    MAX_KV_STALENESS_LIMIT,
    KVVersionAssignment,
    KVVersionHintProblem,
    KVVersionHintWarner,
    KVVersionRun,
    KVVersionWindow,
    parse_kv_version_hints,
    versioned_chunk_hash,
)


@pytest.fixture
def warnings_logged() -> Iterator[list[logging.LogRecord]]:
    """Warnings of the kv_version logger (lmcache loggers do not propagate)."""
    records: list[logging.LogRecord] = []

    class _ListHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = _ListHandler(level=logging.WARNING)
    module_logger = logging.getLogger("lmcache.v1.multiprocess.kv_version")
    module_logger.addHandler(handler)
    try:
        yield records
    finally:
        module_logger.removeHandler(handler)


@pytest.mark.parametrize("configs", [None, {}, {"lmcache.skip_load": True}])
def test_no_hint_means_no_window(configs: dict[str, object] | None) -> None:
    hints = parse_kv_version_hints(configs)
    assert hints.window is None
    assert hints.problems == ()


def test_version_without_staleness_accepts_only_that_version() -> None:
    hints = parse_kv_version_hints({KV_VERSION_CONFIG: 7})
    assert hints.window == KVVersionWindow(version=7, max_staleness=0)
    assert hints.problems == ()


def test_window_bounds() -> None:
    window = parse_kv_version_hints(
        {KV_VERSION_CONFIG: 7, MAX_KV_STALENESS_CONFIG: 2}
    ).window
    assert window is not None
    assert window.oldest == 5
    assert [v for v in range(3, 10) if window.contains(v)] == [5, 6, 7]


def test_window_never_reaches_below_version_zero() -> None:
    window = KVVersionWindow(version=1, max_staleness=5)
    assert window.oldest == 0


@pytest.mark.parametrize("version", [-1, "3", 3.0, True, None, 1 << 63])
def test_malformed_version_disables_versioning(version: object) -> None:
    hints = parse_kv_version_hints(
        {KV_VERSION_CONFIG: version, MAX_KV_STALENESS_CONFIG: 1}
    )
    assert hints.window is None
    assert hints.problems == (KVVersionHintProblem.INVALID_VERSION,)


@pytest.mark.parametrize("staleness", [-1, "2", 2.5, True, None])
def test_malformed_staleness_narrows_to_exact_version(staleness: object) -> None:
    hints = parse_kv_version_hints(
        {KV_VERSION_CONFIG: 4, MAX_KV_STALENESS_CONFIG: staleness}
    )
    assert hints.window == KVVersionWindow(version=4, max_staleness=0)
    assert hints.problems == (KVVersionHintProblem.INVALID_STALENESS,)


def test_large_staleness_is_capped() -> None:
    hints = parse_kv_version_hints(
        {KV_VERSION_CONFIG: 100, MAX_KV_STALENESS_CONFIG: 10_000}
    )
    assert hints.window == KVVersionWindow(
        version=100, max_staleness=MAX_KV_STALENESS_LIMIT
    )
    assert hints.problems == (KVVersionHintProblem.STALENESS_CAPPED,)


def test_staleness_without_version_is_flagged_and_ignored() -> None:
    hints = parse_kv_version_hints({MAX_KV_STALENESS_CONFIG: 2})
    assert hints.window is None
    assert hints.problems == (KVVersionHintProblem.STALENESS_WITHOUT_VERSION,)


@pytest.mark.parametrize("hash_len", [8, 32])
def test_versioned_hash_isolates_versions_and_keeps_length(hash_len: int) -> None:
    base = bytes(range(hash_len))
    hashes = {versioned_chunk_hash(base, v) for v in range(4)}
    assert len(hashes) == 4
    assert base not in hashes
    assert all(len(h) == hash_len for h in hashes)
    # Deterministic: every server derives the same key.
    assert versioned_chunk_hash(base, 2) == versioned_chunk_hash(base, 2)


def test_assignment_keeps_served_versions_and_stamps_the_rest() -> None:
    window = KVVersionWindow(version=9, max_staleness=3)
    runs = (KVVersionRun(1, 3, 9), KVVersionRun(3, 4, 7))
    assignment = KVVersionAssignment.for_window(window, runs)

    # New KV computed on top of version-7 KV is stamped 7, not 9.
    assert assignment.write_version == 7
    assert [assignment.version_of(c) for c in range(6)] == [7, 9, 9, 7, 7, 7]
    hashes = [bytes([c]) * 8 for c in range(2, 5)]
    assert assignment.apply(hashes, start_chunk=2) == [
        versioned_chunk_hash(hashes[0], 9),
        versioned_chunk_hash(hashes[1], 7),
        versioned_chunk_hash(hashes[2], 7),
    ]


def test_assignment_without_runs_stamps_the_request_version() -> None:
    assignment = KVVersionAssignment.for_window(KVVersionWindow(5, 2), ())
    assert assignment.write_version == 5


def test_warner_rate_limits_and_never_logs_raw_values(
    warnings_logged: list[logging.LogRecord],
) -> None:
    warner = KVVersionHintWarner(interval_s=3600.0)
    secret = "client-secret-value"
    for _ in range(3):
        assert warner.parse({KV_VERSION_CONFIG: secret}) is None
    assert len(warnings_logged) == 1
    message = warnings_logged[0].getMessage()
    assert "lmcache.kv_version" in message
    assert secret not in message


def test_warner_reports_each_occurrence_without_rate_limit(
    warnings_logged: list[logging.LogRecord],
) -> None:
    warner = KVVersionHintWarner(interval_s=0.0)
    warner.warn((KVVersionHintProblem.INVALID_STALENESS,))
    warner.warn((KVVersionHintProblem.INVALID_STALENESS,))
    assert [
        r.getMessage().endswith("(1 request(s) since last report)")
        for r in (warnings_logged)
    ] == [True, True]
