# SPDX-License-Identifier: Apache-2.0
"""SequenceTracker and parse_sequence_id contracts."""

# Standard
import threading

# Third Party
import pytest

# First Party
from lmcache.v1.multiprocess.sequence_tracker import (
    SEQUENCE_ID_HINT,
    SEQUENCE_ID_MAX_LEN,
    SequenceKey,
    SequenceTracker,
    SequenceTransition,
    TransitionKind,
    parse_sequence_id,
)


def _chain(*labels: str) -> list[bytes]:
    return [label.encode() for label in labels]


def _key(sequence_id: str, cache_salt: str = "") -> SequenceKey:
    return SequenceKey(
        model_name="m", world_size=1, cache_salt=cache_salt, sequence_id=sequence_id
    )


# ---------------------------------------------------------------------------
# Divergence computation
# ---------------------------------------------------------------------------


def test_first_request_of_a_sequence() -> None:
    tracker = SequenceTracker(1000)
    assert tracker.observe(_key("s"), "r1", _chain("a", "b")) == SequenceTransition(
        TransitionKind.FIRST, 0, 0, (), ()
    )


def test_pure_append_supersedes_nothing() -> None:
    tracker = SequenceTracker(1000)
    tracker.observe(_key("s"), "r1", _chain("a", "b"))
    assert tracker.observe(
        _key("s"), "r2", _chain("a", "b", "c", "d")
    ) == SequenceTransition(TransitionKind.APPEND, 2, 2, (), ())


def test_identical_prompt_is_an_append() -> None:
    tracker = SequenceTracker(1000)
    tracker.observe(_key("s"), "r1", _chain("a", "b"))
    transition = tracker.observe(_key("s"), "r2", _chain("a", "b"))
    assert transition.kind is TransitionKind.APPEND
    assert transition.superseded_hashes == ()


def test_mid_rewrite_supersedes_the_tail() -> None:
    tracker = SequenceTracker(1000)
    tracker.observe(_key("s"), "r1", _chain("a", "b", "c", "d"))
    assert tracker.observe(
        _key("s"), "r2", _chain("a", "b", "x", "y", "z")
    ) == SequenceTransition(TransitionKind.REWRITE, 2, 4, (b"c", b"d"), (b"c", b"d"))


def test_rewrite_at_chunk_zero_supersedes_everything() -> None:
    tracker = SequenceTracker(1000)
    tracker.observe(_key("s"), "r1", _chain("a", "b"))
    assert tracker.observe(_key("s"), "r2", _chain("x")) == SequenceTransition(
        TransitionKind.REWRITE, 0, 2, (b"a", b"b"), (b"a", b"b")
    )


def test_shorter_prefix_is_a_rewrite_of_the_dropped_tail() -> None:
    tracker = SequenceTracker(1000)
    tracker.observe(_key("s"), "r1", _chain("a", "b", "c"))
    assert tracker.observe(_key("s"), "r2", _chain("a")) == SequenceTransition(
        TransitionKind.REWRITE, 1, 3, (b"b", b"c"), (b"b", b"c")
    )


def test_each_request_is_compared_with_the_latest_one() -> None:
    tracker = SequenceTracker(1000)
    tracker.observe(_key("s"), "r1", _chain("a", "b", "c"))
    tracker.observe(_key("s"), "r2", _chain("a", "x"))
    transition = tracker.observe(_key("s"), "r3", _chain("a", "x", "y"))
    assert transition.kind is TransitionKind.APPEND
    assert transition.previous_chunks == 2


def test_repeat_lookup_of_the_same_request_supersedes_nothing() -> None:
    tracker = SequenceTracker(1000)
    tracker.observe(_key("s"), "r1", _chain("a", "b"))
    transition = tracker.observe(_key("s"), "r1", _chain("a"))
    assert transition.kind is TransitionKind.REPEAT
    assert transition.superseded_hashes == ()


def test_sequences_are_scoped_by_cache_salt() -> None:
    tracker = SequenceTracker(1000)
    tracker.observe(_key("s", cache_salt="t1"), "r1", _chain("a", "b"))
    transition = tracker.observe(_key("s", cache_salt="t2"), "r2", _chain("x"))
    assert transition.kind is TransitionKind.FIRST


# ---------------------------------------------------------------------------
# Shared-chain guard
# ---------------------------------------------------------------------------


def test_chunk_still_in_another_sequences_chain_is_not_unreferenced() -> None:
    tracker = SequenceTracker(1000)
    # Two trajectories share a prompt (e.g. GRPO siblings).
    tracker.observe(_key("A"), "a1", _chain("p", "q", "a"))
    tracker.observe(_key("B"), "b1", _chain("p", "q", "b"))

    transition = tracker.observe(_key("A"), "a2", _chain("p", "z"))

    assert transition.superseded_hashes == (b"q", b"a")
    assert transition.unreferenced_hashes == (b"a",)


def test_chunk_becomes_unreferenced_once_no_sequence_holds_it() -> None:
    tracker = SequenceTracker(1000)
    tracker.observe(_key("A"), "a1", _chain("p", "q"))
    tracker.observe(_key("B"), "b1", _chain("p", "q"))
    tracker.observe(_key("A"), "a2", _chain("x"))
    transition = tracker.observe(_key("B"), "b2", _chain("y"))
    assert transition.unreferenced_hashes == (b"p", b"q")


def test_forgotten_sequence_no_longer_protects_its_chunks() -> None:
    tracker = SequenceTracker(4)
    tracker.observe(_key("A"), "a1", _chain("p", "a"))
    tracker.observe(_key("B"), "b1", _chain("p", "b"))
    tracker.observe(_key("C"), "c1", _chain("c", "d"))  # forgets A
    transition = tracker.observe(_key("B"), "b2", _chain("x"))
    assert transition.unreferenced_hashes == (b"p", b"b")


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------


def test_budget_forgets_least_recently_observed_sequences() -> None:
    tracker = SequenceTracker(4)
    tracker.observe(_key("A"), "a1", _chain("a1", "a2"))
    tracker.observe(_key("B"), "b1", _chain("b1", "b2"))
    tracker.observe(_key("A"), "a2", _chain("a1", "a2"))  # A is now newest
    tracker.observe(_key("C"), "c1", _chain("c1", "c2"))  # forgets B

    assert tracker.num_tracked_sequences() == 2
    assert tracker.num_tracked_chunks() == 4
    assert tracker.observe(_key("A"), "a3", _chain("x")).kind is (
        TransitionKind.REWRITE
    )
    assert tracker.observe(_key("B"), "b2", _chain("x")).kind is TransitionKind.FIRST


def test_chain_longer_than_the_budget_is_not_tracked() -> None:
    tracker = SequenceTracker(2)
    tracker.observe(_key("A"), "a1", _chain("a", "b"))
    tracker.observe(_key("B"), "b1", _chain("x", "y", "z"))
    assert tracker.num_tracked_chunks() == 2
    assert tracker.observe(_key("B"), "b2", _chain("x")).kind is TransitionKind.FIRST


def test_empty_chain_forgets_the_sequence() -> None:
    tracker = SequenceTracker(1000)
    tracker.observe(_key("A"), "a1", _chain("a"))
    tracker.observe(_key("A"), "a2", [])
    assert tracker.num_tracked_sequences() == 0
    assert tracker.observe(_key("A"), "a3", _chain("a")).kind is TransitionKind.FIRST


@pytest.mark.parametrize("max_chunks", [0, -1])
def test_non_positive_budget_is_rejected(max_chunks: int) -> None:
    with pytest.raises(ValueError):
        SequenceTracker(max_chunks)


def test_concurrent_observers_keep_the_budget_consistent() -> None:
    """Threads released together by a barrier race on one tracker; the final
    state is deterministic because each thread owns its sequence."""
    tracker = SequenceTracker(10_000)
    num_threads, rounds = 8, 200
    barrier = threading.Barrier(num_threads)

    def run(index: int) -> None:
        barrier.wait()
        for turn in range(rounds):
            chain = [f"{index}:{i}".encode() for i in range(1 + turn % 5)]
            tracker.observe(_key(f"s{index}"), f"r{index}-{turn}", chain)

    threads = [threading.Thread(target=run, args=(i,)) for i in range(num_threads)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    last_len = 1 + (rounds - 1) % 5
    assert tracker.num_tracked_sequences() == num_threads
    assert tracker.num_tracked_chunks() == num_threads * last_len


# ---------------------------------------------------------------------------
# Hint parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("configs", [None, {}, {"lmcache.skip_load": True}])
def test_absent_hint(configs: dict | None) -> None:
    assert parse_sequence_id(configs) is None


@pytest.mark.parametrize(
    "value", ["traj-17", "run.3:sample_2", "a" * SEQUENCE_ID_MAX_LEN]
)
def test_valid_hint(value: str) -> None:
    assert parse_sequence_id({SEQUENCE_ID_HINT: value}) == value


@pytest.mark.parametrize(
    "value",
    ["", "a" * (SEQUENCE_ID_MAX_LEN + 1), "has space", "slash/es", "ünï", 17, None],
)
def test_malformed_hint_raises_without_echoing_the_value(value: object) -> None:
    with pytest.raises(ValueError) as excinfo:
        parse_sequence_id({SEQUENCE_ID_HINT: value})
    if isinstance(value, str) and value:
        assert value not in str(excinfo.value)
