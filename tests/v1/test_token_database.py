# SPDX-License-Identifier: Apache-2.0
# Standard
from collections.abc import Callable
from types import ModuleType, SimpleNamespace
import hashlib
import os
import sys

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.token_database import ChunkedTokenDatabase, SegmentTokenDatabase
import lmcache.v1.token_database as token_database

# Local
from .utils import dumb_metadata, dumb_metadata_with_model_name, generate_tokens


@pytest.fixture
def isolated_vllm_hashes(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Provide deterministic vLLM hash initialization without importing vLLM."""
    modules = {}
    for name in (
        "vllm",
        "vllm.utils",
        "vllm.utils.hashing",
        "vllm.v1",
        "vllm.v1.core",
        "vllm.v1.core.kv_cache_utils",
    ):
        module = ModuleType(name)
        module.__path__ = []
        modules[name] = module
        monkeypatch.setitem(sys.modules, name, module)

    kv_utils = modules["vllm.v1.core.kv_cache_utils"]

    def sha256_hash(value: object) -> bytes:
        return hashlib.sha256(repr(value).encode()).digest()

    def get_hash_fn_by_name(name: str) -> Callable[[object], int | bytes]:
        if name == "builtin":
            return hash
        if name == "sha256":
            return sha256_hash
        raise ValueError(name)

    def init_none_hash(hash_func: Callable[[object], int | bytes]) -> None:
        monkeypatch.setattr(
            kv_utils, "NONE_HASH", hash_func("initial seed"), raising=False
        )

    monkeypatch.setattr(
        modules["vllm.utils.hashing"],
        "get_hash_fn_by_name",
        get_hash_fn_by_name,
        raising=False,
    )
    monkeypatch.setattr(
        modules["vllm.v1.core"], "kv_cache_utils", kv_utils, raising=False
    )
    monkeypatch.setattr(kv_utils, "init_none_hash", init_none_hash, raising=False)
    return kv_utils


@pytest.mark.parametrize("first_algorithm", ["sha256", "builtin"])
@pytest.mark.parametrize("tensor_input", [False, True])
def test_chunk_hashes_survive_another_database_initialization(
    isolated_vllm_hashes: ModuleType,
    first_algorithm: str,
    tensor_input: bool,
) -> None:
    """Constructing another hasher must not change an existing DB's keys."""
    config = LMCacheEngineConfig.from_defaults(
        chunk_size=4, pre_caching_hash_algorithm=first_algorithm
    )
    first = ChunkedTokenDatabase(config, dumb_metadata())
    tokens = torch.arange(8) if tensor_input else list(range(8))
    before = list(first.process_tokens(tokens=tokens))
    other_algorithm = "builtin" if first_algorithm == "sha256" else "sha256"
    other_config = LMCacheEngineConfig.from_defaults(
        chunk_size=4, pre_caching_hash_algorithm=other_algorithm
    )

    ChunkedTokenDatabase(other_config, dumb_metadata())

    assert list(first.process_tokens(tokens=tokens)) == before
    assert len(before) == 2


@pytest.mark.parametrize("first_algorithm", ["sha256", "builtin"])
def test_lazy_chunk_hashing_uses_the_database_seed(
    isolated_vllm_hashes: ModuleType, first_algorithm: str
) -> None:
    """A lazy iterator keeps its owning database's initialization seed."""
    config = LMCacheEngineConfig.from_defaults(
        chunk_size=4, pre_caching_hash_algorithm=first_algorithm
    )
    first = ChunkedTokenDatabase(config)
    before = list(first.process_tokens(tokens=list(range(8)), make_key=False))
    pending = first.process_tokens(tokens=list(range(8)), make_key=False)
    other_algorithm = "builtin" if first_algorithm == "sha256" else "sha256"
    other_config = LMCacheEngineConfig.from_defaults(
        chunk_size=4, pre_caching_hash_algorithm=other_algorithm
    )

    ChunkedTokenDatabase(other_config)

    assert list(pending) == before


@pytest.mark.parametrize("vllm_available", [False, True])
def test_fallback_initialization_does_not_reset_existing_hashes(
    isolated_vllm_hashes: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    vllm_available: bool,
) -> None:
    """The no-vLLM and old-vLLM fallbacks must not reset another DB's seed."""
    config = LMCacheEngineConfig.from_defaults(
        chunk_size=4, pre_caching_hash_algorithm="sha256"
    )
    first = ChunkedTokenDatabase(config)
    before = list(first.process_tokens(tokens=list(range(8)), make_key=False))
    if vllm_available:
        monkeypatch.delattr(isolated_vllm_hashes, "init_none_hash")
    else:
        monkeypatch.setitem(sys.modules, "vllm.v1.core", None)

    ChunkedTokenDatabase()

    assert list(first.process_tokens(tokens=list(range(8)), make_key=False)) == before


@pytest.mark.parametrize("first_algorithm", ["sha256", "builtin"])
def test_segment_hashes_survive_another_database_initialization(
    isolated_vllm_hashes: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    first_algorithm: str,
) -> None:
    """Independent segment hashes also retain their database's seed."""
    tokenizer = SimpleNamespace(encode=lambda text: [0, 99])
    monkeypatch.setattr(
        token_database.AutoTokenizer, "from_pretrained", lambda model: tokenizer
    )
    config = LMCacheEngineConfig.from_defaults(
        pre_caching_hash_algorithm=first_algorithm
    )
    first = SegmentTokenDatabase(config, dumb_metadata())
    tokens = [1, 2, 99, 3, 4]
    before = list(first.process_tokens(tokens=tokens))
    other_algorithm = "builtin" if first_algorithm == "sha256" else "sha256"
    other_config = LMCacheEngineConfig.from_defaults(
        pre_caching_hash_algorithm=other_algorithm
    )

    ChunkedTokenDatabase(other_config)

    assert list(first.process_tokens(tokens=tokens)) == before
    assert len(before) == 2


def hf_credentials_available() -> bool:
    token_env = os.getenv("HF_TOKEN")
    hf_home = os.getenv("HF_HOME")
    default_token_file = os.path.expanduser("~/.cache/huggingface/token")
    token_file = os.path.join(hf_home, "token") if hf_home else ""
    return bool(
        token_env or os.path.exists(default_token_file) or os.path.exists(token_file)
    )


@pytest.mark.parametrize("chunk_length", [16, 64, 256])
@pytest.mark.parametrize("save_unfull_chunk", [False, True])
def test_chunked_token_database(chunk_length, save_unfull_chunk):
    cfg = LMCacheEngineConfig.from_legacy(
        chunk_size=chunk_length, backend="cpu", save_unfull_chunk=save_unfull_chunk
    )
    metadata = dumb_metadata()

    test_length = 2500
    tokens = generate_tokens(test_length, "cpu")
    mask = torch.full([test_length], True, dtype=torch.bool, device="cpu")

    num_falses = [i * chunk_length for i in range(0, test_length // chunk_length)]

    db = ChunkedTokenDatabase(cfg, metadata)

    # Process without mask
    original_results = list(db.process_tokens(tokens=tokens))
    end = (
        test_length if save_unfull_chunk else (test_length - test_length % chunk_length)
    )
    for i in range(0, end, chunk_length):
        st, ed, key = original_results[i // chunk_length]
        assert st == i
        if save_unfull_chunk:
            assert ed == min(i + chunk_length, test_length)
        else:
            assert ed == i + chunk_length

    for i in range(0, test_length // chunk_length):
        mask[: num_falses[i]] = False
        new_results = list(db.process_tokens(tokens=tokens, mask=mask))
        assert len(new_results) == len(original_results) - i

        for j in range(len(new_results)):
            st, ed, key = new_results[j]
            assert st == original_results[j + i][0]
            assert ed == original_results[j + i][1]


@pytest.mark.parametrize("prefix_length", [0, 16, 64, 256])
@pytest.mark.parametrize("chunk_lengths", [[256, 512, 256], [1024, 512, 256]])
@pytest.mark.skipif(
    not hf_credentials_available(), reason="No Hugging Face credentials found"
)
def test_segment_token_database(prefix_length, chunk_lengths):
    cfg = LMCacheEngineConfig.from_legacy(blend_special_str=" # # ")
    metadata = dumb_metadata_with_model_name("facebook/opt-125m")

    db = SegmentTokenDatabase(cfg, metadata)
    sep_tokens = db.sep_tokens

    sys_length = 25
    query_length = 50
    sys_tokens = generate_tokens(sys_length, "cpu", fixed=True)
    query_tokens = generate_tokens(query_length, "cpu", fixed=True)

    token_chunks = []
    starts = [0]
    ends = [sys_length]
    sys_hash = db._hash_tokens(sys_tokens)
    hashes = [sys_hash]
    start = sys_length + len(sep_tokens)
    for idx, chunk_length in enumerate(chunk_lengths):
        token_chunk = generate_tokens(chunk_length, "cpu", fixed=True)

        token_hash = db._hash_tokens(token_chunk)
        hashes.append(token_hash)

        token_chunk = torch.cat([sep_tokens, token_chunk])
        token_chunks.append(token_chunk)
        starts.append(start)
        ends.append(start + chunk_length)
        start += chunk_length + len(sep_tokens)

    query_hash = db._hash_tokens(query_tokens)
    hashes.append(query_hash)
    starts.append(start)
    ends.append(start + query_length)

    tokens = torch.cat([sys_tokens, *token_chunks, sep_tokens, query_tokens])
    total_length = len(tokens)
    mask = torch.full([total_length], True, dtype=torch.bool, device="cpu")
    mask[:prefix_length] = False

    chunk_lists = [sys_tokens, *token_chunks, sep_tokens, query_tokens]
    skip_chunk_num = 0
    cum_length = 0
    for chunk in chunk_lists:
        if prefix_length > cum_length:
            skip_chunk_num += 1
        cum_length += len(chunk)

    starts = starts[skip_chunk_num:]
    ends = ends[skip_chunk_num:]
    hashes = hashes[skip_chunk_num:]

    original_results = list(db.process_tokens(tokens=tokens, mask=mask))
    for i in range(len(original_results)):
        st, ed, key = original_results[i]
        assert st == starts[i]
        assert ed == ends[i]
        assert key.chunk_hash == hashes[i]
        # print(st, starts[i])
        # print(ed, ends[i])


def test_process_tokens_returns_int_keys_for_bytes_hash_func() -> None:
    """process_tokens must produce int chunk_hash keys even when the underlying
    hash function returns bytes (e.g. sha256_cbor). This ensures downstream
    code such as msgpack serialisation always receives plain ints.
    """
    cfg = LMCacheEngineConfig.from_legacy(chunk_size=16, backend="cpu")
    metadata = dumb_metadata()
    db = ChunkedTokenDatabase(cfg, metadata)

    # Stub that returns bytes, mimicking sha256_cbor without requiring cbor2.
    db.hash_func = lambda x: hashlib.sha256(str(x).encode()).digest()

    tokens = generate_tokens(32, "cpu")
    results = list(db.process_tokens(tokens=tokens, make_key=False))

    assert len(results) > 0
    for _, _, hash_val in results:
        assert isinstance(hash_val, int), f"Expected int, got {type(hash_val)}"
        # Must fit in uint64 (msgpack range: 0 to 2**64 - 1)
        assert 0 <= hash_val <= 2**64 - 1
