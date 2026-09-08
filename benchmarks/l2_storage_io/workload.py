# SPDX-License-Identifier: Apache-2.0
"""Deterministic synthetic KV request generation."""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass
import hashlib
import math
import random


@dataclass(frozen=True)
class WorkloadSpec:
    """Parameters for a synthetic chunk-oriented L2 workload."""

    operation: str = "mixed"
    requests: int = 32
    chunks_per_request: int = 8
    chunk_bytes: int = 1024 * 1024
    working_set_chunks: int = 128
    prefix_reuse: float = 0.8
    hot_set_fraction: float = 0.25
    read_ratio: float = 0.5
    warmup_requests: int = 1
    seed: int = 20260908

    def __post_init__(self) -> None:
        if self.operation not in {"read", "write", "mixed"}:
            raise ValueError("operation must be read, write, or mixed")
        for name in (
            "requests",
            "chunks_per_request",
            "chunk_bytes",
            "working_set_chunks",
            "warmup_requests",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.requests == 0 and self.warmup_requests == 0:
            raise ValueError("at least one request is required")
        if self.chunks_per_request == 0:
            raise ValueError("chunks_per_request must be positive")
        if self.operation == "mixed" and self.chunks_per_request < 2:
            raise ValueError("mixed workloads require chunks_per_request >= 2")
        if self.operation == "mixed" and not 0 < self.read_ratio < 1:
            raise ValueError(
                "mixed workloads require read_ratio strictly between 0 and 1"
            )
        if self.chunk_bytes == 0:
            raise ValueError("chunk_bytes must be positive")
        if self.operation in {"read", "mixed"} and self.working_set_chunks == 0:
            raise ValueError("read and mixed workloads require working_set_chunks >= 1")
        for name in ("prefix_reuse", "hot_set_fraction", "read_ratio"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be finite and in [0, 1]")


@dataclass(frozen=True)
class SyntheticChunk:
    """One logical KV chunk and its deterministic payload seed."""

    chunk_id: int
    payload_seed: bytes

    @property
    def key_hex(self) -> str:
        """Return the stable hash used as an ObjectKey chunk hash."""
        return self.payload_seed.hex()


@dataclass(frozen=True)
class SyntheticRequest:
    """One request with per-chunk read/write labels."""

    sequence: int
    operation: str
    chunks: tuple[SyntheticChunk, ...]
    read_indices: tuple[int, ...]
    write_indices: tuple[int, ...]
    warmup: bool = False


def _chunk(seed: int, chunk_id: int) -> SyntheticChunk:
    digest = hashlib.blake2b(
        f"lmcache-l2-synthetic:{seed}:{chunk_id}".encode(), digest_size=16
    ).digest()
    return SyntheticChunk(chunk_id=chunk_id, payload_seed=digest)


def build_workload(
    spec: WorkloadSpec,
) -> tuple[tuple[SyntheticChunk, ...], tuple[SyntheticRequest, ...]]:
    """Build deterministic prepopulation and measured requests.

    Prepopulation stores each working-set key exactly once and is excluded by
    the benchmark. Measured writes use new IDs; measured reads select a hot
    prefix or a deterministic tail prefix according to ``prefix_reuse``.
    """
    rng = random.Random(spec.seed)
    warm_chunks = tuple(
        _chunk(spec.seed, index) for index in range(spec.working_set_chunks)
    )
    requests: list[SyntheticRequest] = []
    next_write = spec.working_set_chunks
    hot_count = min(
        max(1, int(spec.working_set_chunks * spec.hot_set_fraction)),
        max(1, spec.working_set_chunks),
    )

    def read_chunks() -> list[SyntheticChunk]:
        width = min(spec.chunks_per_request, len(warm_chunks))
        if rng.random() <= spec.prefix_reuse:
            start = 0
        else:
            first_cold = min(hot_count, len(warm_chunks) - width)
            last_start = len(warm_chunks) - width
            start = rng.randint(first_cold, last_start)
        return [warm_chunks[start + index] for index in range(width)]

    for sequence in range(spec.warmup_requests):
        chunks = tuple(
            _chunk(spec.seed, next_write + offset)
            for offset in range(spec.chunks_per_request)
        )
        next_write += len(chunks)
        requests.append(
            SyntheticRequest(
                sequence, "write", chunks, (), tuple(range(len(chunks))), True
            )
        )
    for index in range(spec.requests):
        sequence = spec.warmup_requests + index
        if spec.operation == "read":
            chunks = tuple(read_chunks())
            requests.append(
                SyntheticRequest(
                    sequence, "read", chunks, tuple(range(len(chunks))), ()
                )
            )
            continue
        if spec.operation == "write":
            chunks = tuple(
                _chunk(spec.seed, next_write + offset)
                for offset in range(spec.chunks_per_request)
            )
            next_write += len(chunks)
            requests.append(
                SyntheticRequest(
                    sequence, "write", chunks, (), tuple(range(len(chunks)))
                )
            )
            continue
        read_count = round(spec.chunks_per_request * spec.read_ratio)
        read_count = max(1, min(read_count, spec.chunks_per_request - 1))
        reads = read_chunks()[:read_count]
        writes = tuple(
            _chunk(spec.seed, next_write + offset)
            for offset in range(spec.chunks_per_request - len(reads))
        )
        next_write += len(writes)
        chunks = tuple(reads) + writes
        requests.append(
            SyntheticRequest(
                sequence,
                "mixed",
                chunks,
                tuple(range(len(reads))),
                tuple(range(len(reads), len(chunks))),
            )
        )
    return warm_chunks, tuple(requests)
