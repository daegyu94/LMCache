# SPDX-License-Identifier: Apache-2.0
"""Bloom filter over chunk hashes, shared by the coordinator and MP servers.

The coordinator keeps one filter per instance from the key directory's L1
placements; a P2P requester uses its snapshot to skip lookup RPCs to peers
that hold none of a request's chunks. A false positive only costs the RPC the
digest would have saved, so the filter is a hint, never a correctness input.
"""

# Standard
from collections.abc import Iterable
from dataclasses import dataclass
import base64
import hashlib

# Third Party
import numpy as np

_BITS_PER_KEY = 14  # ~0.1% per key: a 16-chunk lookup tests 16 keys, ~1.6%
_NUM_HASHES = 10
_MIN_BITS = 64
_PROBES = np.arange(_NUM_HASHES, dtype=np.uint64)


def _positions(chunk_hashes: list[bytes], num_bits: int) -> np.ndarray:
    """Return the ``(len(chunk_hashes), _NUM_HASHES)`` bit positions.

    Chunk hashes are re-hashed because their algorithm is configurable and
    not every one yields uniform bytes. Kirsch-Mitzenmacher double hashing
    derives all probes from one 128-bit digest.
    """
    raw = b"".join(hashlib.blake2b(h, digest_size=16).digest() for h in chunk_hashes)
    halves = np.frombuffer(raw, dtype="<u8").reshape(-1, 2)
    h1 = halves[:, :1]
    h2 = halves[:, 1:] | np.uint64(1)
    # uint64 arithmetic wraps, which is fine for hashing.
    return (h1 + _PROBES * h2) % np.uint64(num_bits)


def _set_bits(bits: np.ndarray, positions: np.ndarray) -> None:
    """Set ``positions`` (flat, uint64) in the little-endian bit array ``bits``."""
    np.bitwise_or.at(
        bits,
        (positions >> np.uint64(3)).astype(np.intp),
        (np.uint8(1) << (positions & np.uint64(7)).astype(np.uint8)),
    )


@dataclass(frozen=True)
class BloomDigest:
    """An immutable Bloom filter snapshot over chunk hashes.

    Attributes:
        num_bits: Filter size in bits.
        bits: The filter, ``ceil(num_bits / 8)`` bytes, little-endian bit order.
    """

    num_bits: int
    bits: bytes

    @classmethod
    def build(cls, chunk_hashes: Iterable[bytes]) -> "BloomDigest":
        """Build a digest sized for ``chunk_hashes``.

        Args:
            chunk_hashes: The chunk hashes to insert.

        Returns:
            The digest.
        """
        hashes = list(chunk_hashes)
        bloom = BloomFilter(len(hashes))
        bloom.add_many(hashes)
        return bloom.snapshot()

    def might_contain(self, chunk_hash: bytes) -> bool:
        """Return ``False`` only if ``chunk_hash`` was certainly not inserted.

        Args:
            chunk_hash: The chunk hash to test.

        Returns:
            Whether the hash may be in the set.
        """
        return all(
            self.bits[pos >> 3] >> (pos & 7) & 1
            for pos in _positions([chunk_hash], self.num_bits)[0].tolist()
        )

    def might_contain_any(self, chunk_hashes: list[bytes]) -> bool:
        """Return ``False`` only if none of ``chunk_hashes`` was inserted.

        Tests every hash in one vectorized pass, which is what a lookup with
        many chunks needs.

        Args:
            chunk_hashes: The chunk hashes to test.

        Returns:
            Whether any hash may be in the set.
        """
        if not chunk_hashes:
            return False
        positions = _positions(chunk_hashes, self.num_bits)
        bits = np.frombuffer(self.bits, dtype=np.uint8)
        hit = (
            bits[(positions >> np.uint64(3)).astype(np.intp)]
            >> (positions & np.uint64(7)).astype(np.uint8)
        ) & 1
        return bool(hit.all(axis=1).any())

    def with_added(self, chunk_hashes: list[bytes]) -> "BloomDigest":
        """Return a copy of this digest with ``chunk_hashes`` inserted.

        A receiver applies a delta this way and swaps the result in, so
        readers of the old digest are never affected.

        Args:
            chunk_hashes: The chunk hashes to insert.

        Returns:
            The new digest.
        """
        bits = np.frombuffer(self.bits, dtype=np.uint8).copy()
        if chunk_hashes:
            _set_bits(bits, _positions(chunk_hashes, self.num_bits).ravel())
        return BloomDigest(num_bits=self.num_bits, bits=bits.tobytes())

    def to_json(self) -> dict[str, object]:
        """Encode for an HTTP response.

        Returns:
            ``{"num_bits": int, "bits": base64 str}``.
        """
        return {"num_bits": self.num_bits, "bits": base64.b64encode(self.bits).decode()}

    @classmethod
    def from_json(cls, data: dict[str, object]) -> "BloomDigest":
        """Decode :meth:`to_json` output.

        Args:
            data: The encoded digest.

        Returns:
            The digest.

        Raises:
            ValueError: If the payload is malformed.
        """
        num_bits = data.get("num_bits")
        bits = data.get("bits")
        if not isinstance(num_bits, int) or not isinstance(bits, str):
            raise ValueError("malformed Bloom digest")
        raw = base64.b64decode(bits)
        if len(raw) != (num_bits + 7) // 8:
            raise ValueError("Bloom digest size mismatch")
        return cls(num_bits=num_bits, bits=raw)


class BloomFilter:
    """A mutable, insert-only Bloom filter sized for ``capacity`` keys.

    There is no delete: the owner rebuilds a fresh filter when enough keys
    have left that stale bits matter. Not thread-safe; the owner serializes
    access.

    Args:
        capacity: Number of keys the filter is sized for. Inserting more
            raises the false-positive rate above the target.
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.num_bits = max(_MIN_BITS, _BITS_PER_KEY * capacity)
        self._bits = np.zeros((self.num_bits + 7) // 8, dtype=np.uint8)

    def add_many(self, chunk_hashes: list[bytes]) -> None:
        """Insert ``chunk_hashes``.

        Args:
            chunk_hashes: The chunk hashes to insert.
        """
        if not chunk_hashes:
            return
        _set_bits(self._bits, _positions(chunk_hashes, self.num_bits).ravel())

    def snapshot(self) -> BloomDigest:
        """Return an immutable copy of the current bits.

        Returns:
            The digest.
        """
        return BloomDigest(num_bits=self.num_bits, bits=self._bits.tobytes())
