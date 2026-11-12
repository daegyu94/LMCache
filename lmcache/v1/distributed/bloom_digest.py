# SPDX-License-Identifier: Apache-2.0
"""Bloom filter over chunk hashes, shared by the coordinator and MP servers.

The coordinator builds one digest per instance from the key directory's L1
placements; a P2P requester uses it to skip lookup RPCs to peers that hold
none of a request's chunks. A false positive only costs the RPC the digest
would have saved, so the filter is a hint, never a correctness input.
"""

# Standard
from collections.abc import Iterable
from dataclasses import dataclass
import base64
import hashlib

# Third Party
import numpy as np

_BITS_PER_KEY = 10  # ~1% false positives with _NUM_HASHES
_NUM_HASHES = 7
_MIN_BITS = 64


def _positions(chunk_hash: bytes, num_bits: int) -> list[int]:
    """Return the bit positions of ``chunk_hash`` (Kirsch-Mitzenmacher)."""
    digest = hashlib.blake2b(chunk_hash, digest_size=16).digest()
    h1 = int.from_bytes(digest[:8], "little")
    h2 = int.from_bytes(digest[8:], "little") | 1
    return [(h1 + i * h2) % num_bits for i in range(_NUM_HASHES)]


@dataclass(frozen=True)
class BloomDigest:
    """An immutable Bloom filter over chunk hashes.

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
        num_bits = max(_MIN_BITS, _BITS_PER_KEY * len(hashes))
        array = np.zeros((num_bits + 7) // 8, dtype=np.uint8)
        for chunk_hash in hashes:
            for pos in _positions(chunk_hash, num_bits):
                array[pos >> 3] |= 1 << (pos & 7)
        return cls(num_bits=num_bits, bits=array.tobytes())

    def might_contain(self, chunk_hash: bytes) -> bool:
        """Return ``False`` only if ``chunk_hash`` was certainly not inserted.

        Args:
            chunk_hash: The chunk hash to test.

        Returns:
            Whether the hash may be in the set.
        """
        return all(
            self.bits[pos >> 3] >> (pos & 7) & 1
            for pos in _positions(chunk_hash, self.num_bits)
        )

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
