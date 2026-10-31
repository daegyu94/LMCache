# SPDX-License-Identifier: Apache-2.0
"""Canonical key encoding preserves persisted names and matching identity."""

# Standard
import itertools
import unittest

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.key_codec import (
    decode_fs_filename,
    decode_storage_key,
    encode_fs_filename,
    encode_storage_key,
    format_storage_key,
)


class TestCanonicalKeyEncoding(unittest.TestCase):
    """Check the fixed format against established bytes and generic templates."""

    def test_established_wire_bytes(self) -> None:
        """Unsalted and salted names keep rank padding and the group field."""
        for salt, suffix in [("", ""), ("tenant", "@tenant")]:
            with self.subTest(salt=salt):
                key = ObjectKey(b"\xde\xad\xbe\xef", "org/model", 42, 3, salt)
                self.assertEqual(
                    encode_storage_key(key), f"org/model@0000002a@3@deadbeef{suffix}"
                )

    def test_generic_format_equivalence_and_recovery(self) -> None:
        """Identity variations retain exactly the generic encoder's bytes."""
        cases = itertools.product(
            ["", "m", "org/model", "org//model/", "组织/模型", "model%20name"],
            [0, 42, 0xFFFFFFFF, 0x100000000],
            [0, 15, 256],
            [b"", b"\x00", bytes(range(32)), bytes(range(64))],
            ["", "tenant-01", "租户", "a" * 128],
        )
        for model, rank, group, chunk_hash, salt in cases:
            with self.subTest(model=model, rank=rank, group=group, salt=salt):
                key = ObjectKey(chunk_hash, model, rank, group, salt)
                encoded = encode_storage_key(key)
                self.assertEqual(
                    encoded,
                    format_storage_key(key, "{model}@{rank}@{group}@{hash}{suffix}"),
                )
                self.assertEqual(decode_storage_key(encoded), key)

    def test_operational_metadata_does_not_change_names(self) -> None:
        """Operational attributes stay outside the matching identity."""
        key = ObjectKey(b"\x00\x01", "model", 0, extra_attributes={"ttl": 10})
        encoded = encode_storage_key(key)
        key.extra_attributes["ttl"] = 20
        self.assertEqual(encode_storage_key(key), encoded)

    def test_custom_template_escaping_is_preserved(self) -> None:
        """Generic formatting still supports escaped models and custom specs."""
        key = ObjectKey(b"\xde\xad\xbe\xef", "org/model", 42, 3, "tenant")
        self.assertEqual(
            format_storage_key(
                key,
                "{quoted_model}|{model}|{rank:>10}|{group}|{hash_prefix}|{salt}",
                model_separator="--",
                model_suffix="@lg4",
            ),
            "org%2Fmodel|org--model@lg4|  0000002a|3|dead|tenant",
        )
        unsalted = ObjectKey(b"\xde\xad", "model", 0)
        self.assertEqual(
            format_storage_key(unsalted, "{model}@{hash}@{salt}"), "model@dead"
        )

    def test_filesystem_format_is_preserved(self) -> None:
        """Python filesystem filenames retain their independent format."""
        key = ObjectKey(b"\xde\xad\xbe\xef", "org/model", 42, 3, "tenant")
        filename = encode_fs_filename(key)
        self.assertEqual(filename, "org-SEP-model@0x0000002a@3@deadbeef@tenant.data")
        self.assertEqual(decode_fs_filename(filename), key)


if __name__ == "__main__":
    unittest.main()
