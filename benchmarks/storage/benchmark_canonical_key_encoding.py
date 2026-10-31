# SPDX-License-Identifier: Apache-2.0
"""Compare the canonical encoder with its previous generic-template path.

Run from a source checkout with LMCache dependencies installed:
PYTHONPATH=. python benchmarks/storage/benchmark_canonical_key_encoding.py
The result measures key formatting only, with no storage I/O or serving traffic.
"""

# Standard
from statistics import median
import argparse
import itertools
import json
import time

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.key_codec import encode_storage_key, format_storage_key


def encode_with_template(key: ObjectKey) -> str:
    """Return the original canonical encoder's generic-template result.

    Args:
        key: Object whose identity determines the storage name.

    Returns:
        The canonical key formatted with the unchanged generic codec.
    """
    return format_storage_key(key, "{model}@{rank}@{group}@{hash}{suffix}")


def main() -> None:
    """Check output equivalence and print alternating timing samples as JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=100000)
    parser.add_argument("--rounds", type=int, default=9)
    args = parser.parse_args()
    if args.iterations <= 0 or args.rounds <= 0:
        parser.error("iterations and rounds must be positive")
    keys = [
        ObjectKey(bytes(range(length)), model, rank, group, salt)
        for model, length, salt, rank, group in itertools.product(
            ["m", "meta-llama/Llama-3.1-8B-Instruct", "组织/模型"],
            [4, 32, 64],
            ["", "tenant-01"],
            [0, 31],
            [0, 255],
        )
    ]
    for key in keys:
        if encode_with_template(key) != encode_storage_key(key):
            raise RuntimeError("canonical wire bytes changed")
    prepared = [keys[i % len(keys)] for i in range(args.iterations)]
    samples: dict[str, list[float]] = {"template": [], "canonical": []}
    functions = [
        ("template", encode_with_template),
        ("canonical", encode_storage_key),
    ]
    for round_index in range(args.rounds):
        order = functions if round_index % 2 == 0 else list(reversed(functions))
        for name, encode in order:
            start = time.perf_counter_ns()
            checksum = sum(len(encode(key)) for key in prepared)
            elapsed = time.perf_counter_ns() - start
            if checksum == 0:
                raise RuntimeError("empty benchmark output")
            samples[name].append(elapsed / args.iterations)
    print(
        json.dumps(
            {
                "iterations": args.iterations,
                "rounds": args.rounds,
                "median_ns_per_key": {name: median(v) for name, v in samples.items()},
                "samples_ns_per_key": samples,
                "scope": "CPU key formatting only; no storage or serving benchmark",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
