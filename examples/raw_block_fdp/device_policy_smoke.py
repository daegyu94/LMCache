# SPDX-License-Identifier: Apache-2.0
"""Smoke-test raw-block FDP policies against a real FDP NVMe device.

Default mode opens the NVMe namespace character device and fetches FDP status.
Write mode also creates a RawBlockL2Adapter, stores tiny objects, and prints the
adapter FDP status maps. Write mode modifies the target raw-block device.
"""

# Future
from __future__ import annotations

# Standard
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from typing import Any, Literal
import json
import select
import sys

FdpPolicy = Literal["rank_isolation", "domain_isolation", "model_isolation"]
DEFAULT_SLOT_BYTES = 1024 * 1024
DEFAULT_CAPACITY_BYTES = 64 * 1024 * 1024
DEFAULT_BLOCK_ALIGN = 4096
DEFAULT_HEADER_BYTES = 4096
DEFAULT_META_TOTAL_BYTES = 16 * 1024 * 1024
DEFAULT_IOURING_QUEUE_DEPTH = 256
DEFAULT_MAX_RUHS = 256
DEFAULT_PAYLOAD_BYTES = DEFAULT_SLOT_BYTES - DEFAULT_HEADER_BYTES


@dataclass(frozen=True)
class SmokeConfig:
    """Validated smoke-test configuration."""

    device_path: str
    policy: FdpPolicy
    placement_handles: tuple[int, ...]
    max_ruhs: int
    payload_bytes: int


def parse_int_list(raw: str) -> tuple[int, ...]:
    """Parse a comma-separated integer list."""
    if not raw:
        return ()
    values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if any(value < 0 for value in values):
        raise SystemExit("placement handles must be non-negative integers")
    if len(set(values)) != len(values):
        raise SystemExit("placement handles must not contain duplicates")
    return values


def fetch_fdp_status(config: SmokeConfig) -> list[tuple[int, int]]:
    """Open the real NVMe character device and fetch FDP handle status."""
    try:
        # Third Party
        from lmcache_rust_raw_block_io import RawBlockDevice
    except ImportError as e:
        raise SystemExit(
            "lmcache_rust_raw_block_io is required. Build rust/raw_block first."
        ) from e

    dev = RawBlockDevice(
        config.device_path,
        False,
        use_odirect=False,
        alignment=DEFAULT_BLOCK_ALIGN,
        io_engine="io_uring",
        use_uring_cmd=True,
        iouring_queue_depth=DEFAULT_IOURING_QUEUE_DEPTH,
    )
    try:
        return [
            (int(pid), int(ruhid))
            for pid, ruhid in dev.fetch_fdp_status(config.max_ruhs)
        ]
    finally:
        dev.close()


def validate_handles(
    discovered: list[tuple[int, int]], configured_handles: tuple[int, ...]
) -> list[int]:
    """Return selected handles after validating they exist on the device."""
    available = {pid for pid, _ruhid in discovered}
    if not available:
        raise SystemExit("The device did not report any FDP placement handles")
    if configured_handles:
        missing = sorted(set(configured_handles) - available)
        if missing:
            raise SystemExit(
                "Requested placement handles are not reported by the device: "
                + ",".join(str(handle) for handle in missing)
            )
        return list(configured_handles)
    return sorted(available)


def format_fdp_status(status: list[tuple[int, int]] | None) -> dict[str, int] | None:
    """Format FDP status as a readable placement-handle to RUH map."""
    if status is None:
        return None
    return {str(pid): int(ruhid) for pid, ruhid in status}


def make_adapter_config(config: SmokeConfig) -> Any:
    """Build a RawBlockL2AdapterConfig for the selected policy."""
    # First Party
    from lmcache.v1.distributed.l2_adapters.raw_block_l2_adapter import (
        RawBlockL2AdapterConfig,
    )

    return RawBlockL2AdapterConfig(
        device_path=config.device_path,
        capacity_bytes=DEFAULT_CAPACITY_BYTES,
        block_align=DEFAULT_BLOCK_ALIGN,
        header_bytes=DEFAULT_HEADER_BYTES,
        slot_bytes=DEFAULT_SLOT_BYTES,
        meta_total_bytes=DEFAULT_META_TOTAL_BYTES,
        use_odirect=False,
        enable_zero_copy=False,
        meta_enable_periodic=False,
        meta_idle_quiet_ms=0,
        load_checkpoint_on_init=False,
        io_engine="io_uring",
        use_uring_cmd=True,
        iouring_queue_depth=DEFAULT_IOURING_QUEUE_DEPTH,
        fdp_enabled=True,
        fdp_policy=config.policy,
        fdp_placement_handles=list(config.placement_handles) or None,
        num_store_workers=1,
        num_lookup_workers=1,
        num_load_workers=1,
    )


def make_memory_obj(payload: bytes) -> Any:
    """Wrap bytes in a binary TensorMemoryObj for adapter store."""
    # Third Party
    import torch

    # First Party
    from lmcache.v1.memory_management import (
        MemoryFormat,
        MemoryObjMetadata,
        TensorMemoryObj,
    )

    data = bytearray(payload)
    raw_data = torch.frombuffer(data, dtype=torch.uint8)
    metadata = MemoryObjMetadata(
        shape=torch.Size([len(data)]),
        dtype=torch.uint8,
        address=0,
        phy_size=len(data),
        fmt=MemoryFormat.BINARY,
        ref_count=1,
    )
    return TensorMemoryObj(raw_data, metadata, parent_allocator=None)


def make_payload(config: SmokeConfig, index: int) -> bytes:
    """Build a deterministic payload with the configured byte size."""
    prefix = f"raw-block-fdp-smoke-{config.policy}-{index}-".encode()
    if len(prefix) >= config.payload_bytes:
        return prefix[: config.payload_bytes]
    repeats = ((config.payload_bytes - len(prefix)) // len(prefix)) + 1
    return (prefix + (prefix * repeats))[: config.payload_bytes]


def make_keys(config: SmokeConfig) -> list[Any]:
    """Build policy-specific ObjectKeys that should populate FDP maps."""
    # First Party
    from lmcache.v1.distributed.api import ObjectKey

    if config.policy == "rank_isolation":
        ranks = range(max(1, min(2, len(config.placement_handles))))
        return [
            ObjectKey(
                chunk_hash=ObjectKey.IntHash2Bytes(index + 1),
                model_name="raw-block-fdp-smoke",
                kv_rank=ObjectKey.ComputeKVRank(
                    world_size=8,
                    global_rank=rank,
                    local_world_size=4,
                    local_rank=rank,
                ),
            )
            for index, rank in enumerate(ranks)
        ]
    if config.policy == "domain_isolation":
        salts = ["domain-a", "domain-b", "domain-a"]
        return [
            ObjectKey(
                chunk_hash=ObjectKey.IntHash2Bytes(index + 1),
                model_name="raw-block-fdp-smoke",
                kv_rank=0,
                cache_salt=salt,
            )
            for index, salt in enumerate(salts)
        ]
    models = ["model-a", "model-b", "model-a"]
    return [
        ObjectKey(
            chunk_hash=ObjectKey.IntHash2Bytes(index + 1),
            model_name=model,
            kv_rank=0,
        )
        for index, model in enumerate(models)
    ]


def wait_for_event(event_fd: int, timeout_sec: float) -> bool:
    """Wait for an adapter eventfd."""
    readable, _writable, _errors = select.select([event_fd], [], [], timeout_sec)
    if not readable:
        return False
    try:
        # First Party
        from lmcache.v1.platform import consume_fd

        consume_fd(event_fd)
    except Exception:
        pass
    return True


def policy_status(status: dict[str, Any]) -> dict[str, Any]:
    """Extract FDP status fields from adapter status."""
    return {
        "fdp_enabled": status.get("fdp_enabled"),
        "fdp_policy": status.get("fdp_policy"),
        "fdp_placement_handle_to_ruh_id": format_fdp_status(
            status.get("fdp_discovered_status")
        ),
        "fdp_placement_handles": status.get("fdp_placement_handles"),
        "fdp_local_rank_to_placement": status.get("fdp_local_rank_to_placement"),
        "fdp_cache_salt_to_placement": status.get("fdp_cache_salt_to_placement"),
        "fdp_model_to_placement": status.get("fdp_model_to_placement"),
    }


def exercise_policy_writes(config: SmokeConfig) -> dict[str, Any]:
    """Store small objects through RawBlockL2Adapter and return policy status."""
    # First Party
    from lmcache.v1.distributed.l2_adapters.raw_block_l2_adapter import (
        RawBlockL2Adapter,
    )

    adapter = RawBlockL2Adapter(make_adapter_config(config))
    try:
        initial_status = policy_status(adapter.report_status())
        keys = make_keys(config)
        objects = [
            make_memory_obj(make_payload(config, i)) for i, _key in enumerate(keys)
        ]
        task_id = adapter.submit_store_task(keys, objects)
        if not wait_for_event(adapter.get_store_event_fd(), timeout_sec=30.0):
            raise TimeoutError("store task did not complete within 30 seconds")
        store_results = adapter.pop_completed_store_tasks()
        store_result = store_results[task_id]
        if not store_result.is_successful():
            raise RuntimeError("store task completed but at least one object failed")
        final_status = policy_status(adapter.report_status())
        return {
            "initial_policy_status": initial_status,
            "store_successful": store_result.is_successful(),
            "store_bytes_transferred": store_result.bytes_transferred(),
            "stored_key_count": len(keys),
            "final_policy_status": final_status,
        }
    finally:
        adapter.close()


def build_parser() -> ArgumentParser:
    """Build the device smoke-test CLI parser."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device-path",
        required=True,
        help="NVMe char device, e.g. /dev/ng0n1",
    )
    parser.add_argument(
        "--policy",
        choices=("rank_isolation", "domain_isolation", "model_isolation"),
        default="rank_isolation",
    )
    parser.add_argument(
        "--placement-handles",
        default="",
        help="Comma-separated placement handles. Empty means all discovered handles.",
    )
    parser.add_argument("--max-ruhs", type=int, default=DEFAULT_MAX_RUHS)
    parser.add_argument(
        "--payload-bytes",
        type=int,
        default=DEFAULT_PAYLOAD_BYTES,
        help="Payload bytes per stored object in --exercise-writes mode.",
    )
    parser.add_argument(
        "--exercise-writes",
        action="store_true",
        help="Also write small raw-block objects and print adapter policy maps.",
    )
    parser.add_argument(
        "--i-understand-this-writes-to-device",
        action="store_true",
        help=(
            "Required with --exercise-writes because raw-block metadata/data "
            "are written."
        ),
    )
    return parser


def validate_args(args: Namespace) -> SmokeConfig:
    """Validate CLI args and build SmokeConfig."""
    placement_handles = parse_int_list(args.placement_handles)
    if args.exercise_writes and not args.i_understand_this_writes_to_device:
        raise SystemExit(
            "--exercise-writes modifies the raw-block device. Re-run with "
            "--i-understand-this-writes-to-device on a disposable namespace."
        )
    if args.exercise_writes and not placement_handles:
        raise SystemExit(
            "--placement-handles is required with --exercise-writes so the smoke "
            "test does not claim every FDP handle on the device"
        )
    if args.max_ruhs <= 0:
        raise SystemExit("--max-ruhs must be > 0")
    if args.payload_bytes <= 0:
        raise SystemExit("--payload-bytes must be > 0")
    payload_capacity = DEFAULT_SLOT_BYTES - DEFAULT_HEADER_BYTES
    if args.payload_bytes > payload_capacity:
        raise SystemExit(
            "--payload-bytes must be <= slot_bytes - header_bytes "
            f"({payload_capacity} bytes with the current options)"
        )
    return SmokeConfig(
        device_path=args.device_path,
        policy=args.policy,
        placement_handles=placement_handles,
        max_ruhs=args.max_ruhs,
        payload_bytes=args.payload_bytes,
    )


def main() -> None:
    """Run the real-device FDP smoke test."""
    args = build_parser().parse_args()
    config = validate_args(args)
    discovered = fetch_fdp_status(config)
    selected_handles = validate_handles(discovered, config.placement_handles)
    output: dict[str, Any] = {
        "device_path": config.device_path,
        "policy": config.policy,
        "placement_handle_to_ruh_id": format_fdp_status(discovered),
        "selected_placement_handles": selected_handles,
        "payload_bytes": config.payload_bytes,
        "write_exercised": False,
    }
    if args.exercise_writes:
        write_config = SmokeConfig(
            **{**config.__dict__, "placement_handles": tuple(selected_handles)}
        )
        output["write_exercised"] = True
        output["write_result"] = exercise_policy_writes(write_config)
    json.dump(output, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
