# SPDX-License-Identifier: Apache-2.0
"""Measure weighted L2 request scheduling through an L2 adapter."""

# Standard
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import argparse
import json
import select
import socket
import subprocess
import sys
import time
import uuid

DEFAULT_OBJECT_MIB_BY_WEIGHT = {100: 4, 200: 2, 400: 1}


def _positive_int(value: str) -> int:
    """Parse a positive integer command-line argument."""
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


@dataclass
class _Slot:
    """One continuously backlogged L2 request slot."""

    cache_salt: str
    object_mib: int
    obj: Any
    key: Any
    sequence: int


def _scenario(tenants_per_client: int) -> dict[str, list[tuple[str, int]]]:
    """Return the salt and weight registrations for three clients."""
    if tenants_per_client == 1:
        return {
            "client-1": [("client-1-w100", 100)],
            "client-2": [("client-2-w200", 200)],
            "client-3": [("client-3-w400", 400)],
        }
    weights = (100, 400) if tenants_per_client == 2 else (100, 200, 400)
    return {
        f"client-{client}": [
            (f"client-{client}-w{weight}", weight) for weight in weights
        ]
        for client in range(1, 4)
    }


def _object_mib_by_salt(
    weighted_salts: list[tuple[str, int]],
    object_mib_override: int | None,
) -> dict[str, int]:
    """Resolve each tenant's object size from its weight or an override.

    Args:
        weighted_salts: Salt and scheduling-weight pairs in the benchmark.
        object_mib_override: One object size to use for every tenant, or
            ``None`` to use the default 4/2/1 MiB sizes for weights
            100/200/400.

    Returns:
        A cache-salt-to-object-size mapping in MiB.

    Raises:
        ValueError: If the override is invalid or a weight has no default
            object size.
    """
    if object_mib_override is not None:
        if object_mib_override <= 0:
            raise ValueError("object_mib must be positive")
        return {salt: object_mib_override for salt, _weight in weighted_salts}

    object_mib_by_salt: dict[str, int] = {}
    for salt, weight in weighted_salts:
        object_mib = DEFAULT_OBJECT_MIB_BY_WEIGHT.get(weight)
        if object_mib is None:
            supported = sorted(DEFAULT_OBJECT_MIB_BY_WEIGHT)
            raise ValueError(
                f"no default object size for weight {weight}; "
                f"supported weights are {supported} or pass --object-mib"
            )
        object_mib_by_salt[salt] = object_mib
    return object_mib_by_salt


def _recv_json_line(connection: socket.socket) -> dict[str, Any]:
    """Read one newline-delimited JSON object from a socket."""
    payload = bytearray()
    while not payload.endswith(b"\n"):
        chunk = connection.recv(4096)
        if not chunk:
            raise RuntimeError("connection closed before a JSON line was received")
        payload.extend(chunk)
        if len(payload) > 1 << 20:
            raise ValueError("registration exceeds 1 MiB")
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError("registration must be a JSON object")
    return value


def _send_json_line(connection: socket.socket, value: dict[str, Any]) -> None:
    """Write one newline-delimited JSON object to a socket."""
    connection.sendall(json.dumps(value, sort_keys=True).encode() + b"\n")


def _create_memory_obj(size_bytes: int, fill_value: int) -> Any:
    """Allocate a page-aligned TensorMemoryObj suitable for direct I/O."""
    # Third Party
    import torch

    # First Party
    from lmcache.v1.memory_management import (
        MemoryFormat,
        MemoryObjMetadata,
        TensorMemoryObj,
    )

    alignment = 4096
    backing = torch.empty(size_bytes + alignment, dtype=torch.uint8)
    offset = (-backing.data_ptr()) % alignment
    raw = backing[offset : offset + size_bytes]
    raw.fill_(fill_value % 251)
    metadata = MemoryObjMetadata(
        shape=torch.Size([size_bytes]),
        dtype=torch.uint8,
        address=raw.data_ptr(),
        phy_size=size_bytes,
        ref_count=1,
        fmt=MemoryFormat.BINARY,
    )
    return TensorMemoryObj(raw, metadata, parent_allocator=None)


def _make_key(cache_salt: str, salt_index: int, sequence: int) -> Any:
    """Create a unique salted ObjectKey for one benchmark request."""
    # First Party
    from lmcache.v1.distributed.api import ObjectKey

    chunk_id = (salt_index << 28) | sequence
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(chunk_id),
        model_name="l2-qos-benchmark",
        kv_rank=0,
        object_group_id=7,
        cache_salt=cache_salt,
    )


def _wait_for_store(adapter: Any, task_id: int, timeout: float = 60.0) -> None:
    """Wait for one adapter store task and validate its result."""
    # First Party
    from lmcache.v1.platform import consume_fd

    deadline = time.monotonic() + timeout
    poller = select.poll()
    poller.register(adapter.get_store_event_fd(), select.POLLIN)
    while time.monotonic() < deadline:
        for _fd, _events in poller.poll(100):
            try:
                consume_fd(adapter.get_store_event_fd())
            except BlockingIOError:
                pass
        result = adapter.pop_completed_store_tasks().get(task_id)
        if result is None:
            continue
        if not result.is_successful():
            raise RuntimeError("fs setup store failed")
        return
    raise TimeoutError("timed out waiting for fs setup store")


def _measure_store(
    adapter: Any,
    slots: list[_Slot],
    salt_indices: dict[str, int],
    warmup_seconds: float,
    measure_seconds: float,
    sequence_step: int,
) -> dict[str, int]:
    """Continuously store unique keys and count measured completions."""
    # First Party
    from lmcache.v1.platform import consume_fd

    pending = {
        adapter.submit_store_task([slot.key], [slot.obj]): slot for slot in slots
    }
    counts = {salt: 0 for salt in salt_indices}
    measure_start = time.monotonic() + warmup_seconds
    measure_end = measure_start + measure_seconds
    poller = select.poll()
    event_fd = adapter.get_store_event_fd()
    poller.register(event_fd, select.POLLIN)

    while pending:
        for _fd, _events in poller.poll(100):
            try:
                consume_fd(event_fd)
            except BlockingIOError:
                pass
        for task_id, result in adapter.pop_completed_store_tasks().items():
            slot = pending.pop(task_id, None)
            if slot is None:
                continue
            if not result.is_successful():
                raise RuntimeError(f"store failed for {slot.cache_salt}")
            now = time.monotonic()
            if measure_start <= now < measure_end:
                counts[slot.cache_salt] += 1
            if now < measure_end:
                slot.sequence += sequence_step
                slot.key = _make_key(
                    slot.cache_salt,
                    salt_indices[slot.cache_salt],
                    slot.sequence,
                )
                new_task_id = adapter.submit_store_task([slot.key], [slot.obj])
                pending[new_task_id] = slot
    return counts


def _measure_load(
    adapter: Any,
    slots: list[_Slot],
    salts: list[str],
    warmup_seconds: float,
    measure_seconds: float,
) -> dict[str, int]:
    """Continuously load fixed keys and count measured completions."""
    # First Party
    from lmcache.v1.platform import consume_fd

    pending = {adapter.submit_load_task([slot.key], [slot.obj]): slot for slot in slots}
    counts = {salt: 0 for salt in salts}
    measure_start = time.monotonic() + warmup_seconds
    measure_end = measure_start + measure_seconds
    poller = select.poll()
    event_fd = adapter.get_load_event_fd()
    poller.register(event_fd, select.POLLIN)

    while pending:
        for _fd, _events in poller.poll(100):
            try:
                consume_fd(event_fd)
            except BlockingIOError:
                pass
        for task_id in list(pending):
            result = adapter.query_load_result(task_id)
            if result is None:
                continue
            slot = pending.pop(task_id)
            if result.popcount() != 1:
                raise RuntimeError(f"load failed for {slot.cache_salt}")
            now = time.monotonic()
            if measure_start <= now < measure_end:
                counts[slot.cache_salt] += 1
            if now < measure_end:
                new_task_id = adapter.submit_load_task([slot.key], [slot.obj])
                pending[new_task_id] = slot
    return counts


def _create_raw_adapter(
    adapter_name: str,
    run_dir: Path,
    io_mode: str,
) -> Any:
    """Create the configured raw L2 adapter for one benchmark run.

    Currently only the ``fs`` adapter is supported; other L2 adapters are
    TBD (future work).
    """
    if adapter_name != "fs":
        raise ValueError(
            f"unsupported L2 adapter {adapter_name!r}; only fs is supported"
        )

    # First Party
    from lmcache.v1.distributed.l2_adapters import create_l2_adapter
    from lmcache.v1.distributed.l2_adapters.fs_l2_adapter import (
        FSL2AdapterConfig,
    )

    return create_l2_adapter(
        FSL2AdapterConfig(
            base_path=str(run_dir),
            use_odirect=io_mode == "direct",
        )
    )


def _run_benchmark(
    registrations: dict[str, list[tuple[str, int]]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Run one L2 adapter benchmark and return its measurements."""
    # First Party
    from lmcache.v1.distributed.l2_qos_adapter import QosL2Adapter
    from lmcache.v1.multiprocess.qos import CacheSaltQosManager, L2QoSDispatcher

    adapter_name = getattr(args, "adapter", "fs")
    io_mode = getattr(args, "io_mode", "direct")
    if io_mode not in ("direct", "buffered"):
        raise ValueError(f"unsupported io_mode: {io_mode}")
    weighted_salts = [
        item for client in sorted(registrations) for item in registrations[client]
    ]
    run_id = (
        f"{args.operation}-{len(weighted_salts)}-tenant-"
        f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    )
    run_dir = Path(args.base_path).resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    salt_indices = {
        salt: index + 1 for index, (salt, _weight) in enumerate(weighted_salts)
    }
    object_mib_by_salt = _object_mib_by_salt(
        weighted_salts, getattr(args, "object_mib", None)
    )
    slots: list[_Slot] = []
    for salt, _weight in weighted_salts:
        salt_index = salt_indices[salt]
        object_mib = object_mib_by_salt[salt]
        size_bytes = object_mib * (1 << 20)
        for queue_index in range(args.queue_depth):
            sequence = queue_index + 1
            slots.append(
                _Slot(
                    cache_salt=salt,
                    object_mib=object_mib,
                    obj=_create_memory_obj(size_bytes, salt_index + queue_index),
                    key=_make_key(salt, salt_index, sequence),
                    sequence=sequence,
                )
            )

    raw_adapter = _create_raw_adapter(adapter_name, run_dir, io_mode)
    if args.operation == "load":
        setup_id = raw_adapter.submit_store_task(
            [slot.key for slot in slots], [slot.obj for slot in slots]
        )
        _wait_for_store(raw_adapter, setup_id)

    manager = CacheSaltQosManager()
    for salt, weight in weighted_salts:
        manager.set_sched_weight(salt, weight)
    dispatcher = L2QoSDispatcher(
        quantum_bytes=1 << 20,
        max_inflight_tasks=args.max_inflight_tasks,
    )
    manager.register_listener(dispatcher.update_profile)
    adapter = QosL2Adapter(raw_adapter, dispatcher, manager)
    try:
        if args.operation == "store":
            counts = _measure_store(
                adapter,
                slots,
                salt_indices,
                args.warmup_seconds,
                args.measure_seconds,
                args.queue_depth,
            )
        else:
            counts = _measure_load(
                adapter,
                slots,
                list(salt_indices),
                args.warmup_seconds,
                args.measure_seconds,
            )
    finally:
        adapter.close()
        dispatcher.close()

    total_ops = sum(counts.values())
    total_weight = sum(weight for _salt, weight in weighted_salts)
    total_bytes = sum(
        counts[salt] * object_mib_by_salt[salt] * (1 << 20)
        for salt, _weight in weighted_salts
    )
    tenants = {}
    passed = total_bytes > 0
    for salt, weight in weighted_salts:
        tenant_bytes = counts[salt] * object_mib_by_salt[salt] * (1 << 20)
        request_share = counts[salt] / total_ops if total_ops else 0.0
        byte_share = tenant_bytes / total_bytes if total_bytes else 0.0
        expected_byte_share = weight / total_weight
        relative_error = (
            abs(byte_share - expected_byte_share) / expected_byte_share
            if expected_byte_share
            else 0.0
        )
        passed = passed and relative_error <= args.tolerance
        tenants[salt] = {
            "weight": weight,
            "object_mib": object_mib_by_salt[salt],
            "operations": counts[salt],
            "bytes": tenant_bytes,
            "mib_per_second": tenant_bytes / (1 << 20) / args.measure_seconds,
            "request_share": request_share,
            "byte_share": byte_share,
            "expected_byte_share": expected_byte_share,
            "relative_error": relative_error,
        }

    result = {
        "passed": passed,
        "operation": args.operation,
        "client_count": 3,
        "tenant_count": len(weighted_salts),
        "base_path": str(run_dir),
        "adapter": adapter_name,
        "io_mode": io_mode,
        "max_inflight_tasks": args.max_inflight_tasks,
        "object_mib_override": getattr(args, "object_mib", None),
        "object_mib_by_tenant": object_mib_by_salt,
        "queue_depth_per_tenant": args.queue_depth,
        "warmup_seconds": args.warmup_seconds,
        "measure_seconds": args.measure_seconds,
        "tolerance": args.tolerance,
        "total_operations": total_ops,
        "total_bytes": total_bytes,
        "aggregate_mib_per_second": total_bytes / (1 << 20) / args.measure_seconds,
        "tenants": tenants,
    }
    (run_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def _run_client(args: argparse.Namespace) -> int:
    """Register one Docker client and wait for the benchmark result."""
    registration = {
        "client_id": args.client_id,
        "tenants": [
            {
                "cache_salt": value.split("=", 1)[0],
                "weight": int(value.split("=", 1)[1]),
            }
            for value in args.tenant
        ],
    }
    deadline = time.monotonic() + 30.0
    connection: socket.socket | None = None
    while connection is None:
        try:
            connection = socket.create_connection((args.host, args.port), timeout=5)
        except OSError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.2)
    connection.settimeout(None)
    with connection:
        _send_json_line(connection, registration)
        response = _recv_json_line(connection)
    print(json.dumps(response, sort_keys=True))
    return 0 if response.get("status") == "ok" else 1


def _run_host(args: argparse.Namespace) -> int:
    """Launch exactly three Docker clients and run one host benchmark."""
    expected = _scenario(args.tenants_per_client)
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("0.0.0.0", args.port))
    server.listen(3)
    server.settimeout(45)
    port = server.getsockname()[1]

    processes: list[subprocess.Popen[str]] = []
    connections: list[socket.socket] = []
    try:
        for client_id, tenants in expected.items():
            command = [
                "docker",
                "run",
                "--rm",
                "--network",
                "host",
                args.image,
                "client",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--client-id",
                client_id,
            ]
            for salt, weight in tenants:
                command.extend(["--tenant", f"{salt}={weight}"])
            processes.append(
                subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            )

        registrations: dict[str, list[tuple[str, int]]] = {}
        for _index in range(3):
            connection, _address = server.accept()
            connections.append(connection)
            payload = _recv_json_line(connection)
            client_id = str(payload["client_id"])
            registrations[client_id] = [
                (str(item["cache_salt"]), int(item["weight"]))
                for item in payload["tenants"]
            ]
        if registrations != expected:
            raise ValueError(
                f"Docker registrations do not match scenario: {registrations!r}"
            )

        result = _run_benchmark(registrations, args)
        response = {"status": "ok", "result": result}
        for connection in connections:
            _send_json_line(connection, response)
        print(json.dumps(result, indent=2, sort_keys=True))
    except BaseException as exc:
        response = {"status": "error", "error": str(exc)}
        for connection in connections:
            try:
                _send_json_line(connection, response)
            except OSError:
                pass
        raise
    finally:
        server.close()
        for connection in connections:
            connection.close()
        for process in processes:
            output, _unused = process.communicate(timeout=30)
            if process.returncode != 0:
                print(output, file=sys.stderr)
                raise RuntimeError(f"Docker client exited with {process.returncode}")
    return 0 if result["passed"] else 2


def _parse_args() -> argparse.Namespace:
    """Parse benchmark host and Docker-client arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="run host benchmark")
    run.add_argument("--base-path", default="/mnt/nvme/lmcache-l2-qos")
    run.add_argument("--image", default="lmcache-l2-qos-benchmark:local")
    run.add_argument(
        "--adapter",
        choices=("fs",),
        default="fs",
        help="L2 adapter backend. Other L2 adapters are TBD (future work).",
    )
    run.add_argument("--tenants-per-client", type=int, choices=(1, 2, 3), required=True)
    run.add_argument("--operation", choices=("store", "load"), required=True)
    run.add_argument("--port", type=int, default=0)
    run.add_argument("--warmup-seconds", type=float, default=5.0)
    run.add_argument("--measure-seconds", type=float, default=15.0)
    run.add_argument(
        "--io-mode",
        choices=("direct", "buffered"),
        default="direct",
        help="Filesystem I/O mode. Default is direct (O_DIRECT).",
    )
    run.add_argument(
        "--object-mib",
        type=int,
        default=None,
        help=(
            "Use one object size for every tenant. By default, weights "
            "100/200/400 use 4/2/1 MiB respectively."
        ),
    )
    run.add_argument("--queue-depth", type=int, default=8)
    run.add_argument(
        "--max-inflight-tasks",
        type=_positive_int,
        default=1,
        help="Maximum concurrently admitted tasks; defaults to 1.",
    )
    run.add_argument("--tolerance", type=float, default=0.20)

    client = subparsers.add_parser("client", help="register a Docker client")
    client.add_argument("--host", required=True)
    client.add_argument("--port", type=int, required=True)
    client.add_argument("--client-id", required=True)
    client.add_argument("--tenant", action="append", required=True)
    return parser.parse_args()


def main() -> int:
    """Run the selected benchmark role."""
    args = _parse_args()
    if args.command == "client":
        return _run_client(args)
    return _run_host(args)


if __name__ == "__main__":
    raise SystemExit(main())
