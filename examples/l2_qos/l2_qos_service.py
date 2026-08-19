# SPDX-License-Identifier: Apache-2.0
"""Docker integration harness for weighted L2 request scheduling."""

# Future
from __future__ import annotations

# Standard
from collections import Counter
import argparse
import json
import socket
import socketserver
import threading
import time

# Third Party
from qos import CacheSaltQosManager, L2QoSDispatcher


class _ServerState:
    """Shared registry, dispatcher, and admission accounting."""

    def __init__(
        self,
        expected_clients: int,
        default_sched_weight: int,
        weights: dict[str, int],
    ) -> None:
        self.qos_manager = CacheSaltQosManager(default_sched_weight)
        self.dispatcher = L2QoSDispatcher(quantum_bytes=1, max_inflight_tasks=1)
        self.qos_manager.register_listener(self.dispatcher.update_profile)
        for cache_salt, weight in weights.items():
            self.qos_manager.set_sched_weight(cache_salt, weight)
        self.expected_clients = expected_clients
        self.admissions: list[str] = []
        self.completed_clients = 0
        self.lock = threading.Lock()
        self.ready_condition = threading.Condition(self.lock)
        self.ready_clients = 0
        self.submitted_clients = 0
        self.work_ready = threading.Event()
        self.server: _QoSServer | None = None

    def wait_for_clients(self) -> None:
        """Wait until every test client has sent its cache salt."""
        with self.ready_condition:
            self.ready_clients += 1
            if self.ready_clients >= self.expected_clients:
                self.ready_condition.notify_all()
            else:
                self.ready_condition.wait_for(
                    lambda: self.ready_clients >= self.expected_clients
                )

    def mark_submissions_ready(self) -> None:
        """Release the dispatcher after every client queues its batch."""
        with self.lock:
            self.submitted_clients += 1
            if self.submitted_clients >= self.expected_clients:
                self.work_ready.set()

    def record_admission(self, cache_salt: str) -> int:
        """Record one admitted synthetic L2 operation."""
        self.work_ready.wait()
        with self.lock:
            self.admissions.append(cache_salt)
        return 1

    def client_completed(self) -> None:
        """Stop the server after all expected clients drain."""
        with self.lock:
            self.completed_clients += 1
            should_stop = self.completed_clients >= self.expected_clients
        if should_stop and self.server is not None:
            threading.Thread(target=self.server.shutdown, daemon=True).start()


class _QoSRequestHandler(socketserver.StreamRequestHandler):
    """Receive one cache salt and a batch of synthetic L2 operations."""

    state: _ServerState

    def handle(self) -> None:
        hello = self._read_message()
        if hello.get("kind") != "hello":
            raise ValueError("first message must be hello")
        cache_salt = str(hello["cache_salt"])

        self.state.wait_for_clients()
        pending = {}
        submitted_tasks = 0
        while True:
            message = self._read_message()
            kind = message.get("kind")
            if kind == "task":
                task_id = int(message["id"])
                submitted_tasks += 1
                pending[task_id] = self.state.dispatcher.submit(
                    profile=self.state.qos_manager.get_profile(cache_salt),
                    operation="load",
                    cost_bytes=max(1, int(message.get("cost", 1))),
                    action=lambda salt=cache_salt: self.state.record_admission(salt),
                )
            elif kind == "done":
                break
            else:
                raise ValueError(f"unknown message kind: {kind}")

        self.state.mark_submissions_ready()
        while pending:
            progressed = False
            for task_id, handle in list(pending.items()):
                if not handle.future.done():
                    continue
                handle.future.result()
                self.state.dispatcher.complete(handle.task_id)
                self._write_message({"kind": "complete", "id": task_id})
                del pending[task_id]
                progressed = True
            if not progressed:
                time.sleep(0.001)

        self._write_message({"kind": "summary", "tasks": submitted_tasks})
        self.state.client_completed()

    def _read_message(self) -> dict:
        """Read one newline-delimited JSON message."""
        line = self.rfile.readline()
        if not line:
            raise ConnectionError("client disconnected")
        return json.loads(line)

    def _write_message(self, message: dict) -> None:
        """Write one newline-delimited JSON message."""
        payload = json.dumps(message, separators=(",", ":")).encode() + b"\n"
        self.wfile.write(payload)
        self.wfile.flush()


class _QoSServer(socketserver.ThreadingTCPServer):
    """Threading TCP server carrying a shared QoS dispatcher."""

    allow_reuse_address = True
    daemon_threads = True


def _run_server(
    port: int,
    expected_clients: int,
    default_sched_weight: int,
    weights: dict[str, int],
) -> int:
    """Run the synthetic shared-L2 server."""
    state = _ServerState(expected_clients, default_sched_weight, weights)

    class Handler(_QoSRequestHandler):
        pass

    Handler.state = state
    server = _QoSServer(("0.0.0.0", port), Handler)
    state.server = server
    print(
        json.dumps(
            {
                "kind": "ready",
                "port": port,
                "default_sched_weight": default_sched_weight,
                "weights": weights,
            }
        ),
        flush=True,
    )
    try:
        server.serve_forever()
    finally:
        server.server_close()
        state.dispatcher.close()

    counts = Counter(state.admissions)
    prefix_counts = Counter(state.admissions[:32])
    print(
        json.dumps(
            {
                "kind": "server-summary",
                "total": len(state.admissions),
                "counts": dict(counts),
                "prefix_counts": dict(prefix_counts),
            },
            separators=(",", ":"),
        ),
        flush=True,
    )
    return 0


def _run_client(host: str, port: int, tasks: int, cache_salt: str) -> int:
    """Submit synthetic work carrying only a cache salt."""
    print(
        json.dumps(
            {"kind": "client-identity", "cache_salt": cache_salt},
            separators=(",", ":"),
        ),
        flush=True,
    )

    completed: set[int] = set()
    with socket.create_connection((host, port), timeout=10) as connection:
        stream = connection.makefile("rwb")

        def send(message: dict) -> None:
            stream.write(json.dumps(message, separators=(",", ":")).encode())
            stream.write(b"\n")
            stream.flush()

        send({"kind": "hello", "cache_salt": cache_salt})
        for task_id in range(tasks):
            send({"kind": "task", "id": task_id, "cost": 1})
        send({"kind": "done"})

        while len(completed) < tasks:
            line = stream.readline()
            if not line:
                raise ConnectionError("server disconnected before all tasks")
            message = json.loads(line)
            if message.get("kind") == "complete":
                completed.add(int(message["id"]))
        summary = json.loads(stream.readline())
        if summary.get("kind") != "summary":
            raise ValueError(f"unexpected server response: {summary}")

    print(json.dumps({"kind": "client-summary", "tasks": len(completed)}), flush=True)
    return 0


def _parse_weight(value: str) -> tuple[str, int]:
    """Parse CACHE_SALT=WEIGHT server configuration."""
    cache_salt, separator, raw_weight = value.partition("=")
    if not separator:
        raise argparse.ArgumentTypeError("weight must use CACHE_SALT=WEIGHT")
    try:
        weight = int(raw_weight)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("weight must be an integer") from exc
    return cache_salt, weight


def main() -> int:
    """Parse command-line arguments and run one harness role."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)

    server_parser = subparsers.add_parser("server")
    server_parser.add_argument("--port", type=int, default=19090)
    server_parser.add_argument("--clients", type=int, default=2)
    server_parser.add_argument("--default-sched-weight", type=int, default=100)
    server_parser.add_argument(
        "--weight",
        action="append",
        type=_parse_weight,
        default=[],
        metavar="CACHE_SALT=WEIGHT",
    )

    client_parser = subparsers.add_parser("client")
    client_parser.add_argument("host")
    client_parser.add_argument("--port", type=int, default=19090)
    client_parser.add_argument("--tasks", type=int, default=100)
    client_parser.add_argument("--cache-salt", required=True)

    args = parser.parse_args()
    if args.mode == "server":
        return _run_server(
            args.port,
            args.clients,
            args.default_sched_weight,
            dict(args.weight),
        )
    return _run_client(args.host, args.port, args.tasks, args.cache_salt)


if __name__ == "__main__":
    raise SystemExit(main())
