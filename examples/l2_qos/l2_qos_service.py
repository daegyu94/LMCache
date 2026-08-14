# SPDX-License-Identifier: Apache-2.0
"""Minimal Docker integration harness for shared-L2 QoS."""

# Future
from __future__ import annotations

# Standard
from collections import Counter
import argparse
import json
import socket
import socketserver
import sys
import threading
import time

# Third Party
from qos import L2QoSDispatcher, QosProfile


class _ServerState:
    """Shared dispatcher and admission accounting for the test server."""

    def __init__(self, expected_clients: int) -> None:
        self.dispatcher = L2QoSDispatcher(
            quantum_bytes=1,
            max_inflight_tasks=1,
        )
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
        """Wait until every test client has submitted its profile."""
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

    def record_admission(self, domain_id: str) -> int:
        """Record one admitted L2 operation and return a task result."""
        self.work_ready.wait()
        with self.lock:
            self.admissions.append(domain_id)
        return 1

    def client_completed(self) -> None:
        """Stop the server after all expected clients drain."""
        with self.lock:
            self.completed_clients += 1
            should_stop = self.completed_clients >= self.expected_clients
        if should_stop and self.server is not None:
            threading.Thread(target=self.server.shutdown, daemon=True).start()


class _QoSRequestHandler(socketserver.StreamRequestHandler):
    """Receive one client profile and a batch of synthetic L2 operations."""

    state: _ServerState

    def handle(self) -> None:
        hello = self._read_message()
        if hello.get("kind") != "hello":
            raise ValueError("first message must be hello")
        profile = QosProfile(
            domain_id=str(hello["domain"]),
            weight=int(hello["weight"]),
            source=str(hello["source"]),
        )

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
                    profile=profile,
                    operation="load",
                    cost_bytes=max(1, int(message.get("cost", 1))),
                    action=lambda domain=profile.domain_id: self.state.record_admission(
                        domain
                    ),
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
        line = self.rfile.readline()
        if not line:
            raise ConnectionError("client disconnected")
        return json.loads(line)

    def _write_message(self, message: dict) -> None:
        payload = json.dumps(message, separators=(",", ":")).encode() + b"\n"
        self.wfile.write(payload)
        self.wfile.flush()


class _QoSServer(socketserver.ThreadingTCPServer):
    """Threading TCP server carrying a shared QoS dispatcher."""

    allow_reuse_address = True
    daemon_threads = True


def _run_server(port: int, expected_clients: int) -> int:
    state = _ServerState(expected_clients)

    class Handler(_QoSRequestHandler):
        pass

    Handler.state = state
    server = _QoSServer(("0.0.0.0", port), Handler)
    state.server = server
    print(json.dumps({"kind": "ready", "port": port}), flush=True)
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


def _run_client(host: str, port: int, tasks: int) -> int:
    profile = QosProfile.from_environment()
    print(
        json.dumps(
            {
                "kind": "client-profile",
                "domain": profile.domain_id,
                "weight": profile.weight,
                "source": profile.source,
            },
            separators=(",", ":"),
        ),
        flush=True,
    )
    if profile.source != "cgroup":
        print("client did not discover a cgroup profile", file=sys.stderr)
        return 2

    completed: set[int] = set()
    with socket.create_connection((host, port), timeout=10) as connection:
        stream = connection.makefile("rwb")

        def send(message: dict) -> None:
            stream.write(json.dumps(message, separators=(",", ":")).encode())
            stream.write(b"\n")
            stream.flush()

        send(
            {
                "kind": "hello",
                "domain": profile.domain_id,
                "weight": profile.weight,
                "source": profile.source,
            }
        )
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


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)

    server_parser = subparsers.add_parser("server")
    server_parser.add_argument("--port", type=int, default=19090)
    server_parser.add_argument("--clients", type=int, default=2)

    client_parser = subparsers.add_parser("client")
    client_parser.add_argument("host")
    client_parser.add_argument("--port", type=int, default=19090)
    client_parser.add_argument("--tasks", type=int, default=100)

    args = parser.parse_args()
    if args.mode == "server":
        return _run_server(args.port, args.clients)
    return _run_client(args.host, args.port, args.tasks)


if __name__ == "__main__":
    raise SystemExit(main())
