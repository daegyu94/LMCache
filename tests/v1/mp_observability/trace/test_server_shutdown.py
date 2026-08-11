# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import Any
from unittest import mock
import multiprocessing as mp
import os
import signal

# First Party
from lmcache.v1.distributed.config import (
    EvictionConfig,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    StorageManagerConfig,
)
from lmcache.v1.mp_observability.config import ObservabilityConfig
from lmcache.v1.mp_observability.event_bus import EventBus, EventBusConfig
from lmcache.v1.mp_observability.trace import codecs
from lmcache.v1.mp_observability.trace.reader import TraceReader
from lmcache.v1.multiprocess import server as server_module
from lmcache.v1.multiprocess.config import MPServerConfig


class _FakeServerContext:
    def __init__(self, **_kwargs: Any) -> None:
        pass

    def close(self) -> None:
        pass


class _FakeMessageQueueServer:
    def __init__(self, ready: Any) -> None:
        self._ready = ready

    def start(self) -> None:
        self._ready.set()

    def close(self) -> None:
        pass


def _run_server_until_sigterm(trace_path: str, ready: Any) -> None:
    def init_test_observability(
        _config: ObservabilityConfig,
        *,
        start_prometheus_http_server: bool,
    ) -> EventBus:
        del start_prometheus_http_server
        bus = EventBus(EventBusConfig(enabled=True))
        bus.start()
        return bus

    fake_mq_server = _FakeMessageQueueServer(ready)
    storage_config = StorageManagerConfig(
        l1_manager_config=L1ManagerConfig(
            memory_config=L1MemoryManagerConfig(
                size_in_bytes=4096,
                use_lazy=True,
            ),
        ),
        eviction_config=EvictionConfig(eviction_policy="noop"),
    )
    observability_config = ObservabilityConfig(
        enabled=True,
        metrics_enabled=False,
        logging_enabled=False,
        trace_level="l2",
        trace_output=trace_path,
    )

    with (
        mock.patch.object(
            server_module,
            "init_observability",
            init_test_observability,
        ),
        mock.patch.object(
            server_module,
            "MPCacheServerContext",
            _FakeServerContext,
        ),
        mock.patch.object(
            server_module,
            "_build_modules",
            return_value=[],
        ),
        mock.patch.object(
            server_module,
            "MessageQueueServer",
            return_value=fake_mq_server,
        ),
        mock.patch.object(server_module, "torch_dev", object()),
        mock.patch.object(server_module, "torch_device_type", "test"),
    ):
        server_module.run_cache_server(
            mp_config=MPServerConfig(supported_transfer_mode="lmcache_driven"),
            storage_manager_config=storage_config,
            obs_config=observability_config,
            start_prometheus_http_server=False,
        )


def test_sigterm_finalizes_l2_trace(tmp_path) -> None:
    trace_path = tmp_path / "sigterm.lct"
    context = mp.get_context("spawn")
    ready = context.Event()
    process = context.Process(
        target=_run_server_until_sigterm,
        args=(str(trace_path), ready),
    )
    process.start()
    try:
        assert ready.wait(timeout=10)
        assert process.pid is not None
        os.kill(process.pid, signal.SIGTERM)
        process.join(timeout=10)
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.kill()
            process.join(timeout=5)

    with TraceReader(str(trace_path)) as reader:
        records = list(reader.records())

    assert records[-1].qualname == "l2.trace.end"
    footer = codecs.decode_args(records[-1].args)
    assert footer == {
        "recorder_dropped_count": 0,
        "event_bus_dropped_count": 0,
    }
