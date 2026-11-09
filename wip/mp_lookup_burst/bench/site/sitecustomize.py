# SPDX-License-Identifier: Apache-2.0
# Benchmark-only instrumentation. Never shipped.
#   LMC_CAP=none|<int>     override the scheduler adapter's lookup bound
#   LMC_TRACE=<prefix>     append timestamped events to <prefix>.<pid>.log
#   LMC_TRACE_SIDE=client|server   which hooks to install
import functools
import os
import threading
import time

_cap = os.environ.get("LMC_CAP")
_trace = os.environ.get("LMC_TRACE")
_side = os.environ.get("LMC_TRACE_SIDE", "")

_lock = threading.Lock()
_fh = None


def _log(*fields):
    global _fh
    with _lock:
        if _fh is None:
            _fh = open(f"{_trace}.{os.getpid()}.log", "a", buffering=1)
        _fh.write(" ".join(str(f) for f in fields) + "\n")


if _cap is not None:
    from lmcache.integration.vllm import vllm_multi_process_adapter as _m

    _orig_init = _m.LMCacheMPSchedulerAdapter.__init__

    def _init(self, *a, **kw):
        kw["max_pending_lookups"] = None if _cap == "none" else int(_cap)
        import sys

        print(f"LMC_CAP override: max_pending_lookups={kw['max_pending_lookups']}",
              file=sys.stderr, flush=True)
        _orig_init(self, *a, **kw)

    _m.LMCacheMPSchedulerAdapter.__init__ = _init

if _trace and _side == "client":
    from lmcache.v1.client import heartbeat as _hb
    from lmcache.integration.vllm import vllm_multi_process_adapter as _m2

    _orig_ping = _hb.send_ping

    def _send_ping(req_client, timeout, instance_id=None):
        t0 = time.time()
        ok = _orig_ping(req_client, timeout, instance_id)
        _log("HB_PING", f"{t0:.3f}", f"{time.time() - t0:.3f}", ok, instance_id)
        return ok

    _hb.send_ping = _send_ping

    _orig_submit = _m2.LMCacheMPSchedulerAdapter.maybe_submit_lookup_request

    @functools.wraps(_orig_submit)
    def _submit(self, request_id, *a, **kw):
        sent = _orig_submit(self, request_id, *a, **kw)
        if sent:
            _log("SUBMIT", f"{time.time():.3f}")
        return sent

    _m2.LMCacheMPSchedulerAdapter.maybe_submit_lookup_request = _submit

    # Per-step cost: whole schedule() and the connector polls inside it.
    from vllm.v1.core.sched import scheduler as _vs
    from lmcache.integration.vllm import lmcache_mp_connector as _mc

    _poll = {"n": 0, "t": 0.0}
    _orig_gnm = _mc.LMCacheMPConnector.get_num_new_matched_tokens

    @functools.wraps(_orig_gnm)
    def _gnm(self, *a, **kw):
        t0 = time.perf_counter()
        try:
            return _orig_gnm(self, *a, **kw)
        finally:
            _poll["n"] += 1
            _poll["t"] += time.perf_counter() - t0

    _mc.LMCacheMPConnector.get_num_new_matched_tokens = _gnm

    _orig_sched = _vs.Scheduler.schedule

    @functools.wraps(_orig_sched)
    def _sched(self, *a, **kw):
        _poll["n"] = 0
        _poll["t"] = 0.0
        t0 = time.perf_counter()
        r = _orig_sched(self, *a, **kw)
        dt = time.perf_counter() - t0
        if dt > 0.02 or _poll["n"] > 300:
            _log("STEP", f"{time.time():.3f}", f"{dt:.4f}", _poll["n"], f"{_poll['t']:.4f}")
        return r

    _vs.Scheduler.schedule = _sched

if _trace and _side == "server":
    from lmcache.v1.multiprocess.modules import lookup as _lk
    from lmcache.v1.multiprocess.modules import management as _mg

    _orig_srv_ping = _mg.ManagementModule.ping

    @functools.wraps(_orig_srv_ping)
    def _srv_ping(self, instance_id):
        _log("SRV_PING", f"{time.time():.3f}", instance_id)
        return _orig_srv_ping(self, instance_id)

    _mg.ManagementModule.ping = _srv_ping

    _orig_srv_lookup = _lk.LookupModule.lookup

    @functools.wraps(_orig_srv_lookup)
    def _srv_lookup(self, key, tp_size):
        t0 = time.time()
        try:
            return _orig_srv_lookup(self, key, tp_size)
        finally:
            _log("SRV_LOOKUP", f"{t0:.3f}", f"{time.time() - t0:.4f}")

    _lk.LookupModule.lookup = _srv_lookup

    _orig_srv_query = _lk.LookupModule.query_prefetch_status

    @functools.wraps(_orig_srv_query)
    def _srv_query(self, *a, **kw):
        r = _orig_srv_query(self, *a, **kw)
        _log("SRV_QUERY", f"{time.time():.3f}", r is None)
        return r

    _lk.LookupModule.query_prefetch_status = _srv_query
