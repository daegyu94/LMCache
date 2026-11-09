# SPDX-License-Identifier: Apache-2.0
# Send a PING every 0.5 s; print "wall_time rtt_s" (rtt = -1 on 30 s timeout).
import sys, time, zmq
from lmcache.v1.client.transport.factory import RequestClientFactory
c = RequestClientFactory.create(sys.argv[1], context=zmq.Context.instance())
while True:
    t = time.time(); t0 = time.perf_counter()
    try:
        c.ping(None).result(timeout=30); rtt = time.perf_counter() - t0
    except Exception:
        rtt = -1.0
    print(f"{t:.3f} {rtt:.4f}", flush=True)
    time.sleep(max(0.0, 0.5 - (time.perf_counter() - t0)))
