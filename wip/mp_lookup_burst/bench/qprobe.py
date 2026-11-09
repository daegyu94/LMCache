# SPDX-License-Identifier: Apache-2.0
# Scrape vLLM's waiting-queue gauge every 0.5 s; print "wall_time waiting".
import re, sys, time, urllib.request
url = sys.argv[1]
pat = re.compile(r'^vllm:num_requests_waiting\{[^}]*\}\s+([0-9.e+]+)', re.M)
while True:
    t = time.time()
    try:
        body = urllib.request.urlopen(url, timeout=5).read().decode()
        m = pat.search(body); w = float(m.group(1)) if m else -1
    except Exception:
        w = -1
    print(f"{t:.3f} {w:.0f}", flush=True)
    time.sleep(0.5)
