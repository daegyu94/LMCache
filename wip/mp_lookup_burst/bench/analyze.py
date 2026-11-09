# SPDX-License-Identifier: Apache-2.0
"""Summarize run_trace.sh output directories.

usage: python analyze.py <runs/trace_TAG>...

Per burst prints: TTFT stats, degraded-mode entries, peak waiting queue,
LOOKUP submit rate, client-submit -> server-handle lag (FIFO-matched),
server LOOKUP handler time, QUERY_PREFETCH_STATUS count, scheduler step cost,
heartbeat PING round trips (scheduler, instance None) and external PING max.
"""

import glob
import json
import os
import sys


def _lines(pattern):
    out = []
    for f in glob.glob(pattern):
        with open(f) as fh:
            out.extend(line.split() for line in fh if line.strip())
    return out


def _peak_per_sec(ts):
    ts = sorted(ts)
    best = j = 0
    for i in range(len(ts)):
        while ts[i] - ts[j] > 1:
            j += 1
        best = max(best, i - j + 1)
    return best


def summarize(d):
    tag = os.path.basename(d.rstrip("/"))
    bursts = [json.loads(x) for x in open(os.path.join(d, "bursts.jsonl")) if x.startswith("{")]
    cli = _lines(os.path.join(d, "cli.*.log"))
    srv = _lines(os.path.join(d, "srv.*.log"))
    deg = open(os.path.join(d, "degraded.txt")).read().strip() if os.path.exists(os.path.join(d, "degraded.txt")) else "?"
    cap = open(os.path.join(d, "cap.txt")).read().strip() if os.path.exists(os.path.join(d, "cap.txt")) else ""
    sub = sorted(float(x[1]) for x in cli if x[0] == "SUBMIT")
    lk = sorted((float(x[1]), float(x[2])) for x in srv if x[0] == "SRV_LOOKUP")
    qs = [float(x[1]) for x in srv if x[0] == "SRV_QUERY"]
    steps = [(float(x[1]), float(x[2]), int(x[3]), float(x[4])) for x in cli if x[0] == "STEP"]
    hb = [(float(x[1]), float(x[2]), x[3]) for x in cli if x[0] == "HB_PING" and x[4] == "None"]
    queue = [tuple(map(float, x)) for x in _lines(os.path.join(d, "queue.txt")) if len(x) == 2]
    ext = [tuple(map(float, x)) for x in _lines(os.path.join(d, "ping_external.txt")) if len(x) == 2]
    print(f"== {tag}  {cap}  degraded-mode entries (whole run): {deg}")
    for b in bursts:
        t0, t1 = b["t_start"], b["t_end"]
        win = lambda xs, k=lambda v: v: [v for v in xs if t0 <= k(v) <= t1]  # noqa: E731
        s, l = win(sub), win(lk, lambda v: v[0])
        n = min(len(s), len(l))
        lag = max((l[i][0] - s[i] for i in range(n)), default=0.0)
        st_ = win(steps, lambda v: v[0])
        h = win(hb, lambda v: v[0])
        q = win(queue, lambda v: v[0])
        e = [v[1] for v in win(ext, lambda v: v[0])]
        print(
            f"  N={b['N']}: first={b['first_tok']} 256th={b['ttft_256th']} p50={b['ttft_p50']} "
            f"makespan={b['makespan']} | waiting max={int(max((v[1] for v in q), default=-1))}"
        )
        if s or l:
            print(
                f"    LOOKUP submitted={len(s)} peak/s={_peak_per_sec(s)} | submit->server lag max={lag:.2f}s "
                f"| server handler sum={sum(v[1] for v in l):.2f}s | QUERY={len(win(qs))}"
            )
        if st_:
            print(
                f"    schedule() sum={sum(v[1] for v in st_):.1f}s (connector polls {sum(v[3] for v in st_):.1f}s, "
                f"{sum(v[2] for v in st_)} polls) longest step={max(v[1] for v in st_):.2f}s"
            )
        if h:
            print("    heartbeat PING rtt (s):", [round(v[1], 2) for v in h], "ok:", [v[2] for v in h])
        if e:
            to = sum(1 for v in e if v < 0)
            print(f"    external PING max={max(v for v in e if v >= 0):.2f}s timeouts={to}")


if __name__ == "__main__":
    for d in sys.argv[1:]:
        summarize(d)
