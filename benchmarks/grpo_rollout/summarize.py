# SPDX-License-Identifier: Apache-2.0
"""Print one line per run of run_ab.sh: summarize.py RESULTS_DIR"""

# Standard
import glob
import json
import os
import sys

for path in sorted(glob.glob(os.path.join(sys.argv[1], "*.json"))):
    if path.endswith(".status.json"):
        continue
    tag = os.path.basename(path)[: -len(".json")]
    d = json.load(open(path))
    t = d["tokens"]
    lease = ""
    status = path[: -len(".json")] + ".status.json"
    if os.path.exists(status):
        s = json.load(open(status))
        if "prefill_lease_waits" in s:
            lease = " lease waits/served/timeouts=%d/%d/%d" % (
                s["prefill_lease_waits"],
                s["prefill_lease_served"],
                s["prefill_lease_timeouts"],
            )
    print(
        "%-20s wall=%7.1fs gen=%6.0ftok/s ttft_mean=%6dms p90=%6dms "
        "computed=%.2fx lmcache=%d%s"
        % (
            tag,
            d["wall_s"],
            d["gen_tok_s"],
            d["ttft_mean_ms"],
            d["ttft_p90_ms"],
            t.get("local_compute", 0) / d["unique_prompt_tokens"],
            t.get("external_kv_transfer", 0),
            lease,
        )
    )
