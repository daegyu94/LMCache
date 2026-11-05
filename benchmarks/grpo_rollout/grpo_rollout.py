#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Async GRPO-style rollout against several vLLM engines sharing one LMCache server.

`prompts` prompts, each with `group` samples. An async rollout manager keeps at
most `inflight` requests outstanding; a prompt's whole group is admitted at once
and its samples are dealt round-robin over the engines (load balancer).

Prompt lengths:
  --len-dist uniform:A:B   uniform in [A, B] tokens (long-context RL)
A shared system prefix of `--shared-prefix` tokens starts every prompt.
Token content is random: it does not affect prefill or KV cost.

Reports wall time, generated tokens/s, TTFT, and per engine the prompt tokens
vLLM computed vs. served from its prefix cache or from LMCache.
"""

# Standard
import argparse
import json
import random
import re
import threading
import time
import urllib.request


def lengths(dist: str, n: int, rng: random.Random) -> list[int]:
    _, a, b = dist.split(":")
    return [rng.randint(int(a), int(b)) for _ in range(n)]


def tokens(n: int, rng: random.Random) -> list[int]:
    return [rng.randrange(1000, 30000) for _ in range(n)]


def complete(url: str, model: str, toks: list[int], max_tokens: int) -> dict:
    body = json.dumps(
        {
            "model": model,
            "prompt": toks,
            "max_tokens": max_tokens,
            "temperature": 1.0,
            "ignore_eos": True,
            "stream": True,
        }
    ).encode()
    req = urllib.request.Request(
        f"{url}/v1/completions", body, {"Content-Type": "application/json"}
    )
    t0 = time.perf_counter()
    ttft = None
    with urllib.request.urlopen(req, timeout=1800) as r:
        for line in r:
            if ttft is None and line.startswith(b"data:"):
                ttft = time.perf_counter() - t0
    return {"ttft": ttft, "latency": time.perf_counter() - t0}


def vllm_tokens(url: str) -> dict[str, float]:
    text = urllib.request.urlopen(f"{url}/metrics", timeout=10).read().decode()
    out: dict[str, float] = {}
    for line in text.splitlines():
        m = re.match(
            r'vllm:prompt_tokens_by_source_total\{.*source="(\w+)".*\} (\S+)', line
        )
        if m:
            out[m[1]] = out.get(m[1], 0) + float(m[2])
        m = re.match(r"vllm:generation_tokens_total\{.*\} (\S+)", line)
        if m:
            out["generated"] = out.get("generated", 0) + float(m[1])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--urls", required=True, help="comma-separated engine URLs")
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", type=int, default=32)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--inflight", type=int, default=64)
    ap.add_argument("--len-dist", default="uniform:8192:15360")
    ap.add_argument("--shared-prefix", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument(
        "--route",
        choices=["spread", "group"],
        default="spread",
        help="spread: siblings round-robin over engines (verl/AReaL-style); "
        "group: a whole group on one engine (OpenRLHF/ROLL-style)",
    )
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    urls = a.urls.split(",")
    rng = random.Random(a.seed)
    system = tokens(a.shared_prefix, rng)
    prompts = [
        system + tokens(max(1, n - a.shared_prefix), rng)
        for n in lengths(a.len_dist, a.prompts, rng)
    ]

    before = [vllm_tokens(u) for u in urls]
    slots = threading.Semaphore(a.inflight)
    results: list[dict] = []
    lock = threading.Lock()

    def run(url: str, toks: list[int]) -> None:
        try:
            r = complete(url, a.model, toks, a.max_tokens)
            with lock:
                results.append(r)
        finally:
            slots.release()

    t0 = time.perf_counter()
    threads = []
    k = 0
    for p_idx, toks in enumerate(prompts):
        for _ in range(a.group):  # the group is admitted together
            slots.acquire()
            url = urls[(k if a.route == "spread" else p_idx) % len(urls)]
            th = threading.Thread(target=run, args=(url, toks))
            th.start()
            threads.append(th)
            k += 1
    for th in threads:
        th.join()
    wall = time.perf_counter() - t0
    after = [vllm_tokens(u) for u in urls]
    per_engine = [
        {s: after[i].get(s, 0) - before[i].get(s, 0) for s in after[i]}
        for i in range(len(urls))
    ]
    tot: dict[str, float] = {}
    for e in per_engine:
        for s, v in e.items():
            tot[s] = tot.get(s, 0) + v
    ttfts = sorted(r["ttft"] * 1e3 for r in results)
    res = {
        "wall_s": round(wall, 2),
        "gen_tok_s": round(tot.get("generated", 0) / wall, 1),
        "ttft_mean_ms": round(sum(ttfts) / len(ttfts)),
        "ttft_p50_ms": round(ttfts[len(ttfts) // 2]),
        "ttft_p90_ms": round(ttfts[int(0.9 * len(ttfts))]),
        "requests": len(results),
        "unique_prompt_tokens": sum(len(p) for p in prompts),
        "tokens": {s: int(v) for s, v in tot.items()},
        "per_engine": [{s: int(v) for s, v in e.items()} for e in per_engine],
    }
    print(json.dumps(res))
    if a.out:
        json.dump(res, open(a.out, "w"))


if __name__ == "__main__":
    main()
