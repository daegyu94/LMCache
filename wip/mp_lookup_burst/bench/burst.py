# SPDX-License-Identifier: Apache-2.0
# Burst client: send N = P prompts x S samples at once, measure TTFT distribution.
import argparse, asyncio, json, random, time
import aiohttp

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8100)
ap.add_argument("--prompts", type=int, required=True)
ap.add_argument("--samples", type=int, default=16)
ap.add_argument("--plen", type=int, default=1024)
ap.add_argument("--max-tokens", type=int, default=16)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--model", required=True)
ap.add_argument("--tag", default="")
a = ap.parse_args()

rng = random.Random(a.seed)
prompts = [[rng.randint(1000, 100000) for _ in range(a.plen)] for _ in range(a.prompts)]
url = f"http://localhost:{a.port}/v1/completions"


async def one(sess, p, t0):
    body = {"model": a.model, "prompt": p, "max_tokens": a.max_tokens,
            "ignore_eos": True, "stream": True, "temperature": 1.0}
    ttft = None
    async with sess.post(url, json=body) as r:
        async for line in r.content:
            if ttft is None and line.startswith(b"data: {"):
                ttft = time.perf_counter() - t0
    return ttft, time.perf_counter() - t0


async def main():
    conn = aiohttp.TCPConnector(limit=0)
    global wall0
    async with aiohttp.ClientSession(connector=conn, timeout=aiohttp.ClientTimeout(total=3600)) as s:
        t0 = time.perf_counter(); wall0 = time.time()
        res = await asyncio.gather(*[one(s, p, t0) for p in prompts for _ in range(a.samples)])
    tt = sorted(r[0] for r in res)
    n = len(tt)
    out = {"tag": a.tag, "N": n, "first_tok": round(tt[0], 3), "ttft_256th": round(tt[min(255, n - 1)], 3),
           "ttft_p50": round(tt[n // 2], 3), "makespan": round(max(r[1] for r in res), 3), "t_start": round(wall0, 3), "t_end": round(time.time(), 3)}
    print(json.dumps(out), flush=True)


asyncio.run(main())
