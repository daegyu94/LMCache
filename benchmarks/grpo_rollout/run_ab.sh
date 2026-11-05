#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# GRPO rollout A/B for prefill leases + sibling-load coalescing.
#
# One LMCache MP server (L1 only) shared by NENG vLLM engines. Each arm starts
# a fresh stack from its own LMCache checkout, runs grpo_rollout.py, saves the
# result JSON and the server /status (lease counters), then stops everything.
#
# usage: run_ab.sh TAG CODE_DIR "SERVER_ARGS" [grpo_rollout.py args...]
#   CODE_DIR     LMCache checkout to run (dev for the baseline, the branch for
#                the change); it must have its native extensions built
#   SERVER_ARGS  extra `lmcache server` args, e.g. "--prefill-lease-timeout-seconds 10"
#
# env (defaults in brackets):
#   PY        python with lmcache's deps installed           [python]
#   VLLM      vllm executable                                [vllm]
#   MODEL     model path or HF id                            [Qwen/Qwen3-8B]
#   NENG      number of vLLM engines                         [2]
#   PER_GPU   1 = engine e on GPU e-1; 0 = all on GPU 0       [1]
#   GPU_UTIL  --gpu-memory-utilization per engine            [0.85 per GPU, 0.3 shared]
#   L1        LMCache L1 size in GB                          [40]
#   OUT       output directory                               [./results]
#   MAX_LEN   vLLM --max-model-len                           [16384]
set -u
TAG=$1; CODE=$2; SA=$3; shift 3
PY=${PY:-python}; VLLM=${VLLM:-vllm}; MODEL=${MODEL:-Qwen/Qwen3-8B}
NENG=${NENG:-2}; PER_GPU=${PER_GPU:-1}; L1=${L1:-40}; OUT=${OUT:-./results}; MAX_LEN=${MAX_LEN:-16384}
if [ "$PER_GPU" = 1 ]; then GPU_UTIL=${GPU_UTIL:-0.85}; else GPU_UTIL=${GPU_UTIL:-0.3}; fi
MP=${MP_PORT:-6900}; HTTP=${HTTP_PORT:-7900}; BASE_PORT=${BASE_PORT:-8500}
HERE=$(cd "$(dirname "$0")" && pwd); mkdir -p "$OUT"
export PYTHONUNBUFFERED=1

PYTHONPATH=$CODE setsid $PY -c 'import sys; sys.argv[0]="lmcache"; from lmcache.cli.main import main; sys.exit(main())' server \
  --host 127.0.0.1 --port $MP --http-port $HTTP --l1-size-gb $L1 --eviction-policy LRU --chunk-size 256 $SA \
  > "$OUT/$TAG.lmcache.log" 2>&1 &
SERVER=$!
sleep 10
URLS=""; ENGINES=""
for e in $(seq "$NENG"); do
  VP=$((BASE_PORT + e)); GPUENV=""
  [ "$PER_GPU" = 1 ] && GPUENV="CUDA_VISIBLE_DEVICES=$((e - 1))"
  env $GPUENV PYTHONPATH=$CODE setsid $VLLM serve "$MODEL" --port $VP --max-model-len $MAX_LEN \
    --gpu-memory-utilization $GPU_UTIL \
    --kv-transfer-config "{\"kv_connector\":\"LMCacheMPConnector\",\"kv_role\":\"kv_both\",\"kv_load_failure_policy\":\"recompute\",\"kv_connector_extra_config\":{\"lmcache.mp.port\":$MP}}" \
    > "$OUT/$TAG.vllm$e.log" 2>&1 &
  ENGINES="$ENGINES $!"
  # Start engines one at a time when they share a GPU: concurrent memory
  # profiling on one device misreads the free memory.
  if [ "$PER_GPU" != 1 ] || [ "$e" = "$NENG" ]; then
    for p in $(seq $((BASE_PORT + 1)) $VP); do
      for i in $(seq 400); do curl -sf http://127.0.0.1:$p/v1/models >/dev/null && break; sleep 3; done
    done
  fi
  URLS="$URLS${URLS:+,}http://127.0.0.1:$VP"
done

$PY "$HERE/grpo_rollout.py" --urls "$URLS" --model "$MODEL" --out "$OUT/$TAG.json" "$@" 2>&1 | tee "$OUT/$TAG.out"
curl -s http://127.0.0.1:$HTTP/status > "$OUT/$TAG.status.json"

kill $ENGINES $SERVER 2>/dev/null
for i in $(seq 90); do ss -ltn | grep -qE ":($MP|$((BASE_PORT + 1))) " || break; sleep 1; done
kill -9 $ENGINES $SERVER 2>/dev/null
true
