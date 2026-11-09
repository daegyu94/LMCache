#!/bin/bash
# Instrumented LMCache MP + vLLM burst run.
#
# usage: run_trace.sh <tag> <cap> <prompts>...
#   cap      none | <int>   lookup bound forced in the vLLM process (sitecustomize)
#            keep        do not override (use what the connector passes)
#            vanilla     run vLLM without LMCache (no server, no kv connector)
#   prompts  one burst per value; each burst sends prompts x SAMPLES requests
#
# env (defaults in brackets):
#   REPO      LMCache checkout to import            [git toplevel of this script]
#   VENV      venv with vLLM + built LMCache ext    [required]
#   OUT_DIR   where trace_<tag>/ is written         [./runs]
#   MODEL [Qwen/Qwen3-0.6B]  GPU [0]  MAX_NUM_SEQS [256]  SAMPLES [16]
#   PLEN [1024]  L1_SIZE [40]  VLLM_PORT [8100]  MP_PORT [6567]
set -u
TAG=$1; CAP=$2; shift 2
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=${REPO:-$(git -C "$HERE" rev-parse --show-toplevel)}
: "${VENV:?set VENV to the venv that has vLLM and LMCache deps}"
OUT_DIR=${OUT_DIR:-$PWD/runs}
MODEL=${MODEL:-Qwen/Qwen3-0.6B}; GPU=${GPU:-0}; MAX_NUM_SEQS=${MAX_NUM_SEQS:-256}
SAMPLES=${SAMPLES:-16}; PLEN=${PLEN:-1024}; L1_SIZE=${L1_SIZE:-40}
VLLM_PORT=${VLLM_PORT:-8100}; MP_PORT=${MP_PORT:-6567}
VPY=$VENV/bin
LOG=$OUT_DIR/trace_$TAG; rm -rf "$LOG"; mkdir -p "$LOG"
export CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$HERE/site:$REPO
PIDS=()
# Kill only the process groups this script started (setsid below).
cleanup() { for p in "${PIDS[@]}"; do kill -- -"$p" 2>/dev/null; done; wait 2>/dev/null; }
trap cleanup EXIT

KV=()
if [ "$CAP" != vanilla ]; then
  LMC_TRACE=$LOG/srv LMC_TRACE_SIDE=server setsid "$VPY/python" -c \
    'import sys; from lmcache.cli.main import main; sys.argv=["lmcache"]+sys.argv[1:]; sys.exit(main())' \
    server --port "$MP_PORT" --l1-size "$L1_SIZE" --eviction-policy LRU --chunk-size 256 \
    > "$LOG/lmcache.log" 2>&1 &
  PIDS+=($!); sleep 10
  KV=(--kv-transfer-config "{\"kv_connector\":\"LMCacheMPConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"lmcache.mp.port\":$MP_PORT}}")
fi
CAPENV=()
case "$CAP" in vanilla|keep) ;; *) CAPENV=(LMC_CAP=$CAP) ;; esac
env "${CAPENV[@]}" LMC_TRACE=$LOG/cli LMC_TRACE_SIDE=client setsid "$VPY/vllm" serve "$MODEL" \
  --port "$VLLM_PORT" --max-model-len 4096 --max-num-seqs "$MAX_NUM_SEQS" \
  --gpu-memory-utilization 0.8 "${KV[@]}" > "$LOG/vllm.log" 2>&1 &
PIDS+=($!)
for i in $(seq 400); do curl -sf "localhost:$VLLM_PORT/health" >/dev/null && break; sleep 2; done
curl -sf "localhost:$VLLM_PORT/health" >/dev/null || { echo "vllm failed"; tail -40 "$LOG/vllm.log"; exit 1; }
setsid "$VPY/python" "$HERE/qprobe.py" "http://localhost:$VLLM_PORT/metrics" > "$LOG/queue.txt" 2>&1 &
PIDS+=($!)
if [ "$CAP" != vanilla ]; then
  setsid "$VPY/python" "$HERE/pingprobe.py" "tcp://localhost:$MP_PORT" > "$LOG/ping_external.txt" 2>&1 &
  PIDS+=($!)
fi
"$VPY/python" "$HERE/burst.py" --port "$VLLM_PORT" --model "$MODEL" --prompts 4 --samples 4 --seed 999 --tag warm >/dev/null
seed=200
for P in "$@"; do
  "$VPY/python" "$HERE/burst.py" --port "$VLLM_PORT" --model "$MODEL" --prompts "$P" \
    --samples "$SAMPLES" --plen "$PLEN" --seed $seed --tag "$TAG" >> "$LOG/bursts.jsonl"
  seed=$((seed+1)); sleep 5
done
grep -c "entering degraded" "$LOG/vllm.log" > "$LOG/degraded.txt"
grep -o "max_pending_lookups=[A-Za-z0-9]*" "$LOG/vllm.log" | head -1 > "$LOG/cap.txt"
